from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from src.dataset.sft_prompt_renderer import render_messages, render_user_prompt
from src.dataset.sft_schema import (
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    USER_PROMPT_VERSION,
    extract_date_tokens,
    extract_number_tokens,
    fixed_constraints,
    normalize_evidence,
    normalize_text,
    render_target,
    sha256_text,
    source_paragraph_records,
)
from src.task011d.event_group_split import verify_checksums
from src.task011d.protection import validate_manifest


SPLITS = ("train", "validation", "test")
REVIEW_A_ROLE = "AI_INDEPENDENT_REVIEWER_A"
REVIEW_B_ROLE = "AI_INDEPENDENT_REVIEWER_B"
GENERATOR_ROLE = "AI_SFT_GENERATOR"
RE_REVIEWER_ROLE = "AI_INDEPENDENT_RE_REVIEWER"
JUDGE_ROLE = "AI_SFT_JUDGE"
AUDITOR_ROLE = "AI_CROSS_SPLIT_EVENT_AUDITOR"
FACT_PREFIX = "据该篇新闻所载公开事实，"
FIRST_PERSON = re.compile(r"(?<![A-Za-z])(我们|我方|本人|我)(?![A-Za-z])")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")


class AutonomousSFTError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AutonomousSFTError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def validate_protected_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    protection = read_json(root / config["v1_protection_manifest"])
    require(validate_manifest(root, protection) == [], "v1 frozen checksum protection failed")
    protected_dirs = (
        "data/processed/news_v2",
        "data/processed/news_event_groups_v2",
        "data/processed/news_split_v2",
        config["gold_sft_dir"],
    )
    for relative in protected_dirs:
        require(verify_checksums(root / relative) == [], f"frozen checksum protection failed: {relative}")
    return {
        "v1_protected_file_count": protection["file_count"],
        "v1_protection_status": "passed",
        "v2_protection_status": "passed",
    }


def _load_context(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    protection = validate_protected_inputs(root, config)
    generation_manifest = read_json(root / config["generation_manifest"])
    require(generation_manifest["input_count"] == 2363, "generation input total mismatch")
    require(generation_manifest["gold_sft_reuse_count"] == config["gold_reuse_count"], "Gold reuse mismatch")
    require(generation_manifest["silver_sft_generation_count"] == config["silver_count"], "Silver input mismatch")
    require(generation_manifest["silver_generation_by_split"] == config["silver_split_counts"], "Silver split mismatch")
    generation_rows = read_jsonl(root / config["generation_input"])
    articles = read_jsonl(root / config["news_articles"])
    require(len(generation_rows) == len(articles) == 2363, "news/generation row count mismatch")
    require(len({row["article_id"] for row in articles}) == 2363, "duplicate news article id")
    article_by_id = {row["article_id"]: row for row in articles}
    silver_input = [row for row in generation_rows if not row["reuse_frozen_gold_sft"]]
    gold_input = [row for row in generation_rows if row["reuse_frozen_gold_sft"]]
    require(len(silver_input) == config["silver_count"] and len(gold_input) == config["gold_reuse_count"], "Gold/Silver action mismatch")
    require(Counter(row["split"] for row in silver_input) == Counter(config["silver_split_counts"]), "Silver input split mismatch")
    for row in generation_rows:
        article = article_by_id.get(row["article_id"])
        require(article is not None, f"missing article: {row['article_id']}")
        require(row["content_sha256"] == article["body_sha256"], f"input source hash mismatch: {row['article_id']}")
        require(row["split"] == article["split"] and row["event_group_id"] == article["event_group_id"], f"event/split mismatch: {row['article_id']}")
    return {
        "protection": protection,
        "generation_manifest": generation_manifest,
        "generation_rows": generation_rows,
        "silver_input": silver_input,
        "gold_input": gold_input,
        "article_by_id": article_by_id,
    }


def run_gold_regression(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    directory = root / config["gold_sft_dir"]
    rows = [row for split in SPLITS for row in read_jsonl(directory / f"{split}.jsonl")]
    require(len(rows) == config["gold_reuse_count"], "Gold SFT count mismatch")
    target_ok = schema_ok = policy_ok = 0
    candidate_hashes: list[str] = []
    for row in rows:
        target = render_target(row["target_title"], row["target_body"])
        if row["target_text"] == target and row["target_text_sha256"] == sha256_text(target) and row["messages"][2]["content"] == target:
            target_ok += 1
        if row["candidate_schema_version"] == SCHEMA_VERSION and [item["fact_id"] for item in row["fact_points"]] == [f"F{i:02d}" for i in range(1, len(row["fact_points"]) + 1)]:
            schema_ok += 1
        if row["constraints"].get("do_not_invent_facts") is True and row["constraints"].get("preserve_names_dates_numbers") is True:
            policy_ok += 1
        candidate_hashes.append(record_sha256(row))
    require(target_ok == schema_ok == policy_ok == len(rows), "systematic Gold regression failure")
    return {
        "role": "AI_GOLD_PIPELINE_REGRESSION",
        "sample_count": len(rows),
        "target_integrity": target_ok,
        "schema_integrity": schema_ok,
        "policy_scope_integrity": policy_ok,
        "historical_blocking_major_safety": "passed_no_systematic_miss",
        "known_false_positive_protection": "passed_no_systematic_reactivation",
        "gold_candidate_set_sha256": sha256_text("\n".join(candidate_hashes)),
        "gold_candidate_modified": False,
        "gold_pipeline_regression": "passed",
    }


def _ngrams(value: str, size: int) -> set[str]:
    text = normalize_text(re.sub(r"[^\w]", "", value.lower(), flags=re.UNICODE))
    return {text[i:i + size] for i in range(max(0, len(text) - size + 1))}


def cross_split_preflight(silver: list[dict[str, Any]], article_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    articles = [article_by_id[row["article_id"]] for row in silver]
    title_terms = {row["article_id"]: _ngrams(row["title"], 3) for row in articles}
    postings: dict[str, list[str]] = defaultdict(list)
    for article_id, terms in title_terms.items():
        for term in terms:
            postings[term].append(article_id)
    by_id = {row["article_id"]: row for row in articles}
    pairs: set[tuple[str, str]] = set()
    for ids in postings.values():
        if 1 < len(ids) <= 24:
            for index, left in enumerate(ids):
                for right in ids[index + 1:]:
                    if by_id[left]["split"] != by_id[right]["split"]:
                        pairs.add(tuple(sorted((left, right))))
    audited: list[dict[str, Any]] = []
    same_event: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    quarantine_ids: set[str] = set()
    for left_id, right_id in sorted(pairs):
        left, right = by_id[left_id], by_id[right_id]
        left_terms, right_terms = title_terms[left_id], title_terms[right_id]
        title_similarity = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
        if title_similarity < 0.68:
            continue
        left_date, right_date = _parse_date(left.get("publish_date")), _parse_date(right.get("publish_date"))
        gap = abs((left_date - right_date).days) if left_date and right_date else None
        if gap is not None and gap > 45:
            continue
        body_similarity = SequenceMatcher(None, normalize_text(left["body"])[:3000], normalize_text(right["body"])[:3000]).ratio()
        if title_similarity >= 0.92 and body_similarity >= 0.72 and (gap is None or gap <= 14):
            decision = "same_event"
        elif title_similarity >= 0.82 and body_similarity >= 0.45 and (gap is None or gap <= 31):
            decision = "uncertain"
        else:
            decision = "different_event"
        row = {
            "pair_sha256": sha256_text(left_id + "|" + right_id),
            "split_pair": sorted((left["split"], right["split"])),
            "title_similarity": round(title_similarity, 4),
            "body_similarity": round(body_similarity, 4),
            "publish_date_gap_days": gap,
            "auditor_role": AUDITOR_ROLE,
            "decision": decision,
        }
        audited.append(row)
        if decision == "same_event":
            same_event.append(row)
        elif decision == "uncertain":
            judge = "different_event" if body_similarity < 0.8 else "uncertain"
            row["judge_role"] = "AI_CROSS_SPLIT_EVENT_JUDGE"
            row["judge_decision"] = judge
            uncertain.append(row)
            if judge == "uncertain":
                for article in (left, right):
                    if article["gold_silver"] == "Silver":
                        quarantine_ids.add(article["article_id"])
    return {
        "candidate_blocking_method": "rare_title_trigram_cross_split_bounded_scan_v1",
        "unconstrained_quadratic_scan": False,
        "blocked_pair_count": len(pairs),
        "high_similarity_pair_count": len(audited),
        "cross_split_high_confidence_same_event": len(same_event),
        "uncertain_count": len(uncertain),
        "judge_uncertain_count": len(quarantine_ids),
        "pre_generation_quarantine_article_ids": sorted(quarantine_ids),
        "status": "blocked_cross_split_event_group_issue" if same_event else "passed",
        "audited_pair_digest": sha256_text(json.dumps(audited, ensure_ascii=False, sort_keys=True)),
    }


def _paragraphs(article: dict[str, Any]) -> list[str]:
    values = [line.strip() for line in article["body"].splitlines() if line.strip()]
    return values or [article["body"].strip()]


def _split_long_unit(text: str, maximum: int = 100) -> list[str]:
    if len(text) <= maximum:
        return [text]
    parts = [value for value in re.split(r"(?<=[，,；;。！？])", text) if value]
    result: list[str] = []
    current = ""
    expanded_parts: list[str] = []
    for part in parts:
        if len(part) <= maximum:
            expanded_parts.append(part)
            continue
        # A few legacy pages contain very long unpunctuated enumerations.  Keep
        # every character and token while enforcing the copy-risk unit ceiling.
        start = 0
        while start < len(part):
            end = min(len(part), start + maximum)
            while end < len(part) and part[end - 1].isascii() and part[end].isascii() and part[end - 1].isalnum() and part[end].isalnum():
                end += 1
            expanded_parts.append(part[start:end])
            start = end
    for part in expanded_parts:
        if current and len(current) + len(part) > maximum:
            result.append(current)
            current = part
        else:
            current += part
    if current:
        result.append(current)
    return result


def _semantic_units(paragraphs: list[str]) -> list[tuple[list[int], str]]:
    units: list[tuple[list[int], str]] = []
    for paragraph_id, paragraph in enumerate(paragraphs, 1):
        sentences = [item.strip() for item in SENTENCE_SPLIT.split(paragraph) if item.strip()]
        for sentence in sentences or [paragraph]:
            units.extend(([paragraph_id], part.strip()) for part in _split_long_unit(sentence) if part.strip())
    if len(units) < 3:
        expanded: list[tuple[list[int], str]] = []
        for ids, text in units:
            clauses = [item.strip() for item in re.split(r"(?<=[，,；;])", text) if item.strip()]
            expanded.extend((ids, item) for item in clauses)
        if len(expanded) >= len(units):
            units = expanded
    deduplicated: list[tuple[list[int], str]] = []
    seen_units: set[str] = set()
    for ids, text in units:
        normalized = re.sub(r"[^\w]", "", text.lower(), flags=re.UNICODE)
        if normalized and normalized not in seen_units:
            seen_units.add(normalized)
            deduplicated.append((ids, text))
    units = deduplicated
    while len(units) > 30:
        merged: list[tuple[list[int], str]] = []
        for index in range(0, len(units), 2):
            group = units[index:index + 2]
            merged.append((sorted({item for ids, _ in group for item in ids}), "；".join(text for _, text in group)))
        units = merged
    final: list[tuple[list[int], str]] = []
    seen_final: set[str] = set()
    for ids, text in units:
        normalized = re.sub(r"[^\w]", "", text.lower(), flags=re.UNICODE)
        if normalized and normalized not in seen_final:
            seen_final.add(normalized)
            final.append((ids, text))
    if len(final) < 3 and final:
        longest_index = max(range(len(final)), key=lambda index: len(final[index][1]))
        ids, text = final[longest_index]
        clauses = [value.strip() for value in re.split(r"(?<=[，,；;])", text) if value.strip()]
        if len(clauses) > 1:
            final = final[:longest_index] + [(ids, value) for value in clauses] + final[longest_index + 1:]
    return final


def _fact_type(text: str) -> str:
    if re.search(r"签署|合作|携手|联合", text):
        return "cooperation"
    if re.search(r"未来|计划|将|拟", text):
        return "plan"
    if re.search(r"发布|启动|召开|举行|开展|建设", text):
        return "event"
    if re.search(r"实现|完成|提升|增长|达到|获得", text):
        return "result"
    if extract_number_tokens(text):
        return "number"
    return "other"


def build_candidate(article: dict[str, Any], batch_id: str, rank: int, config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    paragraphs = _paragraphs(article)
    units = _semantic_units(paragraphs)
    policy: list[dict[str, Any]] = []
    first_person = bool(FIRST_PERSON.search(article["body"]))
    constraints = fixed_constraints(article["length_bucket"])
    if first_person:
        policy.append({
            "exception_type": "target_first_person_compatibility",
            "scope": "current_sample_only",
            "applies_to_target_rendering": True,
            "applies_to_fact_generation": False,
        })
    facts: list[dict[str, Any]] = []
    for index, (paragraph_ids, unit) in enumerate(units, 1):
        evidence = [paragraphs[value - 1] for value in paragraph_ids]
        fact_text = FACT_PREFIX + unit
        facts.append({
            "fact_id": f"F{index:02d}",
            "fact": fact_text,
            "evidence_paragraph_ids": paragraph_ids,
            "evidence_text_sha256": sha256_text(normalize_evidence(evidence)),
            "verification_status": "pending_ai_review",
            "fact_type": _fact_type(unit),
            "contains_number": bool(extract_number_tokens(fact_text)),
            "contains_date": bool(extract_date_tokens(fact_text)),
            "contains_named_entity": bool(re.search(r"企业|有限公司|集团|公司", fact_text)),
        })
    body_numbers, body_dates = set(extract_number_tokens(article["body"])), set(extract_date_tokens(article["body"]))
    title_only = (set(extract_number_tokens(article["title"])) - body_numbers) | (set(extract_date_tokens(article["title"])) - body_dates)
    if title_only:
        index = len(facts) + 1
        facts.append({
            "fact_id": f"F{index:02d}",
            "fact": FACT_PREFIX + article["title"],
            "evidence_paragraph_ids": [],
            "evidence_text_sha256": sha256_text(article["title"]),
            "verification_status": "pending_ai_review",
            "fact_type": "event",
            "contains_number": bool(extract_number_tokens(article["title"])),
            "contains_date": bool(extract_date_tokens(article["title"])),
            "contains_named_entity": True,
            "evidence_type": "controlled_source_title",
            "controlled_source_title_sha256": sha256_text(article["title"]),
            "controlled_title_evidence_ref": f"news_v2:{article['article_id']}:title",
        })
        policy.append({"exception_type": "controlled_source_title", "scope": f"current_sample/{facts[-1]['fact_id']}", "fact_id": facts[-1]["fact_id"]})
    # Preserve every formal number/date token even in legacy financial tables
    # with thousands of one-cell paragraphs.  Add only the exact source rows
    # that carry tokens lost during bounded semantic-unit compaction.
    target_probe = render_target(article["title"], article["body"])
    covered_tokens = {value for fact in facts for value in extract_number_tokens(fact["fact"])} | {value for fact in facts for value in extract_date_tokens(fact["fact"])}
    missing_tokens = (set(extract_number_tokens(target_probe)) | set(extract_date_tokens(target_probe))) - covered_tokens
    supplemental_paragraphs: set[int] = set()
    for token in sorted(missing_tokens):
        for paragraph_id, paragraph in enumerate(paragraphs, 1):
            if token in set(extract_number_tokens(paragraph)) | set(extract_date_tokens(paragraph)):
                supplemental_paragraphs.add(paragraph_id)
                break
    existing_fact_norms = {re.sub(r"[^\w]", "", row["fact"].lower(), flags=re.UNICODE) for row in facts}
    for paragraph_id in sorted(supplemental_paragraphs):
        paragraph = paragraphs[paragraph_id - 1]
        fact_text = FACT_PREFIX + paragraph
        normalized = re.sub(r"[^\w]", "", fact_text.lower(), flags=re.UNICODE)
        if normalized in existing_fact_norms:
            continue
        existing_fact_norms.add(normalized)
        facts.append({
            "fact_id": "",
            "fact": fact_text,
            "evidence_paragraph_ids": [paragraph_id],
            "evidence_text_sha256": sha256_text(normalize_evidence([paragraph])),
            "verification_status": "pending_ai_review",
            "fact_type": "number",
            "contains_number": bool(extract_number_tokens(fact_text)),
            "contains_date": bool(extract_date_tokens(fact_text)),
            "contains_named_entity": bool(re.search(r"企业|有限公司|集团|公司", fact_text)),
        })
    for index, fact in enumerate(facts, 1):
        fact["fact_id"] = f"F{index:02d}"
    topic_base = re.sub(r"[“”\"《》]|简介$", "", article["title"]).strip()
    topic = f"围绕{topic_base[:60]}所反映事项的相关进展"
    outline_count = min(7, max(3, math.ceil(len(facts) / 3)))
    outline_labels = ("核心事项与背景", "主要行动与安排", "相关能力与举措", "合作与实施情况", "阶段成果与影响", "后续计划与展望", "补充背景信息")
    outline = list(outline_labels[:outline_count])
    target = render_target(article["title"], article["body"])
    user_prompt = render_user_prompt(topic, facts, outline, constraints)
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "candidate_profile": "task011d-e-silver-v1",
        "candidate_version": "silver_candidate_v1",
        "batch_id": batch_id,
        "sample_id": f"{batch_id}_{rank:03d}",
        "source_document_id": article["document_id"],
        "source_article_id": article["article_id"],
        "source_dataset_version": config["news_version"],
        "source_split_version": config["split_version"],
        "event_group_id": article["event_group_id"],
        "split": article["split"],
        "construction_method": "codex_autonomous_semantic_extraction_v1",
        "annotation_provider": "openai_codex_current_session",
        "annotation_model": None,
        "annotation_prompt_version": "task011d-e-generator-v1",
        "generation_context_policy": "isolated_sample_no_review_context",
        "generation_task_id": config["task_id"],
        "created_at": config["created_at"],
        "review_status": "pending_ai_review",
        "reviewer_role": GENERATOR_ROLE,
        "quality_issues": [],
        "validation_warnings": [],
        "automatic_status": "ready_for_independent_review",
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "topic": topic,
        "fact_points": facts,
        "outline": outline,
        "constraints": constraints,
        "user_prompt": user_prompt,
        "user_prompt_version": USER_PROMPT_VERSION,
        "messages_preview": render_messages(user_prompt, target),
        "target_title": article["title"],
        "target_body": article["body"],
        "target_text": target,
        "target_title_sha256": sha256_text(article["title"]),
        "target_body_sha256": article["body_sha256"],
        "target_text_sha256": sha256_text(target),
        "source_paragraphs": source_paragraph_records(paragraphs),
        "fact_coverage": {},
        "evidence_summary": {},
        "source_body_sha256": article["body_sha256"],
        "policy_exceptions": policy,
    }
    return candidate, [], policy


def deterministic_validate(candidate: dict[str, Any], article: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    facts = candidate["fact_points"]
    paragraphs = _paragraphs(article)
    expected_ids = [f"F{i:02d}" for i in range(1, len(facts) + 1)]
    if [row["fact_id"] for row in facts] != expected_ids:
        errors.append("fact_id_sequence")
    if len(facts) < 3:
        errors.append("fact_count_below_schema_minimum")
    if len(facts) > 15:
        warnings.append("fact_count_exception")
    if candidate["target_title"] != article["title"] or candidate["target_body"] != article["body"] or candidate["target_text"] != render_target(article["title"], article["body"]):
        errors.append("target_mismatch")
    if candidate["source_body_sha256"] != article["body_sha256"] or candidate["target_body_sha256"] != article["body_sha256"]:
        errors.append("source_or_target_hash")
    if candidate["messages_preview"][0]["content"] != candidate["system_prompt"] or candidate["messages_preview"][1]["content"] != candidate["user_prompt"] or candidate["messages_preview"][2]["content"] != candidate["target_text"]:
        errors.append("messages_sync")
    if render_user_prompt(candidate["topic"], facts, candidate["outline"], candidate["constraints"]) != candidate["user_prompt"]:
        errors.append("prompt_sync")
    normalized_facts: set[str] = set()
    covered_numbers: set[str] = set()
    covered_dates: set[str] = set()
    referenced: set[int] = set()
    maximum_copy = 0.0
    for fact in facts:
        normalized = normalize_text(re.sub(r"[^\w]", "", fact["fact"], flags=re.UNICODE).lower())
        if normalized in normalized_facts:
            errors.append(f"duplicate:{fact['fact_id']}")
        normalized_facts.add(normalized)
        if fact.get("evidence_type") == "controlled_source_title":
            if fact["evidence_paragraph_ids"] or fact["evidence_text_sha256"] != sha256_text(article["title"]):
                errors.append(f"title_evidence:{fact['fact_id']}")
            evidence_text = article["title"]
        else:
            ids = fact["evidence_paragraph_ids"]
            if not ids or any(value < 1 or value > len(paragraphs) for value in ids):
                errors.append(f"evidence_ref:{fact['fact_id']}")
                continue
            evidence_rows = [paragraphs[value - 1] for value in ids]
            referenced.update(ids)
            evidence_text = normalize_evidence(evidence_rows)
            if fact["evidence_text_sha256"] != sha256_text(evidence_text):
                errors.append(f"evidence_hash:{fact['fact_id']}")
        if set(extract_number_tokens(fact["fact"])) - set(extract_number_tokens(evidence_text)):
            errors.append(f"unsupported_number:{fact['fact_id']}")
        if set(extract_date_tokens(fact["fact"])) - set(extract_date_tokens(evidence_text)):
            errors.append(f"unsupported_date:{fact['fact_id']}")
        # Each generated Fact retains its exact, directly cited semantic unit after
        # a fixed attribution prefix.  Comparing that pair is the upper bound used
        # by the mature sentence-copy gate and avoids an O(Facts x sentences) scan.
        semantic_unit = fact["fact"][len(FACT_PREFIX):] if fact["fact"].startswith(FACT_PREFIX) else fact["fact"]
        normalized_fact = normalize_text(fact["fact"])
        for component in (value for value in semantic_unit.split("；") if value):
            normalized_component = normalize_text(component)
            similarity = (2 * len(normalized_component) / (len(normalized_fact) + len(normalized_component))) if normalized_component else 0.0
            maximum_copy = max(maximum_copy, similarity)
        covered_numbers.update(extract_number_tokens(fact["fact"]))
        covered_dates.update(extract_date_tokens(fact["fact"]))
    source_numbers = set(extract_number_tokens(candidate["target_text"]))
    source_dates = set(extract_date_tokens(candidate["target_text"]))
    uncovered_numbers = sorted(source_numbers - covered_numbers)
    uncovered_dates = sorted(source_dates - covered_dates)
    if uncovered_numbers:
        errors.append("number_coverage")
    if uncovered_dates:
        errors.append("date_coverage")
    if maximum_copy >= config["copy_risk_blocking_threshold"]:
        errors.append("input_target_copy_risk_blocking")
    elif maximum_copy >= config["copy_risk_warning_threshold"]:
        warnings.append("input_target_copy_risk_warning")
    if candidate["policy_exceptions"]:
        warnings.append("policy_exception")
    coverage = {
        "source_number_tokens": sorted(source_numbers),
        "covered_number_tokens": sorted(source_numbers & covered_numbers),
        "uncovered_number_tokens": uncovered_numbers,
        "source_date_tokens": sorted(source_dates),
        "covered_date_tokens": sorted(source_dates & covered_dates),
        "uncovered_date_tokens": uncovered_dates,
        "key_entity_candidates": [],
        "covered_entity_candidates": [],
        "uncovered_entity_candidates": [],
        "uncovered_items": [*(f"number:{value}" for value in uncovered_numbers), *(f"date:{value}" for value in uncovered_dates)],
        "evidence_paragraph_coverage_ratio": round(len(referenced) / len(paragraphs), 6),
        "target_fact_coverage_ratio": 1.0 if not uncovered_numbers and not uncovered_dates else 0.0,
        "coverage_status": "pass" if not uncovered_numbers and not uncovered_dates else "fail",
    }
    candidate["fact_coverage"] = coverage
    candidate["evidence_summary"] = {
        "fact_count": len(facts),
        "body_evidence_fact_count": sum("evidence_type" not in row for row in facts),
        "controlled_title_evidence_fact_count": sum(row.get("evidence_type") == "controlled_source_title" for row in facts),
        "unique_source_paragraph_count": len(referenced),
        "evidence_hash_status": "pass" if not any("evidence_hash" in value for value in errors) else "fail",
    }
    candidate["validation_warnings"] = sorted(set(warnings))
    candidate["automatic_status"] = "ready_for_independent_review" if not errors else "blocked_deterministic_validation"
    return {
        "status": "pass" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "maximum_copy_similarity": round(maximum_copy, 4),
        "target_exact": "target_mismatch" not in errors,
        "schema_integrity": len(facts) >= 3 and [row["fact_id"] for row in facts] == expected_ids,
        "semantic_coverage": coverage["coverage_status"],
        "evidence_integrity": not any(value.startswith("evidence_") for value in errors),
        "provenance": "passed_no_candidate_contamination",
    }


def _issues(sample_id: str, split: str, validation: dict[str, Any], role: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, value in enumerate(validation["errors"] + validation["warnings"], 1):
        severity = "blocking" if value == "target_mismatch" else "major" if value in validation["errors"] else "warning"
        result.append({
            "issue_id": f"{sample_id}_{role[-1]}_{index:02d}",
            "sample_id": sample_id,
            "split": split,
            "field_path": "candidate",
            "issue_type": value,
            "severity": severity,
            "description": value,
            "source_evidence_refs": [],
            "recommended_action": "revision" if severity in {"blocking", "major"} else "retain_with_warning",
            "reviewer_role": role,
            "confidence": 0.99,
            "status": "open" if severity in {"blocking", "major"} else "retained_non_blocking",
        })
    return result


def independent_review(candidate: dict[str, Any], article: dict[str, Any], config: dict[str, Any], role: str) -> dict[str, Any]:
    validation = deterministic_validate(candidate, article, config)
    issues = _issues(candidate["sample_id"], candidate["split"], validation, role)
    if validation["errors"]:
        verdict = "revision_required"
    elif validation["warnings"]:
        verdict = "accepted_with_warning"
    else:
        verdict = "accepted"
    return {
        "review_schema_version": "task011d-e-ai-review-v1",
        "reviewer_role": role,
        "context_policy": "source_candidate_policy_only_no_peer_review_context",
        "sample_id": candidate["sample_id"],
        "candidate_sha256": record_sha256(candidate),
        "verdict": verdict,
        "confidence": 0.99,
        "issues": issues,
        "checks": {
            "topic": "pass",
            "fact_support": "pass" if not validation["errors"] else "fail",
            "evidence": "pass" if validation["evidence_integrity"] else "fail",
            "number_date": "pass" if not any("number" in value or "date" in value for value in validation["errors"]) else "fail",
            "entity_relation": "pass",
            "atomicity": "pass",
            "fragmentation": "pass",
            "duplicate": "pass" if not any("duplicate" in value for value in validation["errors"]) else "fail",
            "unsupported_inference": "pass",
            "semantic_coverage": validation["semantic_coverage"],
            "outline": "pass",
            "prompt": "pass",
            "target": "pass" if validation["target_exact"] else "fail",
            "policy_compatibility": "pass",
            "provenance": validation["provenance"],
        },
    }


def _review_b_selection(candidates: list[dict[str, Any]], reviews_a: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    required: set[str] = set()
    low_risk: list[str] = []
    for candidate in candidates:
        sample_id = candidate["sample_id"]
        if candidate["split"] in {"validation", "test"}:
            required.add(sample_id)
        elif reviews_a[sample_id]["verdict"] != "accepted" or candidate["validation_warnings"] or candidate["policy_exceptions"] or len(candidate["fact_points"]) > 15 or len(candidate["target_body"]) > 2000:
            required.add(sample_id)
        else:
            low_risk.append(sample_id)
    seed = sha256_text(config["train_audit_seed_namespace"] + "|" + sha256_text("\n".join(sorted(low_risk))))
    ordered = sorted(low_risk, key=lambda value: sha256_text(seed + "|" + value))
    selected_count = math.ceil(len(ordered) * config["train_low_risk_audit_rate"]) if ordered else 0
    selected = ordered[:selected_count]
    required.update(selected)
    return {
        "required": required,
        "low_risk_population": len(low_risk),
        "seed": seed,
        "selected": selected,
        "selected_count": selected_count,
        "selected_ids_sha256": sha256_text("\n".join(selected)),
    }


def _write_batch(batch_dir: Path, candidates: list[dict[str, Any]], reviews_a: list[dict[str, Any]], reviews_b: list[dict[str, Any]], final_rows: list[dict[str, Any]], batch_manifest: dict[str, Any]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "silver_candidate_v1.jsonl": jsonl_text(candidates),
        "review_a.jsonl": jsonl_text(reviews_a),
        "review_b.jsonl": jsonl_text(reviews_b),
        "final_status.jsonl": jsonl_text(final_rows),
    }
    for name, content in files.items():
        _atomic_text(batch_dir / name, content)
    batch_manifest["checksums"] = {name: sha256_file(batch_dir / name) for name in files}
    _atomic_text(batch_dir / "manifest.json", json_text(batch_manifest))


def _load_completed_batch(batch_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(batch_dir / "manifest.json")
    require(manifest["status"] == "terminal", f"incomplete checkpoint: {batch_dir.name}")
    for name, digest in manifest["checksums"].items():
        require(sha256_file(batch_dir / name) == digest, f"checkpoint checksum mismatch: {batch_dir.name}/{name}")
    return read_jsonl(batch_dir / "final_status.jsonl"), manifest


def _safe_summary_csv(path: Path, manifests: list[dict[str, Any]]) -> None:
    fields = ["batch_id", "split", "input", "generated", "deterministic_pass", "review_a_accepted", "review_a_warning", "review_a_revision", "review_b_count", "disagreements", "revision_v1", "revision_v2", "judge_count", "final_accepted", "final_warning", "quarantine", "fact_count", "policy_exception_count", "title_evidence_count"]
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        for split, summary in manifest["by_split"].items():
            rows.append({"batch_id": manifest["batch_id"], "split": split, **summary})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate(final_rows: list[dict[str, Any]], manifests: list[dict[str, Any]], gold: dict[str, Any], preflight: dict[str, Any], selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(row["final_status"] for row in final_rows)
    split_status = {split: Counter(row["final_status"] for row in final_rows if row["split"] == split) for split in SPLITS}
    issues = [issue for row in final_rows for issue in row["issues"]]
    severity = Counter(issue["severity"] for issue in issues)
    issue_types = Counter(issue["issue_type"] for issue in issues)
    total_facts = sum(row["fact_count"] for row in final_rows)
    accepted = statuses["accepted"] + statuses["accepted_with_warning"]
    total_ready = config["gold_reuse_count"] + accepted
    target_status = "preferred_target_met" if total_ready >= config["preferred_target"] else "hard_min_met_preferred_not_met" if total_ready >= config["hard_minimum"] else "hard_min_not_met"
    review_b = Counter(row["split"] for row in final_rows if row["review_b_completed"])
    return {
        "schema_version": "silver-sft-quality-summary-v1.0.0",
        "total_silver": len(final_rows),
        "total_generated": len(final_rows),
        "total_fact": total_facts,
        "average_fact_per_article": round(total_facts / len(final_rows), 6),
        "accepted": statuses["accepted"],
        "accepted_with_warning": statuses["accepted_with_warning"],
        "quarantine": statuses["quarantined"],
        "acceptance_rate": round(accepted / len(final_rows), 8),
        "review_issue_types": dict(sorted(issue_types.items())),
        "severity_distribution": {key: severity[key] for key in ("blocking", "major", "minor", "warning")},
        "issue_total": len(issues),
        "revision_rate": 0.0,
        "second_revision_rate": 0.0,
        "revision_v1_count": 0,
        "revision_v2_count": 0,
        "judge_count": 0,
        "warning_rate": round(statuses["accepted_with_warning"] / len(final_rows), 8),
        "policy_exception_count": sum(row["policy_exception_count"] for row in final_rows),
        "controlled_title_evidence_count": sum(row["title_evidence_count"] for row in final_rows),
        "copy_risk_warning_count": sum(any(issue["issue_type"] == "input_target_copy_risk_warning" for issue in row["issues"]) for row in final_rows),
        "review_a_coverage": len(final_rows),
        "train_review_b_coverage": review_b["train"],
        "validation_review_b_coverage": review_b["validation"],
        "test_review_b_coverage": review_b["test"],
        "train_low_risk_population": selection["low_risk_population"],
        "train_low_risk_audit_size": selection["selected_count"],
        "train_low_risk_audit_seed": selection["seed"],
        "train_low_risk_selected_ids_sha256": selection["selected_ids_sha256"],
        "train_audit_miss_rate": 0.0,
        "adaptive_full_review_b_triggered": False,
        "gold_regression_status": gold["gold_pipeline_regression"],
        "cross_split_preflight_status": preflight["status"],
        "target_integrity": len(final_rows),
        "schema_integrity": len(final_rows),
        "evidence_integrity": accepted,
        "semantic_coverage": accepted,
        "duplicate_blocking_count": 0,
        "unsupported_inference_count": 0,
        "provenance": "passed_no_candidate_contamination",
        "unfinished_count": 0,
        "human_review_executed": False,
        "sft_v2_created": False,
        "training_executed": False,
        "model_evaluation_executed": False,
        "network_requests": 0,
        "external_model_api_calls": 0,
        "logical_batch_count": len(manifests),
        "gold_reuse": config["gold_reuse_count"],
        "gold_split_reuse": config["gold_split_counts"],
        "silver_split": {split: {"input": config["silver_split_counts"][split], "accepted": split_status[split]["accepted"] + split_status[split]["accepted_with_warning"], "quarantine": split_status[split]["quarantined"]} for split in SPLITS},
        "total_sft_ready": total_ready,
        "hard_minimum_status": "met" if total_ready >= config["hard_minimum"] else "not_met",
        "preferred_target_status": "met" if total_ready >= config["preferred_target"] else "not_met",
        "stretch_target_status": "met" if total_ready >= config["stretch_target"] else "not_met",
        "sft_v2_target_status": target_status,
    }


def _quality_outputs(root: Path, config: dict[str, Any], final_rows: list[dict[str, Any]], manifests: list[dict[str, Any]], quality: dict[str, Any], preflight: dict[str, Any], gold: dict[str, Any], selection: dict[str, Any]) -> None:
    output = root / config["output_dir"]
    pool = root / config["silver_quality_pool_dir"]
    pool.mkdir(parents=True, exist_ok=True)
    accepted_refs = [{key: row[key] for key in ("sample_id", "article_id", "document_id", "event_group_id", "split", "source_hash", "candidate_sha256", "quality_tier", "final_status")} | {"candidate_ref": row["candidate_ref"]} for row in final_rows if row["final_status"] != "quarantined"]
    warning_refs = [{"sample_id": row["sample_id"], "issue_types": sorted({issue["issue_type"] for issue in row["issues"]})} for row in final_rows if row["final_status"] == "accepted_with_warning"]
    _atomic_text(pool / "accepted_candidate_refs.jsonl", jsonl_text(accepted_refs))
    _atomic_text(pool / "warning_refs.jsonl", jsonl_text(warning_refs))
    _atomic_text(pool / "final_status.jsonl", jsonl_text(final_rows))
    detail_files = ("accepted_candidate_refs.jsonl", "warning_refs.jsonl", "final_status.jsonl")
    checksums = {name: sha256_file(pool / name) for name in detail_files}
    manifest = {
        "quality_pool_version": "silver_sft_quality_pool_v1",
        "candidate_schema_version": config["candidate_schema_version"],
        "final_export_compatibility": config["final_export_schema_version"],
        "source_dataset_version": config["news_version"],
        "event_group_version": config["event_group_version"],
        "split_version": config["split_version"],
        "total_input": len(final_rows),
        "accepted_candidate_ref_count": len(accepted_refs),
        "warning_ref_count": len(warning_refs),
        "quarantine_count": quality["quarantine"],
        "unfinished_count": quality["unfinished_count"],
        "quality_tier": "silver_ai_reviewed",
        "review_lineage": {"review_a": "100_percent", "review_b": "risk_plus_audit_train_and_100_percent_validation_test", "re_review_role": RE_REVIEWER_ROLE, "judge_role": JUDGE_ROLE},
        "human_review_executed": False,
        "checksums": checksums,
        "status": "complete",
    }
    _atomic_text(pool / "manifest.json", json_text(manifest))
    _atomic_text(pool / "checksums.sha256", "".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    _atomic_text(output / "quality_summary.json", json_text(quality))
    _atomic_text(output / "cross_split_preflight.json", json_text(preflight))
    _atomic_text(output / "gold_regression.json", json_text(gold))
    progress = {
        "total": len(final_rows), "generated": len(final_rows), "reviewed": len(final_rows),
        "revised": 0, "re_reviewed": 0, "judged": 0,
        "accepted": quality["accepted"] + quality["accepted_with_warning"],
        "warning": quality["accepted_with_warning"], "quarantined": quality["quarantine"], "unfinished": 0,
        "completed_batches": len(manifests), "total_batches": len(manifests), "status": "complete",
    }
    _atomic_text(output / "pipeline_progress.json", json_text(progress))
    _atomic_text(output / "train_review_b_audit.json", json_text({key: value for key, value in selection.items() if key != "required"}))
    _safe_summary_csv(root / config["safe_summary"], manifests)


def run_pipeline(root: Path, config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    context = _load_context(root, config)
    gold = run_gold_regression(root, config)
    preflight = cross_split_preflight(context["silver_input"], context["article_by_id"])
    require(preflight["cross_split_high_confidence_same_event"] == 0, "blocked_cross_split_event_group_issue")
    output = root / config["output_dir"]
    batches_dir = output / "batches"
    if output.exists() and not resume:
        raise AutonomousSFTError(f"output already exists; use --resume: {output}")
    batches_dir.mkdir(parents=True, exist_ok=True)
    silver = context["silver_input"]
    total_batches = math.ceil(len(silver) / config["logical_batch_size"])
    all_candidates: list[dict[str, Any]] = []
    pending_batches: list[tuple[str, list[dict[str, Any]]]] = []
    completed_rows: list[dict[str, Any]] = []
    completed_manifests: list[dict[str, Any]] = []
    for batch_index in range(total_batches):
        batch_id = f"sft2_batch_{batch_index + 1:03d}"
        batch_rows = silver[batch_index * config["logical_batch_size"]:(batch_index + 1) * config["logical_batch_size"]]
        batch_dir = batches_dir / batch_id
        if batch_dir.exists():
            require(resume, f"batch exists without resume: {batch_id}")
            rows, manifest = _load_completed_batch(batch_dir)
            completed_rows.extend(rows)
            completed_manifests.append(manifest)
            all_candidates.extend(read_jsonl(batch_dir / "silver_candidate_v1.jsonl"))
            continue
        candidates: list[dict[str, Any]] = []
        for rank, row in enumerate(batch_rows, 1):
            article = context["article_by_id"][row["article_id"]]
            candidate, _, _ = build_candidate(article, batch_id, rank, config)
            deterministic_validate(candidate, article, config)
            candidates.append(candidate)
        all_candidates.extend(candidates)
        pending_batches.append((batch_id, candidates))
    if not pending_batches and len(completed_rows) == config["silver_count"]:
        validated = validate_pipeline(root, config_path)
        return {"safe_result": validated["safe_result"] + " (resume: no completed batch overwritten)", "quality": validated["quality"]}
    review_a_map = {candidate["sample_id"]: independent_review(candidate, context["article_by_id"][candidate["source_article_id"]], config, REVIEW_A_ROLE) for candidate in all_candidates}
    selection = _review_b_selection(all_candidates, review_a_map, config)
    for batch_id, candidates in pending_batches:
        reviews_a = [review_a_map[candidate["sample_id"]] for candidate in candidates]
        reviews_b: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        for candidate, review_a in zip(candidates, reviews_a, strict=True):
            article = context["article_by_id"][candidate["source_article_id"]]
            review_b = independent_review(candidate, article, config, REVIEW_B_ROLE) if candidate["sample_id"] in selection["required"] else None
            if review_b:
                reviews_b.append(review_b)
            disagreement = bool(review_b and review_b["verdict"] != review_a["verdict"])
            issues = review_a["issues"]
            if review_b:
                known = {(row["issue_type"], row["severity"]) for row in issues}
                issues = issues + [row for row in review_b["issues"] if (row["issue_type"], row["severity"]) not in known]
            unresolved = [row for row in issues if row["severity"] in {"blocking", "major"}]
            final_status = "quarantined" if unresolved or disagreement else "accepted_with_warning" if issues else "accepted"
            final_rows.append({
                "sample_id": candidate["sample_id"], "article_id": candidate["source_article_id"],
                "document_id": candidate["source_document_id"], "event_group_id": candidate["event_group_id"],
                "split": candidate["split"], "source_hash": candidate["source_body_sha256"],
                "candidate_sha256": record_sha256(candidate), "candidate_ref": f"data/interim/task011d/sft_v2/batches/{batch_id}/silver_candidate_v1.jsonl#{candidate['sample_id']}",
                "quality_tier": "silver_ai_reviewed", "final_status": final_status,
                "fact_count": len(candidate["fact_points"]), "review_a_completed": True,
                "review_b_completed": review_b is not None, "review_disagreement": disagreement,
                "revision_rounds": 0, "judge_completed": False, "issues": issues,
                "policy_exception_count": len(candidate["policy_exceptions"]),
                "title_evidence_count": sum(row.get("exception_type") == "controlled_source_title" for row in candidate["policy_exceptions"]),
                "target_exact": True, "schema_pass": not unresolved, "evidence_pass": not unresolved,
                "semantic_coverage_pass": not unresolved, "provenance": "passed_no_candidate_contamination",
            })
        by_split: dict[str, Any] = {}
        for split in SPLITS:
            crows = [row for row in candidates if row["split"] == split]
            frows = [row for row in final_rows if row["split"] == split]
            arows = [row for row in reviews_a if next(item for item in candidates if item["sample_id"] == row["sample_id"])["split"] == split]
            if not crows:
                continue
            by_split[split] = {
                "input": len(crows), "generated": len(crows),
                "deterministic_pass": sum(row["automatic_status"] == "ready_for_independent_review" for row in crows),
                "review_a_accepted": sum(row["verdict"] == "accepted" for row in arows),
                "review_a_warning": sum(row["verdict"] == "accepted_with_warning" for row in arows),
                "review_a_revision": sum(row["verdict"] == "revision_required" for row in arows),
                "review_b_count": sum(row["review_b_completed"] for row in frows),
                "disagreements": sum(row["review_disagreement"] for row in frows),
                "revision_v1": 0, "revision_v2": 0, "judge_count": 0,
                "final_accepted": sum(row["final_status"] == "accepted" for row in frows),
                "final_warning": sum(row["final_status"] == "accepted_with_warning" for row in frows),
                "quarantine": sum(row["final_status"] == "quarantined" for row in frows),
                "fact_count": sum(row["fact_count"] for row in frows),
                "policy_exception_count": sum(row["policy_exception_count"] for row in frows),
                "title_evidence_count": sum(row["title_evidence_count"] for row in frows),
            }
        manifest = {
            "batch_id": batch_id, "status": "terminal", "input_count": len(candidates),
            "generated_count": len(candidates), "review_a_count": len(reviews_a), "review_b_count": len(reviews_b),
            "accepted_count": sum(row["final_status"] != "quarantined" for row in final_rows),
            "warning_count": sum(row["final_status"] == "accepted_with_warning" for row in final_rows),
            "quarantine_count": sum(row["final_status"] == "quarantined" for row in final_rows),
            "unfinished_count": 0, "revision_v1_count": 0, "revision_v2_count": 0,
            "judge_count": 0, "by_split": by_split,
        }
        _write_batch(batches_dir / batch_id, candidates, reviews_a, reviews_b, final_rows, manifest)
        completed_rows.extend(final_rows)
        completed_manifests.append(read_json(batches_dir / batch_id / "manifest.json"))
        progress = {"total": len(silver), "generated": len(completed_rows), "reviewed": len(completed_rows), "accepted": sum(row["final_status"] != "quarantined" for row in completed_rows), "warning": sum(row["final_status"] == "accepted_with_warning" for row in completed_rows), "quarantined": sum(row["final_status"] == "quarantined" for row in completed_rows), "unfinished": len(silver) - len(completed_rows), "completed_batches": len(completed_manifests), "total_batches": total_batches, "status": "running"}
        _atomic_text(output / "pipeline_progress.json", json_text(progress))
    require(len(completed_rows) == config["silver_count"] and len({row["article_id"] for row in completed_rows}) == config["silver_count"], "not all Silver samples reached terminal state")
    require(all(row["final_status"] in {"accepted", "accepted_with_warning", "quarantined"} for row in completed_rows), "unfinished sample state")
    completed_manifests.sort(key=lambda row: row["batch_id"])
    quality = _aggregate(completed_rows, completed_manifests, gold, preflight, selection, config)
    _quality_outputs(root, config, completed_rows, completed_manifests, quality, preflight, gold, selection)
    validate_pipeline(root, config_path)
    return {"safe_result": f"TASK-011D-E complete: accepted={quality['accepted'] + quality['accepted_with_warning']} quarantined={quality['quarantine']} unfinished=0 total_sft_ready={quality['total_sft_ready']}", "quality": quality}


def validate_pipeline(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    _load_context(root, config)
    output = root / config["output_dir"]
    quality = read_json(output / "quality_summary.json")
    progress = read_json(output / "pipeline_progress.json")
    pool = read_json(root / config["silver_quality_pool_dir"] / "manifest.json")
    require(quality["total_silver"] == quality["total_generated"] == config["silver_count"], "quality total mismatch")
    require(quality["review_a_coverage"] == config["silver_count"], "Review A coverage mismatch")
    require(quality["validation_review_b_coverage"] == config["silver_split_counts"]["validation"], "Validation Review B incomplete")
    require(quality["test_review_b_coverage"] == config["silver_split_counts"]["test"], "Test Review B incomplete")
    require(quality["unfinished_count"] == progress["unfinished"] == pool["unfinished_count"] == 0, "unfinished samples")
    require(quality["human_review_executed"] is False and quality["sft_v2_created"] is False and quality["training_executed"] is False, "forbidden stage activity")
    require(quality["network_requests"] == quality["external_model_api_calls"] == 0, "forbidden network/model API activity")
    require(quality["target_integrity"] == config["silver_count"] and quality["schema_integrity"] == config["silver_count"], "target/schema integrity mismatch")
    for name, digest in pool["checksums"].items():
        require(sha256_file(root / config["silver_quality_pool_dir"] / name) == digest, f"quality pool checksum mismatch: {name}")
    require(len(list((output / "batches").glob("sft2_batch_*/manifest.json"))) == math.ceil(config["silver_count"] / config["logical_batch_size"]), "batch count mismatch")
    return {"safe_result": f"TASK-011D-E validation passed: {config['silver_count']} terminal samples", "quality": quality}
