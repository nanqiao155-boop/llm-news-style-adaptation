from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

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
from src.task011d.autonomous_sft import json_text, read_json, read_jsonl, record_sha256, sha256_file
from src.task011d.event_group_split import verify_checksums
from src.task011d.protection import validate_manifest
from src.task011d.reviewer_taxonomy import canonical_issue_builder, validate_dimension_review
from src.task011d.semantic_provider import SemanticModelProvider, SemanticProviderError, create_semantic_provider


SPLITS = ("train", "validation", "test")
SEVERITY_RANK = {"none": 0, "warning": 1, "minor": 2, "major": 3, "blocking": 4}
FACT_PREFIX = "据该篇新闻所载公开事实，"
TRUE_FIRST_PERSON = re.compile(r"(?<![A-Za-z\u4e00-\u9fff])(我们|我方|本人|我)(?![A-Za-z\u4e00-\u9fff])")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")


class HighPilotError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HighPilotError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        temp_path.write_text(content, encoding="utf-8", newline="")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json_text(value))


def _tree_snapshot(root: Path, relative: str) -> dict[str, Any]:
    directory = root / relative
    rows = [(path.relative_to(directory).as_posix(), sha256_file(path)) for path in sorted(directory.rglob("*")) if path.is_file()]
    return {"path": relative, "file_count": len(rows), "aggregate_sha256": _sha(rows)}


def _length_bucket(character_count: int) -> str:
    if character_count < 500:
        return "short"
    if character_count < 1000:
        return "medium"
    if character_count < 2000:
        return "long"
    return "very_long"


def load_context(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    protection = read_json(root / config["v1_protection_manifest"])
    _require(validate_manifest(root, protection) == [], "v1 protection manifest failed")
    protected = [
        "data/processed/news_v2_1",
        "data/processed/news_event_groups_v2_1",
        "data/processed/news_split_v2_1",
        config["gold_sft_dir"],
    ]
    for relative in protected:
        _require(verify_checksums(root / relative) == [], f"protected checksum failure: {relative}")
    versions = {
        "news": (root / "data/processed/news_v2_1/VERSION").read_text(encoding="utf-8").strip(),
        "event_group": (root / "data/processed/news_event_groups_v2_1/VERSION").read_text(encoding="utf-8").strip(),
        "split": (root / "data/processed/news_split_v2_1/VERSION").read_text(encoding="utf-8").strip(),
    }
    _require(versions["news"] == config["news_version"], "news version mismatch")
    _require(versions["event_group"] == config["event_group_version"], "event group version mismatch")
    _require(versions["split"] == config["split_version"], "split version mismatch")
    manifest = read_json(root / config["generation_manifest"])
    _require(manifest["silver_sft_generation_count"] == config["silver_count"] == 2117, "Silver population mismatch")
    _require(manifest["input_sha256"] == sha256_file(root / config["generation_input"]), "formal input checksum mismatch")
    rows = read_jsonl(root / config["generation_input"])
    silver = [row for row in rows if row["gold_silver"] == "Silver" and not row["reuse_frozen_gold_sft"]]
    _require(len(silver) == 2117, "formal Silver input count mismatch")
    _require(Counter(row["split"] for row in silver) == Counter(config["silver_split_counts"]), "Silver split count mismatch")
    articles: dict[str, dict[str, Any]] = {}
    for row in silver:
        source_path = root / row["source_ref"]
        source = read_json(source_path)
        _require(source["article_id"] == row["article_id"], f"source article mismatch: {row['article_id']}")
        _require(source["body_sha256"] == row["content_sha256"], f"source body mismatch: {row['article_id']}")
        source.update(
            {
                "split": row["split"],
                "event_group_id": row["event_group_id"],
                "document_id": source.get("inventory_id", f"news_v2_1:{row['article_id']}"),
                "length_bucket": _length_bucket(int(source.get("character_count", len(source["body"])))),
                "source_ref": row["source_ref"],
            }
        )
        articles[row["article_id"]] = source
    snapshots = [_tree_snapshot(root, relative) for relative in protected[:3]]
    return {
        "versions": versions,
        "manifest": manifest,
        "silver_rows": silver,
        "articles": articles,
        "silver_population_sha256": sha256_file(root / config["generation_input"]),
        "protected_snapshots": snapshots,
        "old_superseded_input_read": False,
    }


def _stable_rank(seed: str, stratum: str, article_id: str) -> str:
    return hashlib.sha256(f"{seed}|{stratum}|{article_id}".encode("utf-8")).hexdigest()


def _risk_features(article: dict[str, Any]) -> dict[str, Any]:
    paragraphs = article.get("paragraphs") or [row for row in article["body"].splitlines() if row.strip()]
    body = article["body"]
    title = article["title"]
    number_count = len(extract_number_tokens(body)) + len(extract_date_tokens(body))
    entity_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()·-]{2,}(?:公司|集团|中心|平台|实验室|大学|研究院|委员会|联盟)", body))
    relation_count = len(re.findall(r"联合|合作|携手|签署|发布|启动|建设|实现|推动|计划|将", body))
    stage_count = len(re.findall(r"首先|随后|同时|此外|下一步|未来|目前|截至", body))
    max_paragraph = max((len(row) for row in paragraphs), default=0)
    title_tokens = set(extract_number_tokens(title)) | set(extract_date_tokens(title))
    body_tokens = set(extract_number_tokens(body)) | set(extract_date_tokens(body))
    grammatical_first_person = bool(TRUE_FIRST_PERSON.search(body))
    direct_quote_risk = bool(re.search(r"[“\"](?:我们|我方|本人|我)[，。！？、\s]", body))
    title_gap_signal = bool(title_tokens - body_tokens)
    structure_score = max_paragraph + 30 * entity_count + 20 * number_count + 15 * relation_count + 20 * stage_count
    policy_score = 1000 * grammatical_first_person + 800 * direct_quote_risk + 300 * title_gap_signal + 2 * len(title)
    return {
        "character_count": int(article.get("character_count", len(body))),
        "paragraph_count": len(paragraphs),
        "max_paragraph_length": max_paragraph,
        "entity_signal_count": entity_count,
        "number_date_signal_count": number_count,
        "relation_signal_count": relation_count,
        "stage_signal_count": stage_count,
        "grammatical_first_person_signal": grammatical_first_person,
        "direct_quote_signal": direct_quote_risk,
        "title_body_number_date_gap_signal": title_gap_signal,
        "structure_score": structure_score,
        "policy_score": policy_score,
    }


def select_pilot(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    seed = config["pilot_seed"]
    features = {article_id: _risk_features(article) for article_id, article in context["articles"].items()}
    by_split = {split: [row for row in context["silver_rows"] if row["split"] == split] for split in SPLITS}
    quotas = config["selection_quotas"]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for split in SPLITS:
        population = by_split[split]
        lengths = sorted(features[row["article_id"]]["character_count"] for row in population)
        structures = sorted(features[row["article_id"]]["structure_score"] for row in population)
        long_cut = lengths[math.floor(0.75 * (len(lengths) - 1))]
        structure_cut = structures[math.floor(0.80 * (len(structures) - 1))]
        pools = {
            "policy_title_risk": [row for row in population if features[row["article_id"]]["policy_score"] >= 300],
            "source_structure_risk": [row for row in population if features[row["article_id"]]["structure_score"] >= structure_cut],
            "long_high_complexity": [row for row in population if features[row["article_id"]]["character_count"] >= long_cut],
            "ordinary_random": list(population),
        }
        for stratum in ("policy_title_risk", "source_structure_risk", "long_high_complexity", "ordinary_random"):
            available = [row for row in pools[stratum] if row["article_id"] not in used]
            available.sort(key=lambda row: (_stable_rank(seed, stratum, row["article_id"]), row["article_id"]))
            need = int(quotas[stratum][split])
            _require(len(available) >= need, f"insufficient {split}/{stratum} selection pool")
            for row in available[:need]:
                article_id = row["article_id"]
                used.add(article_id)
                selected.append(
                    {
                        "article_id": article_id,
                        "split": split,
                        "primary_risk_stratum": stratum,
                        "risk_features": features[article_id],
                    }
                )
    selected.sort(key=lambda row: (SPLITS.index(row["split"]), row["primary_risk_stratum"], row["article_id"]))
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"task011d_e2_pilot_{index:03d}"
    split_counts = Counter(row["split"] for row in selected)
    strata_counts = Counter(row["primary_risk_stratum"] for row in selected)
    _require(len(selected) == len(used) == 50, "pilot selection must contain 50 unique articles")
    _require(split_counts == Counter(config["pilot_split_counts"]), "pilot split distribution mismatch")
    _require(strata_counts == Counter({"ordinary_random": 30, "long_high_complexity": 10, "source_structure_risk": 5, "policy_title_risk": 5}), "risk stratum mismatch")
    selection_sha = _sha([(row["sample_id"], row["article_id"], row["split"], row["primary_risk_stratum"]) for row in selected])
    return {
        "schema_version": "task011d-e2-pilot-selection-manifest-v2.0.0",
        "task_id": config["task_id"],
        "source_news_version": config["news_version"],
        "event_group_version": config["event_group_version"],
        "split_version": config["split_version"],
        "silver_population_count": config["silver_count"],
        "silver_population_sha256": context["silver_population_sha256"],
        "seed": seed,
        "selection_algorithm": "seeded_sha256_rank_with_fixed_split_and_primary_risk_stratum_quotas_v1",
        "selection_executed": True,
        "pilot_sample_count": 50,
        "selected_article_ids": [row["article_id"] for row in selected],
        "split_distribution": dict(split_counts),
        "risk_strata": dict(strata_counts),
        "selection_sha256": selection_sha,
        "selected_samples": selected,
        "old_superseded_input_read": False,
    }


def _source_view(article: dict[str, Any]) -> dict[str, Any]:
    paragraphs = article.get("paragraphs") or [row for row in article["body"].splitlines() if row.strip()]
    return {
        "article_id": article["article_id"],
        "title": article["title"],
        "publish_date": article.get("publish_date"),
        "paragraphs": [
            {"paragraph_id": index, "text": text, "text_sha256": sha256_text(text)}
            for index, text in enumerate(paragraphs, 1)
        ],
    }


def _copy_risk(candidate: dict[str, Any]) -> dict[str, Any]:
    sentences = [row.strip() for row in SENTENCE_SPLIT.split(candidate["target_body"]) if row.strip()]
    best = (0.0, "", "")
    for fact in candidate["fact_points"]:
        fact_text = fact["fact"].removeprefix(FACT_PREFIX)
        for sentence in sentences:
            ratio = SequenceMatcher(None, normalize_text(fact_text), normalize_text(sentence)).ratio()
            if ratio > best[0]:
                best = (ratio, fact["fact_id"], sha256_text(sentence))
    return {
        "metric_name": "maximum_fact_to_target_body_sentence_sequence_matcher_ratio",
        "maximum_ratio": round(best[0], 6),
        "maximum_fact_id": best[1],
        "maximum_target_sentence_sha256": best[2],
        "warning_threshold": 0.88,
        "blocking_threshold": 0.96,
        "metric_role": "signal_only",
        "signal_level": "blocking" if best[0] >= 0.96 else "warning" if best[0] >= 0.88 else "pass",
    }


def materialize_candidate(
    article: dict[str, Any],
    selection: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    *,
    version: str,
    created_at: str,
) -> dict[str, Any]:
    paragraphs = article.get("paragraphs") or [row for row in article["body"].splitlines() if row.strip()]
    facts_input = list(semantic["facts"])
    controlled = semantic.get("controlled_title_fact")
    if controlled and not any(normalize_text(row["fact"]) == normalize_text(controlled) for row in facts_input):
        facts_input.append({"fact": controlled, "evidence_paragraph_ids": [], "fact_type": "event"})
    facts: list[dict[str, Any]] = []
    for index, row in enumerate(facts_input, 1):
        fact_text = row["fact"].strip()
        if not fact_text.startswith(FACT_PREFIX):
            fact_text = FACT_PREFIX + fact_text
        ids = [int(value) for value in row.get("evidence_paragraph_ids", [])]
        valid_ids = [value for value in ids if 1 <= value <= len(paragraphs)]
        title_evidence = not ids and controlled and normalize_text(row["fact"]) == normalize_text(controlled)
        evidence_text = article["title"] if title_evidence else normalize_evidence([paragraphs[value - 1] for value in valid_ids])
        fact = {
            "fact_id": f"F{index:02d}",
            "fact": fact_text,
            "evidence_paragraph_ids": ids,
            "evidence_text_sha256": sha256_text(evidence_text),
            "verification_status": "pending_ai_review",
            "fact_type": row["fact_type"],
            "contains_number": bool(extract_number_tokens(fact_text)),
            "contains_date": bool(extract_date_tokens(fact_text)),
            "contains_named_entity": bool(re.search(r"企业|有限公司|集团|公司|中心|研究院", fact_text)),
        }
        if title_evidence:
            fact.update(
                {
                    "evidence_type": "controlled_source_title",
                    "controlled_source_title_sha256": sha256_text(article["title"]),
                    "controlled_title_evidence_ref": f"news_v2.1:{article['article_id']}:title",
                }
            )
        facts.append(fact)
    constraints = fixed_constraints(article["length_bucket"])
    policy: list[dict[str, Any]] = []
    if semantic["target_first_person_compatibility"]:
        policy.append(
            {
                "exception_type": "target_first_person_compatibility",
                "scope": "current_sample_only",
                "applies_to_target_rendering": True,
                "applies_to_fact_generation": False,
                "semantic_rationale": semantic["first_person_rationale"],
            }
        )
    for fact in facts:
        if fact.get("evidence_type") == "controlled_source_title":
            policy.append(
                {
                    "exception_type": "controlled_source_title",
                    "scope": f"current_sample/{fact['fact_id']}",
                    "fact_id": fact["fact_id"],
                    "semantic_rationale": semantic["controlled_title_rationale"],
                }
            )
    target = render_target(article["title"], article["body"])
    user_prompt = render_user_prompt(semantic["topic"], facts, semantic["outline"], constraints)
    messages = render_messages(user_prompt, target)
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "candidate_profile": "task011d-e2-high-pilot-only-v1",
        "candidate_version": version,
        "batch_id": "task011d_e2_high_pilot",
        "sample_id": selection["sample_id"],
        "source_document_id": article["document_id"],
        "source_article_id": article["article_id"],
        "source_dataset_version": config["news_version"],
        "source_split_version": config["split_version"],
        "event_group_id": article["event_group_id"],
        "split": article["split"],
        "construction_method": "codexexec_real_semantic_generation_v1",
        "annotation_provider": "codex_exec",
        "annotation_model": config["semantic_model"],
        "annotation_prompt_version": "task011d-e2-generator-v1",
        "generation_context_policy": "isolated_sample_no_review_context",
        "generation_task_id": config["task_id"],
        "created_at": created_at,
        "review_status": "pending_ai_review",
        "reviewer_role": "AI_SFT_GENERATOR",
        "quality_issues": [],
        "validation_warnings": [],
        "automatic_status": "pending_deterministic_validation",
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "topic": semantic["topic"],
        "fact_points": facts,
        "outline": semantic["outline"],
        "constraints": constraints,
        "user_prompt": user_prompt,
        "user_prompt_version": USER_PROMPT_VERSION,
        "messages": messages,
        "messages_preview": messages,
        "target_title": article["title"],
        "target_body": article["body"],
        "target_text": target,
        "target_title_sha256": sha256_text(article["title"]),
        "target_body_sha256": article["body_sha256"],
        "target_text_sha256": sha256_text(target),
        "source_paragraphs": source_paragraph_records(paragraphs),
        "source_body_sha256": article["body_sha256"],
        "policy_exceptions": policy,
        "fact_coverage": {},
        "evidence_summary": {},
        "pilot_namespace": True,
        "full_run_reuse_authorized": False,
    }
    candidate["deterministic_copy_risk_signal"] = _copy_risk(candidate)
    return candidate


def deterministic_validate(candidate: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "topic", "fact_points", "outline", "constraints", "user_prompt", "messages", "target_title",
        "target_body", "target_text", "source_paragraphs", "policy_exceptions",
    }
    missing = sorted(required - set(candidate))
    if missing:
        errors.extend(f"schema_missing:{row}" for row in missing)
    facts = candidate["fact_points"]
    paragraphs = article.get("paragraphs") or [row for row in article["body"].splitlines() if row.strip()]
    if len(facts) < 3:
        errors.append("schema_fact_count_below_minimum")
    expected_ids = [f"F{index:02d}" for index in range(1, len(facts) + 1)]
    if [row["fact_id"] for row in facts] != expected_ids:
        errors.append("fact_id_sequence")
    target = render_target(article["title"], article["body"])
    if candidate["target_title"] != article["title"] or candidate["target_body"] != article["body"] or candidate["target_text"] != target:
        errors.append("target_mismatch")
    if candidate["target_text_sha256"] != sha256_text(target) or candidate["source_body_sha256"] != article["body_sha256"]:
        errors.append("target_or_source_checksum")
    if candidate["messages"] != render_messages(candidate["user_prompt"], target) or candidate["messages_preview"] != candidate["messages"]:
        errors.append("messages_sync")
    if candidate["user_prompt"] != render_user_prompt(candidate["topic"], facts, candidate["outline"], candidate["constraints"]):
        errors.append("prompt_sync")
    exact: set[str] = set()
    normalized: set[str] = set()
    body_evidence = title_evidence = 0
    for fact in facts:
        fact_id = fact["fact_id"]
        exact_value = fact["fact"].strip()
        normalized_value = re.sub(r"[^\w]", "", normalize_text(exact_value).lower(), flags=re.UNICODE)
        if exact_value in exact:
            errors.append(f"exact_duplicate:{fact_id}")
        if normalized_value in normalized:
            errors.append(f"normalized_duplicate:{fact_id}")
        exact.add(exact_value)
        normalized.add(normalized_value)
        if fact.get("evidence_type") == "controlled_source_title":
            title_evidence += 1
            if fact["evidence_paragraph_ids"] or fact["evidence_text_sha256"] != sha256_text(article["title"]):
                errors.append(f"controlled_title_evidence:{fact_id}")
            evidence = article["title"]
        else:
            body_evidence += 1
            ids = fact["evidence_paragraph_ids"]
            if not ids or any(not isinstance(value, int) or value < 1 or value > len(paragraphs) for value in ids):
                errors.append(f"evidence_reference:{fact_id}")
                continue
            evidence = normalize_evidence([paragraphs[value - 1] for value in ids])
            if fact["evidence_text_sha256"] != sha256_text(evidence):
                errors.append(f"evidence_hash:{fact_id}")
        if set(extract_number_tokens(fact["fact"])) - set(extract_number_tokens(evidence)):
            errors.append(f"unsupported_number:{fact_id}")
        if set(extract_date_tokens(fact["fact"])) - set(extract_date_tokens(evidence)):
            errors.append(f"unsupported_date:{fact_id}")
    if len(facts) > 16:
        warnings.append("fact_granularity_review_required")
    if len(facts) > 20:
        warnings.append("high_risk_fact_granularity")
    candidate["automatic_status"] = "ready_for_independent_review" if not errors else "blocked_deterministic_validation"
    candidate["validation_warnings"] = warnings
    candidate["evidence_summary"] = {
        "fact_count": len(facts),
        "body_evidence_fact_count": body_evidence,
        "controlled_title_evidence_fact_count": title_evidence,
        "evidence_hash_status": "pass" if not any("evidence" in row for row in errors) else "fail",
    }
    return {
        "status": "pass" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "schema_integrity": not any(row.startswith("schema_") or row == "fact_id_sequence" for row in errors),
        "target_exact": "target_mismatch" not in errors and "target_or_source_checksum" not in errors,
        "evidence_integrity": not any("evidence" in row for row in errors),
        "prompt_sync": "prompt_sync" not in errors,
        "messages_sync": "messages_sync" not in errors,
        "duplicate_integrity": not any("duplicate" in row for row in errors),
        "candidate_sha256": record_sha256(candidate),
        "provenance": "passed_pilot_only_no_prior_candidate_reuse",
    }


def _candidate_view(candidate: dict[str, Any]) -> dict[str, Any]:
    paragraph_by_id = {row["paragraph_id"]: row for row in candidate["source_paragraphs"]}
    facts = []
    for row in candidate["fact_points"]:
        evidence = []
        if row.get("evidence_type") == "controlled_source_title":
            evidence.append({"evidence_type": "controlled_source_title", "text": candidate["target_title"], "text_sha256": sha256_text(candidate["target_title"])})
        else:
            for paragraph_id in row["evidence_paragraph_ids"]:
                paragraph = paragraph_by_id.get(paragraph_id)
                evidence.append(paragraph or {"paragraph_id": paragraph_id, "missing": True})
        facts.append(
            {
                "fact_id": row["fact_id"],
                "fact": row["fact"],
                "evidence_paragraph_ids": row["evidence_paragraph_ids"],
                "evidence_type": row.get("evidence_type", "source_body"),
                "evidence": evidence,
                "fact_type": row["fact_type"],
            }
        )
    return {
        "sample_id": candidate["sample_id"],
        "topic": candidate["topic"],
        "fact_points": facts,
        "outline": candidate["outline"],
        "constraints": candidate["constraints"],
        "user_prompt": candidate["user_prompt"],
        "messages": candidate["messages"],
        "target_title": candidate["target_title"],
        "target_body": candidate["target_body"],
        "target_text": candidate["target_text"],
        "policy_exceptions": candidate["policy_exceptions"],
        "deterministic_copy_risk_signal": candidate["deterministic_copy_risk_signal"],
    }


def _decorate_issues(issues: list[dict[str, Any]], sample_id: str, role: str) -> list[dict[str, Any]]:
    return [
        dict(issue)
        | {
            "sample_id": sample_id,
            "field": issue["dimension"],
            "type": issue["issue_type"],
            "semantic_evidence": issue["rationale"],
            "recommended_action": issue["recommended_action"],
            "reviewer_role": role,
        }
        for issue in issues
        if issue.get("severity") != "none"
    ]


def _verdict(issues: list[dict[str, Any]]) -> str:
    severities = {row["severity"] for row in issues}
    if severities & {"blocking", "major"}:
        return "revision_required"
    if severities & {"minor", "warning"}:
        return "accepted_with_warning"
    return "accepted"


def _dimension_review(provider: SemanticModelProvider, article: dict[str, Any], candidate: dict[str, Any], role: str) -> dict[str, Any]:
    method = getattr(provider, "review_dimensions", None)
    if not callable(method):
        raise SemanticProviderError("dimension-first semantic reviewer is unavailable")
    raw = method(
        sample_id=candidate["sample_id"],
        split=candidate["split"],
        source=_source_view(article),
        candidate=_candidate_view(candidate),
        reviewer=role,
    )
    validation = validate_dimension_review(raw, [row["fact_id"] for row in candidate["fact_points"]])
    if not validation["passed"]:
        raise SemanticProviderError(f"dimension review incomplete: {candidate['sample_id']}:{role}")
    issues = _decorate_issues(canonical_issue_builder(raw), candidate["sample_id"], role)
    return {
        "sample_id": candidate["sample_id"],
        "reviewer_role": role,
        "verdict": _verdict(issues),
        "issues": issues,
        "review_summary": raw["review_summary"],
        "dimension_validation": validation,
        "dimension_review": raw,
        "output_sha256": _sha(raw),
    }


def _parallel_phase(
    phase: str,
    samples: list[dict[str, Any]],
    directory: Path,
    workers: int,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    phase_dir = directory / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    for sample in samples:
        path = phase_dir / f"{sample['sample_id']}.json"
        if path.exists():
            results[sample["sample_id"]] = read_json(path)
        else:
            pending.append(sample)
    if pending:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=phase) as executor:
            future_map = {executor.submit(worker, sample): sample for sample in pending}
            completed = len(results)
            for future in as_completed(future_map):
                sample = future_map[future]
                result = future.result()
                _write_json(phase_dir / f"{sample['sample_id']}.json", result)
                results[sample["sample_id"]] = result
                completed += 1
                print(f"{phase}: {completed}/{len(samples)} {sample['sample_id']}", flush=True)
    return [results[sample["sample_id"]] for sample in samples]


def _issue_key(issue: dict[str, Any]) -> tuple[Any, ...]:
    return (issue["issue_type"], issue["severity"], tuple(sorted(issue.get("fact_ids", []))))


def _major_issues(review: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in review["issues"] if row["severity"] in {"blocking", "major"}]


def _critical_disagreement(review_a: dict[str, Any], review_b: dict[str, Any]) -> bool:
    return {_issue_key(row) for row in _major_issues(review_a)} != {_issue_key(row) for row in _major_issues(review_b)}


def _confirmed_initial_issues(review_a: dict[str, Any], review_b: dict[str, Any], judge: dict[str, Any] | None) -> tuple[list[dict[str, Any]], bool]:
    if judge:
        if judge["decision"] == "quarantine_recommended":
            return list(judge["confirmed_issues"]), True
        if judge["decision"] == "accept_a":
            return _major_issues(review_a), False
        if judge["decision"] == "accept_b":
            return _major_issues(review_b), False
        combined = _major_issues(review_a) + _major_issues(review_b)
    else:
        keys_b = {_issue_key(row) for row in _major_issues(review_b)}
        combined = [row for row in _major_issues(review_a) if _issue_key(row) in keys_b]
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for issue in combined:
        unique[_issue_key(issue)] = issue
    return list(unique.values()), False


def _technical_issues(validation: dict[str, Any], sample_id: str, role: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "field": "deterministic_validation",
            "type": "deterministic_validation_failure",
            "issue_type": "deterministic_validation_failure",
            "dimension": "other",
            "severity": "blocking",
            "fact_ids": [],
            "semantic_evidence": error,
            "rationale": error,
            "recommended_action": "Repair only the stated deterministic integrity failure.",
            "reviewer_role": role,
        }
        for error in validation["errors"]
    ]


def _provider_failure_issue(sample_id: str, role: str, message: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "field": "semantic_provider",
        "type": "semantic_provider_failure",
        "issue_type": "semantic_provider_failure",
        "dimension": "other",
        "severity": "blocking",
        "fact_ids": [],
        "semantic_evidence": message,
        "rationale": message,
        "recommended_action": "Fail closed; do not substitute a deterministic, template, or alternate-model semantic result.",
        "reviewer_role": role,
    }


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _gold_sanity(root: Path) -> dict[str, Any]:
    gold = read_json(root / "gold_issue_replay_v4.json")
    targeted = read_json(root / "targeted_suite_post_governance.json")
    micro = read_json(root / "micro_regression_v5.json")
    false_positive = read_json(root / "false_positive_regression_v5.json")
    governance = read_json(root / "duplicate_fixture_governance_manifest.json")
    result = {
        "active_gold_blocking": f"{gold.get('blocking_detected')}/{gold.get('active_blocking_count')}",
        "active_gold_major": f"{gold.get('major_detected')}/{gold.get('active_major_count')}",
        "active_gold_overall": f"{gold.get('overall_active_detected')}/{gold.get('active_gold_count')}",
        "active_targeted": f"{targeted.get('active_targeted_detected_count')}/{targeted.get('active_targeted_fixture_count')}",
        "micro_regression": f"{micro.get('core_detected_count')}/{micro.get('case_count')}",
        "false_positive_regression_passed": false_positive.get("passed") is True,
        "duplicate_boundary_status": governance.get("governance_complete") is True and governance.get("historical_gold_preserved") is True,
    }
    result["passed"] = all(
        (
            gold.get("passed") is True,
            targeted.get("active_passed") is True,
            micro.get("passed") is True,
            false_positive.get("passed") is True,
            result["duplicate_boundary_status"],
        )
    )
    return result


def _invocation_audit(path: Path) -> dict[str, Any]:
    records = read_jsonl(path) if path.exists() else []
    role_counts = Counter(row["role"] for row in records if row["status"] == "succeeded")
    role_attempt_counts = Counter(row["role"] for row in records)
    ids = [row["invocation_id"] for row in records]
    return {
        "schema_version": "task011d-e2-pilot-invocation-audit-v1.0.0",
        "task_id": "TASK-011D-E2",
        "semantic_invocation_total": len(records),
        "succeeded": sum(row["status"] == "succeeded" for row in records),
        "failed": sum(row["status"] != "succeeded" for row in records),
        "unique_invocation_id_count": len(set(ids)),
        "all_invocation_ids_unique": len(ids) == len(set(ids)),
        "role_counts": dict(role_counts),
        "role_attempt_counts": dict(role_attempt_counts),
        "generation_samples": 50,
        "generation_calls": role_counts["AI_SFT_GENERATOR"],
        "generation_call_attempts": role_attempt_counts["AI_SFT_GENERATOR"],
        "review_a_samples": 50,
        "review_a_calls": role_counts["AI_INDEPENDENT_REVIEWER_A"],
        "review_a_call_attempts": role_attempt_counts["AI_INDEPENDENT_REVIEWER_A"],
        "review_b_samples": 50,
        "review_b_calls": role_counts["AI_INDEPENDENT_REVIEWER_B"],
        "review_b_call_attempts": role_attempt_counts["AI_INDEPENDENT_REVIEWER_B"],
        "revision_calls": role_counts["AI_REVISION_AGENT"],
        "revision_call_attempts": role_attempt_counts["AI_REVISION_AGENT"],
        "re_review_calls": role_counts["AI_INDEPENDENT_RE_REVIEWER"],
        "re_review_call_attempts": role_attempt_counts["AI_INDEPENDENT_RE_REVIEWER"],
        "judge_calls": role_counts["AI_SFT_JUDGE"],
        "judge_call_attempts": role_attempt_counts["AI_SFT_JUDGE"],
        "pilot_auditor_samples": 50,
        "pilot_auditor_calls": role_counts["PILOT_QUALITY_AUDITOR"],
        "pilot_auditor_call_attempts": role_attempt_counts["PILOT_QUALITY_AUDITOR"],
        "tool_call_count": sum(int(row.get("tool_call_count", 0)) for row in records),
        "fallback_count": sum(bool(row.get("fallback_used")) for row in records),
        "model_requested": sorted({row.get("model_requested") for row in records}),
        "reasoning_effort_requested": sorted({row.get("reasoning_effort_requested") for row in records}),
        "audit_passed": (
            records
            and all(row["status"] == "succeeded" for row in records)
            and len(ids) == len(set(ids))
            and not any(row.get("tool_call_count") or row.get("fallback_used") for row in records)
        ),
        "invocation_ids": ids,
    }


def _summary_for_reviews(role: str, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(row["verdict"] for row in reviews)
    issues = [issue for row in reviews for issue in row["issues"]]
    return {
        "schema_version": "task011d-e2-pilot-review-summary-v1.0.0",
        "task_id": "TASK-011D-E2",
        "reviewer_role": role,
        "sample_count": len(reviews),
        "accepted": verdicts["accepted"],
        "warning": verdicts["accepted_with_warning"],
        "revision": verdicts["revision_required"],
        "issue_count": len(issues),
        "issue_severity_counts": dict(Counter(row["severity"] for row in issues)),
        "issue_type_counts": dict(Counter(row["issue_type"] for row in issues)),
        "dimension_validation_passed": sum(row["dimension_validation"]["passed"] for row in reviews),
        "semantic_invocation": True,
    }


def run_high_pilot(root: Path, config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    _require(config["pilot_count"] == 50 and not config["full_run_execution_allowed"], "unsafe pilot configuration")
    context = load_context(root, config)
    selection_manifest = select_pilot(context, config)
    selection_path = root / "pilot_selection_manifest.json"
    if selection_path.exists() and resume:
        existing = read_json(selection_path)
        _require(existing.get("selection_sha256") == selection_manifest["selection_sha256"], "pilot selection drift")
    _write_json(selection_path, selection_manifest)
    selected = selection_manifest["selected_samples"]
    selected_by_sample = {row["sample_id"]: row for row in selected}
    articles = {row["sample_id"]: context["articles"][row["article_id"]] for row in selected}
    runtime = root / config["pilot_output_dir"]
    runtime.mkdir(parents=True, exist_ok=True)
    invocation_path = root / config["invocation_log"]
    if invocation_path.exists() and not resume:
        raise HighPilotError("invocation log already exists; use --resume")
    provider = create_semantic_provider(config, root=root)
    workers = int(config.get("semantic_concurrency", 4))
    started_at = _timestamp()

    def generate(sample: dict[str, Any]) -> dict[str, Any]:
        article = articles[sample["sample_id"]]
        semantic = provider.generate_sft(sample_id=sample["sample_id"], split=sample["split"], source=_source_view(article))
        candidate = materialize_candidate(article, sample, semantic, config, version="pilot_candidate_v1", created_at=started_at)
        validation = deterministic_validate(candidate, article)
        return {"sample_id": sample["sample_id"], "semantic_generation": semantic, "candidate": candidate, "deterministic_validation": validation}

    generations = _parallel_phase("generation", selected, runtime, workers, generate)
    generation_by_id = {row["sample_id"]: row for row in generations}
    candidates = {row["sample_id"]: row["candidate"] for row in generations}

    def review_a(sample: dict[str, Any]) -> dict[str, Any]:
        return _dimension_review(provider, articles[sample["sample_id"]], candidates[sample["sample_id"]], "AI_INDEPENDENT_REVIEWER_A")

    def review_b(sample: dict[str, Any]) -> dict[str, Any]:
        return _dimension_review(provider, articles[sample["sample_id"]], candidates[sample["sample_id"]], "AI_INDEPENDENT_REVIEWER_B")

    reviews_a = _parallel_phase("review_a", selected, runtime, workers, review_a)
    reviews_b = _parallel_phase("review_b", selected, runtime, workers, review_b)
    review_a_by_id = {row["sample_id"]: row for row in reviews_a}
    review_b_by_id = {row["sample_id"]: row for row in reviews_b}
    judge_samples = [row for row in selected if _critical_disagreement(review_a_by_id[row["sample_id"]], review_b_by_id[row["sample_id"]])]

    def judge(sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        result = provider.judge_candidate(
            sample_id=sample_id,
            split=sample["split"],
            source=_source_view(articles[sample_id]),
            candidate=_candidate_view(candidates[sample_id]),
            reviews=[review_a_by_id[sample_id], review_b_by_id[sample_id]],
        )
        return {"sample_id": sample_id, **result}

    judges = _parallel_phase("judge", judge_samples, runtime, workers, judge) if judge_samples else []
    judge_by_id = {row["sample_id"]: row for row in judges}
    failed_revision_samples = {
        row["sample_id"]
        for row in (read_jsonl(invocation_path) if invocation_path.exists() else [])
        if row.get("role") == "AI_REVISION_AGENT" and row.get("status") != "succeeded"
    }

    def revise(sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        article = articles[sample_id]
        current = candidates[sample_id]
        if sample_id in failed_revision_samples:
            issue = _provider_failure_issue(sample_id, "AI_REVISION_AGENT", "A prior pilot Revision invocation failed or timed out; fail-closed terminal state retained on resume.")
            return {
                "sample_id": sample_id,
                "revision_rounds": [],
                "final_candidate": current,
                "final_validation": generation_by_id[sample_id]["deterministic_validation"],
                "terminal_state": "quarantined",
                "terminal_issues": [issue],
            }
        confirmed, judge_quarantine = _confirmed_initial_issues(review_a_by_id[sample_id], review_b_by_id[sample_id], judge_by_id.get(sample_id))
        if generation_by_id[sample_id]["deterministic_validation"]["status"] != "pass":
            confirmed.extend(_technical_issues(generation_by_id[sample_id]["deterministic_validation"], sample_id, "DETERMINISTIC_VALIDATOR"))
        rounds: list[dict[str, Any]] = []
        if judge_quarantine:
            return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": generation_by_id[sample_id]["deterministic_validation"], "terminal_state": "quarantined", "terminal_issues": confirmed}
        if not confirmed:
            all_nonblocking = review_a_by_id[sample_id]["issues"] + review_b_by_id[sample_id]["issues"]
            state = "accepted_with_warning" if all_nonblocking else "accepted"
            return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": generation_by_id[sample_id]["deterministic_validation"], "terminal_state": state, "terminal_issues": all_nonblocking}
        remaining = confirmed
        final_validation = generation_by_id[sample_id]["deterministic_validation"]
        for round_number in range(1, int(config["max_revision_rounds"]) + 1):
            try:
                semantic = provider.revise_sft(
                    sample_id=sample_id,
                    split=sample["split"],
                    source=_source_view(article),
                    candidate=_candidate_view(current),
                    issues=remaining,
                    revision_round=round_number,
                )
            except SemanticProviderError as exc:
                issue = _provider_failure_issue(sample_id, "AI_REVISION_AGENT", str(exc))
                return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": final_validation, "terminal_state": "quarantined", "terminal_issues": [*remaining, issue]}
            current = materialize_candidate(article, sample, semantic, config, version=f"pilot_candidate_v{round_number + 1}", created_at=_timestamp())
            final_validation = deterministic_validate(current, article)
            try:
                rereview = _dimension_review(provider, article, current, "AI_INDEPENDENT_RE_REVIEWER")
            except SemanticProviderError as exc:
                issue = _provider_failure_issue(sample_id, "AI_INDEPENDENT_RE_REVIEWER", str(exc))
                rounds.append({"revision_round": round_number, "semantic_revision": semantic, "candidate": current, "deterministic_validation": final_validation, "re_review": {"sample_id": sample_id, "reviewer_role": "AI_INDEPENDENT_RE_REVIEWER", "verdict": "quarantine_recommended", "issues": [issue], "provider_failure": True}})
                return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": final_validation, "terminal_state": "quarantined", "terminal_issues": [issue]}
            if final_validation["status"] != "pass":
                rereview["issues"].extend(_technical_issues(final_validation, sample_id, "DETERMINISTIC_VALIDATOR"))
                rereview["verdict"] = "revision_required"
            remaining = [row for row in rereview["issues"] if row["severity"] in {"blocking", "major"}]
            rounds.append(
                {
                    "revision_round": round_number,
                    "semantic_revision": semantic,
                    "candidate": current,
                    "deterministic_validation": final_validation,
                    "re_review": rereview,
                }
            )
            if not remaining:
                state = "accepted_with_warning" if rereview["issues"] else "accepted"
                return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": final_validation, "terminal_state": state, "terminal_issues": rereview["issues"]}
        return {"sample_id": sample_id, "revision_rounds": rounds, "final_candidate": current, "final_validation": final_validation, "terminal_state": "quarantined", "terminal_issues": remaining}

    revisions = _parallel_phase("revision", selected, runtime, workers, revise)
    revision_by_id = {row["sample_id"]: row for row in revisions}

    def audit(sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        try:
            reviewed = _dimension_review(provider, articles[sample_id], revision_by_id[sample_id]["final_candidate"], "PILOT_QUALITY_AUDITOR")
        except SemanticProviderError as exc:
            issue = _provider_failure_issue(sample_id, "PILOT_QUALITY_AUDITOR", str(exc))
            return {"sample_id": sample_id, "audit_status": "blocking", "reviewer_role": "PILOT_QUALITY_AUDITOR", "verdict": "quarantine_recommended", "issues": [issue], "review_summary": str(exc), "provider_failure": True}
        severities = [SEVERITY_RANK[row["severity"]] for row in reviewed["issues"]]
        maximum = max(severities, default=0)
        status = {0: "pass", 1: "minor", 2: "minor", 3: "major", 4: "blocking"}[maximum]
        return {"sample_id": sample_id, "audit_status": status, **reviewed}

    audits = _parallel_phase("independent_audit", selected, runtime, workers, audit)

    review_a_summary = _summary_for_reviews("AI_INDEPENDENT_REVIEWER_A", reviews_a)
    review_b_summary = _summary_for_reviews("AI_INDEPENDENT_REVIEWER_B", reviews_b)
    _write_json(root / "pilot_review_a_summary.json", review_a_summary)
    _write_json(root / "pilot_review_b_summary.json", review_b_summary)
    invocation_audit = _invocation_audit(invocation_path)
    _write_json(root / "pilot_invocation_audit.json", invocation_audit)

    agreement_verdict = sum(review_a_by_id[row["sample_id"]]["verdict"] == review_b_by_id[row["sample_id"]]["verdict"] for row in selected)
    agreement_issue = sum(
        {_issue_key(issue) for issue in review_a_by_id[row["sample_id"]]["issues"]}
        == {_issue_key(issue) for issue in review_b_by_id[row["sample_id"]]["issues"]}
        for row in selected
    )
    exact_output = sum(review_a_by_id[row["sample_id"]]["output_sha256"] == review_b_by_id[row["sample_id"]]["output_sha256"] for row in selected)
    explanation_similarity = [
        SequenceMatcher(None, review_a_by_id[row["sample_id"]]["review_summary"], review_b_by_id[row["sample_id"]]["review_summary"]).ratio()
        for row in selected
    ]
    independence_suspicion = exact_output >= 45 or (agreement_issue >= 48 and statistics.mean(explanation_similarity) >= 0.95)

    final_validations = [row["final_validation"] for row in revisions]
    terminal = Counter(row["terminal_state"] for row in revisions)
    fact_counts = [len(row["final_candidate"]["fact_points"]) for row in revisions]
    all_review_issues = [issue for review in reviews_a + reviews_b for issue in review["issues"]]
    audit_statuses = Counter(row["audit_status"] for row in audits)
    audit_issues = [issue for row in audits for issue in row["issues"]]
    audit_major_samples = {row["sample_id"] for row in audits if row["audit_status"] == "major"}
    audit_blocking_samples = {row["sample_id"] for row in audits if row["audit_status"] == "blocking"}
    overfragmentation_samples = {
        row["sample_id"]
        for row in audits
        if any(issue["issue_type"] in {"fact_micro_fragmentation", "over_fragmentation"} and issue["severity"] in {"major", "blocking"} for issue in row["issues"])
    }
    policy_rows = []
    title_rows = []
    for sample in selected:
        sample_id = sample["sample_id"]
        candidate = revision_by_id[sample_id]["final_candidate"]
        article = articles[sample_id]
        audit_row = next(item for item in audits if item["sample_id"] == sample_id)
        target_policy = audit_row.get("dimension_review", {}).get("target_policy", {})
        first_person = bool(target_policy.get("grammatical_first_person_present", False))
        policy_violation = bool(target_policy.get("first_person_policy_violation", False))
        exception = any(row["exception_type"] == "target_first_person_compatibility" for row in candidate["policy_exceptions"])
        policy_rows.append({"sample_id": sample_id, "first_person_candidate": first_person, "policy_violation": policy_violation, "exception_triggered": exception, "true_exception": first_person and exception and not policy_violation, "false_positive": exception and not first_person, "unnecessary_exception": exception and (not first_person or policy_violation)})
        title_facts = [row for row in candidate["fact_points"] if row.get("evidence_type") == "controlled_source_title"]
        invalid = audit_row.get("provider_failure", False) or any(row["issue_type"] == "title_body_evidence_gap" for row in audit_row["issues"])
        body_supports = (
            audit_row.get("dimension_review", {})
            .get("sample_level_review", {})
            .get("title_body_semantic_consistency", {})
            .get("body_supports_same_semantics", False)
        )
        title_rows.append({"sample_id": sample_id, "triggered": bool(title_facts), "fact_count": len(title_facts), "necessary": bool(title_facts) and not body_supports and not invalid, "unnecessary": bool(title_facts) and body_supports, "body_already_supports": bool(title_facts) and body_supports, "invalid": bool(title_facts) and invalid})

    policy_audit = {
        "schema_version": "task011d-e2-pilot-policy-audit-v1.0.0",
        "first_person_candidate_count": sum(row["first_person_candidate"] for row in policy_rows),
        "triggered": sum(row["exception_triggered"] for row in policy_rows),
        "true_exception": sum(row["true_exception"] for row in policy_rows),
        "false_positive": sum(row["false_positive"] for row in policy_rows),
        "unnecessary_exception": sum(row["unnecessary_exception"] for row in policy_rows),
        "systematic_policy_error": sum(row["false_positive"] for row in policy_rows) >= 3,
        "samples": policy_rows,
    }
    title_audit = {
        "schema_version": "task011d-e2-pilot-title-evidence-audit-v1.0.0",
        "triggered": sum(row["triggered"] for row in title_rows),
        "necessary": sum(row["necessary"] for row in title_rows),
        "valid": sum(row["necessary"] for row in title_rows),
        "unnecessary": sum(row["unnecessary"] for row in title_rows),
        "body_already_supports": sum(row["body_already_supports"] for row in title_rows),
        "invalid": sum(row["invalid"] for row in title_rows),
        "systematic_title_evidence_abuse": sum(row["unnecessary"] or row["invalid"] for row in title_rows) >= 3,
        "samples": title_rows,
    }
    _write_json(root / "pilot_policy_audit.json", policy_audit)
    _write_json(root / "pilot_title_evidence_audit.json", title_audit)

    revision_summary = {
        "schema_version": "task011d-e2-pilot-revision-summary-v1.0.0",
        "samples_revised": sum(bool(row["revision_rounds"]) for row in revisions),
        "revision_v1_count": sum(len(row["revision_rounds"]) >= 1 for row in revisions),
        "revision_v2_count": sum(len(row["revision_rounds"]) >= 2 for row in revisions),
        "re_review_count": sum(len(row["revision_rounds"]) for row in revisions),
        "terminal_states": dict(terminal),
    }
    judge_summary = {"schema_version": "task011d-e2-pilot-judge-summary-v1.0.0", "judge_count": len(judges), "decision_counts": dict(Counter(row["decision"] for row in judges)), "samples": [{"sample_id": row["sample_id"], "decision": row["decision"]} for row in judges]}
    generation_manifest = {
        "schema_version": "task011d-e2-pilot-generation-manifest-v1.0.0",
        "sample_count": len(generations),
        "generation_calls": invocation_audit["generation_calls"],
        "deterministic_pass": sum(row["deterministic_validation"]["status"] == "pass" for row in generations),
        "target_exact": sum(row["deterministic_validation"]["target_exact"] for row in generations),
        "schema_integrity": sum(row["deterministic_validation"]["schema_integrity"] for row in generations),
        "evidence_integrity": sum(row["deterministic_validation"]["evidence_integrity"] for row in generations),
        "candidate_set_sha256": _sha([row["deterministic_validation"]["candidate_sha256"] for row in generations]),
        "old_candidate_reused": False,
        "old_superseded_input_read": False,
    }
    _write_json(root / "pilot_generation_manifest.json", generation_manifest)
    _write_json(root / "pilot_revision_summary.json", revision_summary)
    _write_json(root / "pilot_judge_summary.json", judge_summary)

    auditor_safe_samples = [
        {
            "sample_id": row["sample_id"],
            "audit_status": row["audit_status"],
            "issue_counts": dict(Counter(issue["severity"] for issue in row["issues"])),
            "issue_types": sorted({issue["issue_type"] for issue in row["issues"]}),
            "fact_support": "fail" if any(issue["issue_type"] == "fact_unsupported" for issue in row["issues"]) else "pass",
            "evidence": "fail" if any(issue["issue_type"] == "evidence_incorrect" for issue in row["issues"]) else "pass",
            "atomicity": "fail" if any(issue["issue_type"] == "fact_not_atomic" for issue in row["issues"]) else "pass",
            "over_fragmentation": "fail" if any(issue["issue_type"] == "fact_micro_fragmentation" for issue in row["issues"]) else "pass",
            "duplicate": "fail" if any(issue["issue_type"] == "fact_duplicate" for issue in row["issues"]) else "pass",
            "unsupported_inference": "fail" if any(issue["issue_type"] == "unsupported_inference" for issue in row["issues"]) else "pass",
            "missing_fact": "fail" if any(issue["issue_type"] == "missing_critical_fact" for issue in row["issues"]) else "pass",
            "coverage": "fail" if any(issue["issue_type"] in {"missing_critical_fact", "semantic_coverage_gap"} for issue in row["issues"]) else "pass",
            "policy": "fail" if any(issue["issue_type"] == "policy_compatibility_needs_resolution" for issue in row["issues"]) else "pass",
        }
        for row in audits
    ]
    independent_audit = {
        "schema_version": "task011d-e2-pilot-independent-quality-audit-v1.0.0",
        "auditor_role": "PILOT_QUALITY_AUDITOR",
        "sample_count": len(audits),
        "review_verdicts_visible": False,
        "audit_status_counts": dict(audit_statuses),
        "issue_severity_counts": dict(Counter(row["severity"] for row in audit_issues)),
        "issue_type_counts": dict(Counter(row["issue_type"] for row in audit_issues)),
        "samples": auditor_safe_samples,
    }
    _write_json(root / "pilot_independent_quality_audit.json", independent_audit)

    gold_sanity = _gold_sanity(root)
    hard_gate = (
        not audit_blocking_samples
        and all(row["target_exact"] for row in final_validations)
        and all(row["schema_integrity"] for row in final_validations)
        and all(row["evidence_integrity"] for row in final_validations)
        and all(row["provenance"].startswith("passed") for row in final_validations)
        and sum(terminal.values()) == 50
        and invocation_audit["audit_passed"]
        and not independence_suspicion
    )
    major_rate = len(audit_major_samples) / 50
    overfragmentation_rate = len(overfragmentation_samples) / 50
    quality_gate = major_rate <= 0.05 and overfragmentation_rate <= 0.05 and not policy_audit["systematic_policy_error"] and not title_audit["systematic_title_evidence_abuse"] and gold_sanity["passed"]
    final_status = "passed" if hard_gate and quality_gate else "failed"
    quality = {
        "schema_version": "task011d-e2-50-sample-high-semantic-pilot-summary-v1.0.0",
        "task_id": "TASK-011D-E2",
        "role": config["role"],
        "news_version": config["news_version"],
        "event_group_version": config["event_group_version"],
        "split_version": config["split_version"],
        "silver_population": 2117,
        "pilot_count": 50,
        "pilot_split_counts": selection_manifest["split_distribution"],
        "risk_strata": selection_manifest["risk_strata"],
        "model": config["semantic_model"],
        "reasoning_effort": config["reasoning_effort"],
        "generation_sample_count": 50,
        "generation_calls": invocation_audit["generation_calls"],
        "review_a_sample_count": 50,
        "review_a_calls": invocation_audit["review_a_calls"],
        "review_b_sample_count": 50,
        "review_b_calls": invocation_audit["review_b_calls"],
        "ab_verdict_agreement_count": agreement_verdict,
        "ab_verdict_agreement_rate": agreement_verdict / 50,
        "ab_issue_agreement_count": agreement_issue,
        "ab_issue_agreement_rate": agreement_issue / 50,
        "exact_output_match_count": exact_output,
        "exact_output_match_rate": exact_output / 50,
        "visible_explanation_similarity_mean": round(statistics.mean(explanation_similarity), 6),
        "reviewer_independence_suspicion": independence_suspicion,
        "total_generated_fact": sum(fact_counts),
        "fact_mean": round(statistics.mean(fact_counts), 4),
        "fact_median": statistics.median(fact_counts),
        "fact_p75": round(_quantile(fact_counts, 0.75), 4),
        "fact_p90": round(_quantile(fact_counts, 0.90), 4),
        "fact_max": max(fact_counts),
        "fact_gt_12": sum(value > 12 for value in fact_counts),
        "fact_gt_16": sum(value > 16 for value in fact_counts),
        "fact_gt_20": sum(value > 20 for value in fact_counts),
        "review_issue_total": len(all_review_issues),
        "review_issue_severity_counts": dict(Counter(row["severity"] for row in all_review_issues)),
        "revision_v1_count": revision_summary["revision_v1_count"],
        "revision_v2_count": revision_summary["revision_v2_count"],
        "re_review_count": revision_summary["re_review_count"],
        "judge_count": len(judges),
        "terminal_states": dict(terminal),
        "pilot_auditor_counts": dict(audit_statuses),
        "major_audit_count": len(audit_major_samples),
        "major_audit_rate": major_rate,
        "audit_blocking_count": len(audit_blocking_samples),
        "major_overfragmentation_count": len(overfragmentation_samples),
        "major_overfragmentation_rate": overfragmentation_rate,
        "policy_audit": {key: value for key, value in policy_audit.items() if key != "samples"},
        "title_evidence_audit": {key: value for key, value in title_audit.items() if key != "samples"},
        "target_mismatch": sum(not row["target_exact"] for row in final_validations),
        "schema_failure": sum(not row["schema_integrity"] for row in final_validations),
        "evidence_technical_failure": sum(not row["evidence_integrity"] for row in final_validations),
        "provenance_failure": sum(not row["provenance"].startswith("passed") for row in final_validations),
        "unfinished": 50 - sum(terminal.values()),
        "gold_sanity_regression": gold_sanity,
        "semantic_invocation_total": invocation_audit["semantic_invocation_total"],
        "high_pilot_hard_gate": "passed" if hard_gate else "failed",
        "high_pilot_quality_gate": "passed" if quality_gate else "failed",
        "high_pilot_status": final_status,
        "medium_shadow_ready": final_status == "passed",
        "medium_shadow_executed": False,
        "full_run_executed": False,
        "sft_v2_created": False,
        "training_executed": False,
        "protected_snapshots_before": context["protected_snapshots"],
        "protected_snapshots_after": [_tree_snapshot(root, row["path"]) for row in context["protected_snapshots"]],
    }
    quality["protected_sources_unchanged"] = quality["protected_snapshots_before"] == quality["protected_snapshots_after"]
    if not quality["protected_sources_unchanged"]:
        raise HighPilotError("protected v2.1 source changed during pilot")
    if final_status == "failed":
        quality["failure_classes"] = [
            name
            for name, failed in (
                ("hard_gate", not hard_gate),
                ("major_audit_rate", major_rate > 0.05),
                ("major_overfragmentation", overfragmentation_rate > 0.05),
                ("policy", policy_audit["systematic_policy_error"]),
                ("controlled_title", title_audit["systematic_title_evidence_abuse"]),
                ("gold_regression", not gold_sanity["passed"]),
            )
            if failed
        ]
        quality["affected_sample_count"] = len(audit_major_samples | audit_blocking_samples | overfragmentation_samples)
        quality["issue_families"] = dict(Counter(row["issue_type"] for row in audit_issues if row["severity"] in {"major", "blocking"}))
        quality["minimum_repair_scope"] = "Repair only the failing Generator, Reviewer, or Policy issue families; do not rerun the 50-sample pilot without new authorization."
    _write_json(root / "pilot_quality_summary.json", quality)

    report_path = root / config["report_path"]
    report = f"""# TASK-011D-E2 50-Sample High Semantic Pilot

- Role: `{config['role']}`
- Formal versions: `{config['news_version']}`, `{config['event_group_version']}`, `{config['split_version']}`
- Silver population / pilot: 2117 / 50 (`train=40`, `validation=5`, `test=5`)
- Risk strata: `{json.dumps(selection_manifest['risk_strata'], ensure_ascii=False, sort_keys=True)}`
- Semantic provider: CodexExec `{config['semantic_model']}` / `{config['reasoning_effort']}`
- Generation / Review A / Review B / Auditor samples: 50 / 50 / 50 / 50
- Review A accepted / warning / revision: {review_a_summary['accepted']} / {review_a_summary['warning']} / {review_a_summary['revision']}; Review B: {review_b_summary['accepted']} / {review_b_summary['warning']} / {review_b_summary['revision']}
- A/B verdict agreement: {agreement_verdict}/50; issue agreement: {agreement_issue}/50; exact output match: {exact_output}/50
- Final Fact count: total {sum(fact_counts)}, mean {statistics.mean(fact_counts):.2f}, median {statistics.median(fact_counts)}, p75 {_quantile(fact_counts, 0.75):.2f}, p90 {_quantile(fact_counts, 0.90):.2f}, max {max(fact_counts)}, >12 / >16 / >20 = {sum(value > 12 for value in fact_counts)} / {sum(value > 16 for value in fact_counts)} / {sum(value > 20 for value in fact_counts)}
- Review issues: total {len(all_review_issues)}; severity `{json.dumps(dict(Counter(row['severity'] for row in all_review_issues)), ensure_ascii=False, sort_keys=True)}`
- Revision v1 / v2 / Re-Review / Judge: {revision_summary['revision_v1_count']} / {revision_summary['revision_v2_count']} / {revision_summary['re_review_count']} / {len(judges)}
- Terminal states: `{json.dumps(dict(terminal), ensure_ascii=False, sort_keys=True)}`
- Independent Auditor: `{json.dumps(dict(audit_statuses), ensure_ascii=False, sort_keys=True)}`; major rate `{major_rate:.2%}`; blocking `{len(audit_blocking_samples)}`
- Major over-fragmentation: {len(overfragmentation_samples)}/50 (`{overfragmentation_rate:.2%}`)
- Policy audit: first-person {policy_audit['first_person_candidate_count']}; triggered / true / false-positive / unnecessary = {policy_audit['triggered']} / {policy_audit['true_exception']} / {policy_audit['false_positive']} / {policy_audit['unnecessary_exception']}; systematic error `{policy_audit['systematic_policy_error']}`
- Controlled-title audit: triggered / valid / unnecessary / invalid = {title_audit['triggered']} / {title_audit['valid']} / {title_audit['unnecessary']} / {title_audit['invalid']}; systematic abuse `{title_audit['systematic_title_evidence_abuse']}`
- Deterministic integrity: Target mismatch {quality['target_mismatch']}; Schema failure {quality['schema_failure']}; Evidence technical failure {quality['evidence_technical_failure']}; Provenance failure {quality['provenance_failure']}
- Gold sanity regression: `{'PASS' if gold_sanity['passed'] else 'FAIL'}`
- Invocation audit: {invocation_audit['semantic_invocation_total']} attempts ({invocation_audit['succeeded']} succeeded, {invocation_audit['failed']} failed); Generation {invocation_audit['generation_call_attempts']}/{invocation_audit['generation_calls']}, Revision {invocation_audit['revision_call_attempts']}/{invocation_audit['revision_calls']}, Re-Review {invocation_audit['re_review_call_attempts']}/{invocation_audit['re_review_calls']} attempts/succeeded; unique IDs `{invocation_audit['all_invocation_ids_unique']}`; fallback 0; tool calls 0
- Hard Gate: `{quality['high_pilot_hard_gate']}`
- Quality Gate: `{quality['high_pilot_quality_gate']}`
- Final verdict: `{final_status}`
- Failure classes: `{json.dumps(quality.get('failure_classes', []), ensure_ascii=False)}`; affected samples: {quality.get('affected_sample_count', 0)}
- Major/blocking issue families: `{json.dumps(quality.get('issue_families', {}), ensure_ascii=False, sort_keys=True)}`
- Minimum repair scope: {quality.get('minimum_repair_scope', 'None')}
- Medium Shadow / Full Run / sft_v2 / Training executed: no / no / no / no

Detailed source, candidate, review, revision and auditor payloads are stored only in the Git-ignored pilot runtime namespace.
"""
    _atomic_write(report_path, report)
    csv_path = root / config["safe_summary_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "article_id", "split", "risk_stratum", "fact_count", "review_a_verdict", "review_b_verdict", "revision_rounds", "judge_decision", "terminal_state", "auditor_status"]
    descriptor, temporary = tempfile.mkstemp(prefix=csv_path.name + ".", suffix=".tmp", dir=csv_path.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in selected:
                sample_id = sample["sample_id"]
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "article_id": sample["article_id"],
                        "split": sample["split"],
                        "risk_stratum": sample["primary_risk_stratum"],
                        "fact_count": len(revision_by_id[sample_id]["final_candidate"]["fact_points"]),
                        "review_a_verdict": review_a_by_id[sample_id]["verdict"],
                        "review_b_verdict": review_b_by_id[sample_id]["verdict"],
                        "revision_rounds": len(revision_by_id[sample_id]["revision_rounds"]),
                        "judge_decision": judge_by_id.get(sample_id, {}).get("decision", "not_required"),
                        "terminal_state": revision_by_id[sample_id]["terminal_state"],
                        "auditor_status": next(row["audit_status"] for row in audits if row["sample_id"] == sample_id),
                    }
                )
        temp_path.replace(csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return quality


def validate_high_pilot(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    context = load_context(root, config)
    selection = select_pilot(context, config)
    actual_selection = read_json(root / "pilot_selection_manifest.json")
    _require(actual_selection["selection_sha256"] == selection["selection_sha256"], "selection reproducibility failed")
    quality = read_json(root / "pilot_quality_summary.json")
    invocation = read_json(root / "pilot_invocation_audit.json")
    _require(quality["pilot_count"] == 50, "pilot count mismatch")
    _require(quality["pilot_split_counts"] == config["pilot_split_counts"], "pilot split mismatch")
    _require(quality["unfinished"] == 0, "unfinished pilot samples")
    _require(invocation["generation_calls"] == 50, "generation semantic call mismatch")
    _require(invocation["review_a_calls"] == invocation["review_b_calls"] == 50, "A/B semantic call mismatch")
    _require(invocation["pilot_auditor_calls"] == 50, "auditor semantic call mismatch")
    _require(invocation["fallback_count"] == 0, "semantic fallback detected")
    _require(invocation["tool_call_count"] == 0, "semantic tool call detected")
    _require(invocation["all_invocation_ids_unique"], "duplicate semantic invocation id detected")
    if not invocation["audit_passed"]:
        _require(quality["high_pilot_status"] == "failed", "failed invocation audit must fail closed")
        _require(quality["high_pilot_hard_gate"] == "failed", "failed invocation audit must fail Hard Gate")
    _require(not quality["full_run_executed"] and not quality["sft_v2_created"] and not quality["training_executed"], "forbidden stage executed")
    _require(quality["protected_sources_unchanged"], "v2.1 source protection failed")
    return {"status": "passed", "high_pilot_status": quality["high_pilot_status"], "selection_sha256": selection["selection_sha256"]}
