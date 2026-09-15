from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime
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
    render_target,
    sha256_text,
    source_paragraph_records,
)
from src.task011d.autonomous_sft import json_text, read_json, read_jsonl, record_sha256
from src.task011d.high_semantic_pilot import _risk_features, _source_view, load_context
from src.task011d.semantic_provider import CodexExecSemanticProvider, create_semantic_provider


SPLITS = ("train", "validation", "test")
FACT_PREFIX = "据该篇新闻所载公开事实，"
CORE_ISSUES = (
    "core_fact_error", "evidence_error", "entity_number_date_error",
    "material_coverage_gap", "severe_inference", "severe_structure_error",
)
SOFT_WARNINGS = (
    "minor_atomicity", "minor_fragmentation", "mild_redundancy", "summary_detail_overlap",
    "minor_duplicate", "minor_copy_risk", "target_first_person", "title_supported_fact",
    "non_core_coverage_omission", "style_wording_issue", "minor_fact_count_anomaly",
)


class SilverProductionPilotError(RuntimeError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise SilverProductionPilotError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json_text(value), encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def standard_document(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "silver-production-standard-v1.0.0",
        "standard_id": "silver_production_standard_v1",
        "task_id": config["task_id"],
        "quality_tier": "silver_ai_reviewed",
        "gold_quality_tier_unchanged": "gold_human_reviewed",
        "gold_records_modified": False,
        "source_context": "source_title + source_body",
        "title_and_body_are_first_class_sources": True,
        "evidence_source_types": ["title", "body", "title_and_body"],
        "controlled_source_title_exception_required": False,
        "target_first_person_compatibility_exception_required": False,
        "first_person_policy": {
            "immutable_target_source_quotations_allowed": True,
            "facts_must_be_objective": True,
            "prompt_must_not_introduce_first_person_instruction": True,
            "lexical_non_issues": ["我国", "自我", "自我优化", "产品名称中的‘我’"],
        },
        "required_quality": [
            "core_factual_correctness", "core_source_support", "number_date_correctness",
            "entity_and_entity_relation_correctness", "no_severe_unsupported_inference",
            "material_semantic_coverage", "target_technical_integrity", "schema_integrity", "provenance",
        ],
        "material_coverage": [
            "core_event", "subject", "core_action", "core_object", "material_result",
            "key_number", "key_date", "material_entity_relation", "material_cooperation", "key_future_plan",
        ],
        "non_material_omissions_are_soft": [
            "rhetoric", "promotional_language", "non_core_background", "repeated_explanation", "secondary_sentences",
        ],
        "recommended_fact_counts": {"short": "3-6", "medium": "6-10", "long": "8-14", "complex": "10-16"},
        "fact_count_above_16": "risk_signal_only",
        "hard_failures": [
            "target_technical_mismatch", "schema_invalid", "core_fact_unsupported", "core_number_date_error",
            "core_entity_error", "core_entity_relation_error", "severe_evidence_error", "severe_unsupported_inference",
            "missing_core_event", "missing_material_result_or_relation", "provenance_error", "cross_source_contamination",
        ],
        "soft_warnings": list(SOFT_WARNINGS),
        "reviewer_statuses": ["PASS", "FIX", "DROP"],
        "reviewer_core_issue_types": list(CORE_ISSUES),
        "revision_policy": {"trigger": "FIX_only", "maximum_rounds": 1, "soft_warnings_trigger_revision": False},
        "judge_enabled": False,
        "production_models": {
            "generation_review_revision_rereview": {"model": config["semantic_model"], "reasoning": "medium"},
            "final_independent_auditor": {"model": config["semantic_model"], "reasoning": "high"},
        },
    }


def _rank(seed: str, stratum: str, article_id: str) -> str:
    return hashlib.sha256(f"{seed}|{stratum}|{article_id}".encode()).hexdigest()


def select_pilot(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    features = {key: _risk_features(value) for key, value in context["articles"].items()}
    allocations = {
        "train": {"ordinary": 6, "long": 2, "multi_entity": 2, "number_date": 2, "policy_title": 2},
        "validation": {"ordinary": 1, "long": 1, "multi_entity": 1, "number_date": 0, "policy_title": 0},
        "test": {"ordinary": 1, "long": 0, "multi_entity": 0, "number_date": 1, "policy_title": 1},
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for split in SPLITS:
        rows = [row for row in context["silver_rows"] if row["split"] == split]
        lengths = sorted(features[row["article_id"]]["character_count"] for row in rows)
        long_cut = lengths[math.floor(0.75 * (len(lengths) - 1))]
        pools = {
            "policy_title": [row for row in rows if features[row["article_id"]]["policy_score"] >= 300],
            "number_date": [row for row in rows if features[row["article_id"]]["number_date_signal_count"] >= 3],
            "multi_entity": [row for row in rows if features[row["article_id"]]["entity_signal_count"] >= 3],
            "long": [row for row in rows if features[row["article_id"]]["character_count"] >= long_cut],
            "ordinary": rows,
        }
        for stratum in ("policy_title", "number_date", "multi_entity", "long", "ordinary"):
            available = [row for row in pools[stratum] if row["article_id"] not in used]
            available.sort(key=lambda row: (_rank(config["pilot_seed"], stratum, row["article_id"]), row["article_id"]))
            need = allocations[split][stratum]
            _require(len(available) >= need, f"insufficient selection pool: {split}/{stratum}")
            for row in available[:need]:
                article = context["articles"][row["article_id"]]
                used.add(row["article_id"])
                selected.append({
                    "article_id": row["article_id"], "split": split, "primary_risk_stratum": stratum,
                    "publish_year": str(article.get("publish_year") or str(article.get("publish_date", ""))[:4]),
                    "risk_features": features[row["article_id"]],
                })
    selected.sort(key=lambda row: (SPLITS.index(row["split"]), row["primary_risk_stratum"], row["article_id"]))
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"task011d_e4_pilot_{index:03d}"
    _require(len(selected) == len(used) == config["pilot_count"], "pilot must contain 20 unique samples")
    _require(Counter(row["split"] for row in selected) == Counter(config["pilot_split_counts"]), "split quota mismatch")
    return {
        "schema_version": "task011d-e4-production-selection-v1.0.0", "task_id": config["task_id"],
        "seed": config["pilot_seed"], "pilot_sample_count": len(selected),
        "split_distribution": dict(Counter(row["split"] for row in selected)),
        "risk_strata": dict(Counter(row["primary_risk_stratum"] for row in selected)),
        "year_distribution": dict(Counter(row["publish_year"] for row in selected)),
        "selected_article_ids": [row["article_id"] for row in selected], "selected_samples": selected,
        "selection_sha256": _sha([(row["sample_id"], row["article_id"], row["split"], row["primary_risk_stratum"]) for row in selected]),
        "formal_input": config["generation_input"], "old_superseded_input_read": False,
        "old_50_pilot_candidates_reused": False, "all_candidates_require_fresh_generation": True,
    }


FACT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "fact": {"type": "string"},
        "evidence_source_type": {"type": "string", "enum": ["title", "body", "title_and_body"]},
        "evidence_paragraph_ids": {"type": "array", "items": {"type": "integer"}},
        "fact_type": {"type": "string", "enum": ["event", "cooperation", "plan", "result", "number", "other"]},
    },
    "required": ["fact", "evidence_source_type", "evidence_paragraph_ids", "fact_type"],
}
GENERATION_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "topic": {"type": "string"},
        "facts": {"type": "array", "minItems": 3, "maxItems": 30, "items": FACT_SCHEMA},
        "outline": {"type": "array", "minItems": 3, "maxItems": 7, "items": {"type": "string"}},
    },
    "required": ["topic", "facts", "outline"],
}
REVIEW_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "FIX", "DROP"]},
        "core_issues": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "issue_type": {"type": "string", "enum": list(CORE_ISSUES)}, "fact_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"}, "required_fix": {"type": "string"}},
            "required": ["issue_type", "fact_ids", "rationale", "required_fix"]}},
        "soft_warnings": {"type": "array", "items": {"type": "string", "enum": list(SOFT_WARNINGS)}},
        "rationale": {"type": "string"},
    },
    "required": ["status", "core_issues", "soft_warnings", "rationale"],
}
AUDIT_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "MINOR", "MAJOR", "BLOCKING"]},
        "core_error_types": {"type": "array", "items": {"type": "string", "enum": [
            "core_factual_error", "core_unsupported_fact", "evidence_error", "entity_error", "entity_relation_error",
            "number_date_error", "material_coverage_gap", "severe_inference", "severe_structure_error",
            "target_integrity_error", "schema_error", "provenance_error", "cross_source_contamination",
            "major_overfragmentation", "policy_hard_failure"]}},
        "soft_imperfections": {"type": "array", "items": {"type": "string", "enum": list(SOFT_WARNINGS)}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "core_error_types", "soft_imperfections", "rationale"],
}


GENERATION_SYSTEM = """You generate Chinese-news SFT annotations under silver_production_standard_v1. Use only each sample's source_title and numbered source_body paragraphs. Cover material/core facts: event, actor, action, object, material result, key number/date, material entity relation/cooperation, and key future plan. Do not reconstruct rhetoric or non-core repetition. Facts must be objective, independently training-useful, supported, and not over-fragmented. Title and body are equal formal sources. Set evidence_source_type to title, body, or title_and_body; cite real body paragraph IDs only and never invent a paragraph. Target text is immutable and not generated. Keep samples completely isolated. Return one result for every requested sample ID and no other ID."""
REVIEW_SYSTEM = """You are the single production Reviewer under silver_production_standard_v1. Review each sample independently using source_title + source_body. Return FIX or DROP only for core defects: factual/support/evidence/entity-number-date/material-coverage/severe-inference/severe-structure. Missing rhetoric or non-core detail, mild redundancy/fragmentation, first-person text in the immutable source target, and title-supported facts are soft only and must not trigger FIX. DROP only when one revision cannot safely repair the candidate. Return every requested sample ID exactly once."""
REVISION_SYSTEM = """Perform the only permitted Silver revision round. Resolve only the listed hard/core issues with the smallest change. Preserve immutable Target and provenance, keep supported unaffected facts, use title/body/title_and_body honestly, never invent paragraph IDs, and maintain material coverage without Gold-style over-reconstruction. Return every requested sample ID exactly once."""
REREVIEW_SYSTEM = """Re-review once after the sole revision under silver_production_standard_v1. Return PASS only if all hard/core issues are resolved; otherwise DROP. Soft imperfections do not block and never cause another revision. Return every requested sample ID exactly once."""
AUDIT_SYSTEM = """You are the final independent high-effort Silver auditor. You are blind to production Reviewer and Revision verdicts. Compare only each source and final candidate under silver_production_standard_v1. Distinguish core errors from soft imperfections. BLOCKING is reserved for unusable/system-integrity failures; MAJOR for material semantic defects; MINOR for non-blocking imperfections; PASS for training-grade candidates without notable defect. Title and body are equal formal sources; natural first-person source/Target text is allowed. Missing non-core detail is soft. More than 16 Facts is only a risk signal; major_overfragmentation requires many Facts without independent training value. Return every requested sample ID exactly once."""


def _batch_schema(ids: list[str], item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {
        "results": {"type": "object", "additionalProperties": False,
                    "properties": {sample_id: item_schema for sample_id in ids}, "required": ids}},
        "required": ["results"]}


def _chunks(samples: list[dict[str, Any]], articles: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for sample in samples:
        size = len(articles[sample["sample_id"]]["body"]) + len(articles[sample["sample_id"]]["title"])
        if current and (len(current) >= int(config["batch_size"]) or chars + size > int(config["maximum_batch_source_characters"])):
            batches.append(current); current = []; chars = 0
        current.append(sample); chars += size
    if current:
        batches.append(current)
    return batches


def _invoke_batches(
    provider: CodexExecSemanticProvider, *, phase: str, samples: list[dict[str, Any]], articles: dict[str, dict[str, Any]],
    config: dict[str, Any], runtime: Path, system: str, item_schema: dict[str, Any], payload_builder: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool,
) -> tuple[dict[str, Any], int]:
    results: dict[str, Any] = {}
    batches = _chunks(samples, articles, config)
    phase_dir = runtime / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches, 1):
        path = phase_dir / f"batch_{index:03d}.json"
        ids = [row["sample_id"] for row in batch]
        if resume and path.exists():
            output = read_json(path)
        else:
            task = {sample_id: payload_builder(next(row for row in batch if row["sample_id"] == sample_id)) for sample_id in ids}
            output = provider._invoke(
                role=f"SILVER_PRODUCTION_{phase.upper()}", sample_id="|".join(ids), split="batched",
                prompt_version=f"task011d-e4-{phase}-v1", system_prompt=system,
                user_payload={"sample_ids": ids, "samples": task}, schema_name=f"silver_{phase}",
                schema=_batch_schema(ids, item_schema),
            )
            _write_json(path, output)
        keys = list(output.get("results", {}).keys())
        _require(set(keys) == set(ids) and len(keys) == len(ids), f"{phase} batch ID integrity failure")
        _require(not (set(keys) - set(ids)), f"{phase} cross-sample contamination")
        _require(not (set(results) & set(keys)), f"{phase} duplicate sample output")
        results.update(output["results"])
    _require(set(results) == {row["sample_id"] for row in samples}, f"{phase} output completeness failure")
    return results, len(batches)


def _materialize(article: dict[str, Any], sample: dict[str, Any], semantic: dict[str, Any], config: dict[str, Any], version: str) -> dict[str, Any]:
    paragraphs = article["paragraphs"]
    facts = []
    for index, row in enumerate(semantic["facts"], 1):
        text = row["fact"].strip()
        if not text.startswith(FACT_PREFIX):
            text = FACT_PREFIX + text
        ids = [int(value) for value in row["evidence_paragraph_ids"]]
        source_type = row["evidence_source_type"]
        evidence_parts = []
        if source_type in {"title", "title_and_body"}:
            evidence_parts.append(article["title"])
        if source_type in {"body", "title_and_body"}:
            evidence_parts.extend(paragraphs[value - 1] for value in ids if 1 <= value <= len(paragraphs))
        facts.append({
            "fact_id": f"F{index:02d}", "fact": text, "evidence_source_type": source_type,
            "evidence_paragraph_ids": ids, "evidence_text_sha256": sha256_text(normalize_evidence(evidence_parts)),
            "fact_type": row["fact_type"], "verification_status": "pending_ai_review",
        })
    constraints = fixed_constraints(article["length_bucket"])
    target = render_target(article["title"], article["body"])
    prompt = render_user_prompt(semantic["topic"], facts, semantic["outline"], constraints)
    return {
        "schema_version": SCHEMA_VERSION, "candidate_profile": "silver_production_pilot_v1", "candidate_version": version,
        "sample_id": sample["sample_id"], "source_article_id": article["article_id"], "source_ref": article["source_ref"],
        "source_dataset_version": config["news_version"], "source_split_version": config["split_version"],
        "event_group_id": article["event_group_id"], "split": article["split"], "quality_tier": "silver_ai_reviewed",
        "construction_method": "codexexec_batched_medium_semantic_generation_v1", "created_at": _now(),
        "system_prompt": SYSTEM_PROMPT, "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "topic": semantic["topic"], "fact_points": facts, "outline": semantic["outline"], "constraints": constraints,
        "user_prompt": prompt, "user_prompt_version": USER_PROMPT_VERSION, "messages": render_messages(prompt, target),
        "target_title": article["title"], "target_body": article["body"], "target_text": target,
        "target_title_sha256": sha256_text(article["title"]), "target_body_sha256": article["body_sha256"],
        "target_text_sha256": sha256_text(target), "source_paragraphs": source_paragraph_records(paragraphs),
        "source_body_sha256": article["body_sha256"], "policy_exceptions": [], "pilot_namespace": True,
        "full_run_reuse_authorized": False, "record_sha256": "pending",
    }


_CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}


def _compact_numeric_surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("％", "%")
    value = re.sub(r"(?<=\d)\s+(?=[.%年月日时点分秒万亿GKM])", "", value)
    value = re.sub(r"(?<=\.)\s+(?=\d)", "", value)
    value = re.sub(r"\d{1,3}(?:,\d{3})+", lambda match: match.group(0).replace(",", ""), value)
    return value.replace("—", "-").replace("–", "-").replace("～", "-").replace("~", "-").replace("至", "-").replace("到", "-")


def _parse_chinese_integer(value: str) -> int | None:
    if not value or any(char not in _CHINESE_DIGITS and char not in _CHINESE_UNITS for char in value):
        return None
    total = section = number = 0
    for char in value:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS[char]
        if unit < 10_000:
            section += (number or 1) * unit
        else:
            section = (section + number) * unit
            total += section
            section = 0
        number = 0
    return total + section + number


def _canonical_number_tokens(text: str) -> set[str]:
    """Return numeric tokens after lossless typography/Chinese-numeral normalization."""
    surface = _compact_numeric_surface(text)
    tokens = set(re.findall(r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?:%|％)?(?![A-Za-z\d])", surface))
    # Preserve every token accepted by the original production validator; the
    # canonical layer may add equivalent forms but must never make resume stricter.
    tokens.update(extract_number_tokens(surface))
    tokens.update(re.findall(r"(?<=[A-Za-z])\d+(?:\.\d+)?(?=万|亿|%|个|家|次|名|部|台|座|条|款|项|户)", surface))
    tokens.update(token.lstrip("0") or "0" for token in list(tokens) if token.isdigit())
    numeric_suffix = r"年|月|日|时|点|分|秒|个|家|次|届|辆|车|人|名|部|台|座|条|款|项|户|城|席|楼|号|公里|米|吨|元|美元|万|亿"
    for match in re.finditer(rf"[零〇一二两三四五六七八九十百千万亿]+(?=多?(?:{numeric_suffix}))", surface):
        raw = match.group(0)
        parsed = _parse_chinese_integer(raw)
        if parsed is not None:
            tokens.add(str(parsed or (1 if raw in {"万", "亿"} else 0)))
        for large_unit, divisor in (("万", 10_000), ("亿", 100_000_000)):
            if raw.endswith(large_unit) and parsed is not None and parsed > 0 and parsed % divisor == 0:
                tokens.add(str(parsed // divisor))
    for match in re.finditer(r"百分之([零〇一二两三四五六七八九十百千万亿]+)", surface):
        parsed = _parse_chinese_integer(match.group(1))
        if parsed is not None:
            tokens.add(f"{parsed}%")
    for match in re.finditer(r"([零〇一二两三四五六七八九])点([零〇一二两三四五六七八九]+)", surface):
        left = _CHINESE_DIGITS[match.group(1)]
        right = "".join(str(_CHINESE_DIGITS[char]) for char in match.group(2))
        tokens.add(f"{left}.{right}")
    for match in re.finditer(r"(凌晨|上午|中午|下午|晚|晚上)?([零〇一二两三四五六七八九十]+)点([零〇一二两三四五六七八九十]+)", surface):
        hour = _parse_chinese_integer(match.group(2))
        minute = _parse_chinese_integer(match.group(3))
        if hour is not None:
            if match.group(1) in {"下午", "晚", "晚上"} and hour < 12:
                hour += 12
            tokens.add(str(hour))
        if minute is not None:
            tokens.add(str(minute))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*-+\s*(\d+(?:\.\d+)?)%", surface):
        tokens.update({f"{match.group(1)}%", f"{match.group(2)}%"})
    for match in re.finditer(r"(?:晚|晚上|下午)(\d{1,2})(?:时|点)", surface):
        hour = int(match.group(1))
        tokens.add(str(hour + 12 if hour < 12 else hour))
    for match in re.finditer(r"(\d{1,2})(?:时|点)半", surface):
        tokens.update({str(int(match.group(1))), "30"})
    for match in re.finditer(r"(?<!\d)(\d{5,})(?=余?(?:个|家|次|名|部|台|座|条|款|项|户))", surface):
        coefficient = int(match.group(1)) / 10_000
        tokens.add(str(int(coefficient)) if coefficient.is_integer() else str(coefficient))
    return tokens


def _fact_number_tokens(text: str) -> set[str]:
    """Facts introduce only explicit Arabic quantities; Chinese wording is evidence normalization only."""
    surface = _compact_numeric_surface(text)
    tokens = set(re.findall(r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?:%|％)?(?![A-Za-z\d])", surface))
    tokens.update(token.lstrip("0") or "0" for token in list(tokens) if token.isdigit())
    return tokens


def _enumerated_count_supported(token: str, fact: str, evidence: str) -> bool:
    """Accept an explicit Arabic count only when the cited text mechanically enumerates it."""
    if not token.isdigit() or int(token) < 2:
        return False
    expected = int(token)
    suffixes = [suffix for suffix in ("医院", "有限公司") if suffix in fact]
    if "供应商" in fact:
        suffixes.append("公司")
    if not suffixes:
        return False
    segments = re.split(r"[。；\n]", evidence)
    segments += [match.group(1) for match in re.finditer(r"及(.{1,1200}?)等", evidence)]
    return any(segment.count(suffix) == expected for suffix in suffixes for segment in segments)


def _date_token_supported(token: str, evidence: str) -> bool:
    surface = _compact_numeric_surface(evidence)
    if _compact_numeric_surface(token) in surface:
        return True
    match = re.fullmatch(r"(\d{4})年(?:(\d{1,2})月)?(?:(\d{1,2})日)?", token)
    if not match:
        match = re.fullmatch(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日", token)
        if not match:
            return False
    year, month, day = match.groups()
    year_supported = year is None or year in surface or f"{year[-2:]}年" in surface
    if not year_supported:
        return False
    if month is None:
        return True
    month_value, day_value = int(month), int(day) if day is not None else None
    month_supported = bool(re.search(rf"{month_value}(?:月|-\d{{1,2}}月|月?份)", surface))
    if day_value is None:
        return month_supported
    exact_day = f"{month_value}月{day_value}日" in surface or f"{month_value}月{day_value}号" in surface
    range_day = bool(re.search(rf"{month_value}月(?:{day_value}-+\d{{1,2}}|\d{{1,2}}-+{day_value})日", surface))
    dotted_day = bool(re.search(rf"(?:^|\D)0?{month_value}\.0?{day_value}(?:\D|$)", surface))
    compact_day = month_value >= 2 and bool(re.search(rf"(?:^|\D)0?{month_value}0?{day_value}(?:\D|$)", surface))
    inherited_month = month_supported and bool(re.search(rf"(?:^|\D){day_value}日", surface))
    return exact_day or range_day or dotted_day or compact_day or inherited_month


def deterministic_validate(candidate: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required = {"topic", "fact_points", "outline", "constraints", "user_prompt", "messages", "target_text", "source_ref"}
    if required - set(candidate): errors.append("schema_missing")
    facts = candidate["fact_points"]
    if [row["fact_id"] for row in facts] != [f"F{i:02d}" for i in range(1, len(facts) + 1)]: errors.append("fact_ids")
    target = render_target(article["title"], article["body"])
    if candidate["target_text"] != target or candidate["target_text_sha256"] != sha256_text(target): errors.append("target_exact")
    if candidate["target_body_sha256"] != article["body_sha256"] or candidate["source_ref"] != article["source_ref"]: errors.append("provenance")
    if candidate["messages"] != render_messages(candidate["user_prompt"], target): errors.append("messages_sync")
    if candidate["user_prompt"] != render_user_prompt(candidate["topic"], facts, candidate["outline"], candidate["constraints"]): errors.append("prompt_sync")
    for fact in facts:
        ids = fact["evidence_paragraph_ids"]; source_type = fact["evidence_source_type"]
        if source_type == "title" and ids: errors.append(f"title_ids:{fact['fact_id']}")
        if source_type in {"body", "title_and_body"} and (not ids or any(i < 1 or i > len(article["paragraphs"]) for i in ids)):
            errors.append(f"evidence_refs:{fact['fact_id']}"); continue
        parts = ([article["title"]] if source_type in {"title", "title_and_body"} else []) + ([article["paragraphs"][i-1] for i in ids] if source_type in {"body", "title_and_body"} else [])
        evidence = normalize_evidence(parts)
        if fact["evidence_text_sha256"] != sha256_text(evidence): errors.append(f"evidence_hash:{fact['fact_id']}")
        fact_numbers = _fact_number_tokens(fact["fact"])
        evidence_numbers = _canonical_number_tokens(evidence)
        unsupported_numbers = fact_numbers - evidence_numbers
        legacy_unsupported_numbers = set(extract_number_tokens(fact["fact"])) - set(extract_number_tokens(evidence))
        if not legacy_unsupported_numbers:
            unsupported_numbers = set()
        # A four-digit number can be the year component of an otherwise supported date.
        unsupported_numbers = {
            token for token in unsupported_numbers
            if not any(token in date and _date_token_supported(date, evidence) for date in extract_date_tokens(fact["fact"]))
        }
        unsupported_numbers = {
            token for token in unsupported_numbers
            if not _enumerated_count_supported(token, fact["fact"], evidence)
        }
        if unsupported_numbers: errors.append(f"number:{fact['fact_id']}")
        if any(not _date_token_supported(token, evidence) for token in extract_date_tokens(fact["fact"])):
            errors.append(f"date:{fact['fact_id']}")
    candidate["record_sha256"] = record_sha256({key: value for key, value in candidate.items() if key != "record_sha256"})
    return {
        "status": "pass" if not errors else "blocked", "errors": sorted(set(errors)),
        "schema": not any(row.startswith("schema") or row == "fact_ids" for row in errors),
        "target_integrity": not any(row.startswith("target") or row == "messages_sync" for row in errors),
        "evidence_refs": not any(row.startswith("evidence") or row.startswith("title_ids") for row in errors),
        "number_strings": not any(row.startswith("number") for row in errors), "date_strings": not any(row.startswith("date") for row in errors),
        "prompt_sync": "prompt_sync" not in errors, "checksum": candidate["target_body_sha256"] == article["body_sha256"],
        "provenance": "provenance" not in errors, "candidate_sha256": candidate["record_sha256"],
    }


def _source_candidate(article: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {"source": _source_view(article), "candidate": candidate}


def _provider(root: Path, config: dict[str, Any], effort: str) -> CodexExecSemanticProvider:
    provider_config = dict(config); provider_config["reasoning_effort"] = effort
    provider = create_semantic_provider(provider_config, root=root)
    _require(isinstance(provider, CodexExecSemanticProvider), "CodexExec provider required")
    return provider


def _percentile(values: list[int], percentile: float) -> float:
    values = sorted(values); position = (len(values) - 1) * percentile
    lo, hi = math.floor(position), math.ceil(position)
    return float(values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (position - lo))


def run_pilot(root: Path, config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    _require(config["pilot_count"] == 20 and config["pilot_execution_allowed"], "unsafe pilot configuration")
    _require(not config["full_run_execution_allowed"] and not config["sft_v2_creation_allowed"] and not config["training_allowed"], "forbidden stage enabled")
    runtime = root / config["pilot_output_dir"]
    if runtime.exists() and not resume and any(runtime.iterdir()):
        raise SilverProductionPilotError("pilot output exists; use --resume")
    runtime.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); started_at = _now()
    context = load_context(root, config)
    standard = standard_document(config); selection = select_pilot(context, config)
    _write_json(runtime / "silver_production_standard_v1.json", standard)
    _write_json(runtime / "production_pilot_selection.json", selection)
    selected = selection["selected_samples"]
    articles = {row["sample_id"]: context["articles"][row["article_id"]] for row in selected}
    medium = _provider(root, config, config["production_reasoning_effort"])

    generated, generation_calls = _invoke_batches(
        medium, phase="generation", samples=selected, articles=articles, config=config, runtime=runtime,
        system=GENERATION_SYSTEM, item_schema=GENERATION_ITEM_SCHEMA, resume=resume,
        payload_builder=lambda row: {"source": _source_view(articles[row["sample_id"]])},
    )
    candidates = {sid: _materialize(articles[sid], next(row for row in selected if row["sample_id"] == sid), value, config, "production_candidate_v1") for sid, value in generated.items()}
    validations = {sid: deterministic_validate(candidate, articles[sid]) for sid, candidate in candidates.items()}
    _write_json(runtime / "production_generation_summary.json", {
        "sample_count": 20, "model": config["semantic_model"], "reasoning_effort": "medium", "batch_calls": generation_calls,
        "batch_size_target": config["batch_size"], "deterministic_validation_passed": sum(v["status"] == "pass" for v in validations.values()),
        "deterministic_validation_failed": sum(v["status"] != "pass" for v in validations.values()), "candidate_sha256": {sid: value["candidate_sha256"] for sid, value in validations.items()},
    })
    _write_json(runtime / "candidates_initial.json", candidates)
    _write_json(runtime / "deterministic_validation_initial.json", validations)

    reviews, review_calls = _invoke_batches(
        medium, phase="review", samples=selected, articles=articles, config=config, runtime=runtime,
        system=REVIEW_SYSTEM, item_schema=REVIEW_ITEM_SCHEMA, resume=resume,
        payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]),
    )
    for sid, validation in validations.items():
        if validation["status"] != "pass" and reviews[sid]["status"] == "PASS":
            reviews[sid]["status"] = "FIX"
            reviews[sid]["core_issues"].append({"issue_type": "severe_structure_error", "fact_ids": [], "rationale": ";".join(validation["errors"]), "required_fix": "repair deterministic integrity"})
    fix_samples = [row for row in selected if reviews[row["sample_id"]]["status"] == "FIX"]
    drop_ids = {sid for sid, value in reviews.items() if value["status"] == "DROP"}
    _write_json(runtime / "production_review_summary.json", {
        "sample_count": 20, "model": config["semantic_model"], "reasoning_effort": "medium", "batch_calls": review_calls,
        "status_counts": dict(Counter(value["status"] for value in reviews.values())), "fix_count": len(fix_samples), "drop_count_before_revision": len(drop_ids),
        "double_review_executed": False, "judge_executed": False, "reviews": reviews,
    })

    revised: dict[str, Any] = {}
    revision_calls = rereview_calls = 0
    rereviews: dict[str, Any] = {}
    if fix_samples:
        revisions, revision_calls = _invoke_batches(
            medium, phase="revision", samples=fix_samples, articles=articles, config=config, runtime=runtime,
            system=REVISION_SYSTEM, item_schema=GENERATION_ITEM_SCHEMA, resume=resume,
            payload_builder=lambda row: {**_source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]), "core_issues": reviews[row["sample_id"]]["core_issues"]},
        )
        for sid, semantic in revisions.items():
            sample = next(row for row in selected if row["sample_id"] == sid)
            revised[sid] = _materialize(articles[sid], sample, semantic, config, "production_candidate_v2")
        revised_validations = {sid: deterministic_validate(value, articles[sid]) for sid, value in revised.items()}
        rereviews, rereview_calls = _invoke_batches(
            medium, phase="rereview", samples=fix_samples, articles=articles, config=config, runtime=runtime,
            system=REREVIEW_SYSTEM, item_schema=REVIEW_ITEM_SCHEMA, resume=resume,
            payload_builder=lambda row: {**_source_candidate(articles[row["sample_id"]], revised[row["sample_id"]]), "original_core_issues": reviews[row["sample_id"]]["core_issues"]},
        )
        for sid in revised:
            if revised_validations[sid]["status"] != "pass" or rereviews[sid]["status"] != "PASS":
                drop_ids.add(sid)
    else:
        revised_validations = {}
    final_candidates = {sid: revised.get(sid, candidates[sid]) for sid in candidates}
    final_validations = {sid: deterministic_validate(value, articles[sid]) for sid, value in final_candidates.items()}
    _write_json(runtime / "production_revision_summary.json", {
        "fix_count": len(fix_samples), "revision_sample_count": len(revised), "revision_calls": revision_calls,
        "maximum_revision_rounds": 1, "second_revision_executed": False, "rereview_sample_count": len(rereviews),
        "rereview_calls": rereview_calls, "drop_count": len(drop_ids), "rereviews": rereviews,
    })
    _write_json(runtime / "candidates_final.json", final_candidates)
    _write_json(runtime / "deterministic_validation_final.json", final_validations)

    high = _provider(root, config, config["auditor_reasoning_effort"])
    audits, audit_calls = _invoke_batches(
        high, phase="high_audit", samples=selected, articles=articles, config=config, runtime=runtime,
        system=AUDIT_SYSTEM, item_schema=AUDIT_ITEM_SCHEMA, resume=resume,
        payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], final_candidates[row["sample_id"]]),
    )
    verdicts = Counter(value["verdict"] for value in audits.values())
    core_types = Counter(issue for value in audits.values() for issue in value["core_error_types"])
    hard_gate = all((
        all(value["target_integrity"] and value["schema"] and value["provenance"] for value in final_validations.values()),
        verdicts["BLOCKING"] == 0, core_types["core_unsupported_fact"] == 0,
        core_types["entity_error"] + core_types["entity_relation_error"] + core_types["number_date_error"] == 0,
    ))
    major_classes = Counter(issue for value in audits.values() if value["verdict"] == "MAJOR" for issue in value["core_error_types"])
    systematic_major = any(count >= 2 for count in major_classes.values())
    quality_gate = verdicts["MAJOR"] <= 1 and not systematic_major
    pilot_pass = hard_gate and quality_gate
    _write_json(runtime / "production_high_audit.json", {
        "sample_count": 20, "model": config["semantic_model"], "reasoning_effort": "high", "batch_calls": audit_calls,
        "review_verdicts_visible": False, "revision_verdicts_visible": False, "verdict_counts": dict(verdicts),
        "core_error_type_counts": dict(core_types), "systematic_repeated_major_class": systematic_major,
        "audits": audits,
    })

    fact_counts = [len(value["fact_points"]) for value in final_candidates.values()]
    total_calls = generation_calls + review_calls + revision_calls + rereview_calls + audit_calls
    elapsed = time.monotonic() - started
    efficiency = {
        "generation_calls": generation_calls, "review_calls": review_calls, "revision_calls": revision_calls,
        "rereview_calls": rereview_calls, "high_audit_calls": audit_calls, "judge_calls": 0,
        "total_semantic_calls": total_calls, "wall_time_seconds": round(elapsed, 3), "calls_per_sample": round(total_calls / 20, 4),
        "old_calls_per_sample": config["old_calls_per_sample"],
        "efficiency_improvement_ratio": round(config["old_calls_per_sample"] / (total_calls / 20), 4),
        "semantic_batching_used": True,
    }
    _write_json(runtime / "production_efficiency_summary.json", efficiency)
    fix_rate = len(fix_samples) / 20
    projected = {
        "population": 2117, "assumptions": {"batch_size": 5, "observed_fix_rate": fix_rate, "high_audit_policy": "train_5_percent_plus_all_risk; validation_test_100_percent"},
        "generation_batch_calls": math.ceil(2117 / 5), "review_batch_calls": math.ceil(2117 / 5),
        "expected_revision_calls": math.ceil((2117 * fix_rate) / 5), "expected_rereview_calls": math.ceil((2117 * fix_rate) / 5),
        "high_audit_calls": math.ceil((math.ceil(1697 * 0.05) + 212 + 208) / 5),
    }
    projected["projected_total_calls"] = sum(projected[key] for key in ("generation_batch_calls", "review_batch_calls", "expected_revision_calls", "expected_rereview_calls", "high_audit_calls"))
    projected["projected_wall_time_seconds_at_observed_sequential_rate"] = round(projected["projected_total_calls"] * elapsed / max(total_calls, 1), 1)
    projected["projected_wall_time_hours_at_observed_sequential_rate"] = round(projected["projected_wall_time_seconds_at_observed_sequential_rate"] / 3600, 2)
    projected["projected_wall_time_note"] = "order-of-magnitude projection from observed calls/time; no Codex credit estimate"
    _write_json(runtime / "full_run_cost_projection.json", projected)
    quality = {
        "schema_version": "task011d-e4-production-quality-summary-v1.0.0", "task_id": config["task_id"],
        "started_at": started_at, "finished_at": _now(), "pilot_samples": 20, "split_distribution": selection["split_distribution"],
        "fact_statistics": {"total": sum(fact_counts), "average": round(statistics.mean(fact_counts), 3), "median": statistics.median(fact_counts), "p90": round(_percentile(fact_counts, .9), 3), "above_16": sum(v > 16 for v in fact_counts), "above_20": sum(v > 20 for v in fact_counts)},
        "auditor_verdicts": dict(verdicts), "core_error_type_counts": dict(core_types),
        "target_integrity_failures": sum(not v["target_integrity"] for v in final_validations.values()),
        "schema_failures": sum(not v["schema"] for v in final_validations.values()), "provenance_failures": sum(not v["provenance"] for v in final_validations.values()),
        "pilot_hard_gate": "PASS" if hard_gate else "FAIL", "pilot_quality_gate": "PASS" if quality_gate else "FAIL",
        "final_pilot_verdict": "PASS" if pilot_pass else "FAIL", "silver_production_pipeline_ready": pilot_pass,
        "full_run_executed": False, "sft_v2_created": False, "training_executed": False,
        "next_task": "TASK-011D-E5 Full Silver Production Run" if pilot_pass else "TASK-011D-E4 targeted local hard-issue repair",
    }
    _write_json(runtime / "production_quality_summary.json", quality)
    _write_safe_summary(root / config["safe_summary_csv"], selected, final_candidates, final_validations, audits, drop_ids)
    _write_report(root / config["report_path"], standard, selection, efficiency, projected, quality, len(fix_samples), len(drop_ids), revision_calls, rereview_calls)
    return validate_pilot(root, config_path)


def _write_safe_summary(path: Path, selected: list[dict[str, Any]], candidates: dict[str, Any], validations: dict[str, Any], audits: dict[str, Any], drops: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "article_id", "split", "risk_stratum", "fact_count", "technical_status", "auditor_verdict", "terminal_status"])
        writer.writeheader()
        for row in selected:
            sid = row["sample_id"]
            writer.writerow({"sample_id": sid, "article_id": row["article_id"], "split": row["split"], "risk_stratum": row["primary_risk_stratum"], "fact_count": len(candidates[sid]["fact_points"]), "technical_status": validations[sid]["status"], "auditor_verdict": audits[sid]["verdict"], "terminal_status": "DROP" if sid in drops else "candidate"})


def _write_report(path: Path, standard: dict[str, Any], selection: dict[str, Any], efficiency: dict[str, Any], projection: dict[str, Any], quality: dict[str, Any], fixes: int, drops: int, revision_calls: int, rereview_calls: int) -> None:
    facts = quality["fact_statistics"]; audit = quality["auditor_verdicts"]; core = quality["core_error_type_counts"]
    text = f"""# TASK-011D-E4 Silver Production Standard and 20-Sample Medium Pilot

## 结论

Pilot **{quality['final_pilot_verdict']}**。`silver_production_pipeline_ready = {str(quality['silver_production_pipeline_ready']).lower()}`。未运行 2117 条 Full Run，未构建 `sft_v2`，未训练。

## 标准与门禁

- 标准：`{standard['standard_id']}`；Silver=`silver_ai_reviewed`，Gold=`gold_human_reviewed` 且未修改。
- 正式 Source：`source_title + source_body`；不再需要 controlled-title 或 Target first-person exception。
- Hard Gate：{quality['pilot_hard_gate']}；Quality Gate：{quality['pilot_quality_gate']}。
- Soft warning 不触发 Revision；单 Reviewer，最多一轮 Revision，Judge 禁用。

## Pilot 与效率

- 样本/分布：20；{selection['split_distribution']}。
- Generation：gpt-5.6-sol medium，{efficiency['generation_calls']} calls；Reviewer：gpt-5.6-sol medium，{efficiency['review_calls']} calls。
- FIX：{fixes}；Revision calls：{revision_calls}；Re-Review calls：{rereview_calls}；DROP：{drops}。
- High Auditor：gpt-5.6-sol high，{efficiency['high_audit_calls']} calls。
- 总调用：{efficiency['total_semantic_calls']}；calls/sample={efficiency['calls_per_sample']}；旧值=8.58；效率提升倍数={efficiency['efficiency_improvement_ratio']}；wall time={efficiency['wall_time_seconds']} 秒。

## Fact 与独立审计

- Facts total/avg/median/p90/>16/>20：{facts['total']}/{facts['average']}/{facts['median']}/{facts['p90']}/{facts['above_16']}/{facts['above_20']}。
- Auditor PASS/MINOR/MAJOR/BLOCKING：{audit.get('PASS',0)}/{audit.get('MINOR',0)}/{audit.get('MAJOR',0)}/{audit.get('BLOCKING',0)}。
- core factual errors={core.get('core_factual_error',0) + core.get('core_unsupported_fact',0)}；entity/number/date errors={core.get('entity_error',0) + core.get('entity_relation_error',0) + core.get('number_date_error',0)}；major coverage gaps={core.get('material_coverage_gap',0)}；overfragmentation major={core.get('major_overfragmentation',0)}；policy hard failures={core.get('policy_hard_failure',0)}。
- Target/Schema/Provenance failures：{quality['target_integrity_failures']}/{quality['schema_failures']}/{quality['provenance_failures']}。

## Full Run 投影与下一步

- projected calls：{projection['projected_total_calls']}；projected sequential wall time：{projection['projected_wall_time_seconds_at_observed_sequential_rate']} 秒（约 {projection['projected_wall_time_hours_at_observed_sequential_rate']} 小时，仅按本 Pilot calls/time 外推，不估算 credits）。
- Full Run：否；`sft_v2`：否；Training：否。
- next task：`{quality['next_task']}`。
"""
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8", newline="")


def validate_pilot(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path); runtime = root / config["pilot_output_dir"]
    required = ["silver_production_standard_v1.json", "production_pilot_selection.json", "production_generation_summary.json", "production_review_summary.json", "production_revision_summary.json", "production_high_audit.json", "production_efficiency_summary.json", "full_run_cost_projection.json", "production_quality_summary.json"]
    missing = [name for name in required if not (runtime / name).is_file()]
    _require(not missing, f"missing pilot outputs: {missing}")
    selection = read_json(runtime / "production_pilot_selection.json"); quality = read_json(runtime / "production_quality_summary.json")
    audit = read_json(runtime / "production_high_audit.json"); efficiency = read_json(runtime / "production_efficiency_summary.json")
    _require(selection["pilot_sample_count"] == 20 and audit["sample_count"] == 20, "pilot sample count mismatch")
    _require(audit["review_verdicts_visible"] is False and audit["revision_verdicts_visible"] is False, "auditor not independent")
    _require(efficiency["judge_calls"] == 0, "judge must remain disabled")
    _require(not quality["full_run_executed"] and not quality["sft_v2_created"] and not quality["training_executed"], "forbidden stage executed")
    return {"status": "passed", "final_pilot_verdict": quality["final_pilot_verdict"], "silver_production_pipeline_ready": quality["silver_production_pipeline_ready"], "full_run_executed": False, "sft_v2_created": False, "training_executed": False, "output_dir": config["pilot_output_dir"], "report_path": config["report_path"]}
