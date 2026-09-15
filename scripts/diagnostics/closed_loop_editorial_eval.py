#!/usr/bin/env python3
"""Validation-only closed-loop editorial revision and blind evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import lora_v2 as protocol  # noqa: E402

DEFAULT_VALIDATION = ROOT / "data/processed/sft_v2/validation.jsonl"
DEFAULT_BASE = ROOT / "outputs/lora_v2/E00/E00_base_validation/validation_inference/predictions.jsonl"
DEFAULT_LORA = ROOT / "outputs/lora_v2/auto_recovery/full_validation/strength_0.50/predictions.jsonl"
DEFAULT_FIXED_IDS = ROOT / "outputs/lora_v2/production_decode/validation_judge24/judge24_ids.txt"
DEFAULT_OUTPUT = ROOT / "outputs/lora_v2/closed_loop_editorial"
TEST_SOURCE = ROOT / "data/processed/sft_v2/test.jsonl"
TEST_BASE = ROOT / "outputs/lora_v2/final_test/base/predictions.jsonl"
TEST_LORA = ROOT / "outputs/lora_v2/auto_recovery/final_test/strength_0.50/predictions.jsonl"
TEST_MASK = ROOT / "outputs/lora_v2/final_test/comparison_safe_mask.json"
TEST_AUTHORIZATION = ROOT / "configs/training/lora_v2/final_test_authorization.json"
TEST_OUTPUT = ROOT / "outputs/lora_v2/closed_loop_editorial_test"
FIXED_MODEL = "qwen3-235b-a22b-instruct-2507"
CALIBRATION_IDS = ("task011d_e5_1698", "task011d_e5_1699")
FORMAL_COUNT = 24
TEST_FORMAL_COUNT = 50
SELECTION_SEED = 20250821
MODELS = ("base", "lora_draft", "lora_revised")
LABELS = ("A", "B", "C")
DIMENSIONS = {
    "factual_grounding": 30,
    "key_information_retention": 15,
    "title_quality": 15,
    "formality_objectivity": 15,
    "structure_formatting": 10,
    "conciseness_naturalness": 15,
}
ISSUE_TYPES = set(DIMENSIONS) | {"unsupported_additions", "repetition"}


class WorkflowError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_bytes(path, b"".join(_json_bytes(row) for row in rows))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise WorkflowError(f"required file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid JSON file {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise WorkflowError(f"required file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise WorkflowError(f"{path}:{number} must be a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid JSONL file {path}: {exc}") from exc
    return rows


def _sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise WorkflowError("every row must contain a non-empty sample_id")
    return value


def _index(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in result:
            raise WorkflowError(f"duplicate sample_id in {source}: {sample_id}")
        result[sample_id] = row
    return result


def _prediction(row: dict[str, Any], sample_id: str) -> str:
    for field in ("prediction", "output", "generated_text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise WorkflowError(f"prediction missing for {sample_id}")


def _prompt_messages(row: dict[str, Any], sample_id: str) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise WorkflowError(f"validation messages missing for {sample_id}")
    prompt = messages[:-1]
    clean: list[dict[str, str]] = []
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise WorkflowError(f"invalid prompt messages for {sample_id}")
        clean.append({"role": message["role"], "content": message["content"]})
    return clean


def _load_sources(validation_path: Path, base_path: Path, lora_path: Path) -> tuple[dict[str, dict[str, Any]], ...]:
    return (
        _index(_read_jsonl(validation_path), str(validation_path)),
        _index(_read_jsonl(base_path), str(base_path)),
        _index(_read_jsonl(lora_path), str(lora_path)),
    )


def _authorize_test(authorization: Path) -> tuple[Path, str]:
    resolved = authorization.resolve()
    if resolved != TEST_AUTHORIZATION.resolve():
        raise WorkflowError(f"Test authorization path must be exactly {TEST_AUTHORIZATION}")
    try:
        protocol._read_authorization(resolved)
    except (OSError, protocol.GuardError) as exc:
        raise WorkflowError(str(exc)) from exc
    return resolved, _sha_file(resolved)


def _test_ids(mask_path: Path = TEST_MASK) -> tuple[list[str], dict[str, Any]]:
    mask = _read_json(mask_path)
    rows = mask.get("rows") if isinstance(mask, dict) else None
    if (
        mask.get("schema_version") != "comparison-safe-test-mask-v1.0.0"
        or mask.get("status") != "built_metadata_only"
        or mask.get("safe_count") != 230
        or mask.get("judge_count") != TEST_FORMAL_COUNT
        or not isinstance(rows, list)
        or len(rows) != 230
    ):
        raise WorkflowError("comparison-safe Test mask metadata is inconsistent")
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"sample_id", "document_id", "comparison_safe", "judge_sample"}:
            raise WorkflowError("comparison-safe Test mask row schema is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise WorkflowError("comparison-safe Test mask contains an invalid or duplicate sample_id")
        seen.add(sample_id)
        if row.get("comparison_safe") is not True:
            raise WorkflowError("comparison-safe Test mask contains a non-safe row")
        if not isinstance(row.get("judge_sample"), bool):
            raise WorkflowError("comparison-safe Test mask judge_sample must be boolean")
        if row["judge_sample"]:
            ids.append(sample_id)
    if len(ids) != TEST_FORMAL_COUNT or len(set(ids)) != TEST_FORMAL_COUNT:
        raise WorkflowError("comparison-safe Test mask must contain exactly 50 fixed judge samples")
    return ids, mask


def _load_test_sources(sample_ids: list[str]) -> tuple[dict[str, dict[str, Any]], ...]:
    source, base, lora = _load_sources(TEST_SOURCE, TEST_BASE, TEST_LORA)
    if len(source) != 233 or len(base) != 233 or len(lora) != 233:
        raise WorkflowError("frozen Test source/Base/repaired-LoRA predictions must each contain 233 samples")
    if set(source) != set(base) or set(source) != set(lora):
        raise WorkflowError("frozen Test source/Base/repaired-LoRA sample IDs do not match")
    if any(sample_id not in source for sample_id in sample_ids):
        raise WorkflowError("fixed Test judge sample is absent from a required source")
    return source, base, lora


def _formal_ids(validation: dict[str, dict[str, Any]], fixed_ids_path: Path) -> tuple[list[str], list[str]]:
    if not fixed_ids_path.is_file():
        raise WorkflowError(f"required file does not exist: {fixed_ids_path}")
    fixed = [line.strip() for line in fixed_ids_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(fixed) != len(set(fixed)):
        raise WorkflowError("fixed24 ID file contains duplicates")
    chosen = [sample_id for sample_id in fixed if sample_id not in CALIBRATION_IDS]
    if len(chosen) > FORMAL_COUNT:
        raise WorkflowError("fixed ID file has more than 24 non-calibration IDs")
    pool = [sample_id for sample_id in validation if sample_id not in set(chosen) | set(CALIBRATION_IDS)]
    pool.sort(key=lambda value: hashlib.sha256(f"{SELECTION_SEED}:{value}".encode()).hexdigest())
    supplements = pool[: FORMAL_COUNT - len(chosen)]
    chosen.extend(supplements)
    if len(chosen) != FORMAL_COUNT or len(set(chosen)) != FORMAL_COUNT:
        raise WorkflowError("could not select exactly 24 unique formal Validation samples")
    if any(sample_id not in validation for sample_id in chosen):
        raise WorkflowError("formal sample ID is absent from Validation")
    return chosen, supplements


def _env_api() -> tuple[str, str, str]:
    base = os.environ.get("LLM_JUDGE_API_BASE", "").strip().rstrip("/")
    key = os.environ.get("LLM_JUDGE_API_KEY", "").strip()
    model = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    if not base or not key or not model:
        raise WorkflowError("LLM_JUDGE_API_BASE, LLM_JUDGE_API_KEY, and LLM_JUDGE_MODEL are required")
    if model != FIXED_MODEL:
        raise WorkflowError(f"LLM_JUDGE_MODEL must be exactly {FIXED_MODEL}")
    return base, key, model


def _response_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise WorkflowError("API response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise WorkflowError("API response content is empty")
    return content


def _api_json(messages: list[dict[str, str]], *, attempts: int, retry_seconds: float, timeout: float) -> tuple[dict[str, Any], int]:
    base, key, model = _env_api()
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0,
        "response_format": {"type": "json_object"},
    }, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(endpoint, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        }, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
            content = json.loads(_response_content(raw))
            if not isinstance(content, dict):
                raise WorkflowError("API JSON response must be an object")
            return content, attempt
        except (OSError, urllib.error.URLError, json.JSONDecodeError, WorkflowError) as exc:
            last = exc
            if attempt < attempts:
                time.sleep(retry_seconds * (2 ** (attempt - 1)))
    raise WorkflowError(f"API call failed after {attempts} attempts: {last}")


def _candidate_labels(candidates: dict[str, str]) -> tuple[str, ...]:
    keys = set(candidates)
    if keys == {"A", "B"}:
        return ("A", "B")
    if keys == set(LABELS):
        return LABELS
    raise WorkflowError("candidate labels must be exactly A/B or A/B/C")


def _score_instruction(labels: tuple[str, ...]) -> str:
    label_text = "/".join(labels)
    fields = ", ".join(f"{name} (0-{maximum})" for name, maximum in DIMENSIONS.items())
    candidate_schema = {
        "factual_grounding": 0,
        "key_information_retention": 0,
        "title_quality": 0,
        "formality_objectivity": 0,
        "structure_formatting": 0,
        "conciseness_naturalness": 0,
        "total_score": 0,
        "unsupported_claims": [],
        "major_release_risks": [],
        "publishable": True,
        "short_rationale": "...",
    }
    response_schema = {
        "scores": {label: dict(candidate_schema) for label in labels},
        "winner": f"{'|'.join(labels)}|tie",
        "winner_reason": "...",
    }
    return (
        f"For every candidate {label_text}, return integer scores for: {fields}. "
        "total_score must equal the six-score sum (0-100). Also return unsupported_claims as a string list, "
        "major_release_risks as a string list, publishable as boolean, and short_rationale as a short string. "
        "Return JSON only. All candidate-specific fields must be fully nested under scores.<label>. "
        "DO NOT put unsupported_claims, major_release_risks, publishable, or short_rationale at the top level. "
        "They must exist separately for every candidate under scores.<label>. "
        f"Return winner as one of {label_text}/tie and winner_reason as a short string. "
        "Use exactly this response structure (replace the placeholder values with the actual judgment):\n"
        + json.dumps(response_schema, ensure_ascii=False, indent=2)
    )


def _calibration_request(prompt: list[dict[str, str]], candidates: dict[str, str]) -> list[dict[str, str]]:
    labels = _candidate_labels(candidates)
    payload = {"editorial_request": prompt, "candidates": candidates}
    system = (
        "You are an independent corporate news release Editorial Judge. Judge only the supplied request and drafts. "
        "Unsupported elaboration, invented benefits, causal claims, superlatives, and promotional conclusions are "
        "factual/release risks. In particular, do not reward unsupported phrases such as ‘进一步提升全球服务能力与网络韧性’, "
        "‘为高质量设计服务提供了坚实的人才支撑’, or ‘构建起覆盖全国、响应高效的设计服务网络’. Penalize AI-like, "
        "vague, sensational, slogan-style, or factually unsupported titles; ‘资质实力与组织布局全面展现’ is a poor title. "
        "Reward faithful retention of names, standard abbreviations, times, numbers, and scope when needed. Formality is "
        "not slogan stacking. Do not reward length or imagined reference similarity. Major factual risk has "
        "priority; otherwise compare the 100-point total, and for close scores prefer the more faithful, release-ready draft. "
        + _score_instruction(labels)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _review_request(
    prompt: list[dict[str, str]], draft: str, *, schema_retry: bool = False
) -> list[dict[str, str]]:
    payload = {"editorial_request": prompt, "lora_draft": draft}
    system = (
        "You are a strict corporate news editor. Review only the supplied editorial request and draft. "
        "Check factual grounding, unsupported additions, key information retention, title quality, "
        "formality/objectivity, structure/format, conciseness/naturalness, and repetition. "
        "Each issue.type MUST be exactly one of: factual_grounding, key_information_retention, title_quality, "
        "formality_objectivity, structure_formatting, conciseness_naturalness, unsupported_additions, repetition. "
        "Each issue.severity MUST be exactly one of: minor, major. "
        "Do not invent any other issue type or severity. Return JSON only. "
        "For a failed review, use this complete structure:\n"
        '{\n  "pass": false,\n  "issues": [\n    {\n      "type": "unsupported_additions",\n'
        '      "severity": "major",\n      "evidence": "具体问题文本",\n'
        '      "instruction": "具体修改要求"\n    }\n  ],\n'
        '  "revision_instructions": [\n    "删除无事实依据的宣传性结论"\n  ]\n}.\n'
        "For a passing review, use exactly:\n"
        '{\n  "pass": true,\n  "issues": [],\n  "revision_instructions": []\n}. '
        "If pass is true both lists must be empty; otherwise give actionable, brief items."
    )
    if schema_retry:
        system = (
            "YOUR PREVIOUS RESPONSE FAILED STRICT JSON SCHEMA VALIDATION.\n"
            "Return ONLY the following exact JSON structure.\n"
            "Do not put issue fields at the top level.\n"
            "Do not put strings directly inside issues.\n"
            "Every issue must contain all four fields.\n"
            "Do not omit revision_instructions.\n"
            '{\n  "pass": false,\n  "issues": [\n    {\n'
            '      "type": "unsupported_additions",\n      "severity": "major",\n'
            '      "evidence": "具体原文",\n      "instruction": "具体修改要求"\n'
            '    }\n  ],\n  "revision_instructions": [\n    "具体修改要求"\n  ]\n}\n'
            "For pass=true, return exactly:\n"
            '{\n  "pass": true,\n  "issues": [],\n  "revision_instructions": []\n}\n\n'
            + system
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _revise_request(
    prompt: list[dict[str, str]], draft: str, review: dict[str, Any], *, schema_retry: bool = False
) -> list[dict[str, str]]:
    payload = {
        "editorial_request": prompt, "lora_draft": draft,
        "issues": review["issues"], "revision_instructions": review["revision_instructions"],
    }
    system = (
        "You are a corporate news reviser. Use only the supplied request, draft, and review instructions. Never add "
        "numbers, people, events, evaluations, or conclusions absent from the supplied material. Prefer deleting unsupported "
        "promotional elevation; retain key facts and standard terminology. Fix the title, abnormal line breaks, fragmentation, "
        "and repetition. Write formally, objectively, restrainedly, and concisely as a sound basis for a corporate website "
        "news release. Return JSON only with exactly one field: revised_draft, containing only the finished article text."
    )
    if schema_retry:
        system = (
            "YOUR PREVIOUS RESPONSE FAILED STRICT JSON SCHEMA VALIDATION.\n"
            'Return ONLY this exact JSON structure: {"revised_draft":"完整修改稿"}.\n'
            "Do not add, omit, rename, or move fields. Do not include explanation outside the JSON object.\n\n"
            + system
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _judge_request(prompt: list[dict[str, str]], candidates: dict[str, str]) -> list[dict[str, str]]:
    labels = _candidate_labels(candidates)
    label_text = "/".join(labels)
    payload = {"editorial_request": prompt, "anonymous_candidates": candidates}
    system = (
        f"You are an independent corporate news release Editorial Judge. You know candidates only as {label_text} and have "
        "no additional source of truth. Judge factual grounding and release readiness from the supplied request. Treat unsupported "
        "claims and major factual risks as highest priority. Do not reward length, imagined reference similarity, or "
        "promotional phrases. Otherwise use total_score; for close scores prefer the more faithful, release-ready draft. "
        + _score_instruction(labels)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _validate_score(value: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    expected = {"scores", "winner", "winner_reason"}
    if not isinstance(value, dict) or not expected <= set(value):
        raise WorkflowError(f"judge response must contain fields {sorted(expected)}")
    scores = value["scores"]
    if not isinstance(scores, dict) or set(scores) != set(labels):
        raise WorkflowError("judge score labels do not match candidates")
    required_score_metadata = {"unsupported_claims", "major_release_risks", "publishable", "short_rationale"}
    required_score_fields = set(DIMENSIONS) | required_score_metadata
    allowed_score_fields = required_score_fields | {"total_score"}
    normalized_scores: dict[str, dict[str, Any]] = {}
    for label, score in scores.items():
        if not isinstance(score, dict):
            raise WorkflowError(f"invalid score fields for {label}: score must be an object")
        missing = required_score_fields - set(score)
        additional = set(score) - allowed_score_fields
        if missing:
            raise WorkflowError(
                f"invalid score fields for {label}: missing={sorted(missing)}, extra={sorted(additional)}"
            )
        computed_total = 0
        for dimension, maximum in DIMENSIONS.items():
            number = score[dimension]
            if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= maximum:
                raise WorkflowError(f"invalid {dimension} score for {label}")
            computed_total += number
        for field in ("unsupported_claims", "major_release_risks"):
            if not isinstance(score[field], list) or any(not isinstance(item, str) for item in score[field]):
                raise WorkflowError(f"{field} must be a string list for {label}")
        if not isinstance(score["publishable"], bool):
            raise WorkflowError(f"publishable must be boolean for {label}")
        if not isinstance(score["short_rationale"], str) or not score["short_rationale"].strip():
            raise WorkflowError(f"short_rationale missing for {label}")
        normalized_scores[label] = {
            **{dimension: score[dimension] for dimension in DIMENSIONS},
            "total_score": computed_total,
            "unsupported_claims": score["unsupported_claims"],
            "major_release_risks": score["major_release_risks"],
            "publishable": score["publishable"],
            "short_rationale": score["short_rationale"],
        }
    if value["winner"] not in set(labels) | {"tie"}:
        raise WorkflowError("invalid winner")
    if not isinstance(value["winner_reason"], str) or not value["winner_reason"].strip():
        raise WorkflowError("winner_reason missing")
    return {"scores": normalized_scores, "winner": value["winner"], "winner_reason": value["winner_reason"]}


def _validate_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"pass", "issues", "revision_instructions"}:
        raise WorkflowError("review response schema mismatch")
    if not isinstance(value["pass"], bool) or not isinstance(value["issues"], list):
        raise WorkflowError("invalid review pass/issues")
    if not isinstance(value["revision_instructions"], list) or any(not isinstance(x, str) or not x.strip() for x in value["revision_instructions"]):
        raise WorkflowError("invalid revision_instructions")
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"type", "severity", "evidence", "instruction"}:
            raise WorkflowError("review issue schema mismatch")
        if issue["type"] not in ISSUE_TYPES or issue["severity"] not in {"minor", "major"}:
            raise WorkflowError("invalid review issue type/severity")
        if any(not isinstance(issue[field], str) or not issue[field].strip() for field in ("evidence", "instruction")):
            raise WorkflowError("review issue evidence/instruction missing")
    if value["pass"] and (value["issues"] or value["revision_instructions"]):
        raise WorkflowError("passing review must not contain issues/instructions")
    if not value["pass"] and (not value["issues"] or not value["revision_instructions"]):
        raise WorkflowError("failing review requires issues and revision instructions")
    return value


def _validate_revision(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {"revised_draft"}:
        raise WorkflowError("revision response must contain exactly revised_draft")
    text = value["revised_draft"]
    if not isinstance(text, str) or not text.strip():
        raise WorkflowError("revised_draft is empty")
    return text


def _existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    return _index(_read_jsonl(path), str(path)) if path.exists() else {}


def _safe_new_json(path: Path, value: Any) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise WorkflowError(f"refusing to overwrite different existing artifact: {path}")
        return
    _write_json(path, value)


def calibrate(args: argparse.Namespace, caller: Callable[..., tuple[dict[str, Any], int]] = _api_json) -> None:
    validation, base, lora = _load_sources(args.validation, args.base_predictions, args.lora_predictions)
    path = args.output_root / "calibration_results.json"
    state = _read_json(path) if path.exists() else {"schema_version": "1.0", "status": "in_progress", "model": FIXED_MODEL, "formal_statistics": False, "results": []}
    if not isinstance(state, dict) or state.get("model") != FIXED_MODEL or not isinstance(state.get("results"), list):
        raise WorkflowError("existing calibration state is incompatible")
    result_index = _index(state["results"], "calibration results")
    if not set(result_index) <= set(CALIBRATION_IDS):
        raise WorkflowError("existing calibration contains an unexpected sample")
    for index, sample_id in enumerate(CALIBRATION_IDS):
        if sample_id not in validation or sample_id not in base or sample_id not in lora:
            raise WorkflowError(f"calibration sample absent from required source: {sample_id}")
        mapping = {"A": "base", "B": "lora_draft"} if index % 2 == 0 else {"A": "lora_draft", "B": "base"}
        labels = _candidate_labels(mapping)
        texts = {"base": _prediction(base[sample_id], sample_id), "lora_draft": _prediction(lora[sample_id], sample_id)}
        request = _calibration_request(_prompt_messages(validation[sample_id], sample_id), {label: texts[model] for label, model in mapping.items()})
        input_sha = _sha_value(request)
        if sample_id in result_index:
            existing = result_index[sample_id]
            if existing.get("input_sha256") != input_sha or existing.get("candidate_mapping") != mapping:
                raise WorkflowError(f"calibration input changed for completed sample: {sample_id}")
            normalized = _validate_score(existing.get("result"), labels)
            if existing.get("result") != normalized:
                existing["result"] = normalized
                _write_json(path, state)
            continue
        response, attempts = caller(request, attempts=args.max_attempts, retry_seconds=args.retry_seconds, timeout=args.timeout)
        _write_json(args.output_root / "calibration_last_raw_response.json", response)
        score = _validate_score(response, labels)
        state["results"].append({"sample_id": sample_id, "input_sha256": input_sha, "candidate_mapping": mapping, "attempts": attempts, "result": score})
        _write_json(path, state)
    state["status"] = "completed"
    state["count"] = len(state["results"])
    if state["count"] != len(CALIBRATION_IDS):
        raise WorkflowError("calibration did not complete exactly two samples")
    _write_json(path, state)


def revise(args: argparse.Namespace, caller: Callable[..., tuple[dict[str, Any], int]] = _api_json) -> None:
    validation, base, lora = _load_sources(args.validation, args.base_predictions, args.lora_predictions)
    sample_ids, supplements = _formal_ids(validation, args.fixed_ids)
    for sample_id in sample_ids:
        if sample_id not in base or sample_id not in lora:
            raise WorkflowError(f"formal sample absent from predictions: {sample_id}")
    directory = args.output_root
    review_path, revised_path = directory / "review_results.jsonl", directory / "revised_predictions.jsonl"
    reviews, revised = _existing_rows(review_path), _existing_rows(revised_path)
    if not set(reviews) <= set(sample_ids) or not set(revised) <= set(sample_ids):
        raise WorkflowError("existing revision state contains an unexpected sample")
    for sample_id in sample_ids:
        prompt = _prompt_messages(validation[sample_id], sample_id)
        draft = _prediction(lora[sample_id], sample_id)
        source_sha = _sha_value({"prompt": prompt, "draft": draft})
        if sample_id in reviews:
            if reviews[sample_id].get("source_sha256") != source_sha:
                raise WorkflowError(f"source changed for completed review: {sample_id}")
            review = _validate_review(reviews[sample_id].get("review"))
        else:
            request = _review_request(prompt, draft)
            response, attempts = caller(request, attempts=args.max_attempts, retry_seconds=args.retry_seconds, timeout=args.timeout)
            _write_json(args.output_root / "reviewer_last_raw_response.json", response)
            try:
                review = _validate_review(response)
            except WorkflowError as first_error:
                retry_request = _review_request(prompt, draft, schema_retry=True)
                retry_response, retry_attempts = caller(
                    retry_request,
                    attempts=args.max_attempts,
                    retry_seconds=args.retry_seconds,
                    timeout=args.timeout,
                )
                attempts += retry_attempts
                _write_json(args.output_root / "reviewer_last_retry_response.json", retry_response)
                try:
                    review = _validate_review(retry_response)
                except WorkflowError as retry_error:
                    raise WorkflowError(
                        f"review response schema retry failed: first={first_error}; retry={retry_error}"
                    ) from retry_error
            reviews[sample_id] = {"sample_id": sample_id, "source_sha256": source_sha, "model": FIXED_MODEL, "attempts": attempts, "review": review}
            _write_jsonl(review_path, (reviews[sid] for sid in sample_ids if sid in reviews))
        if sample_id in revised:
            if revised[sample_id].get("source_sha256") != source_sha:
                raise WorkflowError(f"source changed for completed revision: {sample_id}")
            _prediction(revised[sample_id], sample_id)
            if revised[sample_id].get("revision_applied") is not (not review["pass"]):
                raise WorkflowError(f"revision state contradicts review result: {sample_id}")
        else:
            attempts = 0
            if review["pass"]:
                text, applied = draft, False
            else:
                response, attempts = caller(
                    _revise_request(prompt, draft, review),
                    attempts=args.max_attempts,
                    retry_seconds=args.retry_seconds,
                    timeout=args.timeout,
                )
                _write_json(args.output_root / "reviser_last_raw_response.json", response)
                try:
                    text = _validate_revision(response)
                except WorkflowError as first_error:
                    retry_response, retry_attempts = caller(
                        _revise_request(prompt, draft, review, schema_retry=True),
                        attempts=args.max_attempts,
                        retry_seconds=args.retry_seconds,
                        timeout=args.timeout,
                    )
                    attempts += retry_attempts
                    _write_json(args.output_root / "reviser_last_retry_response.json", retry_response)
                    try:
                        text = _validate_revision(retry_response)
                    except WorkflowError as retry_error:
                        raise WorkflowError(
                            f"reviser response schema retry failed: first={first_error}; retry={retry_error}"
                        ) from retry_error
                applied = True
            revised[sample_id] = {"sample_id": sample_id, "prediction": text, "source_sha256": source_sha, "revision_applied": applied, "reviser_attempts": attempts}
            _write_jsonl(revised_path, (revised[sid] for sid in sample_ids if sid in revised))
        progress = {"status": "in_progress", "expected": FORMAL_COUNT, "reviews_completed": len(reviews), "revisions_completed": len(revised)}
        _write_json(directory / "progress.json", progress)
    manifest = {
        "schema_version": "1.0", "status": "completed", "split": "validation", "test_accessed": False,
        "model": FIXED_MODEL, "sample_count": FORMAL_COUNT, "sample_ids": sample_ids,
        "calibration_ids_excluded": list(CALIBRATION_IDS), "supplemental_sample_ids": supplements,
        "sources": {"validation": str(args.validation), "base_predictions": str(args.base_predictions), "lora_predictions": str(args.lora_predictions)},
        "source_sha256": {"validation": _sha_file(args.validation), "base_predictions": _sha_file(args.base_predictions), "lora_predictions": _sha_file(args.lora_predictions)},
        "review_results": str(review_path), "review_results_sha256": _sha_file(review_path),
        "revised_predictions": str(revised_path), "revised_predictions_sha256": _sha_file(revised_path),
    }
    _write_json(directory / "revision_manifest.json", manifest)
    _write_json(directory / "progress.json", {"status": "completed", "expected": FORMAL_COUNT, "reviews_completed": FORMAL_COUNT, "revisions_completed": FORMAL_COUNT})


def _balanced_mapping(sample_ids: list[str]) -> dict[str, dict[str, str]]:
    if len(sample_ids) != FORMAL_COUNT:
        raise WorkflowError("balanced mapping requires exactly 24 samples")
    rotations = (
        {"A": "base", "B": "lora_draft", "C": "lora_revised"},
        {"A": "lora_draft", "B": "lora_revised", "C": "base"},
        {"A": "lora_revised", "B": "base", "C": "lora_draft"},
    )
    return {sample_id: rotations[index % 3] for index, sample_id in enumerate(sample_ids)}


def pack_judge(args: argparse.Namespace) -> None:
    validation, base, lora = _load_sources(args.validation, args.base_predictions, args.lora_predictions)
    revision_dir = judge_dir = args.output_root
    manifest = _read_json(revision_dir / "revision_manifest.json")
    if manifest.get("status") != "completed" or manifest.get("sample_count") != FORMAL_COUNT or manifest.get("test_accessed") is not False:
        raise WorkflowError("revision manifest is not a completed Validation-only 24-sample run")
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != FORMAL_COUNT or any(s in CALIBRATION_IDS for s in sample_ids):
        raise WorkflowError("revision manifest formal sample IDs are invalid")
    expected_ids, _ = _formal_ids(validation, args.fixed_ids)
    if sample_ids != expected_ids:
        raise WorkflowError("revision manifest sample IDs do not match the deterministic formal selection")
    expected_source_hashes = {
        "validation": _sha_file(args.validation), "base_predictions": _sha_file(args.base_predictions),
        "lora_predictions": _sha_file(args.lora_predictions),
    }
    if manifest.get("source_sha256") != expected_source_hashes:
        raise WorkflowError("revision source hashes do not match current explicit inputs")
    revised_path = revision_dir / "revised_predictions.jsonl"
    if manifest.get("revised_predictions_sha256") != _sha_file(revised_path):
        raise WorkflowError("revised predictions hash mismatch")
    revised = _index(_read_jsonl(revised_path), str(revised_path))
    mapping = _balanced_mapping(sample_ids)
    anonymous: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        if sample_id not in validation or sample_id not in base or sample_id not in lora or sample_id not in revised:
            raise WorkflowError(f"pack source missing sample: {sample_id}")
        texts = {"base": _prediction(base[sample_id], sample_id), "lora_draft": _prediction(lora[sample_id], sample_id), "lora_revised": _prediction(revised[sample_id], sample_id)}
        anonymous.append({"sample_id": sample_id, "editorial_request": _prompt_messages(validation[sample_id], sample_id), "candidates": {label: texts[model] for label, model in mapping[sample_id].items()}})
    key = {"schema_version": "1.0", "count": FORMAL_COUNT, "mapping": mapping}
    anonymous_path, key_path = judge_dir / "editorial_judge_24_anonymous.jsonl", judge_dir / "editorial_judge_24_candidate_key.json"
    if anonymous_path.exists():
        if _read_jsonl(anonymous_path) != anonymous:
            raise WorkflowError(f"refusing to overwrite different existing artifact: {anonymous_path}")
    else:
        _write_jsonl(anonymous_path, anonymous)
    _safe_new_json(key_path, key)
    position_counts = {model: {label: sum(mapping[sid][label] == model for sid in sample_ids) for label in LABELS} for model in MODELS}
    if any(count != FORMAL_COUNT // 3 for counts in position_counts.values() for count in counts.values()):
        raise WorkflowError("candidate mapping is not position-balanced")
    _safe_new_json(judge_dir / "pack_manifest.json", {
        "schema_version": "1.0", "status": "completed", "split": "validation", "test_accessed": False,
        "count": FORMAL_COUNT, "calibration_ids_excluded": list(CALIBRATION_IDS), "sample_ids": sample_ids,
        "anonymous": str(anonymous_path), "anonymous_sha256": _sha_file(anonymous_path),
        "candidate_key": str(key_path), "candidate_key_sha256": _sha_file(key_path), "position_counts": position_counts,
    })


def judge(args: argparse.Namespace, caller: Callable[..., tuple[dict[str, Any], int]] = _api_json) -> None:
    judge_dir = args.output_root
    anonymous_path = judge_dir / "editorial_judge_24_anonymous.jsonl"
    pack = _read_json(judge_dir / "pack_manifest.json")
    if pack.get("status") != "completed" or pack.get("count") != FORMAL_COUNT or pack.get("test_accessed") is not False:
        raise WorkflowError("judge pack manifest is not valid")
    if pack.get("anonymous_sha256") != _sha_file(anonymous_path):
        raise WorkflowError("anonymous judge pack hash mismatch")
    rows = _read_jsonl(anonymous_path)
    if len(rows) != FORMAL_COUNT:
        raise WorkflowError("anonymous judge pack must contain exactly 24 samples")
    raw_path = judge_dir / "editorial_judge_24_raw_results.jsonl"
    completed = _existing_rows(raw_path)
    order = [_sample_id(row) for row in rows]
    for row in rows:
        sample_id = _sample_id(row)
        if set(row) != {"sample_id", "editorial_request", "candidates"} or not isinstance(row["candidates"], dict) or set(row["candidates"]) != set(LABELS):
            raise WorkflowError(f"invalid anonymous sample schema: {sample_id}")
        labels = _candidate_labels(row["candidates"])
        input_sha = _sha_value(row)
        if sample_id in completed:
            if completed[sample_id].get("input_sha256") != input_sha or completed[sample_id].get("model") != FIXED_MODEL:
                raise WorkflowError(f"anonymous input/model changed for completed judgment: {sample_id}")
            _validate_score({field: completed[sample_id].get(field) for field in ("scores", "winner", "winner_reason")}, labels)
            continue
        request = _judge_request(row["editorial_request"], row["candidates"])
        response, attempts = caller(request, attempts=args.max_attempts, retry_seconds=args.retry_seconds, timeout=args.timeout)
        _write_json(judge_dir / "editorial_judge_last_raw_response.json", response)
        result = _validate_score(response, labels)
        _write_json(judge_dir / "editorial_judge_last_total_debug.json", {
            "sample_id": sample_id,
            "totals": {
                label: {
                    "reported_total_score": response["scores"][label].get("total_score"),
                    "computed_total_score": result["scores"][label]["total_score"],
                }
                for label in labels
            },
        })
        completed[sample_id] = {"sample_id": sample_id, "input_sha256": input_sha, "model": FIXED_MODEL, "attempts": attempts, **result}
        _write_jsonl(raw_path, (completed[sid] for sid in order if sid in completed))
        _write_json(judge_dir / "judge_progress.json", {"status": "in_progress", "expected": FORMAL_COUNT, "completed": len(completed)})
    if len(completed) != FORMAL_COUNT:
        raise WorkflowError("formal judge results are incomplete")
    _write_json(judge_dir / "judge_manifest.json", {
        "schema_version": "1.0", "status": "completed", "split": "validation", "test_accessed": False,
        "model": FIXED_MODEL, "count": FORMAL_COUNT, "anonymous_sha256": _sha_file(anonymous_path),
        "raw_results": str(raw_path), "raw_results_sha256": _sha_file(raw_path),
    })
    _write_json(judge_dir / "judge_progress.json", {"status": "completed", "expected": FORMAL_COUNT, "completed": FORMAL_COUNT})


def aggregate(args: argparse.Namespace) -> None:
    judge_dir = args.output_root
    anonymous_path = judge_dir / "editorial_judge_24_anonymous.jsonl"
    raw_path = judge_dir / "editorial_judge_24_raw_results.jsonl"
    manifest = _read_json(judge_dir / "judge_manifest.json")
    anonymous = _read_jsonl(anonymous_path)
    raw = _read_jsonl(raw_path)
    if manifest.get("status") != "completed" or manifest.get("count") != FORMAL_COUNT or len(anonymous) != FORMAL_COUNT or len(raw) != FORMAL_COUNT:
        raise WorkflowError("judging must be complete before candidate identities are read")
    if manifest.get("anonymous_sha256") != _sha_file(anonymous_path) or manifest.get("raw_results_sha256") != _sha_file(raw_path):
        raise WorkflowError("completed judge artifact hash mismatch")
    anonymous_index, raw_index = _index(anonymous, "anonymous judge pack"), _index(raw, "raw judge results")
    if set(anonymous_index) != set(raw_index):
        raise WorkflowError("judge result sample IDs do not match anonymous pack")
    for sample_id, row in raw_index.items():
        if row.get("input_sha256") != _sha_value(anonymous_index[sample_id]) or row.get("model") != FIXED_MODEL:
            raise WorkflowError(f"invalid completed judge result: {sample_id}")
        labels = _candidate_labels(anonymous_index[sample_id]["candidates"])
        _validate_score({field: row.get(field) for field in ("scores", "winner", "winner_reason")}, labels)

    # Blindness boundary: candidate identity is opened only after every result above is complete and valid.
    key_path = judge_dir / "editorial_judge_24_candidate_key.json"
    key = _read_json(key_path)
    mapping = key.get("mapping") if isinstance(key, dict) else None
    if key.get("count") != FORMAL_COUNT or not isinstance(mapping, dict) or set(mapping) != set(raw_index):
        raise WorkflowError("candidate key does not match the completed judge set")
    position_counts = {model: {label: 0 for label in LABELS} for model in MODELS}
    values: dict[str, dict[str, list[float]]] = {model: {field: [] for field in (*DIMENSIONS, "total_score", "unsupported_claims_count", "publishable")} for model in MODELS}
    outcomes = {model: {"wins": 0, "ties": 0, "losses": 0} for model in MODELS}
    for sample_id, result_row in raw_index.items():
        sample_mapping = mapping.get(sample_id)
        if not isinstance(sample_mapping, dict) or set(sample_mapping) != set(LABELS) or set(sample_mapping.values()) != set(MODELS):
            raise WorkflowError(f"invalid candidate mapping for {sample_id}")
        result = {field: result_row[field] for field in ("scores", "winner", "winner_reason")}
        for label, model in sample_mapping.items():
            position_counts[model][label] += 1
            score = result["scores"][label]
            for dimension in DIMENSIONS:
                values[model][dimension].append(score[dimension])
            values[model]["total_score"].append(score["total_score"])
            values[model]["unsupported_claims_count"].append(len(score["unsupported_claims"]))
            values[model]["publishable"].append(1 if score["publishable"] else 0)
        winner = result["winner"]
        if winner == "tie":
            for model in MODELS:
                outcomes[model]["ties"] += 1
        else:
            winner_model = sample_mapping[winner]
            outcomes[winner_model]["wins"] += 1
            for model in MODELS:
                if model != winner_model:
                    outcomes[model]["losses"] += 1
    if any(count != FORMAL_COUNT // 3 for counts in position_counts.values() for count in counts.values()):
        raise WorkflowError("candidate key is not position-balanced")
    summaries: dict[str, Any] = {}
    for model in MODELS:
        summaries[model] = {
            "sample_count": FORMAL_COUNT,
            "dimension_means": {dimension: statistics.fmean(values[model][dimension]) for dimension in DIMENSIONS},
            "total_score_mean": statistics.fmean(values[model]["total_score"]),
            "total_score_stddev": statistics.pstdev(values[model]["total_score"]),
            "publishable_rate": statistics.fmean(values[model]["publishable"]),
            "unsupported_claims_mean": statistics.fmean(values[model]["unsupported_claims_count"]),
            **outcomes[model],
        }
        for value in summaries[model]["dimension_means"].values():
            if not math.isfinite(value):
                raise WorkflowError("non-finite aggregate")
    aggregate_value = {"schema_version": "1.0", "status": "completed", "split": "validation", "test_accessed": False, "count": FORMAL_COUNT, "models": summaries}
    json_path = judge_dir / "editorial_judge_aggregate.json"
    csv_path = judge_dir / "editorial_judge_aggregate.csv"
    md_path = judge_dir / "editorial_judge_report.md"
    _write_json(json_path, aggregate_value)
    fields = ["model", "sample_count", *[f"{d}_mean" for d in DIMENSIONS], "total_score_mean", "total_score_stddev", "publishable_rate", "unsupported_claims_mean", "wins", "ties", "losses"]
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        summary = summaries[model]
        rows.append({"model": model, "sample_count": summary["sample_count"], **{f"{d}_mean": summary["dimension_means"][d] for d in DIMENSIONS}, **{field: summary[field] for field in ("total_score_mean", "total_score_stddev", "publishable_rate", "unsupported_claims_mean", "wins", "ties", "losses")}})
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic_bytes(csv_path, buffer.getvalue().encode("utf-8-sig"))
    headings = ["Model", "N", *DIMENSIONS.keys(), "Total mean", "Total SD", "Publishable", "Unsupported claims", "W", "T", "L"]
    md = ["# Editorial Judge Aggregate", "", "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    for row in rows:
        md.append("| " + " | ".join([str(row["model"]), str(row["sample_count"]), *[f'{row[f"{d}_mean"]:.3f}' for d in DIMENSIONS], f'{row["total_score_mean"]:.3f}', f'{row["total_score_stddev"]:.3f}', f'{row["publishable_rate"]:.3f}', f'{row["unsupported_claims_mean"]:.3f}', str(row["wins"]), str(row["ties"]), str(row["losses"])]) + " |")
    _atomic_bytes(md_path, ("\n".join(md) + "\n").encode("utf-8"))


def revise_test(
    args: argparse.Namespace, caller: Callable[..., tuple[dict[str, Any], int]] = _api_json
) -> None:
    authorization, authorization_sha256 = _authorize_test(args.authorization)
    sample_ids, _ = _test_ids()
    source, base, lora = _load_test_sources(sample_ids)
    directory = TEST_OUTPUT
    review_path = directory / "review_results.jsonl"
    revised_path = directory / "revised_predictions.jsonl"
    reviews, revised = _existing_rows(review_path), _existing_rows(revised_path)
    if not set(reviews) <= set(sample_ids) or not set(revised) <= set(sample_ids):
        raise WorkflowError("existing Test revision state contains an unexpected sample")
    for sample_id in sample_ids:
        prompt = _prompt_messages(source[sample_id], sample_id)
        draft = _prediction(lora[sample_id], sample_id)
        source_sha = _sha_value({"prompt": prompt, "draft": draft})
        if sample_id in reviews:
            if reviews[sample_id].get("source_sha256") != source_sha:
                raise WorkflowError(f"Test source changed for completed review: {sample_id}")
            review = _validate_review(reviews[sample_id].get("review"))
        else:
            response, attempts = caller(
                _review_request(prompt, draft), attempts=args.max_attempts,
                retry_seconds=args.retry_seconds, timeout=args.timeout,
            )
            _write_json(directory / "reviewer_last_raw_response.json", response)
            try:
                review = _validate_review(response)
            except WorkflowError as first_error:
                retry_response, retry_attempts = caller(
                    _review_request(prompt, draft, schema_retry=True), attempts=args.max_attempts,
                    retry_seconds=args.retry_seconds, timeout=args.timeout,
                )
                attempts += retry_attempts
                _write_json(directory / "reviewer_last_retry_response.json", retry_response)
                try:
                    review = _validate_review(retry_response)
                except WorkflowError as retry_error:
                    raise WorkflowError(
                        f"Test review response schema retry failed: first={first_error}; retry={retry_error}"
                    ) from retry_error
            reviews[sample_id] = {
                "sample_id": sample_id, "source_sha256": source_sha, "model": FIXED_MODEL,
                "attempts": attempts, "review": review,
            }
            _write_jsonl(review_path, (reviews[sid] for sid in sample_ids if sid in reviews))
        if sample_id in revised:
            if revised[sample_id].get("source_sha256") != source_sha:
                raise WorkflowError(f"Test source changed for completed revision: {sample_id}")
            _prediction(revised[sample_id], sample_id)
            if revised[sample_id].get("revision_applied") is not (not review["pass"]):
                raise WorkflowError(f"Test revision state contradicts review result: {sample_id}")
        else:
            attempts = 0
            if review["pass"]:
                text, applied = draft, False
            else:
                response, attempts = caller(
                    _revise_request(prompt, draft, review), attempts=args.max_attempts,
                    retry_seconds=args.retry_seconds, timeout=args.timeout,
                )
                _write_json(directory / "reviser_last_raw_response.json", response)
                try:
                    text = _validate_revision(response)
                except WorkflowError as first_error:
                    retry_response, retry_attempts = caller(
                        _revise_request(prompt, draft, review, schema_retry=True), attempts=args.max_attempts,
                        retry_seconds=args.retry_seconds, timeout=args.timeout,
                    )
                    attempts += retry_attempts
                    _write_json(directory / "reviser_last_retry_response.json", retry_response)
                    try:
                        text = _validate_revision(retry_response)
                    except WorkflowError as retry_error:
                        raise WorkflowError(
                            f"Test reviser response schema retry failed: first={first_error}; retry={retry_error}"
                        ) from retry_error
                applied = True
            revised[sample_id] = {
                "sample_id": sample_id, "prediction": text, "source_sha256": source_sha,
                "revision_applied": applied, "reviser_attempts": attempts,
            }
            _write_jsonl(revised_path, (revised[sid] for sid in sample_ids if sid in revised))
        _write_json(directory / "progress.json", {
            "status": "in_progress", "expected": TEST_FORMAL_COUNT,
            "reviews_completed": len(reviews), "revisions_completed": len(revised),
            "split": "test", "test_accessed": True,
        })
    manifest = {
        "schema_version": "1.0", "status": "completed", "split": "test", "test_accessed": True,
        "model": FIXED_MODEL, "sample_count": TEST_FORMAL_COUNT, "sample_ids": sample_ids,
        "authorization": str(authorization), "authorization_sha256": authorization_sha256,
        "comparison_safe_mask": str(TEST_MASK.resolve()), "comparison_safe_mask_sha256": _sha_file(TEST_MASK),
        "sources": {
            "test": str(TEST_SOURCE.resolve()), "base_predictions": str(TEST_BASE.resolve()),
            "lora_predictions": str(TEST_LORA.resolve()),
        },
        "source_sha256": {
            "test": _sha_file(TEST_SOURCE), "base_predictions": _sha_file(TEST_BASE),
            "lora_predictions": _sha_file(TEST_LORA),
        },
        "review_results": str(review_path.resolve()), "review_results_sha256": _sha_file(review_path),
        "revised_predictions": str(revised_path.resolve()),
        "revised_predictions_sha256": _sha_file(revised_path),
    }
    _write_json(directory / "revision_manifest.json", manifest)
    _write_json(directory / "progress.json", {
        "status": "completed", "expected": TEST_FORMAL_COUNT,
        "reviews_completed": TEST_FORMAL_COUNT, "revisions_completed": TEST_FORMAL_COUNT,
        "split": "test", "test_accessed": True,
    })


def _balanced_test_mapping(sample_ids: list[str]) -> dict[str, dict[str, str]]:
    if len(sample_ids) != TEST_FORMAL_COUNT:
        raise WorkflowError("Test candidate mapping requires exactly 50 samples")
    rotations = (
        {"A": "base", "B": "lora_draft", "C": "lora_revised"},
        {"A": "lora_draft", "B": "lora_revised", "C": "base"},
        {"A": "lora_revised", "B": "base", "C": "lora_draft"},
    )
    mapping = {sample_id: rotations[index % 3] for index, sample_id in enumerate(sample_ids)}
    counts = {
        model: {label: sum(mapping[sid][label] == model for sid in sample_ids) for label in LABELS}
        for model in MODELS
    }
    if any(max(model_counts.values()) - min(model_counts.values()) > 1 for model_counts in counts.values()):
        raise WorkflowError("Test candidate mapping is not position-balanced within one sample")
    return mapping


def pack_test_judge(args: argparse.Namespace) -> None:
    authorization, authorization_sha256 = _authorize_test(args.authorization)
    sample_ids, _ = _test_ids()
    source, base, lora = _load_test_sources(sample_ids)
    directory = TEST_OUTPUT
    revision_manifest = _read_json(directory / "revision_manifest.json")
    expected_source_hashes = {
        "test": _sha_file(TEST_SOURCE), "base_predictions": _sha_file(TEST_BASE),
        "lora_predictions": _sha_file(TEST_LORA),
    }
    if (
        revision_manifest.get("status") != "completed"
        or revision_manifest.get("split") != "test"
        or revision_manifest.get("test_accessed") is not True
        or revision_manifest.get("sample_count") != TEST_FORMAL_COUNT
        or revision_manifest.get("sample_ids") != sample_ids
        or revision_manifest.get("authorization_sha256") != authorization_sha256
        or revision_manifest.get("source_sha256") != expected_source_hashes
        or revision_manifest.get("comparison_safe_mask_sha256") != _sha_file(TEST_MASK)
    ):
        raise WorkflowError("completed Test revision manifest is inconsistent")
    revised_path = directory / "revised_predictions.jsonl"
    if revision_manifest.get("revised_predictions_sha256") != _sha_file(revised_path):
        raise WorkflowError("Test revised predictions hash mismatch")
    revised = _index(_read_jsonl(revised_path), str(revised_path))
    if set(revised) != set(sample_ids):
        raise WorkflowError("Test revised predictions must contain exactly the fixed 50 samples")
    mapping = _balanced_test_mapping(sample_ids)
    anonymous: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        texts = {
            "base": _prediction(base[sample_id], sample_id),
            "lora_draft": _prediction(lora[sample_id], sample_id),
            "lora_revised": _prediction(revised[sample_id], sample_id),
        }
        anonymous.append({
            "sample_id": sample_id,
            "editorial_request": _prompt_messages(source[sample_id], sample_id),
            "candidates": {label: texts[model] for label, model in mapping[sample_id].items()},
        })
    key = {"schema_version": "1.0", "count": TEST_FORMAL_COUNT, "mapping": mapping}
    anonymous_path = directory / "editorial_judge_50_anonymous.jsonl"
    key_path = directory / "editorial_judge_50_candidate_key.json"
    if anonymous_path.exists():
        if _read_jsonl(anonymous_path) != anonymous:
            raise WorkflowError(f"refusing to overwrite different existing artifact: {anonymous_path}")
    else:
        _write_jsonl(anonymous_path, anonymous)
    _safe_new_json(key_path, key)
    position_counts = {
        model: {label: sum(mapping[sid][label] == model for sid in sample_ids) for label in LABELS}
        for model in MODELS
    }
    _safe_new_json(directory / "pack_manifest.json", {
        "schema_version": "1.0", "status": "completed", "split": "test", "test_accessed": True,
        "count": TEST_FORMAL_COUNT, "sample_ids": sample_ids,
        "authorization": str(authorization), "authorization_sha256": authorization_sha256,
        "comparison_safe_mask_sha256": _sha_file(TEST_MASK),
        "anonymous": str(anonymous_path.resolve()), "anonymous_sha256": _sha_file(anonymous_path),
        "candidate_key": str(key_path.resolve()), "candidate_key_sha256": _sha_file(key_path),
        "position_counts": position_counts,
    })


def judge_test(
    args: argparse.Namespace, caller: Callable[..., tuple[dict[str, Any], int]] = _api_json
) -> None:
    authorization, authorization_sha256 = _authorize_test(args.authorization)
    directory = TEST_OUTPUT
    anonymous_path = directory / "editorial_judge_50_anonymous.jsonl"
    pack = _read_json(directory / "pack_manifest.json")
    if (
        pack.get("status") != "completed"
        or pack.get("split") != "test"
        or pack.get("test_accessed") is not True
        or pack.get("count") != TEST_FORMAL_COUNT
        or pack.get("authorization_sha256") != authorization_sha256
        or pack.get("anonymous_sha256") != _sha_file(anonymous_path)
    ):
        raise WorkflowError("Test judge pack manifest is inconsistent")
    rows = _read_jsonl(anonymous_path)
    if len(rows) != TEST_FORMAL_COUNT:
        raise WorkflowError("anonymous Test judge pack must contain exactly 50 samples")
    raw_path = directory / "editorial_judge_50_raw_results.jsonl"
    completed = _existing_rows(raw_path)
    order = [_sample_id(row) for row in rows]
    if len(order) != len(set(order)):
        raise WorkflowError("anonymous Test judge pack contains duplicate sample IDs")
    if not set(completed) <= set(order):
        raise WorkflowError("existing Test judge results contain an unexpected sample")
    for row in rows:
        sample_id = _sample_id(row)
        if (
            set(row) != {"sample_id", "editorial_request", "candidates"}
            or not isinstance(row["candidates"], dict)
            or set(row["candidates"]) != set(LABELS)
        ):
            raise WorkflowError(f"invalid anonymous Test sample schema: {sample_id}")
        labels = _candidate_labels(row["candidates"])
        input_sha = _sha_value(row)
        if sample_id in completed:
            existing = completed[sample_id]
            if existing.get("input_sha256") != input_sha or existing.get("model") != FIXED_MODEL:
                raise WorkflowError(f"anonymous Test input/model changed for completed judgment: {sample_id}")
            _validate_score(
                {field: existing.get(field) for field in ("scores", "winner", "winner_reason")}, labels
            )
            continue
        request = _judge_request(row["editorial_request"], row["candidates"])
        response, attempts = caller(
            request, attempts=args.max_attempts,
            retry_seconds=args.retry_seconds, timeout=args.timeout,
        )
        _write_json(directory / "editorial_judge_last_raw_response.json", response)
        result = _validate_score(response, labels)
        _write_json(directory / "editorial_judge_last_total_debug.json", {
            "sample_id": sample_id,
            "totals": {
                label: {
                    "reported_total_score": response["scores"][label].get("total_score"),
                    "computed_total_score": result["scores"][label]["total_score"],
                }
                for label in labels
            },
        })
        completed[sample_id] = {
            "sample_id": sample_id, "input_sha256": input_sha, "model": FIXED_MODEL,
            "attempts": attempts, **result,
        }
        _write_jsonl(raw_path, (completed[sid] for sid in order if sid in completed))
        _write_json(directory / "judge_progress.json", {
            "status": "in_progress", "expected": TEST_FORMAL_COUNT,
            "completed": len(completed), "split": "test", "test_accessed": True,
        })
    if len(completed) != TEST_FORMAL_COUNT:
        raise WorkflowError("formal Test judge results are incomplete")
    _write_json(directory / "judge_manifest.json", {
        "schema_version": "1.0", "status": "completed", "split": "test", "test_accessed": True,
        "model": FIXED_MODEL, "count": TEST_FORMAL_COUNT,
        "authorization": str(authorization), "authorization_sha256": authorization_sha256,
        "anonymous_sha256": _sha_file(anonymous_path),
        "raw_results": str(raw_path.resolve()), "raw_results_sha256": _sha_file(raw_path),
    })
    _write_json(directory / "judge_progress.json", {
        "status": "completed", "expected": TEST_FORMAL_COUNT,
        "completed": TEST_FORMAL_COUNT, "split": "test", "test_accessed": True,
    })


def _release_adjusted_score(score: dict[str, Any]) -> int:
    raw_total = score["total_score"]
    major_release_risks = len(score["major_release_risks"])
    unsupported_claim_count = len(score["unsupported_claims"])
    if major_release_risks >= 2:
        cap = 59
    elif major_release_risks == 1:
        cap = 69
    elif score["publishable"] is False:
        cap = 79
    else:
        cap = 100
    return max(0, min(raw_total, cap) - 3 * unsupported_claim_count)


def aggregate_test(args: argparse.Namespace) -> None:
    authorization, authorization_sha256 = _authorize_test(args.authorization)
    sample_ids, _ = _test_ids()
    directory = TEST_OUTPUT
    anonymous_path = directory / "editorial_judge_50_anonymous.jsonl"
    raw_path = directory / "editorial_judge_50_raw_results.jsonl"
    manifest = _read_json(directory / "judge_manifest.json")
    anonymous = _read_jsonl(anonymous_path)
    raw = _read_jsonl(raw_path)
    if (
        manifest.get("status") != "completed"
        or manifest.get("split") != "test"
        or manifest.get("test_accessed") is not True
        or manifest.get("count") != TEST_FORMAL_COUNT
        or manifest.get("authorization_sha256") != authorization_sha256
        or len(anonymous) != TEST_FORMAL_COUNT
        or len(raw) != TEST_FORMAL_COUNT
    ):
        raise WorkflowError("Test judging must be complete before candidate identities are read")
    if (
        manifest.get("anonymous_sha256") != _sha_file(anonymous_path)
        or manifest.get("raw_results_sha256") != _sha_file(raw_path)
    ):
        raise WorkflowError("completed Test judge artifact hash mismatch")
    anonymous_index = _index(anonymous, "anonymous Test judge pack")
    raw_index = _index(raw, "raw Test judge results")
    if set(anonymous_index) != set(raw_index) or set(raw_index) != set(sample_ids):
        raise WorkflowError("Test judge result sample IDs do not match the fixed 50-sample mask")
    for sample_id, row in raw_index.items():
        if row.get("input_sha256") != _sha_value(anonymous_index[sample_id]) or row.get("model") != FIXED_MODEL:
            raise WorkflowError(f"invalid completed Test judge result: {sample_id}")
        labels = _candidate_labels(anonymous_index[sample_id]["candidates"])
        _validate_score(
            {field: row.get(field) for field in ("scores", "winner", "winner_reason")}, labels
        )

    # Blindness boundary: open the candidate key only after all 50 anonymous results validate.
    key_path = directory / "editorial_judge_50_candidate_key.json"
    pack = _read_json(directory / "pack_manifest.json")
    if pack.get("candidate_key_sha256") != _sha_file(key_path):
        raise WorkflowError("Test candidate key hash mismatch")
    key = _read_json(key_path)
    mapping = key.get("mapping") if isinstance(key, dict) else None
    if key.get("count") != TEST_FORMAL_COUNT or not isinstance(mapping, dict) or set(mapping) != set(raw_index):
        raise WorkflowError("Test candidate key does not match the completed judge set")
    position_counts = {model: {label: 0 for label in LABELS} for model in MODELS}
    values: dict[str, dict[str, list[float]]] = {
        model: {
            field: []
            for field in (*DIMENSIONS, "total_score", "release_adjusted", "unsupported_claims_count", "publishable")
        }
        for model in MODELS
    }
    outcomes = {model: {"wins": 0, "ties": 0, "losses": 0} for model in MODELS}
    adjusted_samples: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        result_row = raw_index[sample_id]
        sample_mapping = mapping.get(sample_id)
        if (
            not isinstance(sample_mapping, dict)
            or set(sample_mapping) != set(LABELS)
            or set(sample_mapping.values()) != set(MODELS)
        ):
            raise WorkflowError(f"invalid Test candidate mapping for {sample_id}")
        sample_adjusted: dict[str, Any] = {"sample_id": sample_id, "models": {}}
        for label, model in sample_mapping.items():
            position_counts[model][label] += 1
            score = result_row["scores"][label]
            for dimension in DIMENSIONS:
                values[model][dimension].append(score[dimension])
            adjusted = _release_adjusted_score(score)
            values[model]["total_score"].append(score["total_score"])
            values[model]["release_adjusted"].append(adjusted)
            values[model]["unsupported_claims_count"].append(len(score["unsupported_claims"]))
            values[model]["publishable"].append(1 if score["publishable"] else 0)
            sample_adjusted["models"][model] = {
                "raw_total_score": score["total_score"],
                "release_adjusted_score": adjusted,
                "major_release_risk_count": len(score["major_release_risks"]),
                "unsupported_claim_count": len(score["unsupported_claims"]),
                "publishable": score["publishable"],
            }
        adjusted_samples.append(sample_adjusted)
        winner = result_row["winner"]
        if winner == "tie":
            for model in MODELS:
                outcomes[model]["ties"] += 1
        else:
            winner_model = sample_mapping[winner]
            outcomes[winner_model]["wins"] += 1
            for model in MODELS:
                if model != winner_model:
                    outcomes[model]["losses"] += 1
    if any(
        max(model_counts.values()) - min(model_counts.values()) > 1
        for model_counts in position_counts.values()
    ):
        raise WorkflowError("Test candidate key is not position-balanced within one occurrence")
    summaries: dict[str, Any] = {}
    adjusted_summaries: dict[str, Any] = {}
    for model in MODELS:
        summaries[model] = {
            "sample_count": TEST_FORMAL_COUNT,
            "dimension_means": {
                dimension: statistics.fmean(values[model][dimension]) for dimension in DIMENSIONS
            },
            "total_score_mean": statistics.fmean(values[model]["total_score"]),
            "total_score_stddev": statistics.pstdev(values[model]["total_score"]),
            "publishable_rate": statistics.fmean(values[model]["publishable"]),
            "unsupported_claims_mean": statistics.fmean(values[model]["unsupported_claims_count"]),
            **outcomes[model],
        }
        adjusted_summaries[model] = {
            "sample_count": TEST_FORMAL_COUNT,
            "release_adjusted_score_mean": statistics.fmean(values[model]["release_adjusted"]),
            "release_adjusted_score_stddev": statistics.pstdev(values[model]["release_adjusted"]),
        }

    source, base, lora = _load_test_sources(sample_ids)
    revised = _index(_read_jsonl(directory / "revised_predictions.jsonl"), "Test revised predictions")
    if set(revised) != set(sample_ids):
        raise WorkflowError("Test revised predictions do not match the fixed 50 samples")
    prediction_sets = {"base": base, "lora_draft": lora, "lora_revised": revised}
    automatic_metrics = {
        model: protocol.evaluate_pairs([
            (_prediction(rows[sample_id], sample_id), str(source[sample_id]["target_text"]))
            for sample_id in sample_ids
        ])
        for model, rows in prediction_sets.items()
    }
    aggregate_value = {
        "schema_version": "1.0", "status": "completed", "split": "test",
        "test_accessed": True, "count": TEST_FORMAL_COUNT,
        "authorization": str(authorization), "authorization_sha256": authorization_sha256,
        "models": summaries, "automatic_metrics_auxiliary": automatic_metrics,
    }
    _write_json(directory / "editorial_judge_test_aggregate.json", aggregate_value)
    raw_fields = [
        "model", "sample_count", *[f"{dimension}_mean" for dimension in DIMENSIONS],
        "total_score_mean", "total_score_stddev", "publishable_rate",
        "unsupported_claims_mean", "wins", "ties", "losses",
    ]
    raw_rows: list[dict[str, Any]] = []
    for model in MODELS:
        summary = summaries[model]
        raw_rows.append({
            "model": model, "sample_count": summary["sample_count"],
            **{f"{dimension}_mean": summary["dimension_means"][dimension] for dimension in DIMENSIONS},
            **{field: summary[field] for field in (
                "total_score_mean", "total_score_stddev", "publishable_rate",
                "unsupported_claims_mean", "wins", "ties", "losses",
            )},
        })
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=raw_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(raw_rows)
    _atomic_bytes(directory / "editorial_judge_test_aggregate.csv", buffer.getvalue().encode("utf-8-sig"))
    adjusted_value = {
        "schema_version": "editorial-release-adjusted-test-v1.0.0",
        "status": "completed", "split": "test", "test_accessed": True,
        "count": TEST_FORMAL_COUNT,
        "formula": "max(0, min(raw_total, cap) - 3 * unsupported_claim_count)",
        "caps": {"major_release_risks_gte_2": 59, "major_release_risks_eq_1": 69,
                 "publishable_false": 79, "otherwise": 100},
        "models": adjusted_summaries, "samples": adjusted_samples,
    }
    _write_json(directory / "editorial_release_adjusted_test_v1.json", adjusted_value)
    adjusted_buffer = io.StringIO(newline="")
    adjusted_fields = [
        "model", "sample_count", "release_adjusted_score_mean", "release_adjusted_score_stddev"
    ]
    adjusted_writer = csv.DictWriter(adjusted_buffer, fieldnames=adjusted_fields, lineterminator="\n")
    adjusted_writer.writeheader()
    adjusted_writer.writerows({"model": model, **adjusted_summaries[model]} for model in MODELS)
    _atomic_bytes(
        directory / "editorial_release_adjusted_test_v1.csv",
        adjusted_buffer.getvalue().encode("utf-8-sig"),
    )
    headings = [
        "Model", "Raw Editorial Score", "Release-Adjusted Score", "Publishable Rate",
        "Unsupported Claims", "Win", "Tie", "Loss",
    ]
    report = [
        "# Final Test Closed-Loop Editorial Evaluation", "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for model in MODELS:
        summary, adjusted = summaries[model], adjusted_summaries[model]
        report.append("| " + " | ".join([
            model, f'{summary["total_score_mean"]:.3f}',
            f'{adjusted["release_adjusted_score_mean"]:.3f}',
            f'{summary["publishable_rate"]:.3f}', f'{summary["unsupported_claims_mean"]:.3f}',
            str(summary["wins"]), str(summary["ties"]), str(summary["losses"]),
        ]) + " |")
    report.extend([
        "", "Release-Adjusted Editorial Score v1 uses the frozen cap and unsupported-claim penalty only.",
        "Automatic BLEU/ROUGE metrics are auxiliary and do not affect Editorial Judge scores or winners.", "",
    ])
    _atomic_bytes(
        directory / "editorial_judge_test_report.md", ("\n".join(report)).encode("utf-8")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop editorial revision and blind judge workflow.")
    parser.set_defaults(
        validation=DEFAULT_VALIDATION, base_predictions=DEFAULT_BASE, lora_predictions=DEFAULT_LORA,
        fixed_ids=DEFAULT_FIXED_IDS, output_root=DEFAULT_OUTPUT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function, api in (("calibrate", calibrate, True), ("revise", revise, True), ("pack-judge", pack_judge, False), ("judge", judge, True), ("aggregate", aggregate, False)):
        subparser = subparsers.add_parser(name)
        if api:
            subparser.add_argument("--max-attempts", type=int, default=5)
            subparser.add_argument("--retry-seconds", type=float, default=2.0)
            subparser.add_argument("--timeout", type=float, default=180.0)
        subparser.set_defaults(function=function)
    for name, function, api in (
        ("revise-test", revise_test, True),
        ("pack-test-judge", pack_test_judge, False),
        ("judge-test", judge_test, True),
        ("aggregate-test", aggregate_test, False),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--authorization", type=Path, required=True)
        if api:
            subparser.add_argument("--max-attempts", type=int, default=5)
            subparser.add_argument("--retry-seconds", type=float, default=2.0)
            subparser.add_argument("--timeout", type=float, default=180.0)
        subparser.set_defaults(function=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_attempts", 1) < 1 or getattr(args, "retry_seconds", 0) < 0 or getattr(args, "timeout", 1) <= 0:
        parser.error("retry/timeout values are invalid")
    try:
        args.function(args)
    except (WorkflowError, protocol.GuardError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
