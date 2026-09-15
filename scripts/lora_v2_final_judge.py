#!/usr/bin/env python3
"""Blind two-pass LLM judging and separate keyed aggregation for LoRA V2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = PROJECT_ROOT / "outputs" / "lora_v2" / "final_test" / "judge"
DEFAULT_ANONYMOUS = JUDGE_DIR / "judge_50_anonymous.jsonl"
DEFAULT_RESULTS = JUDGE_DIR / "judge_50_raw_results.jsonl"
DEFAULT_KEY = JUDGE_DIR / "judge_50_candidate_key.json"
DEFAULT_AGGREGATE = JUDGE_DIR / "judge_50_aggregate.json"
EXPECTED_SAMPLE_COUNT = 50
SCHEMA_VERSION = "lora-v2.1-llm-judge-v1"
MODELS = ("base", "final_v1", "final_v2")
CANDIDATES = ("A", "B", "C")
PASS_DIMENSIONS = {
    "news_accr": ("Accuracy", "Completeness", "Conciseness", "Relevance"),
    "cm_style": ("Formality", "Objectivity", "Structure"),
}
FORBIDDEN_BLIND_FIELDS = {
    "candidate_key",
    "candidate_mapping",
    "label_to_model",
    "model_mapping",
}


class JudgeError(RuntimeError):
    """Fail-closed validation or execution error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JudgeError(f"required JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JudgeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise JudgeError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sample_id(row: dict[str, Any]) -> str:
    for field in ("sample_id", "judge_sample_id", "id"):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    raise JudgeError("anonymous sample has no non-empty sample_id, judge_sample_id, or id")


def _find_forbidden_field(value: Any, location: str = "sample") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_BLIND_FIELDS:
                return f"{location}.{key}"
            found = _find_forbidden_field(child, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_forbidden_field(child, f"{location}[{index}]")
            if found:
                return found
    return None


def _candidate_values(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("candidates")
    if isinstance(nested, dict) and all(label in nested for label in CANDIDATES):
        return {label: nested[label] for label in CANDIDATES}
    if isinstance(nested, list) and len(nested) == len(CANDIDATES):
        labeled: dict[str, Any] = {}
        for item in nested:
            if isinstance(item, dict):
                label = next(
                    (item.get(field) for field in ("label", "candidate_id", "id") if item.get(field) in CANDIDATES),
                    None,
                )
                if label is not None:
                    labeled[label] = item
        if set(labeled) == set(CANDIDATES):
            return labeled
        if not labeled:
            return dict(zip(CANDIDATES, nested))
    if all(label in row for label in CANDIDATES):
        return {label: row[label] for label in CANDIDATES}
    prefixed = {label: row.get(f"candidate_{label}") for label in CANDIDATES}
    if all(value is not None for value in prefixed.values()):
        return prefixed
    raise JudgeError("anonymous sample must contain candidates A, B, and C")


def load_anonymous_samples(path: Path) -> list[dict[str, Any]]:
    if "candidate_key" in path.name.lower():
        raise JudgeError("blind judging input cannot be a candidate-key file")
    rows = _read_jsonl(path)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise JudgeError(
            f"anonymous input must contain exactly {EXPECTED_SAMPLE_COUNT} samples; found {len(rows)}"
        )
    seen: set[str] = set()
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in seen:
            raise JudgeError(f"duplicate anonymous sample_id: {sample_id}")
        seen.add(sample_id)
        forbidden = _find_forbidden_field(row)
        if forbidden:
            raise JudgeError(f"anonymous input leaks a candidate identity field: {forbidden}")
        candidates = _candidate_values(row)
        if any(value is None or value == "" for value in candidates.values()):
            raise JudgeError(f"sample {sample_id} has an empty candidate")
    return rows


def _input_hash(row: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(row).encode("utf-8"))


def _rubric(pass_name: str) -> str:
    if pass_name == "news_accr":
        return (
            "Accuracy: factual consistency with the supplied source/reference; "
            "Completeness: coverage of important supplied facts; "
            "Conciseness: economy without losing essential information; "
            "Relevance: focus on the requested news content."
        )
    return (
        "Formality: professional formal register; "
        "Objectivity: neutral, non-promotional presentation; "
        "Structure: clear and coherent organization."
    )


def _messages(row: dict[str, Any], pass_name: str) -> list[dict[str, str]]:
    dimensions = PASS_DIMENSIONS[pass_name]
    shape = {
        "candidates": {
            label: {
                dimension: {"score": "integer 1-5", "rationale": "very short rationale"}
                for dimension in dimensions
            }
            for label in CANDIDATES
        }
    }
    system = (
        "You are an impartial evaluator of anonymous Chinese news-writing outputs. "
        "Judge only the requested dimensions, independently for every candidate. "
        "Do not infer model identity and do not force a ranking or unique winner. "
        "Return one JSON object only, with no Markdown or extra text. Every score "
        "must be an integer from 1 through 5 and every rationale must be very short."
    )
    user = (
        f"Evaluation pass: {pass_name}\n"
        f"Rubric: {_rubric(pass_name)}\n"
        "Score A, B, and C together. Return exactly this JSON shape (replace values):\n"
        f"{_canonical_json(shape)}\n"
        "Anonymous sample:\n"
        f"{_canonical_json(row)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _http_request(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        _endpoint(api_base),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise JudgeError(f"judge API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"judge API request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JudgeError("judge API returned non-JSON HTTP content") from exc
    if not isinstance(decoded, dict):
        raise JudgeError("judge API response must be a JSON object")
    return decoded


def _response_content(api_response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError("judge API response has no choices[0].message.content") from exc
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise JudgeError("judge API message content must be a JSON string or object")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise JudgeError("judge API message content is not JSON-only") from exc
    if not isinstance(parsed, dict):
        raise JudgeError("judge API message content must decode to an object")
    return parsed


def _validate_scores(value: Any, pass_name: str) -> dict[str, Any]:
    dimensions = PASS_DIMENSIONS[pass_name]
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise JudgeError("judge output must contain only the top-level candidates object")
    candidates = value["candidates"]
    if not isinstance(candidates, dict) or set(candidates) != set(CANDIDATES):
        raise JudgeError("judge output must contain exactly candidates A, B, and C")
    normalized: dict[str, Any] = {}
    for label in CANDIDATES:
        candidate = candidates[label]
        if not isinstance(candidate, dict) or set(candidate) != set(dimensions):
            raise JudgeError(f"candidate {label} must contain exactly {', '.join(dimensions)}")
        normalized[label] = {}
        for dimension in dimensions:
            rating = candidate[dimension]
            if not isinstance(rating, dict) or set(rating) != {"score", "rationale"}:
                raise JudgeError(f"{label}/{dimension} must contain score and rationale")
            score = rating["score"]
            rationale = rating["rationale"]
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise JudgeError(f"{label}/{dimension} score must be an integer from 1 to 5")
            if not isinstance(rationale, str) or not rationale.strip() or len(rationale.strip()) > 240:
                raise JudgeError(f"{label}/{dimension} rationale must be 1-240 characters")
            normalized[label][dimension] = {
                "score": score,
                "rationale": rationale.strip(),
            }
    return normalized


RequestFunction = Callable[[str, list[dict[str, str]]], dict[str, Any]]


def _request_with_retries(
    request_fn: RequestFunction,
    pass_name: str,
    messages: list[dict[str, str]],
    max_attempts: int,
    retry_base_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = request_fn(pass_name, messages)
            scores = _validate_scores(_response_content(response), pass_name)
            return response, scores, attempt
        except Exception as exc:  # retry transport and malformed model output alike
            last_error = exc
            if attempt < max_attempts:
                sleep_fn(retry_base_seconds * (2 ** (attempt - 1)))
    raise JudgeError(f"{pass_name} failed after {max_attempts} attempts: {last_error}") from last_error


def _validate_result_record(
    record: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    model: str | None,
) -> tuple[str, str]:
    sample_id = str(record.get("sample_id", ""))
    pass_name = record.get("pass")
    if sample_id not in samples:
        raise JudgeError(f"result references unknown sample_id: {sample_id}")
    if pass_name not in PASS_DIMENSIONS:
        raise JudgeError(f"result has invalid pass for {sample_id}: {pass_name}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise JudgeError(f"result schema mismatch for {sample_id}/{pass_name}")
    if record.get("input_sha256") != _input_hash(samples[sample_id]):
        raise JudgeError(f"anonymous input changed for completed result {sample_id}/{pass_name}")
    if model is not None and record.get("model") != model:
        raise JudgeError(f"fixed judge model changed for completed result {sample_id}/{pass_name}")
    _validate_scores({"candidates": record.get("scores")}, pass_name)
    return sample_id, pass_name


def run_judging(
    anonymous_path: Path,
    results_path: Path,
    *,
    api_base: str,
    api_key: str,
    model: str,
    max_attempts: int = 5,
    retry_base_seconds: float = 2.0,
    timeout_seconds: float = 120.0,
    request_fn: RequestFunction | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    if not api_base or not api_key or not model:
        raise JudgeError("LLM_JUDGE_API_BASE, LLM_JUDGE_API_KEY, and LLM_JUDGE_MODEL are required")
    if max_attempts < 1:
        raise JudgeError("max_attempts must be at least 1")
    samples_list = load_anonymous_samples(anonymous_path)
    samples = {_sample_id(row): row for row in samples_list}
    existing = _read_jsonl(results_path) if results_path.exists() else []
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in existing:
        key = _validate_result_record(record, samples, model)
        if key in completed:
            raise JudgeError(f"duplicate completed result: {key[0]}/{key[1]}")
        completed[key] = record

    if request_fn is None:
        request_fn = lambda _pass, messages: _http_request(
            api_base, api_key, model, messages, timeout_seconds
        )

    added = 0
    for row in samples_list:
        sample_id = _sample_id(row)
        for pass_name in PASS_DIMENSIONS:
            key = (sample_id, pass_name)
            if key in completed:
                continue
            response, scores, attempts = _request_with_retries(
                request_fn,
                pass_name,
                _messages(row, pass_name),
                max_attempts,
                retry_base_seconds,
                sleep_fn,
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "pass": pass_name,
                "model": model,
                "input_sha256": _input_hash(row),
                "completed_at": _utc_now(),
                "attempts": attempts,
                "scores": scores,
                "api_response": response,
            }
            completed[key] = record
            added += 1
            ordered = [
                completed[(sid, judge_pass)]
                for sample in samples_list
                for sid in [_sample_id(sample)]
                for judge_pass in PASS_DIMENSIONS
                if (sid, judge_pass) in completed
            ]
            _atomic_write_jsonl(results_path, ordered)
    return {"samples": len(samples_list), "completed_passes": len(completed), "added_passes": added}


def _mapping_from_value(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if all(label in value for label in CANDIDATES):
        mapping = {label: value[label] for label in CANDIDATES}
        if all(isinstance(item, str) for item in mapping.values()):
            return mapping
    for field in (
        "candidate_key",
        "candidate_mapping",
        "mapping",
        "label_to_model",
        "model_mapping",
        "candidates",
    ):
        nested = value.get(field)
        mapping = _mapping_from_value(nested)
        if mapping is not None:
            return mapping
    return None


def _load_candidate_key(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise JudgeError(f"candidate key does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"candidate key is not valid JSON: {path}") from exc

    entries: list[tuple[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise JudgeError("candidate key list entries must be objects")
            entries.append((_sample_id(item), item))
    elif isinstance(payload, dict):
        list_value = next(
            (payload.get(field) for field in ("samples", "rows", "items") if isinstance(payload.get(field), list)),
            None,
        )
        if list_value is not None:
            for item in list_value:
                if not isinstance(item, dict):
                    raise JudgeError("candidate key entries must be objects")
                entries.append((_sample_id(item), item))
        else:
            container = payload.get("mapping", payload)
            if not isinstance(container, dict):
                raise JudgeError("candidate key mapping must be an object")
            entries.extend((str(sample_id), value) for sample_id, value in container.items())
    else:
        raise JudgeError("candidate key must be a JSON object or array")

    mappings: dict[str, dict[str, str]] = {}
    for sample_id, value in entries:
        if sample_id in mappings:
            raise JudgeError(f"duplicate candidate-key sample_id: {sample_id}")
        mapping = _mapping_from_value(value)
        if mapping is None:
            raise JudgeError(f"candidate key has no A/B/C mapping for sample {sample_id}")
        if set(mapping.values()) != set(MODELS):
            raise JudgeError(
                f"candidate mapping for {sample_id} must map exactly to {', '.join(MODELS)}"
            )
        mappings[sample_id] = mapping
    return mappings


def aggregate_results(
    anonymous_path: Path,
    results_path: Path,
    candidate_key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    samples_list = load_anonymous_samples(anonymous_path)
    samples = {_sample_id(row): row for row in samples_list}
    records = _read_jsonl(results_path)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = _validate_result_record(record, samples, model=None)
        if key in completed:
            raise JudgeError(f"duplicate completed result: {key[0]}/{key[1]}")
        completed[key] = record
    expected = {(sample_id, pass_name) for sample_id in samples for pass_name in PASS_DIMENSIONS}
    missing = sorted(expected - set(completed))
    if missing:
        preview = ", ".join(f"{sample}/{judge_pass}" for sample, judge_pass in missing[:5])
        raise JudgeError(
            f"judging is incomplete ({len(completed)}/{len(expected)} pass records); missing: {preview}"
        )
    judge_models = {record.get("model") for record in records}
    if len(judge_models) != 1 or not all(isinstance(item, str) and item for item in judge_models):
        raise JudgeError("all completed pass records must use one fixed non-empty judge model")

    # The identity key is intentionally opened only after blind judging is complete.
    mappings = _load_candidate_key(candidate_key_path)
    if set(mappings) != set(samples):
        missing_keys = sorted(set(samples) - set(mappings))
        extra_keys = sorted(set(mappings) - set(samples))
        raise JudgeError(f"candidate key sample mismatch; missing={missing_keys}, extra={extra_keys}")

    values: dict[str, dict[str, dict[str, list[int]]]] = {
        model_name: {
            pass_name: {dimension: [] for dimension in dimensions}
            for pass_name, dimensions in PASS_DIMENSIONS.items()
        }
        for model_name in MODELS
    }
    for sample_id in samples:
        mapping = mappings[sample_id]
        for pass_name, dimensions in PASS_DIMENSIONS.items():
            scores = completed[(sample_id, pass_name)]["scores"]
            for label in CANDIDATES:
                model_name = mapping[label]
                for dimension in dimensions:
                    values[model_name][pass_name][dimension].append(scores[label][dimension]["score"])

    model_output: dict[str, Any] = {}
    for model_name in MODELS:
        pass_output: dict[str, Any] = {}
        for pass_name, dimensions in PASS_DIMENSIONS.items():
            dimension_output: dict[str, Any] = {}
            means: list[float] = []
            for dimension in dimensions:
                scores = values[model_name][pass_name][dimension]
                if len(scores) != EXPECTED_SAMPLE_COUNT or not all(math.isfinite(score) for score in scores):
                    raise JudgeError(f"invalid aggregate count for {model_name}/{pass_name}/{dimension}")
                mean = sum(scores) / len(scores)
                means.append(mean)
                dimension_output[dimension] = {"mean": mean, "count": len(scores)}
            pass_output[pass_name] = {
                "dimensions": dimension_output,
                "macro_mean": sum(means) / len(means),
                "count": EXPECTED_SAMPLE_COUNT,
            }
        model_output[model_name] = {"count": EXPECTED_SAMPLE_COUNT, **pass_output}

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "pass_record_count": len(records),
        "judge_model": next(iter(judge_models)),
        "models": model_output,
        "sources": {
            "anonymous_jsonl": str(anonymous_path),
            "anonymous_sha256": _sha256_file(anonymous_path),
            "raw_results_jsonl": str(results_path),
            "raw_results_sha256": _sha256_file(results_path),
            "candidate_key_json": str(candidate_key_path),
            "candidate_key_sha256": _sha256_file(candidate_key_path),
        },
        "aggregated_at": _utc_now(),
    }
    _atomic_write_json(output_path, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    judge = subparsers.add_parser("judge", help="run blind two-pass judging")
    judge.add_argument("--input", type=Path, default=DEFAULT_ANONYMOUS)
    judge.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    judge.add_argument("--max-attempts", type=int, default=5)
    judge.add_argument("--retry-base-seconds", type=float, default=2.0)
    judge.add_argument("--timeout-seconds", type=float, default=120.0)

    aggregate = subparsers.add_parser("aggregate", help="map blind labels and aggregate scores")
    aggregate.add_argument("--input", type=Path, default=DEFAULT_ANONYMOUS)
    aggregate.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    aggregate.add_argument("--candidate-key", type=Path, default=DEFAULT_KEY)
    aggregate.add_argument("--output", type=Path, default=DEFAULT_AGGREGATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "judge":
            summary = run_judging(
                args.input,
                args.output,
                api_base=os.environ.get("LLM_JUDGE_API_BASE", ""),
                api_key=os.environ.get("LLM_JUDGE_API_KEY", ""),
                model=os.environ.get("LLM_JUDGE_MODEL", ""),
                max_attempts=args.max_attempts,
                retry_base_seconds=args.retry_base_seconds,
                timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        else:
            output = aggregate_results(args.input, args.results, args.candidate_key, args.output)
            print(
                json.dumps(
                    {"status": output["status"], "sample_count": output["sample_count"], "output": str(args.output)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    except JudgeError as exc:
        raise SystemExit(f"FAIL-CLOSED: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
