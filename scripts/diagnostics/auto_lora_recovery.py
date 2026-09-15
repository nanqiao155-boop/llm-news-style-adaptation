#!/usr/bin/env python3
"""Autonomous LoRA recovery runner with an explicit authorized final-Test command."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import lora_v2_final_judge as judge_api  # noqa: E402
from scripts import lora_v2_production_decode as production_decode  # noqa: E402
from src.training import lora_v2 as protocol  # noqa: E402


SCHEMA = "lora-v2.1-auto-recovery-v1.0.0"
OUTPUT_ROOT = ROOT / "outputs" / "lora_v2" / "auto_recovery"
E05_RUN_DIR = ROOT / "outputs" / "lora_v2" / "E05" / "E05_rank32_run1"
E05_MANIFEST = E05_RUN_DIR / "run_manifest.json"
EARLY_RUN_DIR = ROOT / "outputs" / "lora_v2" / "diagnostics" / "E05_epoch1_run1"
CHECKPOINT_117 = EARLY_RUN_DIR / "checkpoints" / "checkpoint-117"
JUDGE24_DIR = ROOT / "outputs" / "lora_v2" / "production_decode" / "validation_judge24"
JUDGE24_IDS = JUDGE24_DIR / "judge24_ids.txt"
SEVERE12_IDS = ROOT / "outputs" / "lora_v2" / "production_decode" / "e05_severe12_ids.txt"
EARLYSTOP_JUDGE_DIR = EARLY_RUN_DIR / "judge24_earlystop"
EARLYSTOP_JUDGE_ANONYMOUS = EARLYSTOP_JUDGE_DIR / "judge24_anonymous.jsonl"
EARLYSTOP_JUDGE_KEY = EARLYSTOP_JUDGE_DIR / "judge24_candidate_key.json"
EARLYSTOP_JUDGE_RAW = EARLYSTOP_JUDGE_DIR / "judge24_raw_results.jsonl"
EARLYSTOP_JUDGE_AGGREGATE = EARLYSTOP_JUDGE_DIR / "judge24_aggregate.json"
STRENGTHS = (0.25, 0.50, 0.75, 1.00)
FROZEN_DECODING = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "num_beams": 1,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}
TRAINING_CANDIDATES = {
    "candidate_a": {"learning_rate": 1e-4, "rank": 32, "alpha": 32, "epoch": 0.5},
    "candidate_b": {"learning_rate": 1e-4, "rank": 16, "alpha": 16, "epoch": 0.5},
}
FINAL_TEST_CANDIDATE = "strength_0.50"
FINAL_TEST_STRENGTH = 0.50
FINAL_TEST_COUNT = 233


class RecoveryError(RuntimeError):
    """A recovery safety or integrity guard failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RecoveryError(f"checkpoint contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{_utc_now()}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return list(protocol.iter_jsonl(path))
    except (OSError, protocol.GuardError) as exc:
        raise RecoveryError(str(exc)) from exc


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assert_not_test_path(path: Path) -> None:
    if any(part.lower() == "test" or part.lower().startswith("test.") for part in path.parts):
        raise RecoveryError(f"runner refuses to read a Test path: {path}")


def _fixed_ids(path: Path, expected_count: int, label: str) -> list[str]:
    if not path.is_file():
        raise RecoveryError(f"fixed {label} ID file is missing: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise RecoveryError("judge24 ID JSON must be an array")
        ids = [str(item).strip() for item in value]
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip()]
    if len(ids) != expected_count or len(set(ids)) != expected_count or any(not item for item in ids):
        raise RecoveryError(
            f"{label} ID file must contain exactly {expected_count} unique IDs; found {len(ids)}"
        )
    return ids


def _sample_ids(path: Path) -> list[str]:
    return _fixed_ids(path, 24, "judge24")


def _rows_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise RecoveryError("rows contain missing or duplicate sample_id")
        result[sample_id] = row
    return result


def _prediction_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("predictions.jsonl") if "auto_recovery" not in path.parts]


def _find_predictions(root: Path, ids: Sequence[str], kind: str) -> Path:
    matches: list[Path] = []
    wanted = set(ids)
    for path in _prediction_candidates(root):
        try:
            _assert_not_test_path(path)
            rows = _read_jsonl(path)
        except RecoveryError:
            continue
        indexed = _rows_by_id(rows)
        if not wanted <= set(indexed):
            continue
        checkpoints = {str(indexed[item].get("checkpoint", "")) for item in ids}
        if kind == "base" and not all(value in {"", "None", "base_model"} for value in checkpoints):
            continue
        if kind == "checkpoint-117" and not any("117" in value for value in checkpoints | {str(path)}):
            continue
        matches.append(path)
    if not matches:
        raise RecoveryError(f"could not find {kind} Validation predictions for all fixed 24 IDs under {root}")
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    if len(matches) > 1 and len(matches[0].parts) == len(matches[1].parts):
        raise RecoveryError(f"ambiguous {kind} prediction artifacts: {[str(path) for path in matches[:5]]}")
    return matches[0]


def _stage_status(stage_dir: Path) -> dict[str, Any] | None:
    path = stage_dir / "status.json"
    return _read_json(path) if path.is_file() else None


def _stage_completed(stage_dir: Path, required: Sequence[str]) -> bool:
    status = _stage_status(stage_dir)
    if not status or status.get("status") != "completed":
        return False
    missing = [name for name in required if not (stage_dir / name).is_file()]
    if missing:
        raise RecoveryError(f"completed stage is missing outputs at {stage_dir}: {missing}")
    return True


def _start_stage(
    stage_dir: Path, name: str, manifest: dict[str, Any], *, test_accessed: bool = False
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    status = _stage_status(stage_dir) or {}
    if status.get("status") == "completed":
        raise RecoveryError(f"refusing to restart completed stage: {name}")
    attempt = int(status.get("attempt", 0)) + 1
    _atomic_json(stage_dir / "manifest.json", manifest)
    _atomic_json(stage_dir / "status.json", {
        "schema_version": SCHEMA, "stage": name, "status": "running", "attempt": attempt,
        "started_at": _utc_now(), "test_accessed": test_accessed,
    })
    _append_log(stage_dir / "stage.log", f"START {name} attempt={attempt}")


def _finish_stage(
    stage_dir: Path, name: str, result: dict[str, Any], status: str = "completed", *,
    test_accessed: bool = False,
) -> None:
    previous = _stage_status(stage_dir) or {}
    _atomic_json(stage_dir / "status.json", {
        **previous, "schema_version": SCHEMA, "stage": name, "status": status,
        "finished_at": _utc_now(), "test_accessed": test_accessed, "result": result,
    })
    _append_log(stage_dir / "stage.log", f"END {name} status={status}")


def _fail_stage(stage_dir: Path, name: str, exc: BaseException, *, test_accessed: bool = False) -> None:
    previous = _stage_status(stage_dir) or {}
    _atomic_json(stage_dir / "status.json", {
        **previous, "schema_version": SCHEMA, "stage": name, "status": "failed",
        "failed_at": _utc_now(), "error_type": type(exc).__name__, "error": str(exc),
        "test_accessed": test_accessed,
    })
    _append_log(stage_dir / "stage.log", f"FAIL {name}: {type(exc).__name__}: {exc}")


def _load_plan(output_root: Path) -> dict[str, Any]:
    plan_path = output_root / "plan" / "resolved_plan.json"
    plan = _read_json(plan_path)
    for field in (
        "e05_manifest", "e05_yaml", "checkpoint_117", "judge24_ids", "base_predictions",
        "epoch05_predictions", "train_dataset", "validation_dataset",
    ):
        evidence = plan[field]
        path = Path(evidence["path"] if isinstance(evidence, dict) else evidence)
        if not path.exists():
            raise RecoveryError(f"planned source disappeared: {path}")
        expected_hash = evidence.get("sha256") if isinstance(evidence, dict) else None
        if expected_hash and path.is_file() and _sha256(path) != expected_hash:
            raise RecoveryError(f"planned source changed: {path}")
        expected_tree_hash = evidence.get("tree_sha256") if isinstance(evidence, dict) else None
        if expected_tree_hash and path.is_dir() and _tree_sha256(path) != expected_tree_hash:
            raise RecoveryError(f"planned checkpoint tree changed: {path}")
    return plan


def create_plan(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    stage_dir = output_root / "plan"
    if _stage_completed(stage_dir, ("resolved_plan.json", "manifest.json", "stage.log")):
        return _load_plan(output_root)
    manifest = _read_json(E05_MANIFEST)
    if manifest.get("experiment_id") != "E05" or manifest.get("dataset_version") != "sft_v2.0.0":
        raise RecoveryError("source run must be the frozen E05 sft_v2.0.0 lineage")
    if manifest.get("model_name") != "Qwen/Qwen3-4B-Instruct-2507" or manifest.get("chat_template") != "qwen3_nothink":
        raise RecoveryError("source E05 model/template differs from the frozen protocol")
    if not CHECKPOINT_117.is_dir():
        raise RecoveryError(f"checkpoint-117 is missing: {CHECKPOINT_117}")
    ids = _sample_ids(JUDGE24_IDS)
    validation_path = protocol.split_path(manifest["dataset_version"], "validation", purpose="validation_inference")
    train_path = protocol.split_path(manifest["dataset_version"], "train", purpose="gradient_training")
    _assert_not_test_path(validation_path); _assert_not_test_path(train_path)
    base_predictions = _find_predictions(ROOT / "outputs" / "lora_v2" / "E00", ids, "base")
    epoch05_predictions = _find_predictions(EARLY_RUN_DIR, ids, "checkpoint-117")
    yaml_path = E05_RUN_DIR / "llamafactory.yaml"
    if not yaml_path.is_file():
        raise RecoveryError(f"source E05 training YAML is missing: {yaml_path}")
    resolved = {
        "schema_version": SCHEMA,
        "status": "planned",
        "created_at": _utc_now(),
        "output_root": str(output_root.resolve()),
        "e05_manifest": {"path": str(E05_MANIFEST.resolve()), "sha256": _sha256(E05_MANIFEST)},
        "e05_yaml": {"path": str(yaml_path.resolve()), "sha256": _sha256(yaml_path)},
        "checkpoint_117": {"path": str(CHECKPOINT_117.resolve()), "tree_sha256": _tree_sha256(CHECKPOINT_117)},
        "judge24_ids": {"path": str(JUDGE24_IDS.resolve()), "sha256": _sha256(JUDGE24_IDS), "count": 24},
        "base_predictions": {"path": str(base_predictions.resolve()), "sha256": _sha256(base_predictions)},
        "epoch05_predictions": {"path": str(epoch05_predictions.resolve()), "sha256": _sha256(epoch05_predictions)},
        "train_dataset": {"path": str(train_path.resolve()), "sha256": _sha256(train_path)},
        "validation_dataset": {"path": str(validation_path.resolve()), "sha256": _sha256(validation_path)},
        "strengths": list(STRENGTHS),
        "frozen_decoding": FROZEN_DECODING,
        "training_candidates": TRAINING_CANDIDATES,
        "maximum_new_training_candidates": 2,
        "test_accessed": False,
    }
    try:
        _start_stage(stage_dir, "plan", {"inputs": resolved, "test_accessed": False})
        _atomic_json(stage_dir / "resolved_plan.json", resolved)
        _finish_stage(stage_dir, "plan", {"planned": True})
    except Exception as exc:
        _fail_stage(stage_dir, "plan", exc)
        raise
    return resolved


def _describe_numbers(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values), "min": ordered[0], "max": ordered[-1],
        "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def _target_flags(text: str) -> dict[str, bool]:
    visible_control = any(ord(char) < 32 and char not in "\n\r\t" for char in text)
    return {
        "fullwidth_digit": bool(re.search(r"[０-９]", text)),
        "replacement_or_private_use": "�" in text or bool(re.search(r"[\ue000-\uf8ff]", text)),
        "control_character": visible_control,
        "ocr_like_spacing": bool(re.search(r"(?:[A-Za-zＡ-Ｚａ-ｚ]\s+){3,}[A-Za-zＡ-Ｚａ-ｚ]", text)),
    }


def _dataset_statistics(rows: Sequence[dict[str, Any]], tokenizer: Any, cutoff_len: int) -> dict[str, Any]:
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    newline_counts: list[int] = []
    ratios: list[float] = []
    target_counts: Counter[str] = Counter()
    repeated_line_ids: list[str] = []
    anomalies: list[dict[str, Any]] = []
    overlong: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        target = str(row["target_text"])
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("content") != target:
            raise RecoveryError(f"invalid messages/assistant target alignment: {sample_id}")
        user_text = str(messages[-2].get("content", ""))
        ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        token_count = len(ids)
        char_lengths.append(len(target)); token_lengths.append(token_count)
        newline_counts.append(target.count("\n")); target_counts[target] += 1
        ratios.append(len(target) / max(1, len(user_text)))
        flags = _target_flags(target)
        if any(flags.values()):
            anomalies.append({"sample_id": sample_id, "flags": flags, "snippet": target[:160]})
        lines = [line.strip() for line in target.splitlines() if line.strip()]
        if any(value >= 2 for value in Counter(lines).values()):
            repeated_line_ids.append(sample_id)
        if token_count > cutoff_len:
            overlong.append({"sample_id": sample_id, "tokens": token_count, "characters": len(target)})
    duplicate_groups = [
        {"sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(), "count": count}
        for target, count in target_counts.items() if count > 1
    ]
    return {
        "sample_count": len(rows),
        "target_character_length": _describe_numbers(char_lengths),
        "target_token_length": _describe_numbers(token_lengths),
        "target_newline_count": _describe_numbers(newline_counts),
        "assistant_target_to_user_input_character_ratio": _describe_numbers(ratios),
        "exact_duplicate_target_group_count": len(duplicate_groups),
        "exact_duplicate_target_excess_count": sum(item["count"] - 1 for item in duplicate_groups),
        "exact_duplicate_target_groups": duplicate_groups,
        "target_internal_repeated_line_sample_count": len(repeated_line_ids),
        "target_internal_repeated_line_sample_ids": repeated_line_ids,
        "obvious_character_anomaly_sample_count": len(anomalies),
        "obvious_character_anomalies": anomalies,
        "overlong_threshold_tokens": cutoff_len,
        "overlong_target_count": len(overlong),
        "overlong_targets": overlong,
    }


def _load_tokenizer(manifest: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer  # type: ignore
    model_source, load_kwargs = protocol._inference_model_location(manifest)
    return AutoTokenizer.from_pretrained(model_source, trust_remote_code=False, **load_kwargs)


def _template_sanity(rows: Sequence[dict[str, Any]], tokenizer: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    checked = rows[:64]
    eos_id = tokenizer.eos_token_id
    for row in checked:
        sample_id = str(row["sample_id"])
        messages = row["messages"]
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if eos_id is None or not full_ids or full_ids[-1] != eos_id:
            anomalies.append({"sample_id": sample_id, "issue": "rendered_training_example_does_not_end_in_eos"})
        if full_ids[: len(prompt_ids)] != prompt_ids:
            anomalies.append({"sample_id": sample_id, "issue": "training_prefix_differs_from_inference_prompt"})
    return {
        "checked_sample_count": len(checked),
        "chat_template": manifest.get("chat_template"),
        "chat_template_expected": "qwen3_nothink",
        "enable_thinking": False,
        "training_and_inference_prompt_policy": "same tokenizer.apply_chat_template prefix with enable_thinking=false",
        "eos_token": tokenizer.eos_token,
        "eos_token_id": eos_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "explicit_anomaly_count": len(anomalies),
        "explicit_anomalies": anomalies,
    }


def _key_mapping(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if all(label in value and isinstance(value[label], str) for label in "ABC"):
        return {label: value[label] for label in "ABC"}
    for field in ("candidate_mapping", "candidate_key", "mapping", "label_to_model", "candidates"):
        result = _key_mapping(value.get(field))
        if result:
            return result
    return None


def _existing_judge_material(ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    fixed_paths = (
        EARLYSTOP_JUDGE_ANONYMOUS,
        EARLYSTOP_JUDGE_KEY,
        EARLYSTOP_JUDGE_RAW,
        EARLYSTOP_JUDGE_AGGREGATE,
    )
    missing = [str(path) for path in fixed_paths if not path.is_file()]
    if missing:
        raise RecoveryError(f"fixed early-stop Judge artifacts are missing: {missing}")
    anonymous_rows = _read_jsonl(EARLYSTOP_JUDGE_ANONYMOUS)
    if len(anonymous_rows) != 24 or {str(row.get("sample_id", "")) for row in anonymous_rows} != set(ids):
        raise RecoveryError("fixed early-stop anonymous pack does not match the fixed 24 sample IDs")
    key_payload = _read_json(EARLYSTOP_JUDGE_KEY)
    raw_mapping = key_payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise RecoveryError("fixed early-stop candidate key mapping must be an object")
    if len(raw_mapping) != 24:
        raise RecoveryError(
            f"fixed early-stop candidate key mapping must contain exactly 24 samples; found {len(raw_mapping)}"
        )
    if key_payload.get("count", 24) != 24:
        raise RecoveryError(f"fixed early-stop candidate key count must be 24; found {key_payload.get('count')}")
    allowed_models = {"base", "epoch0.5", "epoch1.0"}
    mappings: dict[str, dict[str, str]] = {}
    for raw_sample_id, value in raw_mapping.items():
        sample_id = str(raw_sample_id)
        if not sample_id or not isinstance(value, dict) or set(value) != set("ABC"):
            raise RecoveryError(f"fixed early-stop mapping must contain exactly A/B/C for sample {sample_id!r}")
        mapping = {label: value[label] for label in "ABC"}
        if any(not isinstance(item, str) or item not in allowed_models for item in mapping.values()):
            raise RecoveryError(f"fixed early-stop mapping has an invalid model value for sample {sample_id}")
        if set(mapping.values()) != allowed_models:
            raise RecoveryError(f"fixed early-stop mapping must assign all three models for sample {sample_id}")
        mappings[sample_id] = mapping
    raw_rows = _read_jsonl(EARLYSTOP_JUDGE_RAW)
    aggregate = _read_json(EARLYSTOP_JUDGE_AGGREGATE)
    aggregate_count = aggregate.get("sample_count", aggregate.get("count"))
    if aggregate_count is not None and aggregate_count != 24:
        raise RecoveryError(f"fixed early-stop Judge aggregate sample_count must be 24; found {aggregate_count}")
    records = {(str(row["sample_id"]), str(row["pass"])): row for row in raw_rows}
    expected_records = {(sample_id, pass_name) for sample_id in ids for pass_name in ("news_accr", "cm_style")}
    if set(records) != expected_records or set(mappings) != set(ids):
        raise RecoveryError("judge24 raw results/key do not cover exactly the fixed 24 IDs and two passes")
    return {
        "anonymous_path": str(EARLYSTOP_JUDGE_ANONYMOUS.resolve()),
        "anonymous_sha256": _sha256(EARLYSTOP_JUDGE_ANONYMOUS),
        "key_path": str(EARLYSTOP_JUDGE_KEY.resolve()), "key_sha256": _sha256(EARLYSTOP_JUDGE_KEY),
        "raw_path": str(EARLYSTOP_JUDGE_RAW.resolve()), "raw_sha256": _sha256(EARLYSTOP_JUDGE_RAW),
        "aggregate_path": str(EARLYSTOP_JUDGE_AGGREGATE.resolve()),
        "aggregate_sha256": _sha256(EARLYSTOP_JUDGE_AGGREGATE),
        "records": records,
    }, mappings


def _alias_kind(value: str) -> str | None:
    normalized = value.lower().replace("_", "").replace("-", "")
    if "base" in normalized:
        return "base"
    if "checkpoint117" in normalized or "epoch0.5" in value.lower() or "epoch05" in normalized or "0.5" in value:
        return "epoch0.5"
    if "checkpoint234" in normalized or "epoch1.0" in value.lower() or "epoch10" in normalized or "1.0" in value:
        return "epoch1.0"
    return None


def _worst_normal_examples(
    ids: Sequence[str], validation_rows: dict[str, dict[str, Any]], base_rows: dict[str, dict[str, Any]],
    epoch_rows: dict[str, dict[str, Any]], judge_material: dict[str, Any], mappings: dict[str, dict[str, str]],
    severe_ids: Sequence[str],
) -> list[dict[str, Any]]:
    severe = set(severe_ids)
    if len(severe) != 12 or not severe <= set(ids):
        raise RecoveryError("fixed severe12 IDs must be 12 unique members of judge24")
    normal_ids = [sample_id for sample_id in ids if sample_id not in severe]
    if len(normal_ids) != 12:
        raise RecoveryError("judge24 minus severe12 must produce exactly normal12")
    examples: list[dict[str, Any]] = []
    for sample_id in normal_ids:
        mapping = mappings[sample_id]
        reverse = {_alias_kind(alias): label for label, alias in mapping.items()}
        if "base" not in reverse or "epoch0.5" not in reverse:
            raise RecoveryError(f"judge key does not identify Base and epoch0.5 for {sample_id}: {mapping}")
        rationales: dict[str, Any] = {}
        gaps: list[float] = []
        for pass_name in ("news_accr", "cm_style"):
            scores = judge_material["records"][(sample_id, pass_name)]["scores"]
            pass_result: dict[str, Any] = {}
            for kind in ("base", "epoch0.5"):
                candidate = scores[reverse[kind]]
                dimension_scores = [float(item["score"]) for item in candidate.values()]
                pass_result[kind] = {"macro_mean": statistics.fmean(dimension_scores), "dimensions": candidate}
            gaps.append(pass_result["base"]["macro_mean"] - pass_result["epoch0.5"]["macro_mean"])
            rationales[pass_name] = pass_result
        row = validation_rows[sample_id]
        examples.append({
            "sample_id": sample_id,
            "base_minus_epoch05_judge_gap": statistics.fmean(gaps),
            "prompt_input": row["messages"][:-1],
            "reference": row["target_text"],
            "base_output": base_rows[sample_id]["prediction"],
            "epoch0.5_output": epoch_rows[sample_id]["prediction"],
            "judge": rationales,
        })
    examples.sort(key=lambda item: (-item["base_minus_epoch05_judge_gap"], item["sample_id"]))
    return examples[:5]


def run_diagnosis(plan: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    stage_dir = output_root / "diagnosis"
    required = ("worst_normal_examples.json", "data_statistics.json", "template_eos_sanity.json", "summary.json")
    if _stage_completed(stage_dir, required):
        return _read_json(stage_dir / "summary.json")
    manifest = _read_json(Path(plan["e05_manifest"]["path"]))
    ids = _sample_ids(Path(plan["judge24_ids"]["path"]))
    severe_ids = _fixed_ids(SEVERE12_IDS, 12, "severe12")
    train_path = Path(plan["train_dataset"]["path"]); validation_path = Path(plan["validation_dataset"]["path"])
    _assert_not_test_path(train_path); _assert_not_test_path(validation_path)
    try:
        _start_stage(stage_dir, "diagnosis", {
            "schema_version": SCHEMA, "sources": plan, "operations": ["read_only_statistics", "tokenizer_sanity"],
            "fixed_earlystop_judge": {
                "anonymous": str(EARLYSTOP_JUDGE_ANONYMOUS.resolve()),
                "candidate_key": str(EARLYSTOP_JUDGE_KEY.resolve()),
                "raw_results": str(EARLYSTOP_JUDGE_RAW.resolve()),
                "aggregate": str(EARLYSTOP_JUDGE_AGGREGATE.resolve()),
            },
            "fixed_severe12_ids": str(SEVERE12_IDS.resolve()),
            "training_performed": False, "test_accessed": False,
        })
        train_rows = _read_jsonl(train_path); validation = _read_jsonl(validation_path)
        validation_rows = _rows_by_id(validation)
        if not set(ids) <= set(validation_rows):
            raise RecoveryError("fixed judge24 IDs are not all in frozen Validation")
        base_rows = _rows_by_id(_read_jsonl(Path(plan["base_predictions"]["path"])))
        epoch_rows = _rows_by_id(_read_jsonl(Path(plan["epoch05_predictions"]["path"])))
        judge_material, mappings = _existing_judge_material(ids)
        worst = _worst_normal_examples(
            ids, validation_rows, base_rows, epoch_rows, judge_material, mappings, severe_ids
        )
        _atomic_json(stage_dir / "worst_normal_examples.json", {
            "schema_version": SCHEMA,
            "normal_id_policy": "fixed judge24 IDs minus fixed e05_severe12 IDs",
            "severe12_ids_file": str(SEVERE12_IDS.resolve()),
            "severe12_ids_file_sha256": _sha256(SEVERE12_IDS),
            "count": len(worst), "examples": worst, "test_accessed": False,
        })
        tokenizer = _load_tokenizer(manifest)
        cutoff_len = int(manifest.get("cutoff_len", 4096))
        statistics_output = {
            "schema_version": SCHEMA,
            "train": _dataset_statistics(train_rows, tokenizer, cutoff_len),
            "validation": _dataset_statistics(validation, tokenizer, cutoff_len),
            "test_accessed": False,
        }
        _atomic_json(stage_dir / "data_statistics.json", statistics_output)
        sanity = _template_sanity([*train_rows[:32], *validation[:32]], tokenizer, manifest)
        sanity["test_accessed"] = False
        _atomic_json(stage_dir / "template_eos_sanity.json", sanity)
        summary = {
            "status": "completed", "worst_normal_count": len(worst),
            "dataset_issue_evidence": {
                "train_duplicate_groups": statistics_output["train"]["exact_duplicate_target_group_count"],
                "train_repeated_line_samples": statistics_output["train"]["target_internal_repeated_line_sample_count"],
                "train_anomaly_samples": statistics_output["train"]["obvious_character_anomaly_sample_count"],
                "train_overlong_targets": statistics_output["train"]["overlong_target_count"],
            },
            "template_eos_explicit_anomaly_count": sanity["explicit_anomaly_count"],
            "existing_judge_material": {key: value for key, value in judge_material.items() if key != "records"},
            "test_accessed": False,
        }
        _atomic_json(stage_dir / "summary.json", summary)
        _finish_stage(stage_dir, "diagnosis", summary)
        return summary
    except Exception as exc:
        _fail_stage(stage_dir, "diagnosis", exc)
        raise


def _set_seed(torch_module: Any, seed: int = 42) -> None:
    random.seed(seed)
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def _load_model(manifest: dict[str, Any], checkpoint: Path) -> tuple[Any, Any, Any]:
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    model_source, load_kwargs = protocol._inference_model_location(manifest)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False, **load_kwargs
    )
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()
    return torch, tokenizer, model


def _lora_scaling(model: Any) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for module_name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name, value in scaling.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[(module_name, str(adapter_name))] = float(value)
    if not values:
        raise RecoveryError("no PEFT LoRA scaling values were found")
    return values


def _apply_lora_strength(model: Any, originals: dict[tuple[str, str], float], strength: float) -> None:
    modules = dict(model.named_modules())
    for (module_name, adapter_name), original in originals.items():
        module = modules.get(module_name)
        scaling = getattr(module, "scaling", None)
        if module is None or not isinstance(scaling, dict) or adapter_name not in scaling:
            raise RecoveryError(f"LoRA scaling topology changed: {module_name}/{adapter_name}")
        scaling[adapter_name] = original * strength


def _prediction_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = production_decode.repetition_diagnostics(rows)
    chars = [len(str(row["prediction"])) for row in rows]
    tokens = [int(row.get("generated_token_count", 0)) for row in rows]
    return {
        "repetition": diagnostics,
        "length_stopping": {
            "prediction_characters": _describe_numbers(chars),
            "generated_tokens": _describe_numbers(tokens),
            "stopped_on_eos": sum(row.get("stopped_on_eos") is True for row in rows),
            "hit_max_new_tokens": sum(row.get("hit_max_new_tokens") is True for row in rows),
            "count": len(rows),
        },
    }


def _generate_rows(
    torch: Any, tokenizer: Any, model: Any, source_rows: Sequence[dict[str, Any]], predictions_path: Path,
    metadata: dict[str, Any], progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    existing = _read_jsonl(predictions_path) if predictions_path.is_file() else []
    expected_ids = [str(row["sample_id"]) for row in source_rows]
    existing_ids = [str(row.get("sample_id", "")) for row in existing]
    if existing_ids != expected_ids[: len(existing_ids)]:
        raise RecoveryError(f"partial predictions are not a valid source-order prefix: {predictions_path}")
    mode = "a" if predictions_path.exists() else "x"
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in source_rows[len(existing):]:
            prompt = tokenizer.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generation = {key: value for key, value in FROZEN_DECODING.items() if key != "no_repeat_ngram_size"}
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            input_length = inputs["input_ids"].shape[1]
            continuation = generated[0, input_length:]
            prediction = tokenizer.decode(continuation, skip_special_tokens=True)
            token_ids = continuation.tolist() if hasattr(continuation, "tolist") else list(continuation)
            record = {
                "sample_id": row["sample_id"], "reference": row["target_text"], "prediction": prediction,
                "generated_token_count": len(token_ids),
                "stopped_on_eos": bool(token_ids and token_ids[-1] == tokenizer.eos_token_id),
                "hit_max_new_tokens": len(token_ids) >= FROZEN_DECODING["max_new_tokens"],
                **metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            existing.append(record)
            if progress:
                progress(len(existing))
    return existing


def _evaluate_adapter(
    stage_dir: Path, name: str, torch: Any, tokenizer: Any, model: Any,
    source_rows: Sequence[dict[str, Any]], metadata: dict[str, Any], manifest: dict[str, Any],
    *, test_accessed: bool = False,
) -> dict[str, Any]:
    required = ("predictions.jsonl", "inference_manifest.json", "repetition_diagnostics.json", "metrics.json", "summary.json")
    if _stage_completed(stage_dir, required):
        return _read_json(stage_dir / "summary.json")
    _start_stage(stage_dir, name, manifest, test_accessed=test_accessed)
    try:
        _set_seed(torch, 42)
        predictions_path = stage_dir / "predictions.jsonl"
        rows = _generate_rows(
            torch, tokenizer, model, source_rows, predictions_path, metadata,
            progress=lambda count: _atomic_json(stage_dir / "progress.json", {"completed": count, "total": len(source_rows)}),
        )
        if len(rows) != len(source_rows):
            raise RecoveryError(f"prediction count mismatch for {name}")
        repeat_and_length = _prediction_summary(rows)
        metrics = protocol.evaluate_pairs([(str(row["prediction"]), str(row["reference"])) for row in rows])
        _atomic_json(stage_dir / "repetition_diagnostics.json", repeat_and_length["repetition"])
        _atomic_json(stage_dir / "metrics.json", metrics)
        inference_manifest = {
            **manifest, "status": "completed", "prediction_count": len(rows),
            "predictions": str(predictions_path.resolve()), "predictions_sha256": _sha256(predictions_path),
            "completed_at": _utc_now(), "test_accessed": test_accessed,
        }
        _atomic_json(stage_dir / "inference_manifest.json", inference_manifest)
        summary = {
            "name": name, **metadata, **repeat_and_length, "metrics": metrics,
            "status": "completed", "test_accessed": test_accessed,
        }
        _atomic_json(stage_dir / "summary.json", summary)
        _finish_stage(
            stage_dir, name, {"prediction_count": len(rows)}, test_accessed=test_accessed
        )
        return summary
    except Exception as exc:
        _fail_stage(stage_dir, name, exc, test_accessed=test_accessed)
        raise


def _verify_strength_one(current: Path, existing: Path, ids: Sequence[str]) -> dict[str, Any]:
    current_rows = _rows_by_id(_read_jsonl(current)); existing_rows = _rows_by_id(_read_jsonl(existing))
    exact = normalized = 0
    for sample_id in ids:
        left, right = str(current_rows[sample_id]["prediction"]), str(existing_rows[sample_id]["prediction"])
        exact += int(left == right)
        normalized += int(left.replace("\r\n", "\n").strip() == right.replace("\r\n", "\n").strip())
    result = {"count": len(ids), "exact_match_count": exact, "normalized_match_count": normalized,
              "minimum_required_normalized_matches": 23, "passed": normalized >= 23}
    if not result["passed"]:
        raise RecoveryError(f"strength=1.0 failed checkpoint-117 parity: {result}")
    return result


def _strength_eligible(summary: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    repeat = summary["repetition"]
    if repeat["severe_repeat_ge_5"] > 1: reasons.append("severe_repeat_gt_1")
    if repeat["consecutive_repeat_ge_3"] > 1: reasons.append("consecutive_repeat_gt_1")
    if summary["length_stopping"]["hit_max_new_tokens"] > 1: reasons.append("max_length_hits_gt_1")
    if summary["metrics"]["BLEU-4"] < 0.85 * baseline["metrics"]["BLEU-4"]: reasons.append("bleu_below_85pct_of_strength1")
    if summary["metrics"]["ROUGE-L"] < 0.90 * baseline["metrics"]["ROUGE-L"]: reasons.append("rouge_l_below_90pct_of_strength1")
    return not reasons, reasons


def run_adapter_strength(plan: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    root = output_root / "adapter_strength"
    if _stage_completed(root, ("summary.json", "manifest.json", "stage.log")):
        return _read_json(root / "summary.json")
    manifest = _read_json(Path(plan["e05_manifest"]["path"]))
    ids = _sample_ids(Path(plan["judge24_ids"]["path"]))
    validation = _rows_by_id(_read_jsonl(Path(plan["validation_dataset"]["path"])))
    source_rows = [validation[sample_id] for sample_id in ids]
    checkpoint = Path(plan["checkpoint_117"]["path"])
    _start_stage(root, "adapter_strength", {
        "schema_version": SCHEMA, "checkpoint": str(checkpoint), "checkpoint_tree_sha256": _tree_sha256(checkpoint),
        "strengths": list(STRENGTHS), "decoding": FROZEN_DECODING, "sample_count": 24,
        "checkpoint_files_modified": False, "test_accessed": False,
        "adapter_scaling_method": "multiply_each_active_PEFT_LoRA_layer_scaling_entry_in_memory",
    })
    try:
        torch, tokenizer, model = _load_model(manifest, checkpoint)
        originals = _lora_scaling(model)
        summaries: dict[str, Any] = {}
        execution_order = (1.00, 0.25, 0.50, 0.75)
        for strength in execution_order:
            label = f"{strength:.2f}"
            _apply_lora_strength(model, originals, strength)
            summaries[label] = _evaluate_adapter(
                root / label, f"adapter_strength_{label}", torch, tokenizer, model, source_rows,
                {"candidate_type": "adapter_strength", "adapter_strength": strength,
                 "source_checkpoint": str(checkpoint.resolve()), "run_id": manifest["run_id"],
                 "checkpoint": str(checkpoint.resolve())},
                {"schema_version": SCHEMA, "adapter_strength": strength, "source_checkpoint": str(checkpoint.resolve()),
                 "source_checkpoint_tree_sha256": _tree_sha256(checkpoint), "decoding": FROZEN_DECODING,
                 "lora_scaling_entry_count": len(originals), "sample_ids_sha256": plan["judge24_ids"]["sha256"],
                 "split": "validation", "test_accessed": False},
            )
            if strength == 1.0:
                parity = _verify_strength_one(
                    root / label / "predictions.jsonl", Path(plan["epoch05_predictions"]["path"]), ids
                )
                _atomic_json(root / label / "strength_1_parity.json", parity)
        baseline = summaries["1.00"]
        ranking: list[dict[str, Any]] = []
        for label, summary in summaries.items():
            eligible, reasons = _strength_eligible(summary, baseline)
            summary["screening"] = {"eligible": eligible, "excluded_reasons": reasons}
            _atomic_json(root / label / "summary.json", summary)
            ranking.append({
                "strength": float(label), "eligible": eligible, "excluded_reasons": reasons,
                "severe": summary["repetition"]["severe_repeat_ge_5"],
                "consecutive": summary["repetition"]["consecutive_repeat_ge_3"],
                "any_repeat": summary["repetition"]["any_line_repeat_ge_3"],
                "BLEU-4": summary["metrics"]["BLEU-4"], "ROUGE-L": summary["metrics"]["ROUGE-L"],
            })
        ranking.sort(key=lambda row: (not row["eligible"], row["severe"], row["consecutive"], row["any_repeat"], -row["ROUGE-L"], -row["BLEU-4"], row["strength"]))
        result = {
            "status": "completed", "summaries": summaries, "ranking": ranking,
            "promising_strengths": [row["strength"] for row in ranking if row["eligible"]][:2],
            "strength_1_parity": _read_json(root / "1.00" / "strength_1_parity.json"),
            "test_accessed": False,
        }
        _atomic_json(root / "summary.json", result)
        _finish_stage(root, "adapter_strength", {"promising_strengths": result["promising_strengths"]})
        del model; gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()
        return result
    except Exception as exc:
        _fail_stage(root, "adapter_strength", exc)
        raise


def _judge_pack(
    judge_dir: Path, candidate_paths: Sequence[tuple[str, Path]], source_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(candidate_paths) != 3 or candidate_paths[0][0] != "base":
        raise RecoveryError("blind judge pack requires Base plus exactly two candidates")
    candidate_rows = {name: _rows_by_id(_read_jsonl(path)) for name, path in candidate_paths}
    aliases = {"base": "base", candidate_paths[1][0]: "final_v1", candidate_paths[2][0]: "final_v2"}
    anonymous: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for row in source_rows:
        sample_id = str(row["sample_id"])
        names = [item[0] for item in candidate_paths]
        digest = int(hashlib.sha256(f"auto-recovery\0{sample_id}".encode()).hexdigest(), 16)
        shift = digest % 3
        rotated = names[shift:] + names[:shift]
        label_to_name = dict(zip("ABC", rotated))
        anonymous.append({
            "sample_id": sample_id, "prompt": row["messages"][:-1], "reference": row["target_text"],
            "candidates": {label: candidate_rows[name][sample_id]["prediction"] for label, name in label_to_name.items()},
        })
        keys.append({"sample_id": sample_id, "candidate_mapping": {
            label: aliases[name] for label, name in label_to_name.items()
        }})
    anonymous_path = judge_dir / "judge_24_anonymous.jsonl"
    key_path = judge_dir / "judge_24_candidate_key.json"
    alias_path = judge_dir / "candidate_aliases.json"
    if anonymous_path.exists() and _read_jsonl(anonymous_path) != anonymous:
        raise RecoveryError("existing anonymous judge pack differs from deterministic reconstruction")
    if not anonymous_path.exists(): _write_jsonl(anonymous_path, anonymous)
    key_payload = {"samples": keys}
    if key_path.exists() and _read_json(key_path) != key_payload:
        raise RecoveryError("existing judge candidate key differs from deterministic reconstruction")
    if not key_path.exists(): _atomic_json(key_path, key_payload)
    _atomic_json(alias_path, {canonical: actual for actual, canonical in aliases.items()})
    return {"anonymous": anonymous_path, "key": key_path, "aliases": alias_path}


def run_optional_judge(plan: dict[str, Any], adapter: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    stage_dir = output_root / "judge" / "adapter_strength"
    completed_summary = stage_dir / "summary.json"
    if _stage_completed(stage_dir, ("summary.json", "judge_24_anonymous.jsonl", "candidate_aliases.json")):
        return _read_json(completed_summary)
    ids = _sample_ids(Path(plan["judge24_ids"]["path"]))
    validation = _rows_by_id(_read_jsonl(Path(plan["validation_dataset"]["path"])))
    source_rows = [validation[sample_id] for sample_id in ids]
    promising = adapter["promising_strengths"]
    if len(promising) < 2:
        promising = [row["strength"] for row in adapter["ranking"][:2]]
    candidate_paths = [("base", Path(plan["base_predictions"]["path"]))] + [
        (f"strength_{strength:.2f}", output_root / "adapter_strength" / f"{strength:.2f}" / "predictions.jsonl")
        for strength in promising[:2]
    ]
    stage_dir.mkdir(parents=True, exist_ok=True)
    pack = _judge_pack(stage_dir, candidate_paths, source_rows)
    env = {name: os.environ.get(name, "") for name in ("LLM_JUDGE_API_BASE", "LLM_JUDGE_API_KEY", "LLM_JUDGE_MODEL")}
    if not all(env.values()):
        summary = {
            "status": "judge_pending", "reason": "LLM judge environment variables are not all set",
            "candidates": [name for name, _ in candidate_paths], "anonymous_pack": str(pack["anonymous"]),
            "required_environment_variables": list(env), "test_accessed": False,
        }
        _atomic_json(completed_summary, summary)
        _atomic_json(stage_dir / "manifest.json", {"schema_version": SCHEMA, **summary})
        _atomic_json(stage_dir / "status.json", {"schema_version": SCHEMA, "stage": "judge_adapter_strength",
            "status": "judge_pending", "finished_at": _utc_now(), "test_accessed": False})
        _append_log(stage_dir / "stage.log", "Judge API env absent; pack created and judging deferred")
        return summary
    _start_stage(stage_dir, "judge_adapter_strength", {
        "schema_version": SCHEMA, "candidates": [name for name, _ in candidate_paths],
        "anonymous_pack": str(pack["anonymous"]), "blind_scoring_reads_candidate_key": False,
        "test_accessed": False,
    })
    old_expected = judge_api.EXPECTED_SAMPLE_COUNT
    try:
        judge_api.EXPECTED_SAMPLE_COUNT = 24
        raw_path = stage_dir / "judge_24_raw_results.jsonl"
        judge_api.run_judging(
            pack["anonymous"], raw_path, api_base=env["LLM_JUDGE_API_BASE"], api_key=env["LLM_JUDGE_API_KEY"],
            model=env["LLM_JUDGE_MODEL"],
        )
        aggregate_path = stage_dir / "judge_24_aggregate.json"
        aggregate = judge_api.aggregate_results(pack["anonymous"], raw_path, pack["key"], aggregate_path)
        canonical_to_actual = _read_json(pack["aliases"])
        translated = {canonical_to_actual[name]: value for name, value in aggregate["models"].items()}
        summary = {"status": "completed", "judge_model": aggregate["judge_model"], "models": translated,
                   "candidate_count": 3, "sample_count": 24, "test_accessed": False}
        _atomic_json(completed_summary, summary)
        _finish_stage(stage_dir, "judge_adapter_strength", {"sample_count": 24})
        return summary
    except Exception as exc:
        _fail_stage(stage_dir, "judge_adapter_strength", exc)
        raise
    finally:
        judge_api.EXPECTED_SAMPLE_COUNT = old_expected


def _scaling_is_sufficient(adapter: dict[str, Any], judge: dict[str, Any]) -> tuple[bool, str]:
    eligible = [row for row in adapter["ranking"] if row["eligible"]]
    if not eligible:
        return False, "no adapter strength passed repetition/length/metric screening"
    if judge.get("status") != "completed":
        best = eligible[0]
        return (best["severe"] == 0 and best["consecutive"] == 0,
                "judge unavailable; conservative metric-only stability gate")
    base = judge["models"]["base"]
    base_style = base["cm_style"]["macro_mean"]; base_accr = base["news_accr"]["macro_mean"]
    for row in eligible:
        name = f"strength_{row['strength']:.2f}"
        scored = judge["models"].get(name)
        if scored and scored["cm_style"]["macro_mean"] >= base_style - 0.35 \
                and scored["news_accr"]["macro_mean"] >= base_accr - 0.40:
            return True, f"{name} passed Base-relative style/accuracy and stability gates"
    return False, "no strength passed Base-relative Judge style/accuracy gates"


def _parse_flat_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise RecoveryError(f"unsupported YAML line at {path}:{line_number}")
        key, raw = line.split(":", 1); raw = raw.strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            if raw == "true": value = True
            elif raw == "false": value = False
            elif raw == "null": value = None
            else:
                try: value = float(raw) if any(char in raw for char in ".eE") else int(raw)
                except ValueError: value = raw
        values[key.strip()] = value
    return values


def _prepare_training_candidate(plan: dict[str, Any], name: str, output_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    candidate = TRAINING_CANDIDATES[name]
    stage_dir = output_root / "training" / name
    source_yaml = _parse_flat_yaml(Path(plan["e05_yaml"]["path"]))
    fixed = {
        "lora_target": "all", "lora_dropout": 0.05, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8, "template": "qwen3_nothink", "train_on_prompt": False,
        "packing": False, "bf16": True, "fp16": False,
    }
    mismatches = {key: (source_yaml.get(key), expected) for key, expected in fixed.items() if source_yaml.get(key) != expected}
    if mismatches:
        raise RecoveryError(f"source E05 YAML fixed fields mismatch: {mismatches}")
    runtime_data = stage_dir / "runtime_data"
    source_dataset_info = E05_RUN_DIR / "runtime_data" / "dataset_info.json"
    dataset_info = _read_json(source_dataset_info)
    serialized = json.dumps(dataset_info, ensure_ascii=False).lower()
    if "test.jsonl" in serialized or '"test"' in serialized:
        raise RecoveryError("source E05 runtime dataset info contains Test")
    stage_dir.mkdir(parents=True, exist_ok=True); runtime_data.mkdir(parents=True, exist_ok=True)
    target_dataset_info = runtime_data / "dataset_info.json"
    if target_dataset_info.exists() and _read_json(target_dataset_info) != dataset_info:
        raise RecoveryError(f"candidate runtime dataset_info changed: {target_dataset_info}")
    if not target_dataset_info.exists(): _atomic_json(target_dataset_info, dataset_info)
    config = dict(source_yaml)
    config.update({
        "dataset_dir": str(runtime_data.resolve()), "output_dir": str((stage_dir / "checkpoints").resolve()),
        "overwrite_output_dir": False, "learning_rate": candidate["learning_rate"],
        "lora_rank": candidate["rank"], "lora_alpha": candidate["alpha"],
        "lora_dropout": 0.05, "num_train_epochs": 0.5,
    })
    yaml_path = stage_dir / "llamafactory.yaml"
    if yaml_path.exists() and _parse_flat_yaml(yaml_path) != config:
        raise RecoveryError(f"candidate YAML already exists with different content: {yaml_path}")
    if not yaml_path.exists(): protocol.write_flat_yaml(yaml_path, config)
    candidate_manifest = {
        "schema_version": SCHEMA, "candidate": name, "source_run": str(E05_MANIFEST.resolve()),
        "source_yaml": plan["e05_yaml"], "config": candidate, "dropout": 0.05,
        "effective_batch": 8, "lora_target": "all", "formal_experiment": False,
        "output_dir": str((stage_dir / "checkpoints").resolve()), "yaml": str(yaml_path.resolve()),
        "split_usage": ["train", "validation"], "test_accessed": False,
    }
    return stage_dir, yaml_path, candidate_manifest


def _checkpoint_from_training(stage_dir: Path) -> Path:
    checkpoints = sorted(
        (path for path in (stage_dir / "checkpoints").glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    valid = [path for path in checkpoints if (path / "adapter_config.json").is_file()
             and any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))]
    if not valid:
        root = stage_dir / "checkpoints"
        if (root / "adapter_config.json").is_file(): return root
        raise RecoveryError(f"training produced no usable LoRA checkpoint: {stage_dir}")
    return valid[-1]


def _checkpoint_epoch(checkpoint: Path) -> float | None:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = _read_json(state_path)
    value = state.get("epoch")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _run_training_subprocess(stage_dir: Path, yaml_path: Path, manifest: dict[str, Any]) -> Path:
    status = _stage_status(stage_dir)
    if status and status.get("status") == "completed":
        checkpoint = _checkpoint_from_training(stage_dir)
        if (_checkpoint_epoch(checkpoint) or 0.0) < 0.49:
            raise RecoveryError(f"completed candidate checkpoint has epoch below 0.5: {checkpoint}")
        return checkpoint
    command_yaml = yaml_path
    process_attempt = int(status.get("training_process_attempt", 0)) if status else 0
    if status and status.get("training_process_started") is True:
        try:
            checkpoint = _checkpoint_from_training(stage_dir)
        except RecoveryError as exc:
            raise RecoveryError(f"previous training process did not complete; automatic retraining is forbidden: {stage_dir}") from exc
        if (_checkpoint_epoch(checkpoint) or 0.0) >= 0.49:
            _finish_stage(stage_dir, manifest["candidate"], {"recovered_completed_checkpoint": str(checkpoint)})
            return checkpoint
        if process_attempt >= 2:
            raise RecoveryError(f"training resume attempt limit reached for {manifest['candidate']}")
        resume_config = _parse_flat_yaml(yaml_path)
        resume_config["resume_from_checkpoint"] = str(checkpoint.resolve())
        command_yaml = stage_dir / f"llamafactory.resume{process_attempt + 1}.yaml"
        if not command_yaml.exists():
            protocol.write_flat_yaml(command_yaml, resume_config)
        _append_log(stage_dir / "stage.log", f"RESUME from {checkpoint}")
    else:
        _start_stage(stage_dir, manifest["candidate"], manifest)
    current = _stage_status(stage_dir) or {}
    process_attempt += 1
    _atomic_json(stage_dir / "status.json", {**current, "training_process_started": True,
                                              "training_process_attempt": process_attempt})
    command = ["llamafactory-cli", "train", str(command_yaml)]
    log_path = stage_dir / ("training.log" if process_attempt == 1 else f"training.resume{process_attempt}.log")
    _append_log(stage_dir / "stage.log", f"EXEC {' '.join(command)}")
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8", errors="replace") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RecoveryError(f"{manifest['candidate']} training failed with return code {result.returncode}")
    checkpoint = _checkpoint_from_training(stage_dir)
    epoch = _checkpoint_epoch(checkpoint)
    if epoch is None or epoch < 0.49:
        raise RecoveryError(f"{manifest['candidate']} ended without a completed 0.5-epoch checkpoint")
    _finish_stage(stage_dir, manifest["candidate"], {
        "returncode": result.returncode, "duration_seconds": round(time.monotonic() - started, 3),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": epoch,
        "checkpoint_tree_sha256": _tree_sha256(checkpoint),
    })
    return checkpoint


def run_training_if_needed(
    plan: dict[str, Any], adapter: dict[str, Any], judge: dict[str, Any], output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    root = output_root / "training"
    if _stage_completed(root, ("summary.json", "manifest.json", "stage.log")):
        return _read_json(root / "summary.json")
    sufficient, reason = _scaling_is_sufficient(adapter, judge)
    if sufficient:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "manifest.json", {"schema_version": SCHEMA, "decision": "skip_training", "reason": reason})
        _atomic_json(root / "status.json", {"schema_version": SCHEMA, "stage": "training", "status": "completed",
            "finished_at": _utc_now(), "test_accessed": False})
        _append_log(root / "stage.log", f"SKIP training: {reason}")
        summary = {"status": "completed", "training_performed": False, "reason": reason, "candidates": {}}
        _atomic_json(root / "summary.json", summary)
        return summary
    _start_stage(root, "training", {"schema_version": SCHEMA, "decision": "run_two_candidates", "reason": reason,
        "maximum_new_training_candidates": 2, "candidates": TRAINING_CANDIDATES, "test_accessed": False})
    try:
        manifest = _read_json(Path(plan["e05_manifest"]["path"]))
        ids = _sample_ids(Path(plan["judge24_ids"]["path"]))
        validation = _rows_by_id(_read_jsonl(Path(plan["validation_dataset"]["path"])))
        source_rows = [validation[sample_id] for sample_id in ids]
        summaries: dict[str, Any] = {}
        for name in TRAINING_CANDIDATES:
            stage_dir, yaml_path, candidate_manifest = _prepare_training_candidate(plan, name, output_root)
            try:
                checkpoint = _run_training_subprocess(stage_dir, yaml_path, candidate_manifest)
                torch, tokenizer, model = _load_model(manifest, checkpoint)
                evaluation = _evaluate_adapter(
                    stage_dir / "validation24", f"{name}_validation24", torch, tokenizer, model, source_rows,
                    {"candidate_type": "diagnostic_training", "candidate": name, "checkpoint": str(checkpoint.resolve()),
                     **TRAINING_CANDIDATES[name]},
                    {"schema_version": SCHEMA, "split": "validation", "sample_count": 24,
                     "checkpoint": str(checkpoint.resolve()), "checkpoint_tree_sha256": _tree_sha256(checkpoint),
                     "training_config": TRAINING_CANDIDATES[name], "decoding": FROZEN_DECODING, "test_accessed": False},
                )
                summaries[name] = evaluation
                del model; gc.collect()
                if hasattr(torch, "cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as exc:
                _fail_stage(stage_dir, name, exc)
                raise
        result = {"status": "completed", "training_performed": True, "training_count": len(summaries),
                  "reason": reason, "candidates": summaries, "test_accessed": False}
        _atomic_json(root / "summary.json", result)
        _finish_stage(root, "training", {"training_count": len(summaries)})
        return result
    except Exception as exc:
        _fail_stage(root, "training", exc)
        raise


def run_optional_training_judge(
    plan: dict[str, Any], training: dict[str, Any], output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    stage_dir = output_root / "judge" / "diagnostic_training"
    if not training.get("training_performed"):
        if _stage_completed(stage_dir, ("summary.json", "manifest.json", "stage.log")):
            return _read_json(stage_dir / "summary.json")
        stage_dir.mkdir(parents=True, exist_ok=True)
        summary = {"status": "not_run", "reason": "diagnostic training was not needed", "test_accessed": False}
        _atomic_json(stage_dir / "summary.json", summary)
        _atomic_json(stage_dir / "manifest.json", {"schema_version": SCHEMA, **summary})
        _atomic_json(stage_dir / "status.json", {"schema_version": SCHEMA, "stage": "judge_diagnostic_training",
            "status": "completed", "finished_at": _utc_now(), "test_accessed": False})
        _append_log(stage_dir / "stage.log", "SKIP judge: diagnostic training was not run")
        return summary
    if _stage_completed(stage_dir, ("summary.json", "judge_24_anonymous.jsonl", "candidate_aliases.json")):
        return _read_json(stage_dir / "summary.json")
    ranked = sorted(
        training["candidates"].items(),
        key=lambda item: (
            item[1]["repetition"]["severe_repeat_ge_5"],
            item[1]["repetition"]["consecutive_repeat_ge_3"],
            item[1]["repetition"]["any_line_repeat_ge_3"],
            -item[1]["metrics"]["ROUGE-L"],
            -item[1]["metrics"]["BLEU-4"],
        ),
    )[:2]
    if len(ranked) != 2:
        raise RecoveryError("training Judge requires the two bounded diagnostic candidates")
    ids = _sample_ids(Path(plan["judge24_ids"]["path"]))
    validation = _rows_by_id(_read_jsonl(Path(plan["validation_dataset"]["path"])))
    source_rows = [validation[sample_id] for sample_id in ids]
    candidate_paths = [("base", Path(plan["base_predictions"]["path"]))] + [
        (name, output_root / "training" / name / "validation24" / "predictions.jsonl") for name, _ in ranked
    ]
    stage_dir.mkdir(parents=True, exist_ok=True)
    pack = _judge_pack(stage_dir, candidate_paths, source_rows)
    env = {name: os.environ.get(name, "") for name in ("LLM_JUDGE_API_BASE", "LLM_JUDGE_API_KEY", "LLM_JUDGE_MODEL")}
    if not all(env.values()):
        summary = {
            "status": "judge_pending", "reason": "LLM judge environment variables are not all set",
            "candidates": [name for name, _ in candidate_paths], "anonymous_pack": str(pack["anonymous"]),
            "required_environment_variables": list(env), "test_accessed": False,
        }
        _atomic_json(stage_dir / "summary.json", summary)
        _atomic_json(stage_dir / "manifest.json", {"schema_version": SCHEMA, **summary})
        _atomic_json(stage_dir / "status.json", {"schema_version": SCHEMA, "stage": "judge_diagnostic_training",
            "status": "judge_pending", "finished_at": _utc_now(), "test_accessed": False})
        _append_log(stage_dir / "stage.log", "Judge API env absent; trained-candidate pack created")
        return summary
    _start_stage(stage_dir, "judge_diagnostic_training", {
        "schema_version": SCHEMA, "candidates": [name for name, _ in candidate_paths],
        "anonymous_pack": str(pack["anonymous"]), "blind_scoring_reads_candidate_key": False,
        "test_accessed": False,
    })
    old_expected = judge_api.EXPECTED_SAMPLE_COUNT
    try:
        judge_api.EXPECTED_SAMPLE_COUNT = 24
        raw_path = stage_dir / "judge_24_raw_results.jsonl"
        judge_api.run_judging(
            pack["anonymous"], raw_path, api_base=env["LLM_JUDGE_API_BASE"], api_key=env["LLM_JUDGE_API_KEY"],
            model=env["LLM_JUDGE_MODEL"],
        )
        aggregate = judge_api.aggregate_results(
            pack["anonymous"], raw_path, pack["key"], stage_dir / "judge_24_aggregate.json"
        )
        canonical_to_actual = _read_json(pack["aliases"])
        translated = {canonical_to_actual[name]: value for name, value in aggregate["models"].items()}
        summary = {"status": "completed", "judge_model": aggregate["judge_model"], "models": translated,
                   "candidate_count": 3, "sample_count": 24, "test_accessed": False}
        _atomic_json(stage_dir / "summary.json", summary)
        _finish_stage(stage_dir, "judge_diagnostic_training", {"sample_count": 24})
        return summary
    except Exception as exc:
        _fail_stage(stage_dir, "judge_diagnostic_training", exc)
        raise
    finally:
        judge_api.EXPECTED_SAMPLE_COUNT = old_expected


def _candidate_rows(adapter: dict[str, Any], training: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, summary in adapter.get("summaries", {}).items():
        rows.append({"name": f"strength_{label}", "source": summary["source_checkpoint"],
            "adapter_strength": float(label), "learning_rate": None, "rank": 32, "alpha": 32, "epoch": 0.5,
            "repetition": summary["repetition"], "metrics": summary["metrics"],
            "length_stopping": summary["length_stopping"], "judge": None,
            "predictions": str((output_root / "adapter_strength" / label / "predictions.jsonl").resolve())})
    for name, summary in training.get("candidates", {}).items():
        rows.append({"name": name, "source": summary["checkpoint"], "adapter_strength": 1.0,
            "learning_rate": summary["learning_rate"], "rank": summary["rank"], "alpha": summary["alpha"],
            "epoch": summary["epoch"], "repetition": summary["repetition"], "metrics": summary["metrics"],
            "length_stopping": summary["length_stopping"], "judge": None,
            "predictions": str((output_root / "training" / name / "validation24" / "predictions.jsonl").resolve())})
    return rows


def _attach_judge(rows: list[dict[str, Any]], judge: dict[str, Any]) -> None:
    if judge.get("status") != "completed": return
    for row in rows:
        if row["name"] in judge["models"]:
            row["judge"] = judge["models"][row["name"]]


def _choose_candidate(rows: Sequence[dict[str, Any]], base_judge: dict[str, Any] | None) -> tuple[dict[str, Any], bool | None, str]:
    candidates = [row for row in rows if row["name"] != "strength_1.00"] or list(rows)
    stable = [row for row in candidates if row["repetition"]["severe_repeat_ge_5"] <= 1
              and row["repetition"]["consecutive_repeat_ge_3"] <= 1
              and row["length_stopping"]["hit_max_new_tokens"] <= 1]
    pool = stable or candidates
    with_judge = [row for row in pool if row.get("judge")]
    if with_judge:
        with_judge.sort(key=lambda row: (-row["judge"]["cm_style"]["macro_mean"],
                                         -row["judge"]["news_accr"]["macro_mean"],
                                         row["repetition"]["severe_repeat_ge_5"],
                                         -row["metrics"]["ROUGE-L"]))
        chosen = with_judge[0]
        exceeds = None
        if base_judge:
            exceeds = (chosen["judge"]["cm_style"]["macro_mean"] > base_judge["cm_style"]["macro_mean"]
                       and chosen["judge"]["news_accr"]["macro_mean"] >= base_judge["news_accr"]["macro_mean"])
        return chosen, exceeds, "Judge available: CM Style first, Accuracy floor second, then repetition and metrics"
    pool.sort(key=lambda row: (row["repetition"]["severe_repeat_ge_5"],
                               row["repetition"]["consecutive_repeat_ge_3"],
                               row["repetition"]["any_line_repeat_ge_3"],
                               -row["metrics"]["ROUGE-L"], -row["metrics"]["BLEU-4"]))
    return pool[0], None, "Judge unavailable: best stable Validation-only candidate; Base superiority is unknown"


def build_report(output_root: Path = OUTPUT_ROOT, failure: BaseException | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    diagnosis = _read_json(output_root / "diagnosis" / "summary.json") if (output_root / "diagnosis" / "summary.json").is_file() else {}
    adapter = _read_json(output_root / "adapter_strength" / "summary.json") if (output_root / "adapter_strength" / "summary.json").is_file() else {}
    judge = _read_json(output_root / "judge" / "adapter_strength" / "summary.json") if (output_root / "judge" / "adapter_strength" / "summary.json").is_file() else {}
    training_judge = _read_json(output_root / "judge" / "diagnostic_training" / "summary.json") if (output_root / "judge" / "diagnostic_training" / "summary.json").is_file() else {}
    training = _read_json(output_root / "training" / "summary.json") if (output_root / "training" / "summary.json").is_file() else {}
    rows = _candidate_rows(adapter, training, output_root) if adapter else []
    _attach_judge(rows, judge)
    _attach_judge(rows, training_judge)
    base_judge = (training_judge.get("models", {}).get("base") if training_judge.get("status") == "completed"
                  else judge.get("models", {}).get("base") if judge else None)
    chosen = exceeds = decision_basis = None
    if rows:
        chosen, exceeds, decision_basis = _choose_candidate(rows, base_judge)
    data_evidence = diagnosis.get("dataset_issue_evidence", {})
    template_anomalies = diagnosis.get("template_eos_explicit_anomaly_count")
    root_cause = {
        "training_overfit": {"supported": True, "evidence": "known checkpoint-117 vs checkpoint-234 repetition and Judge degradation"},
        "adapter_too_strong": {"supported": bool(adapter and any(
            item["repetition"]["severe_repeat_ge_5"] < adapter["summaries"]["1.00"]["repetition"]["severe_repeat_ge_5"]
            for key, item in adapter["summaries"].items() if key != "1.00")) if adapter else None,
            "evidence": "lower-strength fixed-24 comparison" if adapter else "not_run"},
        "dataset_target_issue": {"supported": bool(data_evidence and (
            data_evidence.get("train_duplicate_groups", 0) or data_evidence.get("train_anomaly_samples", 0)
            or data_evidence.get("train_overlong_targets", 0))), "evidence": data_evidence or "not_run"},
        "eos_template_issue": {"supported": template_anomalies > 0 if isinstance(template_anomalies, int) else None,
                               "explicit_anomaly_count": template_anomalies},
    }
    if chosen:
        if chosen["name"].startswith("strength_"):
            full_validation_command = (
                f"python scripts/diagnostics/auto_lora_recovery.py validate --candidate {chosen['name']}"
            )
            test_preparation = (
                "prepare an authorized production decode wrapper that applies the recorded runtime adapter_strength; "
                "this runner intentionally cannot access Test"
            )
        else:
            candidate_manifest = output_root / "training" / chosen["name"] / "candidate_run_manifest.json"
            source_manifest = _read_json(E05_MANIFEST)
            _atomic_json(candidate_manifest, {**source_manifest, "run_id": f"auto_recovery_{chosen['name']}",
                "experiment_id": "E05", "best_checkpoint": chosen["source"],
                "status": "completed", "formal_experiment": False, "test_accessed": False})
            full_validation_command = (
                f"python scripts/lora_v2_production_decode.py --run-manifest {candidate_manifest} "
                f"--output-dir {output_root / 'full_validation' / chosen['name']} --split validation "
                "--repetition-penalty 1.0 --no-repeat-ngram-size 0 --max-new-tokens 2048 --execute"
            )
            test_preparation = None
        test_preparation = (
            f"python scripts/diagnostics/auto_lora_recovery.py prepare-test --candidate {chosen['name']} "
            "--authorization configs/training/lora_v2/final_test_authorization.json"
        )
    else:
        full_validation_command = None; test_preparation = None
    status = "failed" if failure else ("completed" if chosen else "incomplete")
    report = {
        "schema_version": SCHEMA, "status": status, "generated_at": _utc_now(),
        "failure": {"type": type(failure).__name__, "message": str(failure)} if failure else None,
        "root_cause": root_cause, "candidates": rows,
        "recommended_candidate": chosen, "decision_basis": decision_basis,
        "actually_exceeds_base": exceeds,
        "base_comparison_statement": (
            "The repaired LoRA exceeds Base on both judged macro criteria." if exceeds is True else
            "The repaired LoRA does not exceed Base on both judged macro criteria." if exceeds is False else
            "Base superiority cannot be determined without comparable Judge results; no superiority is claimed."
        ),
        "judge_status": {"adapter_strength": judge.get("status", "not_run"),
                         "diagnostic_training": training_judge.get("status", "not_run")},
        "next_steps": {"full_validation": full_validation_command, "prepare_final_production_test": test_preparation,
                       "runner_test_accessed": False},
        "test_accessed": False,
    }
    _atomic_json(output_root / "final_report.json", report)
    lines = [
        "# LoRA V2.1 Auto Recovery Final Report", "", f"Status: `{status}`", "",
        "## Root-cause assessment", "",
    ]
    for name, value in root_cause.items():
        lines.append(f"- {name}: supported={value.get('supported')} — {value.get('evidence', value.get('explicit_anomaly_count'))}")
    lines.extend(["", "## Candidates", "",
        "| Candidate | Source | LR | Rank/Alpha/Epoch | Strength | repeat>=3 / consecutive>=3 / severe>=5 | BLEU-4 | ROUGE-L | Judge ACCR | Judge Style |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|"])
    for row in rows:
        judged = row.get("judge") or {}
        lines.append(
            f"| {row['name']} | {row['source']} | {row['learning_rate']} | {row['rank']}/{row['alpha']}/{row['epoch']} | "
            f"{row['adapter_strength']} | {row['repetition']['any_line_repeat_ge_3']} / "
            f"{row['repetition']['consecutive_repeat_ge_3']} / {row['repetition']['severe_repeat_ge_5']} | "
            f"{row['metrics']['BLEU-4']:.6f} | {row['metrics']['ROUGE-L']:.6f} | "
            f"{judged.get('news_accr', {}).get('macro_mean', 'pending')} | {judged.get('cm_style', {}).get('macro_mean', 'pending')} |"
        )
    lines.extend(["", "## Recommendation", "", f"- Candidate: `{chosen['name'] if chosen else 'none'}`",
                  f"- Decision: {decision_basis}", f"- Base comparison: {report['base_comparison_statement']}", "",
                  "## Next steps", "", f"- Full Validation: `{full_validation_command}`",
                  f"- Final production Test preparation: `{test_preparation}`", "",
                  "Runner Test access: `false`", ""])
    _atomic_text(output_root / "final_report.md", "\n".join(lines))
    report_stage = output_root / "report"
    report_stage.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_stage / "manifest.json", {"schema_version": SCHEMA, "source": "completed stage artifacts",
        "final_report_json": str((output_root / "final_report.json").resolve()),
        "final_report_md": str((output_root / "final_report.md").resolve()), "test_accessed": False})
    _atomic_json(report_stage / "status.json", {"schema_version": SCHEMA, "stage": "report", "status": status,
        "finished_at": _utc_now(), "test_accessed": False})
    _append_log(report_stage / "stage.log", f"REPORT status={status}")
    return report


def run_all(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "runner_status.json", {"schema_version": SCHEMA, "status": "running",
        "started_at": _utc_now(), "test_accessed": False})
    _append_log(output_root / "runner.log", "START autonomous recovery")
    try:
        plan = create_plan(output_root)
        run_diagnosis(plan, output_root)
        adapter = run_adapter_strength(plan, output_root)
        judge = run_optional_judge(plan, adapter, output_root)
        training = run_training_if_needed(plan, adapter, judge, output_root)
        run_optional_training_judge(plan, training, output_root)
        report = build_report(output_root)
        _atomic_json(output_root / "runner_status.json", {"schema_version": SCHEMA, "status": "completed",
            "finished_at": _utc_now(), "test_accessed": False})
        _append_log(output_root / "runner.log", "END autonomous recovery status=completed")
        return report
    except Exception as exc:
        try:
            build_report(output_root, failure=exc)
        finally:
            _atomic_json(output_root / "runner_status.json", {"schema_version": SCHEMA, "status": "failed",
                "failed_at": _utc_now(), "error_type": type(exc).__name__, "error": str(exc), "test_accessed": False})
            _append_log(output_root / "runner.log", f"FAIL autonomous recovery: {type(exc).__name__}: {exc}")
        raise


def _validate_selected(candidate: str, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if not re.fullmatch(r"strength_(?:0\.25|0\.50|0\.75|1\.00)", candidate):
        raise RecoveryError("validate currently accepts strength_0.25/0.50/0.75/1.00 only")
    plan = _load_plan(output_root)
    manifest = _read_json(Path(plan["e05_manifest"]["path"]))
    validation_rows = _read_jsonl(Path(plan["validation_dataset"]["path"]))
    checkpoint = Path(plan["checkpoint_117"]["path"])
    strength = float(candidate.split("_", 1)[1])
    stage_dir = output_root / "full_validation" / candidate
    torch, tokenizer, model = _load_model(manifest, checkpoint)
    originals = _lora_scaling(model); _apply_lora_strength(model, originals, strength)
    result = _evaluate_adapter(
        stage_dir, f"full_validation_{candidate}", torch, tokenizer, model, validation_rows,
        {"candidate_type": "adapter_strength", "adapter_strength": strength,
         "source_checkpoint": str(checkpoint.resolve()), "checkpoint": str(checkpoint.resolve())},
        {"schema_version": SCHEMA, "split": "validation", "sample_count": len(validation_rows),
         "adapter_strength": strength, "checkpoint": str(checkpoint.resolve()), "decoding": FROZEN_DECODING,
         "test_accessed": False},
    )
    return result


def _prepare_test(candidate: str, authorization: Path, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    try:
        protocol._read_authorization(authorization)
    except protocol.GuardError as exc:
        raise RecoveryError(str(exc)) from exc
    adapter_summary_path = output_root / "adapter_strength" / "summary.json"
    training_summary_path = output_root / "training" / "summary.json"
    if candidate.startswith("strength_"):
        if not re.fullmatch(r"strength_(?:0\.25|0\.50|0\.75|1\.00)", candidate):
            raise RecoveryError(f"unsupported strength candidate: {candidate}")
        plan = _load_plan(output_root)
        source = plan["checkpoint_117"]["path"]
        strength: float | None = float(candidate.split("_", 1)[1])
        if not adapter_summary_path.is_file():
            raise RecoveryError("adapter-strength results are unavailable")
    elif candidate in TRAINING_CANDIDATES:
        training = _read_json(training_summary_path)
        if candidate not in training.get("candidates", {}):
            raise RecoveryError(f"training candidate is unavailable: {candidate}")
        source = training["candidates"][candidate]["checkpoint"]
        strength = 1.0
    else:
        raise RecoveryError(f"unknown recovery candidate: {candidate}")
    stage_dir = output_root / "final_test_preparation" / candidate
    result = {
        "schema_version": SCHEMA, "status": "prepared_not_executed", "candidate": candidate,
        "source_checkpoint": source, "adapter_strength": strength, "decoding": FROZEN_DECODING,
        "authorization": str(authorization.resolve()), "authorization_sha256": _sha256(authorization),
        "runner_executes_test": False, "test_content_read": False, "test_accessed": False,
        "note": "Preparation records the authorized candidate only; this runner never loads Test content or executes Test inference.",
    }
    if stage_dir.exists():
        existing = _read_json(stage_dir / "manifest.json")
        if existing != result:
            raise RecoveryError(f"existing Test preparation differs: {stage_dir}")
        return result
    stage_dir.mkdir(parents=True)
    _atomic_json(stage_dir / "manifest.json", result)
    _atomic_json(stage_dir / "status.json", {"schema_version": SCHEMA, "stage": "prepare_test",
        "status": "completed", "finished_at": _utc_now(), "test_accessed": False})
    _append_log(stage_dir / "stage.log", "Prepared metadata only; Test content was not read")
    return result


def _validate_test_prediction_prefix(
    predictions_path: Path, source_rows: Sequence[dict[str, Any]], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    if not predictions_path.is_file():
        return []
    rows = _read_jsonl(predictions_path)
    if len(rows) > len(source_rows):
        raise RecoveryError("partial Test predictions exceed the authorized Test sample count")
    for index, row in enumerate(rows):
        source = source_rows[index]
        expected = {
            "sample_id": source.get("sample_id"),
            "reference": source.get("target_text"),
            "model_name": metadata["model_name"],
            "checkpoint": metadata["checkpoint"],
            "adapter_strength": metadata["adapter_strength"],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise RecoveryError(
                f"partial Test prediction is not the expected immutable source-order prefix at index {index}"
            )
        if not isinstance(row.get("prediction"), str):
            raise RecoveryError(f"partial Test prediction is not a string at index {index}")
    return rows


def _completed_test_result(
    stage_dir: Path, expected_manifest: dict[str, Any], source_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    required = (
        "predictions.jsonl", "inference_manifest.json", "repetition_diagnostics.json",
        "metrics.json", "summary.json", "stage.log",
    )
    if not _stage_completed(stage_dir, required):
        return None
    status = _read_json(stage_dir / "status.json")
    if status.get("test_accessed") is not True:
        raise RecoveryError("completed final Test status must record test_accessed=true")
    if _read_json(stage_dir / "manifest.json") != expected_manifest:
        raise RecoveryError("completed final Test manifest differs from the authorized frozen plan")
    rows = _validate_test_prediction_prefix(stage_dir / "predictions.jsonl", source_rows, metadata)
    if len(rows) != FINAL_TEST_COUNT:
        raise RecoveryError(f"completed final Test predictions count {len(rows)} != {FINAL_TEST_COUNT}")
    inference = _read_json(stage_dir / "inference_manifest.json")
    expected_inference = {
        "status": "completed", "split": "test", "prediction_count": FINAL_TEST_COUNT,
        "test_accessed": True, "predictions_sha256": _sha256(stage_dir / "predictions.jsonl"),
        "authorization_sha256": expected_manifest["authorization_sha256"],
        "preparation_manifest_sha256": expected_manifest["preparation_manifest_sha256"],
        "source_checkpoint_tree_sha256": expected_manifest["source_checkpoint_tree_sha256"],
        "decoding": FROZEN_DECODING,
    }
    if any(inference.get(field) != value for field, value in expected_inference.items()):
        raise RecoveryError("completed final Test inference manifest is inconsistent")
    diagnostics = _read_json(stage_dir / "repetition_diagnostics.json")
    if diagnostics.get("sample_count") != FINAL_TEST_COUNT:
        raise RecoveryError("completed final Test repetition diagnostics count is inconsistent")
    metrics = _read_json(stage_dir / "metrics.json")
    if metrics.get("manifest", {}).get("count") != FINAL_TEST_COUNT:
        raise RecoveryError("completed final Test metrics count is inconsistent")
    summary = _read_json(stage_dir / "summary.json")
    if summary.get("status") != "completed" or summary.get("test_accessed") is not True:
        raise RecoveryError("completed final Test summary is inconsistent")
    return summary


def _run_test(candidate: str, authorization: Path, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if candidate != FINAL_TEST_CANDIDATE:
        raise RecoveryError(f"run-test accepts only {FINAL_TEST_CANDIDATE}")
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise RecoveryError(f"run-test output is fixed to {OUTPUT_ROOT}")

    preparation_path = (
        output_root / "final_test_preparation" / FINAL_TEST_CANDIDATE / "manifest.json"
    )
    preparation = _read_json(preparation_path)
    try:
        protocol._read_authorization(authorization)
    except protocol.GuardError as exc:
        raise RecoveryError(str(exc)) from exc
    authorization_path = authorization.resolve()
    authorization_sha256 = _sha256(authorization_path)
    if preparation.get("status") != "prepared_not_executed":
        raise RecoveryError("final Test preparation status must be prepared_not_executed")
    if preparation.get("candidate") != FINAL_TEST_CANDIDATE:
        raise RecoveryError("final Test preparation candidate differs from strength_0.50")
    if preparation.get("adapter_strength") != FINAL_TEST_STRENGTH:
        raise RecoveryError("final Test preparation adapter_strength must be 0.5")
    if preparation.get("decoding") != FROZEN_DECODING:
        raise RecoveryError("final Test preparation decoding differs from frozen decoding")
    if preparation.get("authorization") != str(authorization_path):
        raise RecoveryError("authorization path differs from final Test preparation")
    if preparation.get("authorization_sha256") != authorization_sha256:
        raise RecoveryError("authorization sha256 differs from final Test preparation")
    if preparation.get("test_content_read") is not False or preparation.get("test_accessed") is not False:
        raise RecoveryError("final Test preparation must remain metadata-only")

    plan = _load_plan(output_root)
    checkpoint = Path(plan["checkpoint_117"]["path"]).resolve()
    prepared_checkpoint = Path(str(preparation.get("source_checkpoint", ""))).resolve()
    if prepared_checkpoint != checkpoint or checkpoint != CHECKPOINT_117.resolve():
        raise RecoveryError("final Test preparation source checkpoint differs from frozen checkpoint-117")
    checkpoint_tree_sha256 = _tree_sha256(checkpoint)
    if checkpoint_tree_sha256 != plan["checkpoint_117"]["tree_sha256"]:
        raise RecoveryError("frozen checkpoint-117 changed after planning")
    manifest_path = Path(plan["e05_manifest"]["path"])
    manifest = _read_json(manifest_path)
    if (
        manifest.get("experiment_id") != "E05"
        or manifest.get("dataset_version") != "sft_v2.0.0"
        or manifest.get("model_name") != "Qwen/Qwen3-4B-Instruct-2507"
        or manifest.get("chat_template") != "qwen3_nothink"
    ):
        raise RecoveryError("source manifest differs from the frozen E05 sft_v2.0.0 lineage")

    # This is the first point at which Test may be resolved, validated, or read.
    try:
        test_path = protocol.split_path(
            "sft_v2.0.0", "test", purpose="final_evaluation",
            authorization_path=authorization_path,
        )
        protocol.validate_dataset("sft_v2.0.0")
    except protocol.GuardError as exc:
        raise RecoveryError(str(exc)) from exc
    source_rows = _read_jsonl(test_path)
    if len(source_rows) != FINAL_TEST_COUNT:
        raise RecoveryError(f"authorized sft_v2 Test count {len(source_rows)} != {FINAL_TEST_COUNT}")
    _rows_by_id(source_rows)

    stage_dir = output_root / "final_test" / FINAL_TEST_CANDIDATE
    preparation_sha256 = _sha256(preparation_path)
    metadata = {
        "candidate": FINAL_TEST_CANDIDATE,
        "candidate_type": "adapter_strength",
        "adapter_strength": FINAL_TEST_STRENGTH,
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint),
        "model_name": manifest["model_name"],
        "model_revision": manifest.get("model_revision"),
        "run_id": manifest.get("run_id"),
    }
    execution_manifest = {
        "schema_version": SCHEMA,
        "stage": "final_test_strength_0.50",
        "candidate": FINAL_TEST_CANDIDATE,
        "split": "test",
        "dataset_version": "sft_v2.0.0",
        "dataset_file": str(test_path.resolve()),
        "dataset_file_sha256": _sha256(test_path),
        "sample_count": FINAL_TEST_COUNT,
        "source_run_manifest": str(manifest_path.resolve()),
        "source_run_manifest_sha256": _sha256(manifest_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_tree_sha256": checkpoint_tree_sha256,
        "adapter_strength": FINAL_TEST_STRENGTH,
        "adapter_scaling_method": "multiply_each_active_PEFT_LoRA_layer_scaling_entry_in_memory",
        "checkpoint_files_modified": False,
        "model_name": manifest["model_name"],
        "model_revision": manifest.get("model_revision"),
        "chat_template": "qwen3_nothink",
        "enable_thinking": False,
        "decoding": FROZEN_DECODING,
        "authorization": str(authorization_path),
        "authorization_sha256": authorization_sha256,
        "preparation_manifest": str(preparation_path.resolve()),
        "preparation_manifest_sha256": preparation_sha256,
        "test_accessed": True,
    }
    if stage_dir.exists():
        contents = list(stage_dir.iterdir())
        stage_manifest = stage_dir / "manifest.json"
        if contents and not stage_manifest.is_file():
            raise RecoveryError("existing final Test output has no bound stage manifest")
        if stage_manifest.is_file() and _read_json(stage_manifest) != execution_manifest:
            raise RecoveryError("existing final Test output differs from the authorized frozen plan")
    completed = _completed_test_result(stage_dir, execution_manifest, source_rows, metadata)
    if completed is not None:
        return completed
    _validate_test_prediction_prefix(stage_dir / "predictions.jsonl", source_rows, metadata)

    torch = model = None
    try:
        torch, tokenizer, model = _load_model(manifest, checkpoint)
        originals = _lora_scaling(model)
        _apply_lora_strength(model, originals, FINAL_TEST_STRENGTH)
        result = _evaluate_adapter(
            stage_dir,
            "final_test_strength_0.50",
            torch,
            tokenizer,
            model,
            source_rows,
            metadata,
            execution_manifest,
            test_accessed=True,
        )
        if result.get("repetition", {}).get("sample_count") != FINAL_TEST_COUNT:
            raise RecoveryError("final Test result count is inconsistent")
        return result
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="resolve and freeze read-only inputs; do not load a model")
    subparsers.add_parser("run", help="run all recovery phases synchronously with resume")
    subparsers.add_parser("report", help="rebuild final reports from completed stage artifacts")
    validate = subparsers.add_parser("validate", help="run full Validation for a selected runtime strength")
    validate.add_argument("--candidate", required=True)
    prepare_test = subparsers.add_parser("prepare-test", help="record an authorized Test candidate without reading Test")
    prepare_test.add_argument("--candidate", required=True)
    prepare_test.add_argument("--authorization", type=Path, required=True)
    run_test = subparsers.add_parser("run-test", help="execute the one explicitly authorized repaired-LoRA final Test")
    run_test.add_argument("--candidate", required=True)
    run_test.add_argument("--authorization", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan": result = create_plan(args.output_root)
        elif args.command == "run": result = run_all(args.output_root)
        elif args.command == "report": result = build_report(args.output_root)
        elif args.command == "validate": result = _validate_selected(args.candidate, args.output_root)
        elif args.command == "prepare-test": result = _prepare_test(args.candidate, args.authorization, args.output_root)
        else: result = _run_test(args.candidate, args.authorization, args.output_root)
        print(json.dumps({"status": result.get("status"), "output_root": str(args.output_root)}, ensure_ascii=False))
        return 0
    except (RecoveryError, protocol.GuardError, judge_api.JudgeError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
