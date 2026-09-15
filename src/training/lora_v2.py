"""Fail-closed infrastructure for the LoRA Experiment Protocol V2.1.

This module prepares and validates experiments.  It never downloads a model,
authorizes final-Test use, or starts a later sweep stage automatically.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "training" / "lora_v2"
DATASETS = {
    "sft_v2.0.0": ROOT / "data" / "processed" / "sft_v2",
    "sft_v1.0.0": ROOT / "data" / "processed" / "sft_v1",
}
TRAINING_VIEW_NAMES = {"sft_v2.0.0": "sft_v2", "sft_v1.0.0": "sft_v1"}
DEFAULT_TRAINING_VIEW_ROOT = ROOT / "outputs" / "lora_v2" / "training_views"
EXPECTED = {
    "sft_v2.0.0": {"train": 1871, "validation": 234, "test": 233, "total": 2338, "facts": 18437},
    "sft_v1.0.0": {"train": 191, "validation": 24, "test": 24, "total": 239, "facts": 3007},
}
UNRESOLVED = "unresolved"
METRIC_IMPLEMENTATION = "cm_style_zh_char_ascii_word_v1"
RESULT_FIELDS = [
    "run_id", "experiment_id", "dataset_version", "model_name", "model_revision",
    "training_enabled", "finetuning_type", "training_method", "rank", "alpha", "dropout", "learning_rate", "micro_batch",
    "gradient_accumulation", "effective_batch", "epoch", "cutoff_len", "seed",
    "precision", "best_eval_loss", "best_checkpoint", "checkpoint", "last_eval_loss", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "training_time", "peak_gpu_memory", "status",
    "formal_model_inference_performed", "test_accessed",
]


class GuardError(RuntimeError):
    """A fail-closed protocol guard rejected an operation."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GuardError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GuardError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise GuardError(f"non-object JSONL record at {path}:{line_number}")
            yield value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    checksum_file = dataset_dir / "checksums.sha256"
    if not checksum_file.is_file():
        return [f"missing checksum file: {checksum_file}"]
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"malformed checksum line: {line}")
            continue
        expected_hash, relative = parts
        target = dataset_dir / relative.strip().lstrip("*")
        if not target.is_file():
            errors.append(f"missing checksummed file: {target.name}")
        elif sha256(target) != expected_hash:
            errors.append(f"checksum mismatch: {target.name}")
    return errors


def _validate_messages(row: dict[str, Any], split: str, errors: list[str]) -> None:
    sample_id = str(row.get("sample_id", "<missing>"))
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{split}/{sample_id}: messages must contain exactly 3 entries")
        return
    roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append(f"{split}/{sample_id}: invalid message roles {roles}")
        return
    if any(not isinstance(item.get("content"), str) or not item["content"] for item in messages):
        errors.append(f"{split}/{sample_id}: message content must be non-empty strings")
    if messages[-1].get("content") != row.get("target_text"):
        errors.append(f"{split}/{sample_id}: assistant message differs from target_text")


def _validate_usage(row: dict[str, Any], split: str, errors: list[str]) -> None:
    expected = {
        "train": {"gradient_usage": True, "training_usage": True, "training_eligible": True,
                  "model_selection_usage": False, "hyperparameter_selection_usage": False,
                  "checkpoint_selection_usage": False, "final_evaluation_usage": False},
        "validation": {"gradient_usage": False, "training_usage": False, "training_eligible": False,
                       "model_selection_usage": True, "hyperparameter_selection_usage": True,
                       "checkpoint_selection_usage": True, "final_evaluation_usage": False},
        "test": {"gradient_usage": False, "training_usage": False, "training_eligible": False,
                 "model_selection_usage": False, "hyperparameter_selection_usage": False,
                 "checkpoint_selection_usage": False, "final_evaluation_usage": True},
    }[split]
    sample_id = str(row.get("sample_id", "<missing>"))
    for field, value in expected.items():
        if row.get(field) is not value:
            errors.append(f"{split}/{sample_id}: {field} must be {value}")


def validate_dataset(version: str) -> dict[str, Any]:
    """Validate a frozen dataset without changing it; raise on every failure."""

    if version not in DATASETS:
        raise GuardError(f"unsupported dataset version: {version}")
    dataset_dir, expected = DATASETS[version], EXPECTED[version]
    errors = verify_checksums(dataset_dir)
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    if manifest.get("dataset_version") != version or manifest.get("build_status") != "frozen":
        errors.append("dataset manifest version/status is not frozen as expected")
    if manifest.get("training_started") is not False or manifest.get("model_test_evaluation_performed") is not False:
        errors.append("dataset manifest indicates training or Test evaluation already occurred")
    if version == "sft_v2.0.0":
        if manifest.get("provenance") != "passed_no_candidate_contamination":
            errors.append("SFT_v2 provenance gate did not pass")
        gate = read_json(dataset_dir / "pre_training_gate.json")
        if gate.get("status") != "ready_for_user_pre_training_approval":
            errors.append("SFT_v2 pre-training gate is not ready")
    schema = read_json(dataset_dir / "schema.json")
    required = set(schema.get("required", []))
    counts: dict[str, int] = {}
    total_facts = 0
    all_ids: set[str] = set()
    all_docs: set[str] = set()
    for split in ("train", "validation", "test"):
        rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
        counts[split] = len(rows)
        if len(rows) != expected[split]:
            errors.append(f"{split} count {len(rows)} != {expected[split]}")
        for row in rows:
            missing = sorted(required - row.keys())
            if missing:
                errors.append(f"{split}/{row.get('sample_id')}: missing fields {missing}")
            if row.get("split") != split:
                errors.append(f"{split}/{row.get('sample_id')}: split field mismatch")
            sample_id, document_id = row.get("sample_id"), row.get("document_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in all_ids:
                errors.append(f"{split}: invalid or duplicate sample_id {sample_id!r}")
            else:
                all_ids.add(sample_id)
            if not isinstance(document_id, str) or document_id in all_docs:
                errors.append(f"{split}/{sample_id}: invalid or duplicate document_id")
            else:
                all_docs.add(document_id)
            facts = row.get("fact_points")
            if not isinstance(facts, list) or not facts:
                errors.append(f"{split}/{sample_id}: fact_points must be non-empty")
            else:
                total_facts += len(facts)
            target = row.get("target_text")
            if isinstance(target, str) and hashlib.sha256(target.encode("utf-8")).hexdigest() != row.get("target_text_sha256"):
                errors.append(f"{split}/{sample_id}: target_text checksum mismatch")
            _validate_messages(row, split, errors)
            _validate_usage(row, split, errors)
    if sum(counts.values()) != expected["total"]:
        errors.append(f"total count {sum(counts.values())} != {expected['total']}")
    if total_facts != expected["facts"]:
        errors.append(f"fact count {total_facts} != {expected['facts']}")
    if version == "sft_v1.0.0" and counts.get("train") != 191:
        errors.append("E08 gradient set must be the historical 191-row Train split")
    result = {"dataset_version": version, "status": "passed" if not errors else "failed",
              "counts": counts, "total": sum(counts.values()), "facts": total_facts,
              "checksums": "passed" if not verify_checksums(dataset_dir) else "failed", "errors": errors}
    if errors:
        raise GuardError("dataset preflight failed:\n- " + "\n- ".join(errors[:50]))
    return result


def training_view_dir(version: str, output_root: Path | None = None) -> Path:
    if version not in TRAINING_VIEW_NAMES:
        raise GuardError(f"unsupported training view dataset version: {version}")
    return (output_root or DEFAULT_TRAINING_VIEW_ROOT) / TRAINING_VIEW_NAMES[version]


def _validate_training_messages(messages: Any, split: str, index: int) -> None:
    if not isinstance(messages, list) or len(messages) != 3:
        raise GuardError(f"training view {split}/{index}: messages must contain exactly 3 entries")
    roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
    if roles != ["system", "user", "assistant"]:
        raise GuardError(f"training view {split}/{index}: invalid message roles {roles}")
    if any(not isinstance(item.get("content"), str) or not item["content"] for item in messages):
        raise GuardError(f"training view {split}/{index}: message content must be non-empty strings")


def validate_training_view(version: str, view_dir: Path) -> dict[str, Any]:
    """Verify a disposable messages-only view against its authoritative frozen SFT source."""

    dataset_validation = validate_dataset(version)
    manifest_path = view_dir / "training_view_manifest.json"
    if not manifest_path.is_file():
        raise GuardError(f"training view manifest is missing: {manifest_path}")
    if (view_dir / "test.jsonl").exists():
        raise GuardError("Test must not be materialized in a training view")
    unexpected = sorted(path.name for path in view_dir.glob("*.jsonl")
                        if path.name not in {"train.jsonl", "validation.jsonl"})
    if unexpected:
        raise GuardError(f"unexpected training view JSONL files: {unexpected}")
    manifest = read_json(manifest_path)
    expected_manifest = {
        "schema_version": "lora-training-view-v1.0.0", "status": "ready",
        "source_dataset_version": version, "transformation": "messages_only_training_view",
        "semantic_transformation": False, "sample_order_preserved": True,
        "message_content_preserved": True, "assistant_target_preserved": True,
        "test_materialized": False,
    }
    mismatches = [key for key, value in expected_manifest.items() if manifest.get(key) != value]
    dataset_dir = DATASETS[version]
    source_manifest_path = dataset_dir / "dataset_manifest.json"
    source_checksums_path = dataset_dir / "checksums.sha256"
    if manifest.get("source_manifest_reference") != str(source_manifest_path.resolve()):
        mismatches.append("source_manifest_reference")
    if manifest.get("source_checksum_reference") != str(source_checksums_path.resolve()):
        mismatches.append("source_checksum_reference")
    if manifest.get("source_manifest_sha256") != sha256(source_manifest_path):
        mismatches.append("source_manifest_sha256")
    if manifest.get("source_checksums_sha256") != sha256(source_checksums_path):
        mismatches.append("source_checksums_sha256")
    source_manifest = read_json(source_manifest_path)
    if manifest.get("source_split_version") != source_manifest.get("split_version"):
        mismatches.append("source_split_version")
    if mismatches:
        raise GuardError(f"training view manifest mismatch: {sorted(set(mismatches))}")
    split_records = manifest.get("splits")
    if not isinstance(split_records, dict) or set(split_records) != {"train", "validation"}:
        raise GuardError("training view manifest must contain only Train and Validation splits")
    for split in ("train", "validation"):
        source_path = dataset_dir / f"{split}.jsonl"
        output_path = view_dir / f"{split}.jsonl"
        if not output_path.is_file():
            raise GuardError(f"training view split is missing: {output_path}")
        record = split_records.get(split)
        if not isinstance(record, dict):
            raise GuardError(f"training view manifest split record is invalid: {split}")
        source_rows = list(iter_jsonl(source_path))
        output_rows = list(iter_jsonl(output_path))
        expected_count = EXPECTED[version][split]
        metadata = {
            "source_file": str(source_path.resolve()), "source_sha256": sha256(source_path),
            "source_count": expected_count, "output_file": str(output_path.resolve()),
            "output_sha256": sha256(output_path), "output_count": expected_count,
        }
        if any(record.get(key) != value for key, value in metadata.items()):
            raise GuardError(f"training view {split} lineage/checksum mismatch")
        if len(source_rows) != expected_count or len(output_rows) != expected_count:
            raise GuardError(f"training view {split} count mismatch")
        for index, (source, output) in enumerate(zip(source_rows, output_rows)):
            if set(output) != {"messages"}:
                raise GuardError(f"training view {split}/{index} contains non-training fields")
            _validate_training_messages(output.get("messages"), split, index)
            if output["messages"] != source.get("messages"):
                raise GuardError(f"training view {split}/{index} changed message content or order")
            if output["messages"][-1]["content"] != source.get("target_text"):
                raise GuardError(f"training view {split}/{index} changed assistant target")
    return {**manifest, "validation_status": "passed", "dataset_validation": dataset_validation["status"]}


def prepare_training_view(version: str, output_root: Path | None = None) -> dict[str, Any]:
    """Build or validate/reuse a deterministic messages-only Train/Validation view."""

    validate_dataset(version)
    view_dir = training_view_dir(version, output_root)
    if view_dir.exists():
        result = validate_training_view(version, view_dir)
        return {**result, "reuse_status": "validated_existing_view"}
    view_dir.mkdir(parents=True)
    dataset_dir = DATASETS[version]
    source_manifest_path = dataset_dir / "dataset_manifest.json"
    source_checksums_path = dataset_dir / "checksums.sha256"
    source_manifest = read_json(source_manifest_path)
    split_records: dict[str, dict[str, Any]] = {}
    try:
        for split in ("train", "validation"):
            source_path = dataset_dir / f"{split}.jsonl"
            output_path = view_dir / f"{split}.jsonl"
            source_rows = list(iter_jsonl(source_path))
            if len(source_rows) != EXPECTED[version][split]:
                raise GuardError(f"frozen {version} {split} count changed before view generation")
            with output_path.open("x", encoding="utf-8", newline="\n") as handle:
                for index, row in enumerate(source_rows):
                    _validate_training_messages(row.get("messages"), split, index)
                    if row["messages"][-1]["content"] != row.get("target_text"):
                        raise GuardError(f"frozen {split}/{index} assistant message differs from target")
                    handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")) + "\n")
            output_rows = list(iter_jsonl(output_path))
            if len(output_rows) != len(source_rows) or any(
                    output.get("messages") != source.get("messages")
                    for source, output in zip(source_rows, output_rows)):
                raise GuardError(f"generated training view failed exact preservation: {split}")
            split_records[split] = {
                "source_file": str(source_path.resolve()), "source_sha256": sha256(source_path),
                "source_count": len(source_rows), "output_file": str(output_path.resolve()),
                "output_sha256": sha256(output_path), "output_count": len(output_rows),
            }
        manifest = {
            "schema_version": "lora-training-view-v1.0.0", "status": "ready",
            "source_dataset_version": version, "source_split_version": source_manifest.get("split_version"),
            "source_manifest_reference": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_checksum_reference": str(source_checksums_path.resolve()),
            "source_checksums_sha256": sha256(source_checksums_path),
            "transformation": "messages_only_training_view", "semantic_transformation": False,
            "sample_order_preserved": True, "message_content_preserved": True,
            "assistant_target_preserved": True, "test_materialized": False,
            "splits": split_records, "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_to_output_lineage": [
                {"split": split, "source": split_records[split]["source_file"],
                 "output": split_records[split]["output_file"]}
                for split in ("train", "validation")
            ],
        }
        write_json(view_dir / "training_view_manifest.json", manifest)
        return validate_training_view(version, view_dir)
    except Exception:
        write_json(view_dir / "PREPARATION_FAILED.json", {
            "status": "failed", "source_dataset_version": version,
            "frozen_dataset_modified": False, "test_materialized": False,
        })
        raise


def load_experiment(experiment_id: str) -> dict[str, Any]:
    path = CONFIG_DIR / "experiments" / f"{experiment_id.upper()}.json"
    if not path.is_file():
        raise GuardError(f"unknown experiment: {experiment_id}")
    return read_json(path)


def load_selection_state(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIG_DIR / "selection_state.json")


def _require_selection(name: str, state: dict[str, Any]) -> Any:
    value = state.get(name)
    if value in (None, UNRESOLVED):
        raise GuardError(f"{name} is unresolved; explicit user selection is required")
    return value


def resolve_experiment(experiment_id: str, selection_path: Path | None = None) -> dict[str, Any]:
    """Resolve a manifest, preserving unresolved state instead of guessing."""

    exp = load_experiment(experiment_id)
    state = load_selection_state(selection_path)
    resolved = dict(exp)
    bindings = {"LR_STAR": "learning_rate", "RANK_STAR": "rank", "BATCH_STAR": "effective_batch",
                "CUTOFF_LEN": "cutoff_len"}
    for selector, field in bindings.items():
        if resolved.get(field) == selector:
            resolved[field] = state.get(selector, UNRESOLVED)
    if resolved.get("model_revision") == "unresolved_pending_cloud_verification":
        resolved["model_revision"] = state.get("MODEL_REVISION", UNRESOLVED)
    if resolved.get("training_enabled"):
        resolved["training_method"] = state.get("training_method", resolved.get("training_method"))
    resolved["qlora_fallback_authorized"] = state.get("qlora_fallback_authorized", False)
    resolved["qlora_fallback_evidence"] = state.get("qlora_fallback_evidence")
    if resolved.get("cutoff_len") in (None, "CUTOFF_LEN"):
        resolved["cutoff_len"] = state.get("CUTOFF_LEN", UNRESOLVED)
    if isinstance(resolved.get("effective_batch"), int):
        resolved["gradient_accumulation"] = resolved["effective_batch"] // resolved["micro_batch"]
    if resolved.get("experiment_id") == "E08" and resolved.get("effective_batch") not in (UNRESOLVED, None):
        resolved["gradient_accumulation"] = int(resolved["effective_batch"])
    unresolved = sorted(k for k in ("learning_rate", "rank", "effective_batch", "cutoff_len", "model_revision")
                        if resolved.get(k) == UNRESOLVED)
    resolved["resolution_status"] = "resolved" if not unresolved else "unresolved"
    resolved["unresolved_fields"] = unresolved
    return resolved


def assert_launchable(config: dict[str, Any]) -> None:
    if config.get("training_enabled") is not True:
        raise GuardError(f"{config.get('experiment_id')} is not a training experiment")
    if config.get("resolution_status") != "resolved":
        raise GuardError(f"unresolved formal config fields: {config.get('unresolved_fields')}")
    if config.get("cutoff_len") not in (2048, 4096):
        raise GuardError("cutoff_len must be frozen to 2048 or 4096 by formal token statistics")
    if config.get("model_revision") in (None, UNRESOLVED, "unresolved_pending_cloud_verification"):
        raise GuardError("the official model revision must be verified and frozen during cloud preflight")
    if config.get("training_method") not in {"standard_bf16_lora", "uniform_qlora"}:
        raise GuardError("training_method must be standard_bf16_lora or uniform_qlora")
    if config.get("training_method") == "uniform_qlora" and not (
        config.get("qlora_fallback_authorized") is True
        and config.get("qlora_fallback_evidence") == "standard_bf16_lora_24gb_failed"
    ):
        raise GuardError("QLoRA requires explicit pre-E01 authorization backed by failed 24GB BF16 smoke")
    if config.get("micro_batch") != 1:
        raise GuardError("protocol fixes micro_batch at 1")
    if config.get("effective_batch") != config.get("micro_batch") * config.get("gradient_accumulation"):
        raise GuardError("effective batch arithmetic mismatch")
    if config.get("experiment_id") == "E08" and config.get("dataset_version") != "sft_v1.0.0":
        raise GuardError("E08 must use only frozen sft_v1.0.0")


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,79}", run_id):
        raise GuardError("run_id must be 3-80 safe filename characters")


def _verified_model_evidence(config: dict[str, Any], selection_path: Path | None) -> dict[str, Any]:
    if selection_path is None:
        raise GuardError("E00 requires the formal runtime --selection artifact")
    artifact_path = selection_path.resolve().parent / "model_verification.json"
    if not artifact_path.is_file():
        raise GuardError(f"verified model artifact is required beside runtime selection: {artifact_path}")
    verification = read_json(artifact_path)
    expected = {
        "status": "verified",
        "model_identifier": config["model_name"],
        "model_revision": config["model_revision"],
        "tokenizer_identifier": config["model_name"],
        "chat_template": config["chat_template"],
        "model_inference_performed": False,
    }
    mismatches = [key for key, value in expected.items() if verification.get(key) != value]
    tokenizer_revision = verification.get("tokenizer_revision")
    if not isinstance(tokenizer_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
        mismatches.append("tokenizer_revision")
    local_value = verification.get("local_model_path")
    if not isinstance(local_value, str) or not local_value.strip():
        mismatches.append("local_model_path")
    if mismatches:
        raise GuardError(f"model verification does not match resolved E00 config: {sorted(set(mismatches))}")
    local_model_path = Path(local_value).resolve()
    if not local_model_path.is_dir():
        raise GuardError(f"verified local model snapshot is unavailable: {local_model_path}")
    return {
        "local_model_path": str(local_model_path),
        "tokenizer_identifier": verification["tokenizer_identifier"],
        "tokenizer_revision": tokenizer_revision,
        "model_verification_artifact": str(artifact_path),
        "model_verification_sha256": sha256(artifact_path),
    }


def differing_fields(left: dict[str, Any], right: dict[str, Any], ignored: Sequence[str] = ()) -> set[str]:
    ignored_set = {"experiment_id", "run_id", "status", "parent", "reused_run", "created_at",
                   "stage", "description", *ignored}
    return {key for key in left.keys() | right.keys() if key not in ignored_set and left.get(key) != right.get(key)}


def assert_single_variable(left: dict[str, Any], right: dict[str, Any], allowed: set[str]) -> None:
    difference = differing_fields(left, right)
    if not difference or not difference <= allowed:
        raise GuardError(f"non-single-variable comparison: changed={sorted(difference)}, allowed={sorted(allowed)}")


def validate_experiment_matrix() -> dict[str, Any]:
    experiments = {f"E{i:02d}": load_experiment(f"E{i:02d}") for i in range(9)}
    assert_single_variable(experiments["E01"], experiments["E02"], {"learning_rate"})
    assert_single_variable(experiments["E02"], experiments["E03"], {"learning_rate"})
    rank_base = dict(experiments["E01"], learning_rate="LR_STAR")
    assert_single_variable(rank_base, experiments["E04"], {"rank"})
    assert_single_variable(rank_base, experiments["E05"], {"rank"})
    batch_base = dict(rank_base, rank="RANK_STAR")
    assert_single_variable(batch_base, experiments["E06"], {"gradient_accumulation", "effective_batch"})
    assert_single_variable(batch_base, experiments["E07"], {"gradient_accumulation", "effective_batch"})
    final_reference = dict(batch_base, effective_batch="BATCH_STAR", gradient_accumulation="BATCH_STAR")
    # E08 has its own dataset lineage, but every training variable must match the final reference.
    dataset_difference = differing_fields(experiments["E08"], final_reference, ignored=("stage", "description"))
    allowed_dataset_fields = {"dataset_version", "dataset_manifest_reference", "dataset_checksum_reference"}
    if dataset_difference != allowed_dataset_fields:
        raise GuardError(f"E08 changes fields beyond dataset identity/lineage: {sorted(dataset_difference)}")
    return {"status": "passed", "experiments": sorted(experiments), "guards": ["lr", "rank", "batch", "dataset"]}


def _yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:+-]+", str(value)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_flat_yaml(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}: {_yaml_scalar(value)}\n" for key, value in values.items()), encoding="utf-8")


def llama_factory_config(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    assert_launchable(config)
    view_dir = Path(config["training_view_dir"])
    train_count = EXPECTED[config["dataset_version"]]["train"]
    steps_per_epoch = math.ceil(train_count / config["effective_batch"])
    cadence = max(1, math.ceil(steps_per_epoch / 2))
    runtime_data = run_dir / "runtime_data"
    dataset_info = {
        "formal_train": {"file_name": str((view_dir / "train.jsonl").resolve()), "formatting": "sharegpt",
                         "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                         "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
        "formal_validation": {"file_name": str((view_dir / "validation.jsonl").resolve()), "formatting": "sharegpt",
                              "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                              "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
    }
    write_json(runtime_data / "dataset_info.json", dataset_info)
    output_dir = run_dir / "checkpoints"
    generated = {
        "model_name_or_path": config["model_name"], "trust_remote_code": False, "stage": "sft", "do_train": True,
        "finetuning_type": config["finetuning_type"], "lora_rank": config["rank"], "lora_alpha": config["alpha"],
        "lora_dropout": config["dropout"], "lora_target": config["lora_target"], "dataset": "formal_train",
        "eval_dataset": "formal_validation", "dataset_dir": str(runtime_data.resolve()), "template": config["chat_template"],
        "cutoff_len": config["cutoff_len"], "train_on_prompt": config["train_on_prompt"], "packing": config["packing"],
        "output_dir": str(output_dir.resolve()), "overwrite_output_dir": False, "per_device_train_batch_size": config["micro_batch"],
        "per_device_eval_batch_size": 1, "gradient_accumulation_steps": config["gradient_accumulation"],
        "learning_rate": config["learning_rate"], "num_train_epochs": config["epoch"], "lr_scheduler_type": config["scheduler"],
        "warmup_ratio": config["warmup_ratio"], "seed": config["seed"], "bf16": config["precision"] == "bf16",
        "fp16": False, "gradient_checkpointing": True, "do_eval": True, "eval_strategy": "steps", "eval_steps": cadence,
        "save_strategy": "steps", "save_steps": cadence, "save_total_limit": 3, "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", "greater_is_better": False, "logging_steps": max(1, cadence // 10),
        "report_to": "none", "plot_loss": True,
    }
    if config["training_method"] == "uniform_qlora":
        generated.update({"quantization_bit": 4, "quantization_method": "bitsandbytes",
                          "double_quantization": True})
    return generated


def prepare_run(experiment_id: str, run_id: str, output_root: Path,
                selection_path: Path | None = None,
                training_view_root: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    _validate_run_id(run_id)
    config = resolve_experiment(experiment_id, selection_path)
    assert_launchable(config)
    validate_dataset(config["dataset_version"])
    validate_experiment_matrix()
    view_dir = training_view_dir(config["dataset_version"], training_view_root)
    view = validate_training_view(config["dataset_version"], view_dir)
    run_dir = (output_root / experiment_id.upper() / run_id).resolve()
    if run_dir.exists():
        raise GuardError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        resolved = dict(config, run_id=run_id, status="prepared", created_at=datetime.now(timezone.utc).isoformat(),
                        git_commit=_git_commit(), training_view_dir=str(view_dir.resolve()),
                        training_view_manifest=str((view_dir / "training_view_manifest.json").resolve()),
                        training_view_manifest_sha256=sha256(view_dir / "training_view_manifest.json"),
                        training_view_transformation=view["transformation"],
                        frozen_dataset_authority=config["dataset_version"], test_materialized=False)
        write_json(run_dir / "resolved_manifest.json", resolved)
        yaml_path = run_dir / "llamafactory.yaml"
        write_flat_yaml(yaml_path, llama_factory_config(resolved, run_dir))
        write_json(run_dir / "environment_manifest.json", environment_manifest())
        return run_dir, yaml_path, resolved
    except Exception:
        # Preserve a clear failure marker; never silently reuse this directory.
        write_json(run_dir / "PREPARATION_FAILED.json", {"status": "failed", "experiment_id": experiment_id})
        raise


def prepare_baseline_run(run_id: str, output_root: Path, selection_path: Path | None = None) -> dict[str, Any]:
    """Create an immutable E00 Base Validation manifest without training or inference."""

    _validate_run_id(run_id)
    config = resolve_experiment("E00", selection_path)
    if config.get("training_enabled") is not False or config.get("training_method") != "base_validation":
        raise GuardError("E00 must remain a non-training base_validation experiment")
    if config.get("finetuning_type") != "none" or config.get("best_checkpoint") is not None:
        raise GuardError("E00 cannot contain an adapter or best checkpoint")
    if config.get("resolution_status") != "resolved":
        raise GuardError(f"unresolved E00 config fields: {config.get('unresolved_fields')}")
    if config.get("cutoff_len") not in (2048, 4096):
        raise GuardError("cutoff_len must be frozen to 2048 or 4096 by formal token statistics")
    if not isinstance(config.get("model_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise GuardError("the official model revision must be a verified 40-character commit")
    evidence = _verified_model_evidence(config, selection_path)
    validate_dataset(config["dataset_version"])
    validate_experiment_matrix()
    authorization = read_json(CONFIG_DIR / "final_test_authorization.json")
    if authorization.get("final_test_authorized") is not False:
        raise GuardError("E00 preparation requires final_test_authorized=false")
    run_dir = (output_root / "E00" / run_id).resolve()
    if run_dir.exists():
        raise GuardError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        manifest = dict(
            config,
            **evidence,
            run_id=run_id,
            status="prepared",
            checkpoint=None,
            formal_training_started=False,
            formal_model_inference_performed=False,
            final_test_authorized=False,
            test_accessed=False,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(),
            runtime_selection_artifact=str(selection_path.resolve()),
            runtime_selection_sha256=sha256(selection_path.resolve()),
        )
        write_json(run_dir / "resolved_manifest.json", manifest)
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "baseline_plan.json", {
            "status": "prepared_not_executed", "training": False, "inference": False,
            "test_access": False, "checkpoint": None,
        })
        write_json(run_dir / "environment_manifest.json", environment_manifest())
        return manifest
    except Exception:
        write_json(run_dir / "PREPARATION_FAILED.json", {"status": "failed", "experiment_id": "E00"})
        raise


def launch_training(experiment_id: str, run_id: str, output_root: Path, *, execute: bool = False,
                    selection_path: Path | None = None, cli: str = "llamafactory-cli",
                    training_view_root: Path | None = None) -> dict[str, Any]:
    if experiment_id.upper() == "E00":
        if execute:
            raise GuardError("E00 is never executable through the training launcher")
        return prepare_baseline_run(run_id, output_root, selection_path)
    run_dir, yaml_path, manifest = prepare_run(experiment_id, run_id, output_root, selection_path,
                                               training_view_root)
    command = [cli, "train", str(yaml_path)]
    write_json(run_dir / "launch_plan.json", {"command": command, "execute": execute, "test_access": False})
    if not execute:
        return dict(manifest, status="prepared_not_executed", command=command)
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        result = dict(manifest, status="failed", returncode=completed.returncode)
        write_json(run_dir / "run_manifest.json", result)
        raise GuardError(f"training command failed with exit code {completed.returncode}; see {run_dir / 'training.log'}")
    return finalize_run(run_dir, manifest)


def finalize_run(run_dir: Path, prepared_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Populate tracked run fields from trainer artifacts without manual copying."""

    manifest = dict(prepared_manifest or read_json(run_dir / "resolved_manifest.json"))
    if manifest.get("experiment_id") == "E00":
        run_manifest_path = run_dir / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise GuardError("E00 finalization requires its prepared run_manifest.json")
        current = read_json(run_manifest_path)
        identity_fields = ("experiment_id", "run_id", "dataset_version", "model_name", "model_revision",
                           "training_enabled", "finetuning_type", "training_method", "cutoff_len", "chat_template")
        if any(current.get(field) != manifest.get(field) for field in identity_fields):
            raise GuardError("E00 resolved and run manifests contradict each other")
        manifest = current
        return _finalize_baseline_run(run_dir, manifest)
    checkpoint_root = run_dir / "checkpoints"
    trainer_state_path = checkpoint_root / "trainer_state.json"
    if not trainer_state_path.is_file():
        candidates = sorted(checkpoint_root.glob("checkpoint-*/trainer_state.json"))
        trainer_state_path = candidates[-1] if candidates else trainer_state_path
    if not trainer_state_path.is_file():
        raise GuardError(f"successful command did not produce trainer_state.json under {checkpoint_root}")
    state = read_json(trainer_state_path)
    history = state.get("log_history", [])
    eval_losses = [float(row["eval_loss"]) for row in history if isinstance(row, dict) and "eval_loss" in row]
    checkpoints = sorted(checkpoint_root.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    metrics: dict[str, Any] = {}
    for name in ("all_results.json", "train_results.json"):
        path = checkpoint_root / name
        if path.is_file():
            metrics.update(read_json(path))
    manifest.update({
        "status": "training_completed_validation_generation_pending",
        "returncode": 0,
        "best_checkpoint": state.get("best_model_checkpoint"),
        "best_eval_loss": state.get("best_metric"),
        "last_eval_loss": eval_losses[-1] if eval_losses else None,
        "last_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "training_time": metrics.get("train_runtime"),
        "peak_gpu_memory": metrics.get("train_mem_gpu_peaked_delta") or metrics.get("train_mem_gpu_alloc_delta"),
        "trainer_state": str(trainer_state_path),
    })
    if manifest["best_checkpoint"] is None or manifest["best_eval_loss"] is None:
        raise GuardError("trainer state lacks best Validation checkpoint/eval loss")
    _finalize_training_validation_if_present(run_dir, manifest)
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _finalize_training_validation_if_present(run_dir: Path, manifest: dict[str, Any]) -> None:
    inference_path = run_dir / "validation_inference" / "inference_manifest.json"
    predictions_path = run_dir / "validation_inference" / "predictions.jsonl"
    metrics_path = run_dir / "validation_metrics.json"
    if not all(path.is_file() for path in (inference_path, predictions_path, metrics_path)):
        return
    expected_count = EXPECTED[manifest["dataset_version"]]["validation"]
    inference = read_json(inference_path)
    mismatches = [
        field for field, value in {
            "status": "completed", "split": "validation", "prediction_count": expected_count,
        }.items() if inference.get(field) != value
    ]
    if mismatches:
        raise GuardError(f"training Validation inference metadata mismatch: {mismatches}")
    prediction_count = sum(1 for _ in iter_jsonl(predictions_path))
    if prediction_count != expected_count:
        raise GuardError(f"training Validation predictions count {prediction_count} != {expected_count}")
    metrics = read_json(metrics_path)
    metrics_manifest = metrics.get("manifest")
    if not isinstance(metrics_manifest, dict) or metrics_manifest.get("count") != expected_count:
        raise GuardError(f"training Validation metrics count must be {expected_count}")
    metric_names = ("BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L")
    if any(isinstance(metrics.get(name), bool) or not isinstance(metrics.get(name), (int, float))
           or not math.isfinite(float(metrics[name])) for name in metric_names):
        raise GuardError("training Validation metrics are missing or non-finite")
    manifest.update({
        "status": "completed", "formal_model_inference_performed": True,
        "validation_inference_manifest": str(inference_path.resolve()),
        "validation_inference_manifest_sha256": sha256(inference_path),
        "validation_predictions": str(predictions_path.resolve()),
        "validation_predictions_sha256": sha256(predictions_path),
        "validation_metrics": str(metrics_path.resolve()),
        "validation_metrics_sha256": sha256(metrics_path),
        **{name: float(metrics[name]) for name in metric_names},
    })


def _finalize_baseline_run(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "experiment_id": "E00", "dataset_version": "sft_v2.0.0", "training_enabled": False,
        "finetuning_type": "none", "training_method": "base_validation", "best_checkpoint": None,
        "cutoff_len": 4096, "chat_template": "qwen3_nothink", "formal_training_started": False,
        "final_test_authorized": False, "test_accessed": False,
    }
    mismatches = [field for field, value in expected.items() if manifest.get(field) != value]
    if manifest.get("checkpoint") not in (None, "base_model"):
        mismatches.append("checkpoint")
    if manifest.get("rank") is not None or manifest.get("alpha") is not None or manifest.get("learning_rate") is not None:
        mismatches.append("training_hyperparameters")
    if mismatches:
        raise GuardError(f"run is not the formal E00 non-training baseline: {sorted(set(mismatches))}")
    _inference_model_location(manifest)

    inference_path = run_dir / "validation_inference" / "inference_manifest.json"
    predictions_path = run_dir / "validation_inference" / "predictions.jsonl"
    metrics_path = run_dir / "validation_metrics.json"
    for label, path in (("inference manifest", inference_path), ("predictions", predictions_path),
                        ("validation metrics", metrics_path)):
        if not path.is_file():
            raise GuardError(f"E00 finalization requires existing {label}: {path}")

    inference = read_json(inference_path)
    inference_expected = {
        "status": "completed", "split": "validation", "prediction_count": 234,
        "run_id": manifest["run_id"], "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"], "checkpoint": None,
    }
    inference_mismatches = [field for field, value in inference_expected.items() if inference.get(field) != value]
    if inference.get("test_accessed", False) is not False:
        inference_mismatches.append("test_accessed")
    dataset_file = inference.get("dataset_file")
    if not isinstance(dataset_file, str) or Path(dataset_file).name != "validation.jsonl":
        inference_mismatches.append("dataset_file")
    output_value = inference.get("output")
    if not isinstance(output_value, str) or Path(output_value).resolve() != predictions_path.resolve():
        inference_mismatches.append("output")
    if inference_mismatches:
        raise GuardError(f"E00 Validation inference metadata mismatch: {sorted(set(inference_mismatches))}")

    predictions = list(iter_jsonl(predictions_path))
    if len(predictions) != 234:
        raise GuardError(f"E00 Validation predictions count {len(predictions)} != 234")
    sample_ids: set[str] = set()
    for row in predictions:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise GuardError("E00 Validation predictions contain missing or duplicate sample_id")
        sample_ids.add(sample_id)
        if row.get("run_id") != manifest["run_id"] or row.get("model_name") != manifest["model_name"]:
            raise GuardError("E00 prediction metadata contradicts the run manifest")
        if row.get("model_revision", manifest["model_revision"]) != manifest["model_revision"]:
            raise GuardError("E00 prediction model revision contradicts the run manifest")
        if row.get("checkpoint") not in (None, "base_model"):
            raise GuardError("E00 predictions must use base_model checkpoint semantics")

    metrics = read_json(metrics_path)
    metrics_manifest = metrics.get("manifest")
    if not isinstance(metrics_manifest, dict) or metrics_manifest.get("count") != 234:
        raise GuardError("E00 Validation metrics count must be 234")
    metric_names = ("BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L")
    if any(isinstance(metrics.get(name), bool) or not isinstance(metrics.get(name), (int, float))
           or not math.isfinite(float(metrics[name]))
           for name in metric_names):
        raise GuardError("E00 Validation metrics are missing or non-finite")

    manifest.update({
        "status": "completed", "formal_model_inference_performed": True,
        "formal_training_started": False, "training_time": None, "peak_gpu_memory": None,
        "best_checkpoint": None, "best_eval_loss": None, "checkpoint": None,
        "last_checkpoint": None, "last_eval_loss": None, "trainer_state": None,
        "validation_inference_manifest": str(inference_path.resolve()),
        "validation_inference_manifest_sha256": sha256(inference_path),
        "validation_predictions": str(predictions_path.resolve()),
        "validation_predictions_sha256": sha256(predictions_path),
        "validation_metrics": str(metrics_path.resolve()),
        "validation_metrics_sha256": sha256(metrics_path),
        **{name: float(metrics[name]) for name in metric_names},
    })
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _read_authorization(path: Path | None = None) -> dict[str, Any]:
    artifact = read_json(path or CONFIG_DIR / "final_test_authorization.json")
    if artifact.get("final_test_authorized") is not True:
        raise GuardError("final Test is not authorized")
    required = ("base_model_frozen", "method_frozen", "lr_star_frozen", "rank_star_frozen", "batch_star_frozen",
                "checkpoint_frozen", "chat_template_frozen", "prompt_frozen", "decoding_parameters_frozen")
    if any(artifact.get(key) is not True for key in required):
        raise GuardError("final Test authorization is incomplete")
    return artifact


def split_path(version: str, split: str, *, purpose: str, authorization_path: Path | None = None) -> Path:
    if version not in DATASETS or split not in {"train", "validation", "test"}:
        raise GuardError("unsupported dataset or split")
    allowed = {"gradient_training": "train", "model_selection": "validation", "token_statistics": "train",
               "validation_inference": "validation", "comparison_mask_metadata": "test", "final_evaluation": "test"}
    if allowed.get(purpose) != split:
        raise GuardError(f"split {split} is forbidden for purpose {purpose}")
    if split == "test" and purpose != "comparison_mask_metadata":
        _read_authorization(authorization_path)
    return DATASETS[version] / f"{split}.jsonl"


def token_statistics(model_path: str, output: Path, *, version: str = "sft_v2.0.0") -> dict[str, Any]:
    """Use a local official tokenizer; record pending instead of downloading/faking."""

    source = split_path(version, "train", purpose="token_statistics")
    base = {"dataset_version": version, "split": "train", "model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_path": model_path, "local_files_only": True, "chat_template": "qwen3_nothink"}
    try:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        result = dict(base, status="pending_cloud_tokenizer_run", reason=f"local tokenizer unavailable: {type(exc).__name__}")
        write_json(output, result)
        return result
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    for row in iter_jsonl(source):
        messages = row["messages"]
        prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
        total_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                  enable_thinking=False)
        target_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_ids)); target_lengths.append(len(target_ids)); total_lengths.append(len(total_ids))
    p95 = _summary(total_lengths)["p95"]
    result = dict(base, status="completed", prompt_tokens=_summary(prompt_lengths),
                  assistant_target_tokens=_summary(target_lengths), total_tokens=_summary(total_lengths),
                  recommended_cutoff_len=2048 if p95 <= 2048 else 4096,
                  cutoff_rule="2048 if Train total-token P95 <= 2048, otherwise 4096")
    write_json(output, result)
    return result


def verify_model_snapshot(model_path: Path, repo_id: str, revision: str, tokenizer_revision: str,
                          output: Path) -> dict[str, Any]:
    """Verify an already-downloaded official snapshot without model inference."""

    if repo_id != "Qwen/Qwen3-4B-Instruct-2507":
        raise GuardError(f"unexpected model repository: {repo_id}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
        raise GuardError("model and tokenizer revisions must be actual 40-character commit SHAs")
    required = (model_path / "config.json", model_path / "tokenizer_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    weights = sorted(model_path.glob("*.safetensors"))
    if missing or not weights:
        raise GuardError(f"incomplete local model snapshot; missing={missing}, weight_files={len(weights)}")
    config = read_json(model_path / "config.json")
    tokenizer_config = read_json(model_path / "tokenizer_config.json")
    try:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        probe = [{"role": "system", "content": "You are a formatting probe."},
                 {"role": "user", "content": "Return one short sentence."}]
        rendered = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        token_ids = tokenizer.apply_chat_template(probe, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
    except Exception as exc:
        raise GuardError(f"official tokenizer/chat-template verification failed: {type(exc).__name__}: {exc}") from exc
    if not rendered or not token_ids:
        raise GuardError("tokenizer/chat-template probe produced empty output")
    result = {
        "schema_version": "lora-v2-model-verification-v1.0.0", "status": "verified",
        "model_identifier": repo_id, "model_revision": revision,
        "tokenizer_identifier": repo_id, "tokenizer_revision": tokenizer_revision,
        "local_model_path": str(model_path.resolve()), "local_files_only": True,
        "config_sha256": sha256(model_path / "config.json"),
        "tokenizer_config_sha256": sha256(model_path / "tokenizer_config.json"),
        "chat_template": "qwen3_nothink", "native_chat_template_sha256": hashlib.sha256(
            str(tokenizer_config.get("chat_template", "")).encode("utf-8")).hexdigest(),
        "render_probe_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "render_probe_token_count": len(token_ids), "model_type": config.get("model_type"),
        "architectures": config.get("architectures"), "weight_file_count": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "model_inference_performed": False, "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, result)
    return result


def token_statistics_verified(model_path: str, verification_path: Path, output: Path,
                              *, version: str = "sft_v2.0.0") -> dict[str, Any]:
    verification = read_json(verification_path)
    if verification.get("status") != "verified":
        raise GuardError("model/tokenizer verification artifact is not verified")
    if Path(verification.get("local_model_path", "")).resolve() != Path(model_path).resolve():
        raise GuardError("tokenizer path differs from verified model snapshot")
    result = token_statistics(model_path, output, version=version)
    if result.get("status") != "completed":
        raise GuardError("verified cloud tokenizer statistics did not complete")
    result.update({
        "model_revision": verification["model_revision"],
        "tokenizer_identifier": verification["tokenizer_identifier"],
        "tokenizer_revision": verification["tokenizer_revision"],
        "model_verification_artifact": str(verification_path.resolve()),
        "model_verification_sha256": sha256(verification_path),
    })
    write_json(output, result)
    return result


def freeze_cutoff(statistics_path: Path, model_verification_path: Path, output: Path) -> dict[str, Any]:
    """Create the final cutoff artifact only from completed formal Train statistics."""

    if output.exists():
        raise GuardError(f"refusing to overwrite cutoff freeze artifact: {output}")
    stats, verification = read_json(statistics_path), read_json(model_verification_path)
    if stats.get("status") != "completed" or stats.get("dataset_version") != "sft_v2.0.0" or stats.get("split") != "train":
        raise GuardError("cutoff requires completed SFT_v2 Train statistics")
    if stats.get("total_tokens", {}).get("count") != 1871:
        raise GuardError("cutoff requires all 1871 SFT_v2 Train samples")
    if verification.get("status") != "verified":
        raise GuardError("cutoff requires verified model/tokenizer metadata")
    if stats.get("model_revision") != verification.get("model_revision") or stats.get("tokenizer_revision") != verification.get("tokenizer_revision"):
        raise GuardError("statistics and model/tokenizer revisions differ")
    p95 = float(stats["total_tokens"]["p95"])
    selected = 2048 if p95 <= 2048 else 4096
    if stats.get("recommended_cutoff_len") != selected:
        raise GuardError("statistics recommendation violates the frozen cutoff rule")
    dataset_checksums = DATASETS["sft_v2.0.0"] / "checksums.sha256"
    artifact = {
        "schema_version": "lora-v2-cutoff-freeze-v1.0.0", "status": "frozen",
        "dataset_version": "sft_v2.0.0", "dataset_checksums_sha256": sha256(dataset_checksums),
        "model_identifier": verification["model_identifier"], "model_revision": verification["model_revision"],
        "tokenizer_identifier": verification["tokenizer_identifier"], "tokenizer_revision": verification["tokenizer_revision"],
        "chat_template": "qwen3_nothink", "statistics_artifact": str(statistics_path.resolve()),
        "statistics_artifact_sha256": sha256(statistics_path), "train_total_token_p95": p95,
        "selected_cutoff_len": selected,
        "selection_rule": "2048 if SFT_v2 Train total-token P95 <= 2048, otherwise 4096",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, artifact)
    return artifact


def prepare_smoke_dataset(output_dir: Path, *, train_count: int = 32, validation_count: int = 8,
                          seed: str = "lora-v2.1-smoke-v1") -> dict[str, Any]:
    """Materialize small deterministic Train/Validation subsets; Test is unreachable."""

    if output_dir.exists():
        raise GuardError(f"refusing to overwrite smoke directory: {output_dir}")
    if not 1 <= train_count <= 64 or not 1 <= validation_count <= 32:
        raise GuardError("smoke subset bounds are Train 1..64 and Validation 1..32")
    validate_dataset("sft_v2.0.0")
    output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    for split, count in (("train", train_count), ("validation", validation_count)):
        source = split_path("sft_v2.0.0", split, purpose="gradient_training" if split == "train" else "model_selection")
        rows = sorted(iter_jsonl(source), key=lambda row: (
            hashlib.sha256(f"{seed}\0{row['sample_id']}".encode()).hexdigest(), row["sample_id"]))[:count]
        target = output_dir / f"smoke_{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        counts[split] = len(rows)
    manifest = {"schema_version": "lora-v2-smoke-data-v1.0.0", "status": "prepared_not_trained",
                "source_dataset": "sft_v2.0.0", "selection_seed": seed, "counts": counts,
                "included_splits": ["train", "validation"], "test_accessed": False}
    write_json(output_dir / "smoke_data_manifest.json", manifest)
    return manifest


def prepare_cloud_smoke(model_path: Path, model_verification_path: Path, cutoff_freeze_path: Path,
                        output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve the BF16 smoke config from verified cloud artifacts, never Test."""

    if output_dir.exists():
        raise GuardError(f"refusing to overwrite cloud smoke directory: {output_dir}")
    verification, cutoff = read_json(model_verification_path), read_json(cutoff_freeze_path)
    if verification.get("status") != "verified" or cutoff.get("status") != "frozen":
        raise GuardError("cloud smoke requires verified model and frozen cutoff artifacts")
    if cutoff.get("model_revision") != verification.get("model_revision"):
        raise GuardError("smoke model revision differs from cutoff freeze")
    if Path(verification.get("local_model_path", "")).resolve() != model_path.resolve():
        raise GuardError("smoke model path differs from verified snapshot")
    cutoff_len = cutoff.get("selected_cutoff_len")
    if cutoff_len not in (2048, 4096):
        raise GuardError("smoke cutoff must be formally frozen to 2048 or 4096")
    runtime_data = output_dir / "runtime_data"
    prepare_smoke_dataset(runtime_data, train_count=32, validation_count=8)
    dataset_info = {
        "smoke_train": {"file_name": str((runtime_data / "smoke_train.jsonl").resolve()), "formatting": "sharegpt",
                        "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                        "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
        "smoke_validation": {"file_name": str((runtime_data / "smoke_validation.jsonl").resolve()), "formatting": "sharegpt",
                             "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                             "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
    }
    write_json(runtime_data / "dataset_info.json", dataset_info)
    config = {
        "model_name_or_path": str(model_path.resolve()), "trust_remote_code": False, "stage": "sft", "do_train": True,
        "finetuning_type": "lora", "lora_target": "all", "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "dataset": "smoke_train", "eval_dataset": "smoke_validation", "dataset_dir": str(runtime_data.resolve()),
        "template": "qwen3_nothink", "cutoff_len": cutoff_len, "train_on_prompt": False, "packing": False,
        "max_samples": 32, "output_dir": str((output_dir / "adapter").resolve()), "overwrite_output_dir": False,
        "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 0.00005, "max_steps": 20, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
        "seed": 42, "bf16": True, "fp16": False, "gradient_checkpointing": True, "do_eval": True,
        "eval_strategy": "steps", "eval_steps": 10, "save_strategy": "steps", "save_steps": 10,
        "save_total_limit": 2, "logging_steps": 1, "report_to": "none", "plot_loss": False,
    }
    config_path = output_dir / "smoke.yaml"
    write_flat_yaml(config_path, config)
    plan = {
        "schema_version": "lora-v2-cloud-smoke-plan-v1.0.0", "status": "prepared_not_executed",
        "model_identifier": verification["model_identifier"], "model_revision": verification["model_revision"],
        "tokenizer_revision": verification["tokenizer_revision"], "model_path": str(model_path.resolve()),
        "cutoff_len": cutoff_len, "training_method": "standard_bf16_lora", "precision": "bf16",
        "max_steps": 20, "train_count": 32, "validation_count": 8, "test_accessed": False,
        "config": str(config_path.resolve()), "formal_experiment": False, "next_experiment_started": False,
    }
    write_json(output_dir / "smoke_plan.json", plan)
    return config_path, plan


def _gpu_memory_mib() -> tuple[int | None, int | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None
    completed = subprocess.run([executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=False)
    try:
        used, total = completed.stdout.strip().splitlines()[0].split(",")
        return int(used.strip()), int(total.strip())
    except (ValueError, IndexError):
        return None, None


def run_cloud_smoke(model_path: Path, model_verification_path: Path, cutoff_freeze_path: Path,
                    output_dir: Path, *, cli: str = "llamafactory-cli") -> dict[str, Any]:
    """Run the bounded BF16 smoke only; OOM blocks and never changes method/cutoff."""

    config_path, plan = prepare_cloud_smoke(model_path, model_verification_path, cutoff_freeze_path, output_dir)
    command = [cli, "train", str(config_path)]
    log_path = output_dir / "smoke_training.log"
    baseline, total = _gpu_memory_mib(); peak = baseline or 0
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            used, observed_total = _gpu_memory_mib()
            if used is not None: peak = max(peak, used)
            if observed_total is not None: total = observed_total
            time.sleep(.25)
        returncode = process.wait()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    common = dict(plan, command=command, returncode=returncode, duration_seconds=round(time.monotonic() - started, 3),
                  baseline_gpu_memory_mib=baseline, peak_gpu_memory_mib=peak, total_gpu_memory_mib=total,
                  log_path=str(log_path.resolve()))
    if returncode != 0:
        oom = "out of memory" in log_text.lower() or "cuda oom" in log_text.lower()
        result = dict(common, status="blocked_pending_training_method_decision" if oom else "failed",
                      oom_detected=oom, forward_succeeded=False, backward_succeeded=False, loss_finite=False,
                      grad_norm_finite=False, bf16_enabled=True, memory_acceptable=False, checkpoint_saved=False,
                      adapter_reloaded=False, inference_succeeded=False, output_format_valid=False,
                      test_accessed=False, next_experiment_started=False)
        write_json(output_dir / "smoke_result.json", result)
        raise GuardError(f"cloud smoke {'OOM; user method decision required' if oom else 'failed'}; see {log_path}")
    adapter_dir = output_dir / "adapter"
    state_paths = sorted(adapter_dir.rglob("trainer_state.json"))
    if not state_paths:
        raise GuardError("smoke command succeeded without trainer_state.json")
    state = read_json(state_paths[-1]); history = state.get("log_history", [])
    losses = [float(row["loss"]) for row in history if isinstance(row, dict) and isinstance(row.get("loss"), (int, float))]
    grad_norms = [float(row["grad_norm"]) for row in history if isinstance(row, dict) and isinstance(row.get("grad_norm"), (int, float))]
    checkpoint_saved = (adapter_dir / "adapter_config.json").is_file() and any(adapter_dir.glob("adapter_model.*"))
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",
                                                    local_files_only=True, trust_remote_code=False)
        model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True).eval()
        validation_row = next(iter(iter_jsonl(output_dir / "runtime_data" / "smoke_validation.jsonl")))
        prompt = tokenizer.apply_chat_template(validation_row["messages"][:-1], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        output_record = {"sample_id": validation_row["sample_id"], "prediction": prediction,
                         "model_revision": plan["model_revision"], "smoke": True}
        write_json(output_dir / "smoke_validation_output.json", output_record)
        adapter_reloaded = True; inference_succeeded = bool(prediction)
        output_format_valid = set(output_record) == {"sample_id", "prediction", "model_revision", "smoke"}
    except Exception as exc:
        adapter_reloaded = False; inference_succeeded = False; output_format_valid = False
        write_json(output_dir / "smoke_reload_failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
    result = dict(common, status="pending_validation", oom_detected=False, forward_succeeded=bool(losses),
                  backward_succeeded=bool(grad_norms), loss_finite=bool(losses) and all(math.isfinite(x) for x in losses),
                  grad_norm_finite=bool(grad_norms) and all(math.isfinite(x) for x in grad_norms), bf16_enabled=True,
                  memory_acceptable=bool(total and peak < total), checkpoint_saved=checkpoint_saved,
                  adapter_reloaded=adapter_reloaded, inference_succeeded=inference_succeeded,
                  output_format_valid=output_format_valid, test_accessed=False, next_experiment_started=False)
    required = ("forward_succeeded", "backward_succeeded", "loss_finite", "grad_norm_finite", "bf16_enabled",
                "memory_acceptable", "checkpoint_saved", "adapter_reloaded", "inference_succeeded", "output_format_valid")
    result["status"] = "passed" if all(result[field] is True for field in required) else "failed"
    write_json(output_dir / "smoke_result.json", result)
    validate_smoke_manifest(output_dir / "smoke_result.json")
    return result


def _percentile(values: Sequence[int], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GuardError("cannot summarize an empty sequence")
    position = (len(ordered) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: Sequence[int]) -> dict[str, Any]:
    count = len(values)
    le2048 = sum(value <= 2048 for value in values)
    le4096 = sum(value <= 4096 for value in values)
    return {"count": count, "mean": statistics.fmean(values), "median": statistics.median(values),
            "p90": _percentile(values, .90), "p95": _percentile(values, .95), "p99": _percentile(values, .99),
            "max": max(values), "le_2048_count": le2048, "le_2048_ratio": le2048 / count,
            "gt_2048_count": count - le2048, "gt_2048_ratio": (count - le2048) / count,
            "le_4096_count": le4096, "le_4096_ratio": le4096 / count,
            "gt_4096_count": count - le4096, "gt_4096_ratio": (count - le4096) / count}


def zh_tokens(text: str) -> list[str]:
    """CJK characters plus lower-cased contiguous ASCII alphanumeric words."""

    return [part.lower() for part in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+", text)]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1)))


def _lcs(left: Sequence[str], right: Sequence[str]) -> int:
    row = [0] * (len(right) + 1)
    for l_token in left:
        previous = 0
        for index, r_token in enumerate(right, 1):
            old = row[index]
            row[index] = previous + 1 if l_token == r_token else max(row[index], row[index - 1])
            previous = old
    return row[-1]


def evaluate_pairs(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Deterministic corpus BLEU-4 and macro ROUGE F1 for Chinese text."""

    if not pairs:
        raise GuardError("no prediction/reference pairs")
    matches = [0, 0, 0, 0]; totals = [0, 0, 0, 0]
    ref_length = pred_length = 0
    rouge: dict[str, list[float]] = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for prediction, reference in pairs:
        pred, ref = zh_tokens(prediction), zh_tokens(reference)
        pred_length += len(pred); ref_length += len(ref)
        for n in range(1, 5):
            pred_grams, ref_grams = _ngrams(pred, n), _ngrams(ref, n)
            matches[n - 1] += sum((pred_grams & ref_grams).values())
            totals[n - 1] += sum(pred_grams.values())
        for name, overlap, pred_den, ref_den in (
            ("ROUGE-1", sum((_ngrams(pred, 1) & _ngrams(ref, 1)).values()), len(pred), len(ref)),
            ("ROUGE-2", sum((_ngrams(pred, 2) & _ngrams(ref, 2)).values()), max(0, len(pred)-1), max(0, len(ref)-1)),
            ("ROUGE-L", _lcs(pred, ref), len(pred), len(ref)),
        ):
            precision = overlap / pred_den if pred_den else 0.0
            recall = overlap / ref_den if ref_den else 0.0
            rouge[name].append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    # Corpus BLEU, add-one smoothing on n-gram orders 1..4.
    precisions = [(match + 1) / (total + 1) for match, total in zip(matches, totals)]
    brevity = 0.0 if pred_length == 0 else min(1.0, math.exp(1 - ref_length / pred_length))
    bleu = brevity * math.exp(sum(math.log(value) for value in precisions) / 4)
    return {"BLEU-4": bleu, **{name: statistics.fmean(values) for name, values in rouge.items()},
            "manifest": {"implementation": METRIC_IMPLEMENTATION, "implementation_version": "1.0.0",
                         "tokenization": "each CJK character; lower-case contiguous ASCII alphanumeric word; punctuation/space removed",
                         "bleu": "corpus modified precision BLEU-4, uniform weights, add-one smoothing",
                         "rouge": "macro-average F1; multiset overlap for ROUGE-1/2; LCS for ROUGE-L",
                         "python": platform.python_version(), "count": len(pairs)}}


def evaluate_predictions(path: Path, output: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    required = {"sample_id", "prediction", "reference", "run_id", "checkpoint"}
    if any(not required <= row.keys() for row in rows):
        raise GuardError("prediction rows are missing required metadata")
    result = evaluate_pairs([(str(row["prediction"]), str(row["reference"])) for row in rows])
    write_json(output, result)
    return result


def build_comparison_safe_mask(output: Path, *, judge_count: int = 50, seed: str = "lora-v2.1-safe-judge-v1") -> dict[str, Any]:
    """Exclude v2 Test documents exposed to v1 Train/Validation; perform no inference."""

    v1_test_path = split_path("sft_v1.0.0", "test", purpose="comparison_mask_metadata")
    v2_path = split_path("sft_v2.0.0", "test", purpose="comparison_mask_metadata")
    v1_test_docs = {row["document_id"] for row in iter_jsonl(v1_test_path)}
    # Metadata-only governance access: a fair v1/v2 comparison must exclude any
    # v2 Test item that the v1 adapter saw in Train or Validation. New v2 Silver
    # items were absent from v1 and are therefore comparison-safe.
    v1_exposed_docs = {
        row["document_id"]
        for split in ("train", "validation")
        for row in iter_jsonl(DATASETS["sft_v1.0.0"] / f"{split}.jsonl")
    }
    v2_rows = list(iter_jsonl(v2_path))
    safe = sorted((row["sample_id"], row["document_id"]) for row in v2_rows if row["document_id"] not in v1_exposed_docs)
    excluded = sorted(row["sample_id"] for row in v2_rows if row["document_id"] in v1_exposed_docs)
    ranked = sorted(safe, key=lambda item: (hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest(), item[0]))
    judges = {item[0] for item in ranked[:min(judge_count, len(ranked))]}
    rows = [{"sample_id": sample_id, "document_id": document_id, "comparison_safe": True,
             "judge_sample": sample_id in judges} for sample_id, document_id in safe]
    result = {"schema_version": "comparison-safe-test-mask-v1.0.0", "status": "built_metadata_only",
              "v1_test_count": len(v1_test_docs), "v1_train_validation_exposure_count": len(v1_exposed_docs),
              "v2_test_count": len(v2_rows), "safe_count": len(rows), "excluded_count": len(excluded),
              "excluded_sample_ids": excluded,
              "judge_count": len(judges), "selection_seed": seed, "rows": rows,
              "model_inference_performed": False, "test_evaluation_performed": False}
    write_json(output, result)
    return result


def parse_trainer_log(path: Path) -> list[dict[str, Any]]:
    """Parse JSON/JSONL trainer logs and add comparable exposure axes."""

    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        rows = list(iter_jsonl(path))
    else:
        rows = document.get("log_history", []) if isinstance(document, dict) else []
    for row in rows:
        if "step" not in row:
            continue
        effective_batch = int(row.get("effective_batch", 0))
        enriched = dict(row)
        if effective_batch:
            enriched["examples_seen"] = int(row["step"]) * effective_batch
        if "epoch" in row:
            enriched["epoch_progress"] = float(row["epoch"])
        records.append(enriched)
    return records


def aggregate_results(run_root: Path, output: Path) -> list[dict[str, Any]]:
    manifests = sorted(run_root.glob("*/*/run_manifest.json")) if run_root.exists() else []
    rows: list[dict[str, Any]] = []
    for path in manifests:
        manifest = read_json(path)
        metrics_path = path.parent / "evaluation_manifest.json"
        metrics = read_json(metrics_path) if metrics_path.is_file() else {}
        row = {field: metrics.get(field, manifest.get(field, "")) for field in RESULT_FIELDS}
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return rows


def plot_results(experiments_csv: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise GuardError("matplotlib is required for plotting") from exc
    with experiments_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise GuardError("no real experiment results to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for field, label in (("learning_rate", "LR"), ("rank", "Rank"), ("effective_batch", "Batch")):
        points = [(float(row[field]), float(row["best_eval_loss"])) for row in rows if row.get(field) and row.get("best_eval_loss")]
        if len(points) < 2:
            continue
        points.sort(); fig, axis = plt.subplots(); axis.plot([p[0] for p in points], [p[1] for p in points], marker="o")
        axis.set(xlabel=label, ylabel="Best validation loss", title=f"{label} comparison")
        target = output_dir / f"{field}_comparison.png"; fig.savefig(target, dpi=160, bbox_inches="tight"); plt.close(fig)
        generated.append(target)
    metric_rows = [row for row in rows if row.get("BLEU-4") or row.get("ROUGE-L")]
    if metric_rows:
        labels = [row.get("run_id") or row.get("experiment_id") or "unknown" for row in metric_rows]
        fig, axis = plt.subplots(figsize=(max(6, len(labels) * .8), 4))
        x = list(range(len(labels)))
        width = .35
        bleu = [float(row.get("BLEU-4") or 0) for row in metric_rows]
        rouge_l = [float(row.get("ROUGE-L") or 0) for row in metric_rows]
        axis.bar([value - width/2 for value in x], bleu, width, label="BLEU-4")
        axis.bar([value + width/2 for value in x], rouge_l, width, label="ROUGE-L")
        axis.set_xticks(x, labels, rotation=30, ha="right"); axis.set_ylabel("Score"); axis.legend()
        axis.set_title("Base / sweep / final comparison")
        target = output_dir / "model_metric_comparison.png"; fig.savefig(target, dpi=160, bbox_inches="tight"); plt.close(fig)
        generated.append(target)
    return generated


def plot_loss_curves(trainer_state: Path, output: Path, *, effective_batch: int) -> Path:
    """Plot train/eval loss against comparable examples_seen (with raw steps retained)."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise GuardError("matplotlib is required for plotting") from exc
    text = trainer_state.read_text(encoding="utf-8")
    document = json.loads(text)
    rows = document.get("log_history", [])
    train = [(int(row["step"]) * effective_batch, float(row["loss"])) for row in rows if "step" in row and "loss" in row]
    validation = [(int(row["step"]) * effective_batch, float(row["eval_loss"])) for row in rows if "step" in row and "eval_loss" in row]
    if not train and not validation:
        raise GuardError("trainer state contains no loss records")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots()
    if train: axis.plot([x for x, _ in train], [y for _, y in train], label="training loss")
    if validation: axis.plot([x for x, _ in validation], [y for _, y in validation], marker="o", label="validation loss")
    axis.set(xlabel="examples_seen (raw step × effective batch)", ylabel="Loss", title="Training / Validation loss")
    axis.legend(); fig.savefig(output, dpi=160, bbox_inches="tight"); plt.close(fig)
    return output


def validation_inference_plan(run_manifest: Path, output_dir: Path, *, split: str = "validation",
                              authorization_path: Path | None = None) -> dict[str, Any]:
    manifest = read_json(run_manifest)
    version = manifest["dataset_version"]
    purpose = "validation_inference" if split == "validation" else "final_evaluation"
    source = split_path(version, split, purpose=purpose, authorization_path=authorization_path)
    if split == "validation" and manifest.get("training_enabled", True) and not manifest.get("best_checkpoint"):
        raise GuardError("best Validation checkpoint must be recorded before generation")
    if output_dir.exists():
        raise GuardError(f"refusing to overwrite inference output: {output_dir}")
    decoding = {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "num_beams": 1,
                "max_new_tokens": 2048, "repetition_penalty": 1.0}
    model_source, _ = _inference_model_location(manifest)
    return {"status": "prepared_not_executed", "split": split, "dataset_file": str(source),
            "run_id": manifest["run_id"], "checkpoint": manifest.get("best_checkpoint"),
            "model_name": manifest["model_name"], "model_revision": manifest["model_revision"],
            "local_model_path": manifest.get("local_model_path"),
            "model_verification_artifact": manifest.get("model_verification_artifact"),
            "model_load_source": "verified_local_snapshot" if model_source != manifest["model_name"] else "official_identifier_revision",
            "chat_template": manifest["chat_template"],
            "decoding": decoding, "prediction_schema": ["sample_id", "reference", "prediction", "model_name", "run_id", "checkpoint"],
            "output": str(output_dir / "predictions.jsonl")}


def _inference_model_location(manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    local_value = manifest.get("local_model_path")
    if not local_value:
        return manifest["model_name"], {"revision": manifest["model_revision"]}
    artifact_value = manifest.get("model_verification_artifact")
    artifact_hash = manifest.get("model_verification_sha256")
    if not artifact_value or not artifact_hash:
        raise GuardError("local model path requires bound model verification evidence")
    artifact_path = Path(artifact_value)
    if not artifact_path.is_file() or sha256(artifact_path) != artifact_hash:
        raise GuardError("bound model verification artifact is missing or changed")
    verification = read_json(artifact_path)
    expected = {
        "status": "verified", "model_identifier": manifest["model_name"],
        "model_revision": manifest["model_revision"], "local_model_path": local_value,
        "chat_template": manifest["chat_template"], "model_inference_performed": False,
    }
    if any(verification.get(key) != value for key, value in expected.items()):
        raise GuardError("local model path no longer matches its verification artifact")
    local_path = Path(local_value)
    if not local_path.is_dir():
        raise GuardError(f"verified local model snapshot is unavailable: {local_path}")
    return str(local_path), {"local_files_only": True}


def run_validation_inference(run_manifest: Path, output_dir: Path, *, split: str = "validation",
                             authorization_path: Path | None = None) -> dict[str, Any]:
    """Run an explicitly requested generation job using the frozen rendering policy."""

    plan = validation_inference_plan(run_manifest, output_dir, split=split, authorization_path=authorization_path)
    manifest = read_json(run_manifest)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "inference_manifest.json", dict(plan, status="running"))
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        model_source, load_kwargs = _inference_model_location(manifest)
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.bfloat16, device_map="auto",
                                                     trust_remote_code=False, **load_kwargs)
        checkpoint = manifest.get("best_checkpoint")
        if checkpoint:
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(model, checkpoint)
        source = Path(plan["dataset_file"])
        predictions = output_dir / "predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as handle:
            for row in iter_jsonl(source):
                prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **plan["decoding"])
                continuation = generated[0, inputs["input_ids"].shape[1]:]
                prediction = tokenizer.decode(continuation, skip_special_tokens=True)
                record = {"sample_id": row["sample_id"], "reference": row["target_text"], "prediction": prediction,
                          "model_name": manifest["model_name"], "model_revision": manifest["model_revision"],
                          "run_id": manifest["run_id"], "checkpoint": checkpoint or "base_model"}
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        finished = dict(plan, status="completed", prediction_count=sum(1 for _ in iter_jsonl(predictions)))
        write_json(output_dir / "inference_manifest.json", finished)
        return finished
    except Exception as exc:
        write_json(output_dir / "inference_manifest.json", dict(plan, status="failed", error_type=type(exc).__name__))
        raise


def validate_smoke_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    required_true = ("forward_succeeded", "backward_succeeded", "loss_finite", "grad_norm_finite", "bf16_enabled",
                     "memory_acceptable", "checkpoint_saved", "adapter_reloaded", "inference_succeeded", "output_format_valid")
    errors = [field for field in required_true if manifest.get(field) is not True]
    if manifest.get("test_accessed") is not False:
        errors.append("test_accessed")
    if errors:
        raise GuardError(f"smoke manifest failed fields: {errors}")
    return {"status": "passed", "checked": list(required_true)}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {"schema_version": "lora-cloud-environment-v1.0.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": platform.python_version(), "git_commit": _git_commit(),
        "packages": {name: _package_version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "llamafactory")}}
    disk = shutil.disk_usage(ROOT)
    manifest["disk"] = {"path": str(ROOT), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}
    source_version_path = ROOT / "CLOUD_SOURCE_VERSION.json"
    if source_version_path.is_file():
        manifest["bundled_source_version"] = read_json(source_version_path)
    try:
        import torch  # type: ignore
        cuda = bool(torch.cuda.is_available())
        manifest["pytorch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": cuda,
                               "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported())}
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            manifest["gpu"] = {"name": properties.name, "vram_bytes": properties.total_memory}
    except ImportError:
        manifest["pytorch"] = {"status": "not_installed", "cuda_available": False, "bf16_supported": False}
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        result = subprocess.run([nvidia, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=False)
        manifest["nvidia_smi"] = {"returncode": result.returncode, "summary": result.stdout.strip()}
    else:
        manifest["nvidia_smi"] = {"status": "not_found"}
    return manifest
