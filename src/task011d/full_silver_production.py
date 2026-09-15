from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.task011d.autonomous_sft import json_text, read_json, sha256_file
from src.task011d.high_semantic_pilot import _risk_features, _source_view, load_context
from src.task011d.semantic_provider import CodexExecSemanticProvider, SemanticProviderError, create_semantic_provider
from src.task011d.silver_production_pilot import (
    AUDIT_ITEM_SCHEMA,
    AUDIT_SYSTEM,
    CORE_ISSUES,
    GENERATION_ITEM_SCHEMA,
    GENERATION_SYSTEM,
    REVIEW_ITEM_SCHEMA,
    REVIEW_SYSTEM,
    REVISION_SYSTEM,
    REREVIEW_SYSTEM,
    SOFT_WARNINGS,
    _batch_schema,
    _materialize,
    _source_candidate,
    deterministic_validate,
    standard_document,
)


SPLITS = ("train", "validation", "test")
TERMINAL = {"accepted", "accepted_with_warning", "dropped"}
HARD_AUDIT_VERDICTS = {"MAJOR", "BLOCKING"}
AUDIT_TO_REVIEW = {
    "core_factual_error": "core_fact_error",
    "core_unsupported_fact": "evidence_error",
    "evidence_error": "evidence_error",
    "entity_error": "entity_number_date_error",
    "entity_relation_error": "entity_number_date_error",
    "number_date_error": "entity_number_date_error",
    "material_coverage_gap": "material_coverage_gap",
    "severe_inference": "severe_inference",
    "severe_structure_error": "severe_structure_error",
    "target_integrity_error": "severe_structure_error",
    "schema_error": "severe_structure_error",
    "provenance_error": "severe_structure_error",
    "cross_source_contamination": "evidence_error",
    "major_overfragmentation": "severe_structure_error",
    "policy_hard_failure": "severe_structure_error",
}


class FullSilverProductionError(RuntimeError):
    pass


class ProviderCapacityPause(FullSilverProductionError):
    pass


class ProviderRuntimePause(FullSilverProductionError):
    pass


class DeferredSemanticTimeoutPause(FullSilverProductionError):
    pass


CAPACITY_CLASSES = {"explicit_rate_limit", "explicit_capacity", "quota", "authentication"}
RUNTIME_FAILURE_CLASSES = {"timeout", *CAPACITY_CLASSES, "parse", "child_process_error", "unknown_provider_error"}


def classify_provider_failure(exc: BaseException) -> str:
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "explicit_rate_limit" in text or "rate limit" in text or "rate_limit" in text or "429" in text:
        return "explicit_rate_limit"
    if "quota" in text or "credit exhausted" in text or "credits exhausted" in text:
        return "quota"
    if "explicit_capacity" in text or "capacity" in text or "overloaded" in text:
        return "explicit_capacity"
    if "authentication" in text or "unauthorized" in text or "login required" in text:
        return "authentication"
    if "invalid" in text or "structured output" in text or "parse" in text:
        return "parse"
    if "exit code" in text or "child" in text or "tool" in text:
        return "child_process_error"
    return "unknown_provider_error"


class RuntimeTimeoutGovernor:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.config = config
        self.path = root / config["runtime_timeout_governor"]
        self.failure_log = root / config["runtime_failure_log"]
        self.queue_path = root / config["deferred_timeout_queue"]
        self.lock = threading.Lock()
        if self.path.exists():
            self.data = read_json(self.path)
        else:
            invocation_path = root / config["invocation_log"]
            rows = _read_jsonl(invocation_path) if invocation_path.exists() else []
            timeouts = [row for row in rows if row.get("error_type") == "TimeoutExpired"]
            explicit = [row for row in rows if row.get("provider_failure_class") in CAPACITY_CLASSES]
            self.data = {
                "schema_version": "task011d-e5-runtime-timeout-governor-v1.0.0",
                "default_timeout": int(config["semantic_timeout_seconds"]),
                "split_retry_timeout": int(config["split_retry_timeout_seconds"]),
                "single_sample_timeout": int(config["single_sample_timeout_seconds"]),
                "timeout_batch_count": len(timeouts), "split_batch_count": 0, "single_sample_retry_count": 0,
                "deferred_queue_count": 0, "explicit_capacity_pause_count": len(explicit),
                "failure_class_counts": dict(Counter("timeout" if row.get("error_type") == "TimeoutExpired" else row.get("provider_failure_class", "unknown_provider_error") for row in rows if row.get("status") == "failed")),
                "current_concurrency": 1 if timeouts else int(config["max_concurrent_semantic_workers"]),
                "maximum_concurrency": int(config["max_concurrent_semantic_workers"]),
                "normal_success_streak": 0, "recovery_success_window": int(config["concurrency_recovery_success_window"]),
                "legacy_timeout_pause_count_migrated": len(timeouts), "updated_at": _now(),
            }
        if self.queue_path.exists():
            self.queue = read_json(self.queue_path)
        else:
            self.queue = {"schema_version": "task011d-e5-deferred-timeout-queue-v1.0.0", "samples": {}}
        self._save()

    @property
    def concurrency(self) -> int:
        return max(1, min(2, int(self.data.get("current_concurrency", 1))))

    def _save(self) -> None:
        self.data["deferred_queue_count"] = len(self.queue["samples"])
        self.data["updated_at"] = _now()
        _write_json(self.path, self.data)
        _write_json(self.queue_path, self.queue)

    def record_success(self) -> None:
        with self.lock:
            self.data["normal_success_streak"] = int(self.data.get("normal_success_streak", 0)) + 1
            if self.data["normal_success_streak"] >= int(self.data["recovery_success_window"]):
                self.data["current_concurrency"] = min(2, int(self.data["maximum_concurrency"]))
            self._save()

    def record_failure(self, *, failure_class: str, phase: str, sample_ids: list[str], timeout_seconds: int, invocation_ids: list[str]) -> None:
        _require(failure_class in RUNTIME_FAILURE_CLASSES, f"unknown runtime failure class: {failure_class}")
        with self.lock:
            counts = Counter(self.data.get("failure_class_counts", {})); counts[failure_class] += 1
            self.data["failure_class_counts"] = dict(counts)
            self.data["normal_success_streak"] = 0
            if failure_class == "timeout":
                self.data["timeout_batch_count"] = int(self.data.get("timeout_batch_count", 0)) + 1
                self.data["current_concurrency"] = 1
            elif failure_class in CAPACITY_CLASSES:
                self.data["explicit_capacity_pause_count"] = int(self.data.get("explicit_capacity_pause_count", 0)) + 1
            else:
                self.data["current_concurrency"] = 1
            record = {"recorded_at": _now(), "failure_class": failure_class, "phase": phase, "sample_ids": sample_ids, "timeout_seconds": timeout_seconds, "invocation_ids": invocation_ids}
            self.failure_log.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_log.open("a", encoding="utf-8", newline="") as handle:
                handle.write(_canonical(record) + "\n")
            self._save()

    def record_split(self) -> None:
        with self.lock:
            self.data["split_batch_count"] = int(self.data.get("split_batch_count", 0)) + 1; self._save()

    def record_single_retry(self) -> None:
        with self.lock:
            self.data["single_sample_retry_count"] = int(self.data.get("single_sample_retry_count", 0)) + 1; self._save()

    def defer(self, sample_id: str, history: list[dict[str, Any]]) -> None:
        with self.lock:
            self.queue["samples"][sample_id] = {"sample_id": sample_id, "status": "deferred_semantic_timeout", "attempts": len(history), "timeout_history": history, "invocation_ids": [value for row in history for value in row.get("invocation_ids", [])], "last_error": "semantic_call_timeout", "updated_at": _now()}
            self._save()

    def resolve(self, sample_id: str) -> None:
        with self.lock:
            self.queue["samples"].pop(sample_id, None); self._save()


def _require(value: bool, message: str) -> None:
    if not value:
        raise FullSilverProductionError(message)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(5):
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(path)
            return
        except OSError as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))
        finally:
            if temporary.exists():
                temporary.unlink()
    if last_error is not None:  # pragma: no cover - loop either returns or raises
        raise last_error


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json_text(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_canonical(row) + "\n" for row in rows))


def _provider(root: Path, config: dict[str, Any], effort: str, timeout_seconds: int | None = None) -> CodexExecSemanticProvider:
    provider_config = dict(config)
    provider_config["reasoning_effort"] = effort
    if timeout_seconds is not None:
        provider_config["semantic_timeout_seconds"] = timeout_seconds
    provider = create_semantic_provider(provider_config, root=root)
    _require(isinstance(provider, CodexExecSemanticProvider), "CodexExec provider required")
    return provider


def _matching_failure_invocation_ids(root: Path, config: dict[str, Any], sample_key: str, failure_class: str) -> list[str]:
    path = root / config["invocation_log"]
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    matched = []
    for row in rows:
        row_class = "timeout" if row.get("error_type") == "TimeoutExpired" else row.get("provider_failure_class")
        if row.get("sample_id") == sample_key and row.get("status") == "failed" and row_class == failure_class:
            matched.append(row.get("invocation_id"))
    return [value for value in matched[-1:] if value]


def build_samples(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(context["silver_rows"], key=lambda row: (SPLITS.index(row["split"]), row["article_id"]))
    result = []
    for index, row in enumerate(rows, 1):
        article = context["articles"][row["article_id"]]
        result.append({
            "sample_id": f"task011d_e5_{index:04d}",
            "article_id": row["article_id"],
            "split": row["split"],
            "event_group_id": row["event_group_id"],
            "source_ref": row["source_ref"],
            "source_sha256": row["content_sha256"],
            "risk_features": _risk_features(article),
        })
    return result


def _batches(samples: list[dict[str, Any]], articles: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    characters = 0
    for sample in samples:
        article = articles[sample["sample_id"]]
        size = len(article["title"]) + len(article["body"])
        if current and (len(current) >= int(config["batch_size"]) or characters + size > int(config["maximum_batch_source_characters"])):
            result.append(current)
            current, characters = [], 0
        current.append(sample)
        characters += size
    if current:
        result.append(current)
    return result


def _initial_state(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: {
        "article_id": row["article_id"], "split": row["split"], "status": "pending_generation", "updated_at": _now(),
        "generated": False, "reviewed": False, "fix": False, "revised": False, "re_reviewed": False,
    } for row in samples}


def _mark_phase_state(state: dict[str, Any], status: str) -> None:
    state["status"] = status
    if status == "generated":
        state["generated"] = True
    elif status == "revised":
        state["generated"] = state["reviewed"] = state["fix"] = state["revised"] = True
    elif status == "re_reviewed":
        state["generated"] = state["reviewed"] = state["fix"] = state["revised"] = state["re_reviewed"] = True


def _load_run_meta(runtime: Path, resume: bool) -> dict[str, Any]:
    path = runtime / "run_metadata.json"
    if path.exists():
        meta = read_json(path)
        if resume:
            meta["resume_count"] = int(meta.get("resume_count", 0)) + 1
            meta["last_resumed_at"] = _now()
        return meta
    invocation_path = runtime / "invocations.jsonl"
    invocations = _read_jsonl(invocation_path) if invocation_path.exists() else []
    started_at = invocations[0].get("started_at", _now()) if invocations else _now()
    elapsed = 0.0
    if invocations:
        try:
            elapsed = max(0.0, (datetime.fromisoformat(invocations[-1]["finished_at"]) - datetime.fromisoformat(started_at)).total_seconds())
        except (KeyError, TypeError, ValueError):
            elapsed = 0.0
    return {
        "started_at": started_at, "resume_count": 1 if resume else 0,
        "active_wall_time_seconds": round(elapsed, 3),
        "provider_pause_count": sum(row.get("status") == "failed" for row in invocations),
        "recovered_after_interrupted_pause": bool(invocations),
    }


def progress_document(
    states: dict[str, dict[str, Any]],
    high_audited: set[str],
    pipeline_status: str,
    *,
    governor_data: dict[str, Any] | None = None,
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in states.values())
    governor_data = governor_data or {}
    failure_counts = Counter(governor_data.get("failure_class_counts", {}))
    return {
        "schema_version": "task011d-e5-progress-v1.0.0",
        "task_id": "TASK-011D-E5",
        "pipeline_status": pipeline_status,
        "updated_at": _now(),
        "total": len(states),
        "generated": sum(bool(row.get("generated")) for row in states.values()),
        "reviewed": sum(bool(row.get("reviewed")) for row in states.values()),
        "fix": sum(bool(row.get("fix")) for row in states.values()),
        "revised": sum(bool(row.get("revised")) for row in states.values()),
        "re_reviewed": sum(bool(row.get("re_reviewed")) for row in states.values()),
        "accepted": statuses["accepted"],
        "warning": statuses["accepted_with_warning"],
        "accepted_with_warning": statuses["accepted_with_warning"],
        "dropped": statuses["dropped"],
        "high_audited": len(high_audited),
        "unfinished": sum(row["status"] not in TERMINAL for row in states.values()),
        "status_counts": dict(statuses),
        "provider_pauses": int(governor_data.get("explicit_capacity_pause_count", 0)),
        "runtime_errors": sum(count for name, count in failure_counts.items() if name != "timeout" and name not in CAPACITY_CLASSES),
        "timeouts": int(governor_data.get("timeout_batch_count", failure_counts["timeout"])),
        "resume_count": int((run_meta or {}).get("resume_count", 0)),
    }


def _checkpoint(runtime: Path, states: dict[str, dict[str, Any]], high_audited: set[str], pipeline_status: str) -> None:
    _write_json(runtime / "sample_states.json", states)
    governor_data = read_json(runtime / "runtime_timeout_governor.json") if (runtime / "runtime_timeout_governor.json").exists() else {}
    run_meta = read_json(runtime / "run_metadata.json") if (runtime / "run_metadata.json").exists() else {}
    _write_json(
        runtime / "pipeline_progress.json",
        progress_document(states, high_audited, pipeline_status, governor_data=governor_data, run_meta=run_meta),
    )


PROVIDER_HEALTH_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["ok"]}},
    "required": ["status"],
    "additionalProperties": False,
}


def validate_resume_integrity(root: Path, config: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fail closed before a resume if formal generation lineage has drifted."""
    runtime = root / config["output_dir"]
    phase_dir = runtime / "semantic_batches" / "generation"
    output_ids: list[str] = []
    output_hashes: dict[str, str] = {}
    batch_hashes: dict[str, str] = {}
    for path in sorted(phase_dir.glob("batch_*.json")):
        payload = read_json(path)
        results = payload.get("results")
        _require(isinstance(results, dict), f"invalid generation checkpoint: {path.name}")
        batch_hashes[path.name] = sha256_file(path)
        for sample_id, candidate in results.items():
            _require(sample_id in states, f"unknown generation output: {sample_id}")
            output_ids.append(sample_id)
            output_hashes[sample_id] = _sha(candidate)
    _require(len(output_ids) == len(set(output_ids)), "duplicate generation output detected")
    generated_state_ids = {sample_id for sample_id, state in states.items() if state.get("generated")}
    _require(generated_state_ids == set(output_ids), "generation output/checkpoint state mismatch")

    invocation_path = root / config["invocation_log"]
    invocations = _read_jsonl(invocation_path) if invocation_path.exists() else []
    formal = [row for row in invocations if row.get("parent_task") == config["task_id"]]
    invocation_ids = [row.get("invocation_id") for row in formal]
    _require(all(invocation_ids), "formal invocation without invocation_id")
    _require(len(invocation_ids) == len(set(invocation_ids)), "duplicate formal invocation_id detected")
    for row in formal:
        if row.get("status") == "succeeded":
            _require(bool(row.get("input_hash") or row.get("input_sha256")), "successful invocation missing input hash")
            _require(bool(row.get("output_hash") or row.get("output_sha256")), "successful invocation missing output hash")

    selection_path = runtime / "full_production_selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        _require(selection["formal_input"] == config["generation_input"], "formal input path changed")
        _require(selection["formal_input_sha256"] == sha256_file(root / config["generation_input"]), "formal input checksum changed")
    return {
        "checked_at": _now(),
        "generated_count": len(output_ids),
        "duplicate_output_count": len(output_ids) - len(set(output_ids)),
        "batch_count": len(batch_hashes),
        "batch_files_sha256": _sha(batch_hashes),
        "candidate_payloads_sha256": _sha(output_hashes),
        "formal_invocation_count": len(formal),
        "formal_invocation_ids_sha256": _sha(invocation_ids),
        "formal_input_sha256": sha256_file(root / config["generation_input"]),
        "checkpoint_health": "PASS",
    }


def run_provider_health_smoke(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run one synthetic CodexExec call without touching formal candidates or invocations."""
    runtime = root / config["output_dir"]
    health_config = dict(config)
    health_config["task_id"] = "TASK-011D-E5-PROVIDER-HEALTH"
    health_config["invocation_log"] = config["provider_health_invocation_log"]
    timeout_seconds = int(config.get("provider_health_timeout_seconds", 300))
    health_log = root / health_config["invocation_log"]
    before_count = len(_read_jsonl(health_log)) if health_log.exists() else 0
    failure_class = None
    error = None
    output: dict[str, Any] | None = None
    try:
        provider = _provider(root, health_config, config["production_reasoning_effort"], timeout_seconds)
        output = provider._invoke(
            role="SILVER_PRODUCTION_PROVIDER_HEALTH",
            sample_id="task011d_e5_provider_health_synthetic",
            split="health_check",
            prompt_version="task011d-e5-provider-health-v1",
            system_prompt="Return status ok. This is a synthetic provider runtime health check, not a dataset sample.",
            user_payload={"probe": "provider_runtime_health"},
            schema_name="provider_health",
            schema=PROVIDER_HEALTH_SCHEMA,
        )
    except (SemanticProviderError, TimeoutError, OSError) as exc:
        failure_class = classify_provider_failure(exc)
        error = str(exc)
    rows = _read_jsonl(health_log) if health_log.exists() else []
    new_rows = rows[before_count:]
    record = new_rows[-1] if new_rows else {}
    passed = (
        output == {"status": "ok"}
        and record.get("status") == "succeeded"
        and record.get("exit_code") == 0
        and record.get("structured_output_status") == "passed"
        and bool(record.get("invocation_id"))
        and not record.get("provider_failure_class")
    )
    summary = {
        "schema_version": "task011d-e5-provider-health-v1.0.0",
        "checked_at": _now(),
        "provider_runtime_health": "restored" if passed else "unavailable",
        "passed": passed,
        "formal_candidate_pollution": False,
        "formal_invocation_pollution": False,
        "model": config["semantic_model"],
        "reasoning_effort": config["production_reasoning_effort"],
        "timeout_seconds": timeout_seconds,
        "invocation_id": record.get("invocation_id"),
        "child_started": bool(record.get("started_at")),
        "exit_code": record.get("exit_code"),
        "structured_output_status": record.get("structured_output_status", "not_recorded"),
        "failure_class": record.get("provider_failure_class") or failure_class,
        "error": error,
        "cli_version": config.get("codex_cli_version", "unknown"),
        "authentication_status": config.get("authentication_status", "unknown"),
        "safe_command_arguments": record.get("safe_command_arguments", []),
    }
    _write_json(runtime / "provider_health_summary.json", summary)
    return summary


def _invoke_phase(
    provider: CodexExecSemanticProvider,
    *,
    phase: str,
    samples: list[dict[str, Any]],
    articles: dict[str, dict[str, Any]],
    config: dict[str, Any],
    runtime: Path,
    system: str,
    item_schema: dict[str, Any],
    payload_builder: Callable[[dict[str, Any]], dict[str, Any]],
    states: dict[str, dict[str, Any]],
    completed_status: str | None,
    high_audited: set[str],
    resume: bool,
    governor: RuntimeTimeoutGovernor | None = None,
    provider_factory: Callable[[int], CodexExecSemanticProvider] | None = None,
) -> tuple[dict[str, Any], int]:
    phase_dir = runtime / "semantic_batches" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    calls = 0
    missing: list[tuple[int, list[dict[str, Any]], Path]] = []
    requested_ids = {row["sample_id"] for row in samples}
    existing_paths = sorted(phase_dir.glob("batch_*.json"))
    if resume and existing_paths:
        seen_checkpoint_ids: set[str] = set()
        for path in existing_paths:
            output = read_json(path)
            ids = set(output.get("results", {}))
            _require(not (ids & seen_checkpoint_ids), f"{phase} duplicate sample output across checkpoints")
            seen_checkpoint_ids.update(ids)
            reusable_ids = ids & requested_ids
            outputs.update({sample_id: output["results"][sample_id] for sample_id in reusable_ids})
            if completed_status:
                for sample_id in reusable_ids:
                    _mark_phase_state(states[sample_id], completed_status)
            if "audit" in phase:
                high_audited.update(ids)
        missing_samples = [row for row in samples if row["sample_id"] not in outputs]
        next_index = max(int(path.stem.split("_")[-1]) for path in existing_paths) + 1
        for offset, batch in enumerate(_batches(missing_samples, articles, config)):
            index = next_index + offset
            missing.append((index, batch, phase_dir / f"batch_{index:04d}.json"))
    else:
        batches = _batches(samples, articles, config)
        for index, batch in enumerate(batches, 1):
            path = phase_dir / f"batch_{index:04d}.json"
            if path.exists():
                output = read_json(path)
                ids = [row["sample_id"] for row in batch]
                _require(set(output.get("results", {})) == set(ids), f"{phase} checkpoint alignment failure: {index}")
                outputs.update(output["results"])
                if completed_status:
                    for sample_id in ids:
                        _mark_phase_state(states[sample_id], completed_status)
                if "audit" in phase:
                    high_audited.update(ids)
            else:
                missing.append((index, batch, path))
    if missing and not resume and outputs:
        raise FullSilverProductionError(f"partial {phase} checkpoint exists; use --resume")

    calls_lock = threading.Lock()

    def invoke(item: tuple[int, list[dict[str, Any]], Path], *, final_deferred_attempt: bool = False) -> tuple[int, list[dict[str, Any]], Path, dict[str, Any] | None]:
        index, batch, path = item
        ids = [row["sample_id"] for row in batch]
        adaptive_dir = phase_dir / "adaptive" / f"batch_{index:04d}"

        def semantic_call(part: list[dict[str, Any]], timeout_seconds: int, label: str) -> dict[str, Any] | None:
            nonlocal calls
            part_path = adaptive_dir / f"{label}.json"
            part_ids = [row["sample_id"] for row in part]
            if part_path.exists():
                saved = read_json(part_path)
                _require(set(saved.get("results", {})) == set(part_ids), f"{phase} adaptive checkpoint alignment failure: {index}/{label}")
                return saved
            selected_provider = provider if timeout_seconds == int(config["semantic_timeout_seconds"]) else provider_factory(timeout_seconds) if provider_factory else provider
            task = {row["sample_id"]: payload_builder(row) for row in part}
            try:
                value = selected_provider._invoke(
                    role=f"SILVER_PRODUCTION_{phase.upper()}", sample_id="|".join(part_ids), split="batched",
                    prompt_version=f"task011d-e5-{phase}-v1", system_prompt=system,
                    user_payload={"sample_ids": part_ids, "samples": task}, schema_name=f"silver_{phase}",
                    schema=_batch_schema(part_ids, item_schema),
                )
            except (SemanticProviderError, TimeoutError, OSError) as exc:
                failure_class = classify_provider_failure(exc)
                invocation_ids = _matching_failure_invocation_ids(Path(config["_root"]), config, "|".join(part_ids), failure_class)
                if governor:
                    governor.record_failure(failure_class=failure_class, phase=phase, sample_ids=part_ids, timeout_seconds=timeout_seconds, invocation_ids=invocation_ids)
                if failure_class == "timeout":
                    return None
                if failure_class in CAPACITY_CLASSES:
                    raise ProviderCapacityPause(f"{failure_class}: explicit provider failure") from exc
                raise ProviderRuntimePause(f"{failure_class}: semantic runtime failure") from exc
            _require(set(value.get("results", {})) == set(part_ids), f"{phase} batch output alignment failure: {index}/{label}")
            _write_json(part_path, value)
            with calls_lock:
                calls += 1
            if governor:
                governor.record_success()
            return value

        def single_call(row: dict[str, Any], label: str, history: list[dict[str, Any]]) -> dict[str, Any] | None:
            if governor:
                governor.record_single_retry()
            value = semantic_call([row], int(config["single_sample_timeout_seconds"]), label)
            if value is None:
                sample_id = row["sample_id"]
                invocation_ids = _matching_failure_invocation_ids(Path(config["_root"]), config, sample_id, "timeout")
                history.append({"timeout_seconds": int(config["single_sample_timeout_seconds"]), "invocation_ids": invocation_ids, "at": _now()})
                if governor:
                    governor.defer(sample_id, history)
                states[sample_id]["status"] = "deferred_semantic_timeout"
                return None
            if governor:
                governor.resolve(row["sample_id"])
            return value

        original_key = "|".join(ids)
        invocation_path = Path(config["_root"]) / config["invocation_log"]
        prior_timeout = False
        prior_child_process_error = False
        if invocation_path.exists():
            prior_rows = _read_jsonl(invocation_path)
            prior_timeout = any(row.get("sample_id") == original_key and row.get("error_type") == "TimeoutExpired" for row in prior_rows)
            prior_child_process_error = any(row.get("sample_id") == original_key and row.get("provider_failure_class") == "child_process_error" for row in prior_rows)
        if not prior_timeout and not prior_child_process_error:
            direct = semantic_call(batch, int(config["semantic_timeout_seconds"]), "default")
            if direct is not None:
                return index, batch, path, direct
        if len(batch) == 1:
            history = [{"timeout_seconds": int(config["semantic_timeout_seconds"]), "invocation_ids": _matching_failure_invocation_ids(Path(config["_root"]), config, original_key, "timeout"), "at": _now()}]
            value = single_call(batch[0], f"single_{ids[0]}", history)
            if value is None and final_deferred_attempt:
                raise DeferredSemanticTimeoutPause(f"unresolved single-sample timeout in {phase} batch {index}")
            return index, batch, path, value

        if governor:
            governor.record_split()
        if len(batch) == 5:
            parts = [batch[:2], batch[2:]]
        elif len(batch) == 4:
            parts = [batch[:2], batch[2:]]
        else:
            parts = [batch]
        combined: dict[str, Any] = {"results": {}}
        unresolved = False
        for part_index, part in enumerate(parts, 1):
            part_ids = [row["sample_id"] for row in part]
            part_value = semantic_call(part, int(config["split_retry_timeout_seconds"]), f"split_{part_index:02d}")
            if part_value is not None:
                combined["results"].update(part_value["results"]); continue
            for row in part:
                history = [
                    {"timeout_seconds": int(config["default_timeout"] if "default_timeout" in config else config["semantic_timeout_seconds"]), "invocation_ids": _matching_failure_invocation_ids(Path(config["_root"]), config, original_key, "timeout"), "at": _now()},
                    {"timeout_seconds": int(config["split_retry_timeout_seconds"]), "invocation_ids": _matching_failure_invocation_ids(Path(config["_root"]), config, "|".join(part_ids), "timeout"), "at": _now()},
                ]
                single = single_call(row, f"single_{row['sample_id']}", history)
                if single is None:
                    unresolved = True
                else:
                    combined["results"].update(single["results"])
        if unresolved:
            if final_deferred_attempt:
                raise DeferredSemanticTimeoutPause(f"unresolved single-sample timeout in {phase} batch {index}")
            return index, batch, path, None
        _require(set(combined["results"]) == set(ids), f"{phase} adaptive output completeness failure: {index}")
        return index, batch, path, combined

    config["_root"] = str(runtime.parents[3])
    worker_count = governor.concurrency if governor else int(config["max_concurrent_semantic_workers"])
    executor = ThreadPoolExecutor(max_workers=worker_count)
    remaining = iter(missing)
    futures = {}
    for _ in range(worker_count):
        item = next(remaining, None)
        if item is not None:
            futures[executor.submit(invoke, item)] = item
    try:
        while futures:
            future = next(as_completed(futures))
            current_item = futures.pop(future)
            try:
                _, batch, path, output = future.result()
            except (ProviderCapacityPause, ProviderRuntimePause, DeferredSemanticTimeoutPause):
                # Preserve any concurrent invocation that already succeeded so resume
                # never duplicates a successful semantic call.
                # Cancel the full queue first. Otherwise a worker that finishes while
                # this loop is waiting could take another queued invocation.
                for pending in futures:
                    pending.cancel()
                for pending in futures:
                    if pending.cancelled():
                        continue
                    try:
                        _, pending_batch, pending_path, pending_output = pending.result()
                    except Exception:
                        continue
                    if not pending_path.exists():
                        _write_json(pending_path, pending_output)
                        pending_ids = [row["sample_id"] for row in pending_batch]
                        outputs.update(pending_output["results"])
                        if completed_status:
                            for sample_id in pending_ids:
                                _mark_phase_state(states[sample_id], completed_status)
                        if "audit" in phase:
                            high_audited.update(pending_ids)
                for row in current_item[1]:
                    states[row["sample_id"]]["resume_status"] = states[row["sample_id"]]["status"]
                raise
            if output is None:
                next_item = next(remaining, None)
                if next_item is not None:
                    futures[executor.submit(invoke, next_item)] = next_item
                continue
            _write_json(path, output)
            ids = [row["sample_id"] for row in batch]
            outputs.update(output["results"])
            if completed_status:
                for sample_id in ids:
                    _mark_phase_state(states[sample_id], completed_status)
                    states[sample_id]["updated_at"] = _now()
            if "audit" in phase:
                high_audited.update(ids)
            _checkpoint(runtime, states, high_audited, f"running_{phase}")
            next_item = next(remaining, None)
            if next_item is not None:
                futures[executor.submit(invoke, next_item)] = next_item
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    unresolved_items = [item for item in missing if not item[2].exists() and any(states[row["sample_id"]]["status"] == "deferred_semantic_timeout" for row in item[1])]
    for item in unresolved_items:
        _, batch, path, output = invoke(item, final_deferred_attempt=True)
        _require(output is not None, f"unresolved deferred batch: {phase}/{item[0]}")
        _write_json(path, output); outputs.update(output["results"])
        if completed_status:
            for row in batch:
                _mark_phase_state(states[row["sample_id"]], completed_status)
                states[row["sample_id"]]["updated_at"] = _now()
        _checkpoint(runtime, states, high_audited, f"running_{phase}_deferred_resolution")
    _require(set(outputs) == {row["sample_id"] for row in samples}, f"{phase} output completeness failure")
    return outputs, calls


def _phase_call_count(runtime: Path, phase: str) -> int:
    directory = runtime / "semantic_batches" / phase
    return len(list(directory.glob("batch_*.json"))) if directory.exists() else 0


def _phase_result_ids(runtime: Path, phase: str) -> set[str]:
    directory = runtime / "semantic_batches" / phase
    result: set[str] = set()
    if not directory.exists():
        return result
    for path in sorted(directory.glob("batch_*.json")):
        ids = set(read_json(path).get("results", {}))
        _require(not result.intersection(ids), f"{phase} duplicate sample output across checkpoints")
        result.update(ids)
    return result


def _materialize_all(samples: list[dict[str, Any]], articles: dict[str, dict[str, Any]], semantic: dict[str, Any], config: dict[str, Any], version: str) -> dict[str, dict[str, Any]]:
    result = {}
    by_id = {row["sample_id"]: row for row in samples}
    for sample_id, value in semantic.items():
        candidate = _materialize(articles[sample_id], by_id[sample_id], value, config, version)
        candidate["candidate_profile"] = "silver_full_production_v2"
        candidate["pilot_namespace"] = False
        candidate["full_run_reuse_authorized"] = True
        candidate["construction_method"] = "codexexec_batched_medium_full_silver_production_v1"
        result[sample_id] = candidate
    return result


def _stable_pick(seed: str, rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (hashlib.sha256(f"{seed}|{row['sample_id']}".encode()).hexdigest(), row["sample_id"]))
    return ranked[:count]


def _audit_stratum(sample: dict[str, Any]) -> str:
    features = sample["risk_features"]
    if features["character_count"] >= 1000:
        return "long"
    if features["number_date_signal_count"] >= 3:
        return "number_date"
    if features["entity_signal_count"] >= 3:
        return "multi_entity"
    return "ordinary"


def _select_train_audit(samples: list[dict[str, Any]], candidates: dict[str, Any], reviews: dict[str, Any], revised_ids: set[str], config: dict[str, Any]) -> dict[str, Any]:
    train = [row for row in samples if row["split"] == "train"]
    risk_ids = set(revised_ids)
    reasons: dict[str, list[str]] = defaultdict(list)
    for row in train:
        sample_id = row["sample_id"]
        if sample_id in revised_ids:
            reasons[sample_id].append("revision_accepted")
        if len(candidates[sample_id]["fact_points"]) > 16:
            risk_ids.add(sample_id); reasons[sample_id].append("fact_count_above_16")
        if "minor_fact_count_anomaly" in reviews[sample_id].get("soft_warnings", []):
            risk_ids.add(sample_id); reasons[sample_id].append("reviewer_fact_count_risk")
    low = [row for row in train if row["sample_id"] not in risk_ids]
    count = math.ceil(len(low) * float(config["train_low_risk_audit_rate"]))
    random_first = _stable_pick(config["train_audit_seed"] + "|round1", low, count)
    for row in random_first:
        reasons[row["sample_id"]].append("low_risk_random_5_percent_round1")
    selected_ids = risk_ids | {row["sample_id"] for row in random_first}
    return {
        "selected": [row for row in train if row["sample_id"] in selected_ids],
        "risk_ids": sorted(risk_ids),
        "low_population": [row["sample_id"] for row in low],
        "random_round1": [row["sample_id"] for row in random_first],
        "reasons": dict(reasons),
    }


def _audit_issue_payload(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "issue_type": AUDIT_TO_REVIEW.get(issue, "severe_structure_error"),
        "fact_ids": [],
        "rationale": audit["rationale"],
        "required_fix": f"resolve High Audit issue: {issue}",
    } for issue in audit["core_error_types"]] or [{
        "issue_type": "severe_structure_error", "fact_ids": [], "rationale": audit["rationale"], "required_fix": "resolve High Audit hard issue",
    }]


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return float(ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))


def _full_run_quality_gate(*, semantic_quality_gate: bool, unfinished: int, total_sft_ready: int) -> bool:
    """TASK-011D-E5 can pass only when quality, completeness, and scale all pass."""
    return semantic_quality_gate and unfinished == 0 and total_sft_ready >= 2000


def _invocation_stats(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path) if path.exists() else []
    task_rows = [row for row in rows if row.get("parent_task") == "TASK-011D-E5"]
    succeeded = [row for row in task_rows if row.get("status") == "succeeded"]
    return {
        "attempted": len(task_rows), "succeeded": len(succeeded), "failed": len(task_rows) - len(succeeded),
        "tool_calls": sum(int(row.get("tool_call_count", 0)) for row in task_rows),
        "fallback_calls": sum(bool(row.get("fallback_used")) for row in task_rows),
        "usage": {
            key: sum(int((row.get("usage") or {}).get(key, 0) or 0) for row in succeeded)
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
        },
    }


def _write_safe_csv(path: Path, samples: list[dict[str, Any]], candidates: dict[str, Any], states: dict[str, dict[str, Any]], reviews: dict[str, Any], audits: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "article_id", "split", "event_group_id", "fact_count", "primary_review", "high_audit", "terminal_status", "soft_warning_count", "candidate_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in samples:
            sample_id = row["sample_id"]
            writer.writerow({
                "sample_id": sample_id, "article_id": row["article_id"], "split": row["split"], "event_group_id": row["event_group_id"],
                "fact_count": len(candidates[sample_id]["fact_points"]), "primary_review": reviews[sample_id]["status"],
                "high_audit": audits.get(sample_id, {}).get("verdict", "not_selected"), "terminal_status": states[sample_id]["status"],
                "soft_warning_count": len(reviews[sample_id].get("soft_warnings", [])), "candidate_sha256": candidates[sample_id]["record_sha256"],
            })


def run_production(root: Path, config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    _require(config["role"] == "FULL_SILVER_PRODUCTION_OPERATOR" and config["full_run_execution_allowed"], "full production not authorized")
    _require(not config["sft_v2_creation_allowed"] and not config["training_allowed"] and not config["judge_allowed"], "forbidden stage enabled")
    _require(1 <= int(config["max_concurrent_semantic_workers"]) <= 2, "semantic concurrency must be <=2")
    runtime = root / config["output_dir"]
    if runtime.exists() and any(runtime.iterdir()) and not resume:
        raise FullSilverProductionError("production output exists; use --resume")
    runtime.mkdir(parents=True, exist_ok=True)
    previous_progress = read_json(runtime / "pipeline_progress.json") if (runtime / "pipeline_progress.json").exists() else {}
    run_started = time.monotonic()
    meta = _load_run_meta(runtime, resume)
    governor = RuntimeTimeoutGovernor(root, config)
    legacy_pauses = int(meta.get("provider_pause_count", 0))
    if legacy_pauses and not meta.get("timeout_capacity_migration_applied"):
        meta["legacy_timeout_misclassified_pause_count"] = legacy_pauses
        meta["timeout_capacity_migration_applied"] = True
    meta["provider_pause_count"] = int(governor.data.get("explicit_capacity_pause_count", 0))
    config["max_concurrent_semantic_workers"] = governor.concurrency
    meta["runtime_governor_concurrency"] = governor.concurrency
    context = load_context(root, config)
    samples = build_samples(context)
    _require(len(samples) == config["silver_count"] == 2117, "formal sample count mismatch")
    articles = {row["sample_id"]: context["articles"][row["article_id"]] for row in samples}
    state_path = runtime / "sample_states.json"
    states = read_json(state_path) if state_path.exists() else _initial_state(samples)
    high_audited: set[str] = set()
    if (runtime / "high_audited_ids.json").exists():
        high_audited = set(read_json(runtime / "high_audited_ids.json"))
    _write_json(runtime / "run_metadata.json", meta)
    _checkpoint(runtime, states, high_audited, "running_preflight")
    _write_json(runtime / "silver_production_standard_v1.json", standard_document(config))
    _write_json(runtime / "full_production_selection.json", {
        "sample_count": 2117, "split_counts": dict(Counter(row["split"] for row in samples)),
        "formal_input": config["generation_input"], "formal_input_sha256": context["silver_population_sha256"],
        "old_input_read": False, "pilot_candidates_reused": False,
        "sample_mapping_sha256": _sha([(row["sample_id"], row["article_id"], row["split"]) for row in samples]),
    })
    try:
        if resume:
            integrity = validate_resume_integrity(root, config, states)
            integrity["protected_snapshots"] = context["protected_snapshots"]
            integrity["formal_input_path"] = config["generation_input"]
            _write_json(runtime / "resume_integrity_check.json", integrity)
        if resume and previous_progress.get("pipeline_status") == "paused_provider_runtime_error":
            health = run_provider_health_smoke(root, config)
            meta["provider_runtime_health"] = health["provider_runtime_health"]
            meta["last_provider_health_check_at"] = health["checked_at"]
            _write_json(runtime / "run_metadata.json", meta)
            if not health["passed"]:
                failure_class = health.get("failure_class") or "unknown_provider_error"
                governor.record_failure(
                    failure_class=failure_class,
                    phase="provider_health",
                    sample_ids=["task011d_e5_provider_health_synthetic"],
                    timeout_seconds=int(health["timeout_seconds"]),
                    invocation_ids=[health["invocation_id"]] if health.get("invocation_id") else [],
                )
                raise ProviderRuntimePause(f"provider health smoke failed: {failure_class}")
            _checkpoint(runtime, states, high_audited, "resuming")
        medium = _provider(root, config, config["production_reasoning_effort"])
        generated, _ = _invoke_phase(
            medium, phase="generation", samples=samples, articles=articles, config=config, runtime=runtime,
            system=GENERATION_SYSTEM, item_schema=GENERATION_ITEM_SCHEMA,
            payload_builder=lambda row: {"source": _source_view(articles[row["sample_id"]])}, states=states,
            completed_status="generated", high_audited=high_audited, resume=resume, governor=governor,
            provider_factory=lambda timeout: _provider(root, config, config["production_reasoning_effort"], timeout),
        )
        initial = _materialize_all(samples, articles, generated, config, "production_candidate_v1")
        validations_initial = {sample_id: deterministic_validate(candidate, articles[sample_id]) for sample_id, candidate in initial.items()}
        for sample_id in states:
            states[sample_id]["status"] = "deterministic_checked"
        _write_json(runtime / "deterministic_validation_initial.json", validations_initial)
        _checkpoint(runtime, states, high_audited, "running_deterministic_validation")

        reviews, _ = _invoke_phase(
            medium, phase="review", samples=samples, articles=articles, config=config, runtime=runtime,
            system=REVIEW_SYSTEM, item_schema=REVIEW_ITEM_SCHEMA,
            payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], initial[row["sample_id"]]), states=states,
            completed_status=None, high_audited=high_audited, resume=resume, governor=governor,
            provider_factory=lambda timeout: _provider(root, config, config["production_reasoning_effort"], timeout),
        )
        historical_revision_ids = _phase_result_ids(runtime, "revision") if resume else set()
        for sample_id, validation in validations_initial.items():
            if validation["status"] != "pass" and reviews[sample_id]["status"] == "PASS":
                reviews[sample_id]["status"] = "FIX"
                reviews[sample_id]["core_issues"].append({"issue_type": "severe_structure_error", "fact_ids": [], "rationale": ";".join(validation["errors"]), "required_fix": "repair deterministic integrity"})
            # Completed semantic phase membership is immutable across validator
            # releases, so resume can reuse every prior result without rebatching it.
            if sample_id in historical_revision_ids and reviews[sample_id]["status"] != "FIX":
                reviews[sample_id]["status"] = "FIX"
                reviews[sample_id]["core_issues"].append({
                    "issue_type": "severe_structure_error", "fact_ids": [],
                    "rationale": "historical revision checkpoint membership preserved after deterministic equivalence normalization",
                    "required_fix": "reuse the completed revision checkpoint",
                })
            states[sample_id]["status"] = "reviewed_" + reviews[sample_id]["status"].lower()
            states[sample_id]["reviewed"] = True
            states[sample_id]["fix"] = reviews[sample_id]["status"] == "FIX"
        fix_samples = [row for row in samples if reviews[row["sample_id"]]["status"] == "FIX"]
        drop_ids = {sample_id for sample_id, review in reviews.items() if review["status"] == "DROP"}
        _write_json(runtime / "primary_reviews.json", reviews)
        _checkpoint(runtime, states, high_audited, "running_primary_review")

        revised: dict[str, Any] = {}
        rereviews: dict[str, Any] = {}
        if fix_samples:
            revisions, _ = _invoke_phase(
                medium, phase="revision", samples=fix_samples, articles=articles, config=config, runtime=runtime,
                system=REVISION_SYSTEM, item_schema=GENERATION_ITEM_SCHEMA,
                payload_builder=lambda row: {**_source_candidate(articles[row["sample_id"]], initial[row["sample_id"]]), "core_issues": reviews[row["sample_id"]]["core_issues"]},
                states=states, completed_status="revised", high_audited=high_audited, resume=resume,
                governor=governor, provider_factory=lambda timeout: _provider(root, config, config["production_reasoning_effort"], timeout),
            )
            revised = _materialize_all(fix_samples, articles, revisions, config, "production_candidate_v2")
            revised_validations = {sample_id: deterministic_validate(candidate, articles[sample_id]) for sample_id, candidate in revised.items()}
            rereviews, _ = _invoke_phase(
                medium, phase="rereview", samples=fix_samples, articles=articles, config=config, runtime=runtime,
                system=REREVIEW_SYSTEM, item_schema=REVIEW_ITEM_SCHEMA,
                payload_builder=lambda row: {**_source_candidate(articles[row["sample_id"]], revised[row["sample_id"]]), "original_core_issues": reviews[row["sample_id"]]["core_issues"]},
                states=states, completed_status="re_reviewed", high_audited=high_audited, resume=resume,
                governor=governor, provider_factory=lambda timeout: _provider(root, config, config["production_reasoning_effort"], timeout),
            )
            for sample_id in revised:
                if revised_validations[sample_id]["status"] != "pass" or rereviews[sample_id]["status"] != "PASS":
                    drop_ids.add(sample_id)
        else:
            revised_validations = {}
        candidates = {sample_id: revised.get(sample_id, initial[sample_id]) for sample_id in initial}
        validations_final = {sample_id: deterministic_validate(candidate, articles[sample_id]) for sample_id, candidate in candidates.items()}
        _write_json(runtime / "primary_rereviews.json", rereviews)
        _write_json(runtime / "deterministic_validation_final.json", validations_final)

        accepted_after_primary = [row for row in samples if row["sample_id"] not in drop_ids]
        train_selection = _select_train_audit(accepted_after_primary, candidates, reviews, set(revised), config)
        # Validation and Test are audited 100%, including a candidate already rejected by
        # Primary Review. Such a record remains dropped; the audit is retained as lineage.
        validation_samples = [row for row in samples if row["split"] == "validation"]
        test_samples = [row for row in samples if row["split"] == "test"]
        audit_samples = train_selection["selected"] + validation_samples + test_samples
        audit_samples.sort(key=lambda row: (SPLITS.index(row["split"]), row["sample_id"]))
        high = _provider(root, config, config["auditor_reasoning_effort"])
        audits, _ = _invoke_phase(
            high, phase="high_audit", samples=audit_samples, articles=articles, config=config, runtime=runtime,
            system=AUDIT_SYSTEM, item_schema=AUDIT_ITEM_SCHEMA,
            payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]),
            states=states, completed_status=None, high_audited=high_audited, resume=resume,
            governor=governor, provider_factory=lambda timeout: _provider(root, config, config["auditor_reasoning_effort"], timeout),
        )
        high_audited.update(audits)

        random_first_ids = set(train_selection["random_round1"])
        random_first_hard = [sample_id for sample_id in random_first_ids if audits[sample_id]["verdict"] in HARD_AUDIT_VERDICTS]
        first_major_rate = len(random_first_hard) / max(len(random_first_ids), 1)
        round2_samples: list[dict[str, Any]] = []
        adaptive = first_major_rate > 0.02 or any(audits[sample_id]["verdict"] == "BLOCKING" for sample_id in random_first_ids)
        if adaptive:
            remaining_low = [row for row in accepted_after_primary if row["split"] == "train" and row["sample_id"] in set(train_selection["low_population"]) - random_first_ids]
            round2_samples = _stable_pick(config["train_audit_seed"] + "|round2", remaining_low, math.ceil(len(train_selection["low_population"]) * 0.05))
            round2, _ = _invoke_phase(
                high, phase="train_high_audit_round2", samples=round2_samples, articles=articles, config=config, runtime=runtime,
                system=AUDIT_SYSTEM, item_schema=AUDIT_ITEM_SCHEMA,
                payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]),
                states=states, completed_status=None, high_audited=high_audited, resume=resume,
                governor=governor, provider_factory=lambda timeout: _provider(root, config, config["auditor_reasoning_effort"], timeout),
            )
            audits.update(round2); high_audited.update(round2)
            round2_hard = [sample_id for sample_id in round2 if round2[sample_id]["verdict"] in HARD_AUDIT_VERDICTS]
            round2_rate = len(round2_hard) / max(len(round2), 1)
            if round2_rate > 0.02:
                bad_strata = {_audit_stratum(next(row for row in round2_samples if row["sample_id"] == sample_id)) for sample_id in round2_hard}
                targeted = [row for row in remaining_low if row["sample_id"] not in round2 and _audit_stratum(row) in bad_strata]
                if targeted:
                    targeted_audits, _ = _invoke_phase(
                        high, phase="train_high_audit_targeted", samples=targeted, articles=articles, config=config, runtime=runtime,
                        system=AUDIT_SYSTEM, item_schema=AUDIT_ITEM_SCHEMA,
                        payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]),
                        states=states, completed_status=None, high_audited=high_audited, resume=resume,
                        governor=governor, provider_factory=lambda timeout: _provider(root, config, config["auditor_reasoning_effort"], timeout),
                    )
                    audits.update(targeted_audits); high_audited.update(targeted_audits)
        _write_json(runtime / "high_audited_ids.json", sorted(high_audited))
        _checkpoint(runtime, states, high_audited, "running_high_audit")

        hard_audit_samples = [row for row in accepted_after_primary if row["sample_id"] in audits and audits[row["sample_id"]]["verdict"] in HARD_AUDIT_VERDICTS]
        repairable = [row for row in hard_audit_samples if row["sample_id"] not in revised]
        high_rereviews: dict[str, Any] = {}
        if repairable:
            high_revisions, _ = _invoke_phase(
                medium, phase="high_issue_revision", samples=repairable, articles=articles, config=config, runtime=runtime,
                system=REVISION_SYSTEM, item_schema=GENERATION_ITEM_SCHEMA,
                payload_builder=lambda row: {**_source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]), "core_issues": _audit_issue_payload(audits[row["sample_id"]])},
                states=states, completed_status="revised", high_audited=high_audited, resume=resume,
                governor=governor, provider_factory=lambda timeout: _provider(root, config, config["production_reasoning_effort"], timeout),
            )
            repaired_candidates = _materialize_all(repairable, articles, high_revisions, config, "production_candidate_v2")
            repaired_validations = {sample_id: deterministic_validate(candidate, articles[sample_id]) for sample_id, candidate in repaired_candidates.items()}
            candidates.update(repaired_candidates)
            high_rereviews, _ = _invoke_phase(
                high, phase="high_rereview", samples=repairable, articles=articles, config=config, runtime=runtime,
                system=AUDIT_SYSTEM, item_schema=AUDIT_ITEM_SCHEMA,
                payload_builder=lambda row: _source_candidate(articles[row["sample_id"]], candidates[row["sample_id"]]),
                states=states, completed_status="re_reviewed", high_audited=high_audited, resume=resume,
                governor=governor, provider_factory=lambda timeout: _provider(root, config, config["auditor_reasoning_effort"], timeout),
            )
            for sample_id in high_rereviews:
                audits[sample_id] = high_rereviews[sample_id]
                if repaired_validations[sample_id]["status"] != "pass" or high_rereviews[sample_id]["verdict"] in HARD_AUDIT_VERDICTS:
                    drop_ids.add(sample_id)
        for row in hard_audit_samples:
            if row["sample_id"] in revised:
                drop_ids.add(row["sample_id"])
        validations_final = {sample_id: deterministic_validate(candidate, articles[sample_id]) for sample_id, candidate in candidates.items()}

        for row in samples:
            sample_id = row["sample_id"]
            warnings = list(reviews[sample_id].get("soft_warnings", []))
            if sample_id in audits:
                warnings += audits[sample_id].get("soft_imperfections", [])
                if audits[sample_id]["verdict"] == "MINOR":
                    warnings.append("high_audit_minor")
            states[sample_id]["soft_warnings"] = sorted(set(warnings))
            if sample_id in drop_ids:
                states[sample_id]["status"] = "dropped"
            elif warnings:
                states[sample_id]["status"] = "accepted_with_warning"
            else:
                states[sample_id]["status"] = "accepted"
            states[sample_id]["updated_at"] = _now()
        _checkpoint(runtime, states, high_audited, "finalizing")

        pool_rows = []
        drop_rows = []
        for row in samples:
            sample_id = row["sample_id"]
            lineage = {
                "primary_review": reviews[sample_id]["status"], "primary_revision": sample_id in revised,
                "primary_rereview": rereviews.get(sample_id), "high_audit": audits.get(sample_id),
                "high_revision": sample_id in {item["sample_id"] for item in repairable}, "high_rereview": high_rereviews.get(sample_id),
            }
            if states[sample_id]["status"] == "dropped":
                drop_rows.append({"sample_id": sample_id, "source_article_id": row["article_id"], "split": row["split"], "hard_reason": lineage, "candidate_sha256": candidates[sample_id]["record_sha256"]})
            else:
                candidate = candidates[sample_id]
                pool_rows.append({
                    "sample_id": sample_id, "split": row["split"], "event_group_id": row["event_group_id"], "source_hash": row["source_sha256"],
                    "candidate_version": candidate["candidate_version"], "fact_count": len(candidate["fact_points"]),
                    "quality_status": states[sample_id]["status"], "soft_warnings": states[sample_id]["soft_warnings"],
                    "review_lineage": {"status": reviews[sample_id]["status"]}, "revision_lineage": {"primary": sample_id in revised, "high": sample_id in {item["sample_id"] for item in repairable}},
                    "high_audit_lineage": audits.get(sample_id), "target_hash": candidate["target_text_sha256"], "final_candidate_hash": candidate["record_sha256"],
                    "candidate": candidate,
                })
        pool_dir = runtime / "silver_quality_pool_v2"
        pool_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(pool_dir / "silver_sft_quality_pool_v2.jsonl", pool_rows)
        _write_jsonl(runtime / "silver_drop_ledger.jsonl", drop_rows)

        # The global quality gate describes the final quality pool, not candidates that
        # were correctly removed by an earlier or High gate.
        final_audits = {sample_id: audit for sample_id, audit in audits.items() if sample_id not in drop_ids}
        audit_verdicts = Counter(audit["verdict"] for audit in final_audits.values())
        audit_core = Counter(issue for audit in final_audits.values() for issue in audit["core_error_types"])
        blocking_rate = audit_verdicts["BLOCKING"] / max(len(final_audits), 1)
        major_rate = audit_verdicts["MAJOR"] / max(len(final_audits), 1)
        systematic_core = Counter(
            issue for audit in final_audits.values() if audit["verdict"] in HARD_AUDIT_VERDICTS
            for issue in audit["core_error_types"]
        )
        systematic = any(value >= 2 for value in systematic_core.values())
        technical_failures = Counter()
        for value in validations_final.values():
            technical_failures["target"] += not value["target_integrity"]
            technical_failures["schema"] += not value["schema"]
            technical_failures["provenance"] += not value["provenance"]
        semantic_quality_gate = blocking_rate == 0 and major_rate <= 0.02 and not systematic and not sum(technical_failures.values())
        fact_counts = [len(candidates[row["sample_id"]]["fact_points"]) for row in samples]
        soft_counts = Counter(warning for value in states.values() for warning in value.get("soft_warnings", []))
        soft_counts["noncore_coverage"] += soft_counts["non_core_coverage_omission"]
        soft_counts["fact_count_signal"] += soft_counts["minor_fact_count_anomaly"]
        soft_counts["style_warning"] += soft_counts["style_wording_issue"]
        hard_counts = Counter(issue for review in reviews.values() for issue in (item["issue_type"] for item in review["core_issues"]))
        core_hard_counts = Counter({
            "core_fact_error": hard_counts["core_fact_error"] + audit_core["core_factual_error"] + audit_core["core_unsupported_fact"],
            "evidence_error": hard_counts["evidence_error"] + audit_core["evidence_error"],
            "entity_number_date_error": hard_counts["entity_number_date_error"] + audit_core["entity_error"] + audit_core["entity_relation_error"] + audit_core["number_date_error"],
            "material_coverage_gap": hard_counts["material_coverage_gap"] + audit_core["material_coverage_gap"],
            "severe_inference": hard_counts["severe_inference"] + audit_core["severe_inference"],
            "severe_structure_error": hard_counts["severe_structure_error"] + audit_core["severe_structure_error"],
            "target_technical_error": technical_failures["target"], "schema_error": technical_failures["schema"],
            "provenance_error": technical_failures["provenance"],
        })
        terminal_counts = Counter(value["status"] for value in states.values())
        total_ready = 239 + len(pool_rows)
        unfinished = sum(value["status"] not in TERMINAL for value in states.values())
        quality_gate = _full_run_quality_gate(
            semantic_quality_gate=semantic_quality_gate,
            unfinished=unfinished,
            total_sft_ready=total_ready,
        )
        audited_pool_samples = [row for row in samples if row["sample_id"] in final_audits]
        global_ids: list[dict[str, Any]] = []
        used_global: set[str] = set()
        # Seed one sample for each required coverage dimension, then fill to 100 by
        # the same stable rank. Existing High results are reused; no extra call occurs.
        selectors = [
            ("train", lambda row: row["split"] == "train"),
            ("validation", lambda row: row["split"] == "validation"),
            ("test", lambda row: row["split"] == "test"),
            ("ordinary", lambda row: _audit_stratum(row) == "ordinary"),
            ("long", lambda row: _audit_stratum(row) == "long"),
            ("number_date", lambda row: _audit_stratum(row) == "number_date"),
            ("multi_entity", lambda row: _audit_stratum(row) == "multi_entity"),
            ("risk", lambda row: row["sample_id"] in set(train_selection["risk_ids"])),
        ]
        for label, selector in selectors:
            choices = [row for row in audited_pool_samples if selector(row) and row["sample_id"] not in used_global]
            if choices:
                chosen = _stable_pick(config["global_audit_seed"] + "|" + label, choices, 1)[0]
                global_ids.append(chosen); used_global.add(chosen["sample_id"])
        remainder = [row for row in audited_pool_samples if row["sample_id"] not in used_global]
        global_ids += _stable_pick(config["global_audit_seed"] + "|fill", remainder, max(0, min(100, len(audited_pool_samples)) - len(global_ids)))
        global_id_set = {row["sample_id"] for row in global_ids}
        global_audits = {sample_id: audit for sample_id, audit in audits.items() if sample_id in global_id_set}
        _write_json(runtime / "train_high_audit.json", {"selection": train_selection, "adaptive_round2": [row["sample_id"] for row in round2_samples], "audits": {key: value for key, value in audits.items() if states[key]["split"] == "train"}})
        _write_json(runtime / "validation_high_audit.json", {"input_count": 212, "audited_count": sum(key in audits for key, state in states.items() if state["split"] == "validation"), "audits": {key: value for key, value in audits.items() if states[key]["split"] == "validation"}})
        _write_json(runtime / "test_high_audit.json", {"input_count": 208, "audited_count": sum(key in audits for key, state in states.items() if state["split"] == "test"), "audits": {key: value for key, value in audits.items() if states[key]["split"] == "test"}})
        _write_json(runtime / "global_quality_audit.json", {"sample_count": len(global_audits), "sample_ids": sorted(global_audits), "split_counts": dict(Counter(states[key]["split"] for key in global_audits)), "strata": dict(Counter(_audit_stratum(next(row for row in samples if row["sample_id"] == key)) for key in global_audits)), "audits": global_audits})
        _write_json(runtime / "full_generation_summary.json", {"samples": 2117, "calls": _phase_call_count(runtime, "generation"), "deterministic_pass": sum(value["status"] == "pass" for value in validations_initial.values()), "fresh_generation": True})
        _write_json(runtime / "full_review_summary.json", {"samples": 2117, "calls": _phase_call_count(runtime, "review"), "status_counts": dict(Counter(review["status"] for review in reviews.values())), "soft_warning_counts": dict(soft_counts), "hard_issue_counts": dict(core_hard_counts), "double_review": False, "judge": False})
        _write_json(runtime / "full_revision_summary.json", {"primary_fix_samples": len(fix_samples), "primary_revision_samples": len(revised), "high_issue_revision_samples": len(repairable), "maximum_revision_per_sample": 1, "second_revision": False, "primary_rereview_samples": len(rereviews), "high_rereview_samples": len(high_rereviews)})
        pool_manifest = {
            "schema_version": "silver-sft-quality-pool-v2.0.0", "version": "silver_sft_quality_pool_v2", "record_count": len(pool_rows),
            "terminal_counts": dict(terminal_counts), "split_counts": dict(Counter(row["split"] for row in pool_rows)),
            "quality_pool_sha256": sha256_file(pool_dir / "silver_sft_quality_pool_v2.jsonl"), "drop_ledger_sha256": sha256_file(runtime / "silver_drop_ledger.jsonl"),
            "gold_reuse": 239, "total_sft_ready": total_ready, "sft_v2_created": False, "training_executed": False,
        }
        _write_json(pool_dir / "manifest.json", pool_manifest)
        invocation = _invocation_stats(root / config["invocation_log"])
        meta["active_wall_time_seconds"] = round(float(meta.get("active_wall_time_seconds", 0)) + time.monotonic() - run_started, 3)
        meta["finished_at"] = _now()
        _write_json(runtime / "run_metadata.json", meta)
        phase_calls = {phase: _phase_call_count(runtime, phase) for phase in ("generation", "review", "revision", "rereview", "high_audit", "train_high_audit_round2", "train_high_audit_targeted", "high_issue_revision", "high_rereview")}
        total_calls = sum(phase_calls.values())
        efficiency = {"phase_calls": phase_calls, "total_semantic_calls": total_calls, "calls_per_sample": round(total_calls / 2117, 6), "wall_time_seconds": meta["active_wall_time_seconds"], "provider_pauses": meta.get("provider_pause_count", 0), "resume_count": meta["resume_count"], "projected_calls": config["projected_calls"], "actual_minus_projected": total_calls - int(config["projected_calls"]["total"]), "invocation_audit": invocation}
        _write_json(runtime / "full_efficiency_summary.json", efficiency)
        gold_rows = []
        for split in SPLITS:
            gold_rows.extend(_read_jsonl(root / config["gold_sft_dir"] / f"{split}.jsonl"))
        gold_fact_counts = [len(row["fact_points"]) for row in gold_rows]
        split_terminal = {
            split: dict(Counter(states[row["sample_id"]]["status"] for row in samples if row["split"] == split))
            for split in SPLITS
        }
        quality = {
            "schema_version": "task011d-e5-quality-summary-v1.0.0", "task_id": config["task_id"], "role": config["role"],
            "versions": context["versions"], "silver_input": config["generation_input"], "split_input": config["silver_split_counts"],
            "terminal_counts": dict(terminal_counts), "unfinished": unfinished,
            "acceptance_rate": round(len(pool_rows) / 2117, 6), "gold_reuse": 239, "total_sft_ready": total_ready,
            "hard_minimum_met": total_ready >= 2000, "preferred_target_met": total_ready >= 2200, "stretch_target_met": total_ready >= 3000,
            "fact_statistics": {"total": sum(fact_counts), "average": round(statistics.mean(fact_counts), 4), "median": statistics.median(fact_counts), "p75": round(_percentile(fact_counts, .75), 3), "p90": round(_percentile(fact_counts, .90), 3), "p95": round(_percentile(fact_counts, .95), 3), "max": max(fact_counts), "above_12": sum(value > 12 for value in fact_counts), "above_16": sum(value > 16 for value in fact_counts), "above_20": sum(value > 20 for value in fact_counts)},
            "split_terminal_counts": split_terminal, "gold_average_fact_per_article": round(statistics.mean(gold_fact_counts), 4),
            "soft_warning_counts": dict(soft_counts), "hard_issue_counts": dict(core_hard_counts), "audit_verdict_counts": dict(audit_verdicts), "audit_core_error_counts": dict(audit_core),
            "global_high_blocking_rate": blocking_rate, "global_high_major_rate": major_rate, "systematic_hard_error": systematic,
            "technical_failures": dict(technical_failures), "semantic_quality_gate": "PASS" if semantic_quality_gate else "FAIL",
            "quality_gate": "PASS" if quality_gate else "FAIL",
            "pipeline_status": "completed" if quality_gate else "completed_quality_gate_failed", "silver_quality_pool": config["output_dir"] + "/silver_quality_pool_v2", "sft_v2_created": False, "training_executed": False,
        }
        _write_json(runtime / "production_quality_audit.json", quality)
        _checkpoint(runtime, states, high_audited, quality["pipeline_status"])
        manifest = {
            "schema_version": "task011d-e5-full-production-manifest-v1.0.0", "task_id": config["task_id"], "role": config["role"],
            "started_at": meta["started_at"], "finished_at": meta["finished_at"], "pipeline_status": quality["pipeline_status"],
            "input_sha256": context["silver_population_sha256"], "protected_snapshots": context["protected_snapshots"],
            "outputs": {name: sha256_file(runtime / name) for name in ("silver_drop_ledger.jsonl", "full_generation_summary.json", "full_review_summary.json", "full_revision_summary.json", "train_high_audit.json", "validation_high_audit.json", "test_high_audit.json", "global_quality_audit.json", "full_efficiency_summary.json", "pipeline_progress.json", "production_quality_audit.json")},
            "quality_gate": quality["quality_gate"], "unfinished": quality["unfinished"], "total_sft_ready": total_ready, "local_commit_required": quality_gate,
            "sft_v2_created": False, "training_executed": False, "push_executed": False,
        }
        _write_json(runtime / "full_production_manifest.json", manifest)
        _write_safe_csv(root / config["safe_summary_csv"], samples, candidates, states, reviews, audits)
        _write_report(root / config["report_path"], quality, efficiency, pool_manifest, phase_calls)
        return validate_production(root, config_path)
    except ProviderCapacityPause as exc:
        meta["provider_pause_count"] = int(governor.data.get("explicit_capacity_pause_count", 0))
        meta["active_wall_time_seconds"] = round(float(meta.get("active_wall_time_seconds", 0)) + time.monotonic() - run_started, 3)
        meta["last_provider_error"] = str(exc)
        meta["last_pause_reason"] = "explicit_provider_capacity"
        _write_json(runtime / "run_metadata.json", meta)
        _checkpoint(runtime, states, high_audited, "paused_provider_capacity")
        return {"pipeline_status": "paused_provider_capacity", "resume_required": True, "unfinished": progress_document(states, high_audited, "paused_provider_capacity")["unfinished"], "error": str(exc)}
    except DeferredSemanticTimeoutPause as exc:
        meta["active_wall_time_seconds"] = round(float(meta.get("active_wall_time_seconds", 0)) + time.monotonic() - run_started, 3)
        meta["last_provider_error"] = str(exc); meta["last_pause_reason"] = "provider_runtime_unresolved"
        _write_json(runtime / "run_metadata.json", meta)
        _checkpoint(runtime, states, high_audited, "paused_provider_runtime_unresolved")
        return {"pipeline_status": "paused_provider_runtime_unresolved", "resume_required": True, "unfinished": progress_document(states, high_audited, "paused_provider_runtime_unresolved")["unfinished"], "deferred_samples": sorted(governor.queue["samples"]), "error": str(exc)}
    except ProviderRuntimePause as exc:
        meta["active_wall_time_seconds"] = round(float(meta.get("active_wall_time_seconds", 0)) + time.monotonic() - run_started, 3)
        meta["last_provider_error"] = str(exc); meta["last_pause_reason"] = "provider_runtime_error"
        meta["provider_runtime_health"] = "unavailable"
        _write_json(runtime / "run_metadata.json", meta)
        _checkpoint(runtime, states, high_audited, "paused_provider_runtime_error")
        return {"pipeline_status": "paused_provider_runtime_error", "resume_required": True, "unfinished": progress_document(states, high_audited, "paused_provider_runtime_error")["unfinished"], "error": str(exc)}


def _write_report(path: Path, quality: dict[str, Any], efficiency: dict[str, Any], pool: dict[str, Any], calls: dict[str, int]) -> None:
    facts = quality["fact_statistics"]; terminal = quality["terminal_counts"]
    text = f"""# TASK-011D-E5 Full Silver Production Run

## 结论

Full Run Quality Gate：**{quality['quality_gate']}**；pipeline status：`{quality['pipeline_status']}`。2117 条 Silver 已全部进入终态，Silver Quality Pool 为 `silver_sft_quality_pool_v2`。未构建完整 `sft_v2`，未训练，未 push。

## 正式版本与规模

- news/event/split：`news_v2.1.0` / `news_event_groups_v2.1.0` / `news_split_v2.1.0`。
- Silver input：1697/212/208；accepted/warning/dropped：{terminal.get('accepted',0)}/{terminal.get('accepted_with_warning',0)}/{terminal.get('dropped',0)}。
- Gold reuse：239；Silver accepted：{pool['record_count']}；Total SFT-ready：{quality['total_sft_ready']}。
- hard/preferred/stretch：{quality['hard_minimum_met']}/{quality['preferred_target_met']}/{quality['stretch_target_met']}。

## 调用与质量

- phase calls：{calls}；total={efficiency['total_semantic_calls']}；calls/sample={efficiency['calls_per_sample']}；wall time={efficiency['wall_time_seconds']} 秒。
- provider pauses/resumes：{efficiency['provider_pauses']}/{efficiency['resume_count']}。
- Fact total/avg/median/p90/>16/>20：{facts['total']}/{facts['average']}/{facts['median']}/{facts['p90']}/{facts['above_16']}/{facts['above_20']}。
- Global High blocking/major：{quality['global_high_blocking_rate']}/{quality['global_high_major_rate']}；systematic hard error={quality['systematic_hard_error']}。
- Target/Schema/Provenance failures：{quality['technical_failures'].get('target',0)}/{quality['technical_failures'].get('schema',0)}/{quality['technical_failures'].get('provenance',0)}。

## 边界与下一步

本任务只生成 Silver Quality Pool。`sft_v2 created=false`，`training=false`。质量与规模门禁通过时，下一任务为 `TASK-011D-F Complete SFT v2 Build and Pre-Training Gate`。
"""
    _atomic_text(path, text)


def validate_production(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path); runtime = root / config["output_dir"]
    progress = read_json(runtime / "pipeline_progress.json")
    if progress["pipeline_status"].startswith("paused_provider_"):
        return {"pipeline_status": progress["pipeline_status"], "unfinished": progress["unfinished"], "resume_required": True}
    required = ["full_production_manifest.json", "full_generation_summary.json", "full_review_summary.json", "full_revision_summary.json", "train_high_audit.json", "validation_high_audit.json", "test_high_audit.json", "global_quality_audit.json", "full_efficiency_summary.json", "production_quality_audit.json", "silver_drop_ledger.jsonl", "silver_quality_pool_v2/manifest.json", "silver_quality_pool_v2/silver_sft_quality_pool_v2.jsonl"]
    missing = [name for name in required if not (runtime / name).is_file()]
    _require(not missing, f"missing production outputs: {missing}")
    quality = read_json(runtime / "production_quality_audit.json")
    manifest = read_json(runtime / "full_production_manifest.json")
    pool = read_json(runtime / "silver_quality_pool_v2/manifest.json")
    _require(progress["total"] == 2117 and progress["unfinished"] == 0, "unfinished Silver samples")
    _require(sum(progress[key] for key in ("accepted", "warning", "dropped")) == 2117, "terminal count mismatch")
    _require(pool["record_count"] + progress["dropped"] == 2117, "quality pool count mismatch")
    _require(not quality["sft_v2_created"] and not quality["training_executed"], "forbidden stage executed")
    _require(manifest["input_sha256"] == sha256_file(root / config["generation_input"]), "formal input changed")
    load_context(root, config)
    return {"pipeline_status": quality["pipeline_status"], "quality_gate": quality["quality_gate"], "unfinished": 0, "silver_accepted": pool["record_count"], "dropped": progress["dropped"], "total_sft_ready": quality["total_sft_ready"], "local_commit_required": manifest["local_commit_required"], "sft_v2_created": False, "training_executed": False, "push_executed": False, "output_dir": config["output_dir"], "report_path": config["report_path"]}
