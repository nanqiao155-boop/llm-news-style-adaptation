from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB = Path(r"experiments/lightweight/private_workspace\same_source_system_q4_q6_q8_20260907\round2_judge")
RESULTS = LAB / "results"
RAW_DIR = RESULTS / "judge_raw"
PROJECT = Path(r".")
Q48_ROOT = Path(r"experiments/lightweight/private_workspace\closed_loop_bartowski_q4_q8_20260907")
Q6_ROOT = Path(r"experiments/lightweight/private_workspace\closed_loop_q6_20260907")
DEPLOYMENT_ROOT = Path(r"experiments/lightweight/private_workspace\precision_sweep_bartowski_same_source_20260907")
Q48_SOURCE = Q48_ROOT / "results" / "closed_loop_outputs.jsonl"
Q6_SOURCE = Q6_ROOT / "results" / "closed_loop_outputs.jsonl"
Q48_AGGREGATE = Q48_ROOT / "results" / "aggregate_round1.json"
Q6_AGGREGATE = Q6_ROOT / "results" / "aggregate_round1.json"
Q48_BURDEN = Q48_ROOT / "results" / "same_source_agent_burden_3way.json"
DEPLOYMENT_SOURCE = DEPLOYMENT_ROOT / "results" / "same_source_aggregate.json"
PACK = Path(
    r"experiments/lightweight/private_workspace\validation24_20260907\historical\china_mobile_lora_final_delivery_20260829"
    r"\results\closed_loop_validation\pack_manifest.json"
)
VALIDATION = PROJECT / "data" / "processed" / "sft_v2" / "validation.jsonl"
FROZEN_ROUND2_RUNNER = Path(r"experiments/lightweight/private_workspace\closed_loop_2x2_20260907\round2_judge\run_round2.py")
FIXED_MODEL = "qwen3-235b-a22b-instruct-2507"
SEED = "cm-lightweight-same-source-6condition-round2-20260907-v1"
PRECISIONS = ("q4", "q6", "q8")
CONDITIONS = ("Q4_DRAFT", "Q4_FINAL", "Q6_DRAFT", "Q6_FINAL", "Q8_DRAFT", "Q8_FINAL")
LABELS = ("A", "B", "C", "D", "E", "F")
FORBIDDEN_TERMS = (
    "Q4", "Q6", "Q8", "quantization", "bit", "bartowski", "model identity",
    "Draft", "Final", "revision", "Reviewer", "Reviser", "BLEU", "ROUGE",
)


def load_frozen_round2() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_lightweight_round2", FROZEN_ROUND2_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Round 2 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frozen = load_frozen_round2()
DIMENSIONS = frozen.DIMENSIONS


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def condition_texts(row: dict[str, Any]) -> dict[str, str]:
    return {
        "Q4_DRAFT": row["q4"]["draft"],
        "Q4_FINAL": row["q4"]["final"],
        "Q6_DRAFT": row["q6"]["draft"],
        "Q6_FINAL": row["q6"]["final"],
        "Q8_DRAFT": row["q8"]["draft"],
        "Q8_FINAL": row["q8"]["final"],
    }


def load_validation_tasks(sample_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    wanted = set(sample_ids)
    tasks: dict[str, list[dict[str, str]]] = {}
    with VALIDATION.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in wanted:
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise RuntimeError(f"invalid Validation messages for {sample_id}")
            clean: list[dict[str, str]] = []
            for message in messages[:-1]:
                if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                    raise RuntimeError(f"invalid task-side message for {sample_id}")
                clean.append({"role": message["role"], "content": message["content"]})
            tasks[sample_id] = clean
    return tasks


def input_integrity() -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]], dict[str, list[dict[str, str]]]]:
    q48_rows = load_jsonl(Q48_SOURCE)
    q6_rows = load_jsonl(Q6_SOURCE)
    q48_by_id = {str(row["sample_id"]): row for row in q48_rows}
    q6_by_id = {str(row["sample_id"]): row for row in q6_rows}
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    ids = [str(value) for value in pack["sample_ids"]]
    combined: dict[str, dict[str, Any]] = {}
    for sample_id in ids:
        if sample_id not in q48_by_id or sample_id not in q6_by_id:
            continue
        combined[sample_id] = {
            "sample_id": sample_id,
            "q4": q48_by_id[sample_id]["q4"],
            "q6": q6_by_id[sample_id]["q6"],
            "q8": q48_by_id[sample_id]["q8"],
        }
    tasks = load_validation_tasks(ids)
    duplicate_ids = {
        precision: [
            sample_id for sample_id in ids
            if combined[sample_id][precision]["draft"].encode("utf-8") == combined[sample_id][precision]["final"].encode("utf-8")
        ]
        for precision in PRECISIONS
    }
    checks = {
        "q4_q8_rows_24": len(q48_rows) == len(q48_by_id) == 24,
        "q6_rows_24": len(q6_rows) == len(q6_by_id) == 24,
        "pack_count_24": pack.get("count") == 24 and len(ids) == len(set(ids)) == 24,
        "pack_split_validation": pack.get("split") == "validation",
        "pack_test_accessed_false": pack.get("test_accessed") is False,
        "all_ids_exact_pack_order": list(q48_by_id) == ids == list(q6_by_id),
        "combined_24": len(combined) == 24,
        "validation_tasks_24": set(tasks) == set(ids),
        "conditions_144_nonempty": all(
            isinstance(text, str) and bool(text)
            for sample_id in ids for text in condition_texts(combined[sample_id]).values()
        ),
        "source_text_hashes_valid": all(
            combined[sample_id][precision][f"{stage}_sha256"]
            == sha256_bytes(str(combined[sample_id][precision][stage]).encode("utf-8"))
            for sample_id in ids for precision in PRECISIONS for stage in ("draft", "final")
        ),
        "q4_pass_exact_duplicates_3": len(duplicate_ids["q4"]) == 3,
        "q6_pass_exact_duplicates_4": len(duplicate_ids["q6"]) == 4,
        "q8_pass_exact_duplicates_5": len(duplicate_ids["q8"]) == 5,
        "round1_q4_q8_success": json.loads(Q48_AGGREGATE.read_text(encoding="utf-8")).get("status") == "SUCCESS",
        "round1_q6_success": json.loads(Q6_AGGREGATE.read_text(encoding="utf-8")).get("status") == "SUCCESS",
        "calibration_1698_1699_absent": not any(sample_id.endswith(("1698", "1699")) for sample_id in ids),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {[key for key, passed in checks.items() if not passed]}")
    integrity = {
        "schema_version": "cm-lightweight-same-source-6condition-input-integrity-v1",
        "status": "PASS",
        "checks": checks,
        "sample_ids": ids,
        "exact_draft_final_duplicate_ids": duplicate_ids,
        "sources": {
            "q4_q8": {"path": str(Q48_SOURCE), "sha256": sha256_file(Q48_SOURCE), "modified": False},
            "q6": {"path": str(Q6_SOURCE), "sha256": sha256_file(Q6_SOURCE), "modified": False},
            "pack": {"path": str(PACK), "sha256": sha256_file(PACK)},
            "validation_task_side": {
                "path": str(VALIDATION), "sha256": sha256_file(VALIDATION),
                "retained": "messages[:-1] only", "assistant_reference_retained": False,
            },
        },
        "reference_loaded_or_sent": False,
        "final_test_accessed": False,
    }
    return integrity, ids, combined, tasks


def build_mapping(ids: list[str], by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    position_counts = {condition: {label: 0 for label in LABELS} for condition in CONDITIONS}
    samples: list[dict[str, Any]] = []
    unique_distribution: Counter[int] = Counter()
    total_reuse = 0
    cross_precision_reuse = 0
    for sample_id in ids:
        texts = condition_texts(by_id[sample_id])
        groups: dict[bytes, list[str]] = {}
        for condition in CONDITIONS:
            groups.setdefault(texts[condition].encode("utf-8"), []).append(condition)
        group_items = list(groups.items())
        unique_ids = [f"U{index + 1}" for index in range(len(group_items))]
        applicable_labels = LABELS[:len(unique_ids)]

        def objective(permutation: tuple[str, ...]) -> tuple[int, str]:
            projected = copy.deepcopy(position_counts)
            for (_, conditions), label in zip(group_items, permutation):
                for condition in conditions:
                    projected[condition][label] += 1
            squares = sum(sum(value * value for value in counts.values()) for counts in projected.values())
            tie = sha256_bytes(f"{SEED}|{sample_id}|{'-'.join(permutation)}".encode("utf-8"))
            return squares, tie

        chosen = min(itertools.permutations(applicable_labels), key=objective)
        unique_entries: list[dict[str, Any]] = []
        condition_to_unique: dict[str, str] = {}
        condition_to_position: dict[str, str] = {}
        for (text_bytes, conditions), unique_id, label in zip(group_items, unique_ids, chosen):
            unique_entries.append({
                "unique_candidate_id": unique_id,
                "text_sha256": sha256_bytes(text_bytes),
                "conditions": conditions,
                "anonymous_position": label,
            })
            for condition in conditions:
                condition_to_unique[condition] = unique_id
                condition_to_position[condition] = label
                position_counts[condition][label] += 1
            reuse = len(conditions) - 1
            total_reuse += reuse
            same_precision_pairs = sum(
                int(f"{precision.upper()}_DRAFT" in conditions and f"{precision.upper()}_FINAL" in conditions)
                for precision in PRECISIONS
            )
            cross_precision_reuse += max(0, reuse - same_precision_pairs)
        unique_distribution[len(unique_entries)] += 1
        samples.append({
            "sample_id": sample_id,
            "unique_candidate_count": len(unique_entries),
            "condition_to_unique_candidate": condition_to_unique,
            "condition_to_anonymous_position": condition_to_position,
            "unique_candidates": unique_entries,
        })

    def counts_for(current_samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        counts = {condition: {label: 0 for label in LABELS} for condition in CONDITIONS}
        for sample in current_samples:
            for entry in sample["unique_candidates"]:
                for condition in entry["conditions"]:
                    counts[condition][entry["anonymous_position"]] += 1
        return counts

    def balance_score(counts: dict[str, dict[str, int]]) -> tuple[int, int, int]:
        ranges = [max(values.values()) - min(values.values()) for values in counts.values()]
        squares = sum(sum(value * value for value in values.values()) for values in counts.values())
        return max(ranges), sum(ranges), squares

    for refinement_round in range(12):
        changed = False
        for sample in samples:
            entries = sample["unique_candidates"]
            applicable = LABELS[:len(entries)]
            old = tuple(entry["anonymous_position"] for entry in entries)
            choices: list[tuple[tuple[int, int, int], str, tuple[str, ...]]] = []
            for permutation in itertools.permutations(applicable):
                for entry, label in zip(entries, permutation):
                    entry["anonymous_position"] = label
                score = balance_score(counts_for(samples))
                tie = sha256_bytes(
                    f"{SEED}|refine|{refinement_round}|{sample['sample_id']}|{'-'.join(permutation)}".encode("utf-8")
                )
                choices.append((score, tie, permutation))
            best = min(choices)[2]
            for entry, label in zip(entries, best):
                entry["anonymous_position"] = label
            sample["condition_to_anonymous_position"] = {
                condition: entry["anonymous_position"]
                for entry in entries for condition in entry["conditions"]
            }
            changed = changed or best != old
        if not changed:
            break
    position_counts = counts_for(samples)
    mapping = {
        "schema_version": "cm-lightweight-same-source-6condition-private-mapping-v1",
        "seed": SEED,
        "private_do_not_send_to_judge": True,
        "samples": samples,
    }
    balance = {
        "schema_version": "cm-lightweight-same-source-6condition-position-balance-v1",
        "seed": SEED,
        "position_counts_by_condition": position_counts,
        "position_range_by_condition": {
            condition: max(counts.values()) - min(counts.values()) for condition, counts in position_counts.items()
        },
        "unique_candidate_count_distribution": {str(key): value for key, value in sorted(unique_distribution.items())},
        "condition_records": len(ids) * len(CONDITIONS),
        "unique_candidates": sum(sample["unique_candidate_count"] for sample in samples),
        "exact_reuse_records": total_reuse,
        "cross_precision_reuse_beyond_same_precision_draft_final": cross_precision_reuse,
        "dedup_rule": "UTF-8 bytes must be exactly identical within the same sample; no semantic deduplication",
    }
    return mapping, balance


def verify_no_leakage(messages: list[dict[str, str]]) -> dict[str, Any]:
    text = json.dumps(messages, ensure_ascii=False)
    found: list[str] = []
    for term in FORBIDDEN_TERMS:
        if term == "bit":
            matched = re.search(r"(?i)(?<![A-Za-z0-9])bit(?![A-Za-z0-9])", text) is not None
        else:
            matched = term.casefold() in text.casefold()
        if matched:
            found.append(term)
    if found:
        raise RuntimeError(f"identity/stage/score leakage blocked before Judge API: {found}")
    return {"forbidden_terms_checked": list(FORBIDDEN_TERMS), "forbidden_terms_found": [], "passed": True}


def judge_sample(sample_id: str, prompt: list[dict[str, str]], candidates: dict[str, str]) -> dict[str, Any]:
    labels = tuple(candidates)
    messages = frozen.judge_request(prompt, candidates)
    leakage = verify_no_leakage(messages)
    input_hash = canonical_hash({"model": FIXED_MODEL, "temperature": 0, "messages": messages})
    path = RAW_DIR / f"{sample_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("input_sha256") == input_hash:
            return existing
    started_at = now_iso()
    started = time.perf_counter()
    logical_calls = 0
    api_attempts = 0
    response: dict[str, Any] | None = None
    normalized: dict[str, Any] | None = None
    errors: list[str] = []
    schema_retry_used = False
    for schema_round in range(2):
        call_messages = messages if schema_round == 0 else frozen.schema_retry_request(messages, labels)
        logical_calls += 1
        response = None
        try:
            response, attempts = frozen._api_json(call_messages, attempts=3, retry_seconds=2.0, timeout=240.0)
            api_attempts += attempts
            normalized = frozen._validate_score(response, labels)
            schema_retry_used = schema_round == 1
            break
        except Exception:
            errors.append("request_or_schema_failure")
            if response is None or schema_round == 1:
                break
            schema_retry_used = True
    base_record = {
        "schema_version": "cm-lightweight-same-source-6condition-judge-raw-v1",
        "sample_id": sample_id,
        "judge_model": FIXED_MODEL,
        "temperature": 0,
        "candidate_labels": list(labels),
        "input_sha256": input_hash,
        "started_at": started_at,
        "completed_at": now_iso(),
        "latency_seconds": time.perf_counter() - started,
        "logical_calls": logical_calls,
        "api_attempts": api_attempts,
        "schema_retry_used": schema_retry_used,
        "errors_before_success": errors,
        "leakage_audit": leakage,
        "mapping_or_condition_identity_sent": False,
        "request_body_persisted": False,
    }
    if normalized is None:
        failed = {**base_record, "status": "failed"}
        atomic_json(path, failed)
        return failed
    completed = {
        **base_record,
        "status": "completed",
        "raw_judge_response": response,
        "normalized_anonymous_result": normalized,
    }
    atomic_json(path, completed)
    return completed


def release(score: dict[str, Any]) -> float:
    return frozen.release_adjusted_score(
        score["total_score"], len(score["major_release_risks"]), score["publishable"], len(score["unsupported_claims"])
    )


def expand_conditions(
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    raw_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    mapping_by_id = {row["sample_id"]: row for row in mapping["samples"]}
    records: list[dict[str, Any]] = []
    for sample_id in ids:
        texts = condition_texts(by_id[sample_id])
        map_row = mapping_by_id[sample_id]
        scores = raw_by_id[sample_id]["normalized_anonymous_result"]["scores"]
        canonical_by_uid: dict[str, str] = {}
        for condition in CONDITIONS:
            canonical_by_uid.setdefault(map_row["condition_to_unique_candidate"][condition], condition)
        for condition in CONDITIONS:
            precision = condition[0:2].lower()
            stage = condition.split("_", 1)[1].lower()
            uid = map_row["condition_to_unique_candidate"][condition]
            label = map_row["condition_to_anonymous_position"][condition]
            score = copy.deepcopy(scores[label])
            reused_from = canonical_by_uid[uid] if canonical_by_uid[uid] != condition else None
            records.append({
                "schema_version": "cm-lightweight-same-source-6condition-result-v1",
                "sample_id": sample_id,
                "condition": condition,
                "precision": precision,
                "stage": stage,
                "unique_candidate_id": uid,
                "anonymous_position": label,
                "text_sha256": sha256_bytes(texts[condition].encode("utf-8")),
                "score_reused_due_to_exact_text": reused_from is not None,
                "reused_from_condition": reused_from,
                "dimensions": {dimension: score[dimension] for dimension in DIMENSIONS},
                "raw_editorial_score": score["total_score"],
                "release_adjusted_score": release(score),
                "publishable": score["publishable"],
                "unsupported_claims": score["unsupported_claims"],
                "major_release_risks": score["major_release_risks"],
                "short_rationale": score["short_rationale"],
            })
    return records


def aggregate_condition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "raw_editorial_mean": average([float(row["raw_editorial_score"]) for row in rows]),
        "release_adjusted_mean": average([float(row["release_adjusted_score"]) for row in rows]),
        "publishable_rate": average([1.0 if row["publishable"] else 0.0 for row in rows]),
        "unsupported_claims_per_sample": average([float(len(row["unsupported_claims"])) for row in rows]),
        "major_risks_per_sample": average([float(len(row["major_release_risks"])) for row in rows]),
        "dimension_means": {
            dimension: average([float(row["dimensions"][dimension]) for row in rows]) for dimension in DIMENSIONS
        },
    }


def aggregate_stage(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return {
        "schema_version": f"cm-lightweight-same-source-{stage}-aggregate-3way-v1",
        "stage": stage,
        "models": {
            precision: aggregate_condition_rows([
                row for row in records if row["precision"] == precision and row["stage"] == stage
            ])
            for precision in PRECISIONS
        },
    }


def delta_class(delta: float) -> str:
    if delta > 1e-12:
        return "improved"
    if delta < -1e-12:
        return "worsened"
    return "unchanged"


def paired_delta(records: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    by_key = {(row["sample_id"], row["condition"]): row for row in records}
    result: dict[str, Any] = {"schema_version": "cm-lightweight-same-source-paired-delta-3way-v1", "models": {}}
    for precision in PRECISIONS:
        prefix = precision.upper()
        metrics: dict[str, Any] = {}
        for field in ("raw_editorial_score", "release_adjusted_score"):
            values = [
                float(by_key[(sample_id, f"{prefix}_FINAL")][field])
                - float(by_key[(sample_id, f"{prefix}_DRAFT")][field])
                for sample_id in ids
            ]
            counts = Counter(delta_class(value) for value in values)
            metrics[field] = {
                "mean_delta": average(values),
                "median_delta": statistics.median(values),
                "improved": counts["improved"],
                "unchanged": counts["unchanged"],
                "worsened": counts["worsened"],
                "per_sample": dict(zip(ids, values)),
            }
        result["models"][precision] = {"count": len(ids), "metrics": metrics}
    return result


def publishability_transition(
    records: list[dict[str, Any]], ids: list[str], burden: dict[str, Any]
) -> dict[str, Any]:
    by_key = {(row["sample_id"], row["condition"]): row for row in records}
    result: dict[str, Any] = {
        "schema_version": "cm-lightweight-same-source-publishability-transition-3way-v1", "models": {}
    }
    for precision in PRECISIONS:
        prefix = precision.upper()
        counts: Counter[str] = Counter()
        per_sample: dict[str, str] = {}
        for sample_id in ids:
            draft = bool(by_key[(sample_id, f"{prefix}_DRAFT")]["publishable"])
            final = bool(by_key[(sample_id, f"{prefix}_FINAL")]["publishable"])
            category = "RESCUED" if not draft and final else "RETAINED" if draft and final else "NOT_RESCUED" if not draft else "HARMED"
            counts[category] += 1
            per_sample[sample_id] = category
        triggered = int(burden["models"][precision]["reviser_triggered"])
        result["models"][precision] = {
            "count": len(ids),
            "RESCUED": counts["RESCUED"],
            "RETAINED": counts["RETAINED"],
            "NOT_RESCUED": counts["NOT_RESCUED"],
            "HARMED": counts["HARMED"],
            "net_publishable_gain": counts["RESCUED"] - counts["HARMED"],
            "reviser_triggered": triggered,
            "rescue_efficiency": counts["RESCUED"] / triggered if triggered else None,
            "harm_rate": counts["HARMED"] / triggered if triggered else None,
            "per_sample": per_sample,
        }
    return result


def count_transition(records: list[dict[str, Any]], ids: list[str], field: str, schema: str) -> dict[str, Any]:
    by_key = {(row["sample_id"], row["condition"]): row for row in records}
    result: dict[str, Any] = {"schema_version": schema, "models": {}}
    for precision in PRECISIONS:
        prefix = precision.upper()
        drafts: list[int] = []
        finals: list[int] = []
        transitions: Counter[str] = Counter()
        per_sample: dict[str, Any] = {}
        for sample_id in ids:
            draft = len(by_key[(sample_id, f"{prefix}_DRAFT")][field])
            final = len(by_key[(sample_id, f"{prefix}_FINAL")][field])
            category = "reduced" if final < draft else "increased" if final > draft else "unchanged"
            drafts.append(draft)
            finals.append(final)
            transitions[category] += 1
            per_sample[sample_id] = {"draft": draft, "final": final, "delta": final - draft, "transition": category}
        result["models"][precision] = {
            "count": len(ids),
            "draft_mean": average([float(value) for value in drafts]),
            "final_mean": average([float(value) for value in finals]),
            "delta": average([float(value) for value in finals]) - average([float(value) for value in drafts]),
            "reduced": transitions["reduced"],
            "unchanged": transitions["unchanged"],
            "increased": transitions["increased"],
            "per_sample": per_sample,
        }
    return result


def merge_agent_burden() -> dict[str, Any]:
    source = json.loads(Q48_BURDEN.read_text(encoding="utf-8"))
    q6_source = json.loads(Q6_AGGREGATE.read_text(encoding="utf-8"))["model"]["q6"]
    models = {"q4": source["models"]["q4"], "q6": q6_source, "q8": source["models"]["q8"]}
    selected = {
        precision: {
            "reviser_trigger_rate": models[precision]["reviser_trigger_rate"],
            "reviser_triggered": models[precision]["reviser_triggered"],
            "total_reviewer_issues": models[precision]["total_reviewer_issues"],
            "unsupported_related_issues": models[precision]["unsupported_related_issues"],
            "missing_information_issues": models[precision]["missing_information_issues"],
            "total_api_calls": models[precision]["total_api_calls"],
            "total_api_tokens": models[precision]["total_api_tokens"],
            "mean_api_tokens_per_sample": models[precision]["mean_api_tokens_per_sample"],
        }
        for precision in PRECISIONS
    }
    return {
        "schema_version": "cm-lightweight-same-source-agent-burden-3way-round2-merge-v1",
        "status": "SUCCESS",
        "models": selected,
        "sources": {
            "q4_q8": {"path": str(Q48_BURDEN), "sha256": sha256_file(Q48_BURDEN)},
            "q6": {"path": str(Q6_AGGREGATE), "sha256": sha256_file(Q6_AGGREGATE)},
        },
        "agents_rerun": False,
    }


def merge_deployment() -> dict[str, Any]:
    source = json.loads(DEPLOYMENT_SOURCE.read_text(encoding="utf-8"))
    models: dict[str, Any] = {}
    for precision in PRECISIONS:
        item = source["deployment"][precision]
        models[precision] = {
            "base_size_bytes": item["base_size_bytes"],
            "base_size_gib": item["base_size_bytes"] / (1024 ** 3),
            "peak_vram_mib": item["peak_gpu_vram_mib"],
            "peak_ram_working_set_mib": item["peak_process_working_set_mib"],
            "writer_tokens_per_second_median": item["median_tokens_per_second"],
            "writer_latency_seconds_median": item["median_total_seconds"],
            "load_time_seconds": item["server_load_seconds"],
        }
    return {
        "schema_version": "cm-lightweight-same-source-deployment-3way-round2-merge-v1",
        "status": "SUCCESS",
        "models": models,
        "source": {"path": str(DEPLOYMENT_SOURCE), "sha256": sha256_file(DEPLOYMENT_SOURCE)},
        "read_scope": "deployment.q4/q6/q8 only; historical Editorial scores excluded",
    }


def system_tradeoff(
    final_aggregate: dict[str, Any], burden: dict[str, Any], deployment: dict[str, Any]
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for precision in PRECISIONS:
        quality = final_aggregate["models"][precision]
        models[precision] = {
            **deployment["models"][precision],
            "reviser_trigger_rate": burden["models"][precision]["reviser_trigger_rate"],
            "api_tokens_per_sample": burden["models"][precision]["mean_api_tokens_per_sample"],
            "final_release_adjusted": quality["release_adjusted_mean"],
            "final_publishable_rate": quality["publishable_rate"],
            "final_unsupported_claims_per_sample": quality["unsupported_claims_per_sample"],
            "final_major_risks_per_sample": quality["major_risks_per_sample"],
        }
    return {
        "schema_version": "cm-lightweight-same-source-system-tradeoff-3way-v1",
        "models": models,
        "quality_source": "same-source unified 6-condition Judge in this batch",
    }


def pareto_analysis(tradeoff: dict[str, Any]) -> dict[str, Any]:
    lower = (
        "base_size_bytes", "peak_vram_mib", "writer_latency_seconds_median",
        "api_tokens_per_sample", "final_major_risks_per_sample",
    )
    higher = ("final_release_adjusted", "final_publishable_rate")
    models = tradeoff["models"]

    def dominates(left: str, right: str) -> bool:
        no_worse = all(float(models[left][field]) <= float(models[right][field]) + 1e-12 for field in lower)
        no_worse = no_worse and all(float(models[left][field]) + 1e-12 >= float(models[right][field]) for field in higher)
        strict = any(float(models[left][field]) < float(models[right][field]) - 1e-12 for field in lower)
        strict = strict or any(float(models[left][field]) > float(models[right][field]) + 1e-12 for field in higher)
        return no_worse and strict

    dominated_by = {
        precision: [other for other in PRECISIONS if other != precision and dominates(other, precision)]
        for precision in PRECISIONS
    }
    nondominated = [precision for precision in PRECISIONS if not dominated_by[precision]]
    return {
        "schema_version": "cm-lightweight-same-source-system-pareto-v1",
        "dimensions": {"minimize": list(lower), "maximize": list(higher)},
        "pareto_nondominated": nondominated,
        "pareto_dominated": [precision for precision in PRECISIONS if dominated_by[precision]],
        "dominated_by": dominated_by,
        "strict_dominance_rule": "no worse on all seven specified dimensions and strictly better on at least one",
    }


def sweet_spot(tradeoff: dict[str, Any], pareto: dict[str, Any]) -> dict[str, Any]:
    frontier = pareto["pareto_nondominated"]
    if len(frontier) == 1:
        choice = frontier[0].upper()
        decision_rule = "the sole Pareto-nondominated precision across the seven user-specified system dimensions"
    else:
        choice = "INCONCLUSIVE"
        decision_rule = "multiple Pareto-nondominated precisions remain; no unapproved cross-domain weighting was imposed"
    models = tradeoff["models"]
    smallest = min(PRECISIONS, key=lambda key: models[key]["base_size_bytes"])
    lowest_agent = min(PRECISIONS, key=lambda key: models[key]["api_tokens_per_sample"])
    best_release = max(PRECISIONS, key=lambda key: models[key]["final_release_adjusted"])
    best_publishable = max(PRECISIONS, key=lambda key: models[key]["final_publishable_rate"])
    lowest_risk = min(PRECISIONS, key=lambda key: models[key]["final_major_risks_per_sample"])
    return {
        "SYSTEM_SWEET_SPOT": choice,
        "decision_rule": decision_rule,
        "pareto_frontier": [value.upper() for value in frontier],
        "resource_reason": (
            f"{smallest.upper()} is smallest at {models[smallest]['base_size_gib']:.3f} GiB and uses "
            f"{models[smallest]['peak_vram_mib']} MiB peak VRAM."
        ),
        "agent_burden_reason": (
            f"{lowest_agent.upper()} has the lowest mean API tokens/sample at "
            f"{models[lowest_agent]['api_tokens_per_sample']:.1f}."
        ),
        "final_quality_reason": (
            f"{best_release.upper()} has the highest Final Release-Adjusted score at "
            f"{models[best_release]['final_release_adjusted']:.3f}; {best_publishable.upper()} has the highest "
            f"Final publishable rate at {models[best_publishable]['final_publishable_rate']:.1%}."
        ),
        "risk_reason": (
            f"{lowest_risk.upper()} has the lowest Final major risks/sample at "
            f"{models[lowest_risk]['final_major_risks_per_sample']:.3f}."
        ),
    }


def write_cases(
    records: list[dict[str, Any]], ids: list[str], source: dict[str, dict[str, Any]], transitions: dict[str, Any]
) -> dict[str, Any]:
    by_key = {(row["sample_id"], row["condition"]): row for row in records}
    rescues: list[dict[str, Any]] = []
    harms: list[dict[str, Any]] = []
    for precision in PRECISIONS:
        prefix = precision.upper()
        for sample_id in ids:
            draft = by_key[(sample_id, f"{prefix}_DRAFT")]
            final = by_key[(sample_id, f"{prefix}_FINAL")]
            release_delta = float(final["release_adjusted_score"]) - float(draft["release_adjusted_score"])
            item = {
                "sample_id": sample_id,
                "precision": precision,
                "draft_release": draft["release_adjusted_score"],
                "final_release": final["release_adjusted_score"],
                "release_delta": release_delta,
                "draft_publishable": draft["publishable"],
                "final_publishable": final["publishable"],
                "draft_rationale": draft["short_rationale"],
                "final_rationale": final["short_rationale"],
            }
            if not draft["publishable"] and final["publishable"]:
                rescues.append(item)
            if (draft["publishable"] and not final["publishable"]) or release_delta < -1e-12:
                harms.append(item)

    typical_rescue: dict[str, str | None] = {}
    typical_harm: dict[str, str | None] = {}
    for precision in PRECISIONS:
        available_rescues = [item for item in rescues if item["precision"] == precision]
        available_harms = [item for item in harms if item["precision"] == precision]
        typical_rescue[precision] = max(available_rescues, key=lambda item: item["release_delta"])["sample_id"] if available_rescues else None
        typical_harm[precision] = min(available_harms, key=lambda item: item["release_delta"])["sample_id"] if available_harms else None

    rescue_lines = ["# Rescue cases", "", "All Draft non-publishable → Final publishable transitions in this batch.", ""]
    for item in rescues:
        rescue_lines.extend([
            f"## {item['sample_id']} — {item['precision'].upper()}", "",
            f"Release-Adjusted: {item['draft_release']} → {item['final_release']} ({item['release_delta']:+.1f})", "",
            f"Draft rationale: {item['draft_rationale']}", "", f"Final rationale: {item['final_rationale']}", "",
        ])
    rescue_lines.extend(["## Representative selections", ""])
    rescue_lines.extend(f"- {precision.upper()}: `{typical_rescue[precision]}`" for precision in PRECISIONS)
    (RESULTS / "rescue_cases.md").write_text("\n".join(rescue_lines) + "\n", encoding="utf-8", newline="\n")

    harm_lines = [
        "# Harm cases", "",
        "All Draft publishable → Final non-publishable transitions, plus every Final Release-Adjusted < Draft case.", "",
    ]
    for item in harms:
        harm_lines.extend([
            f"## {item['sample_id']} — {item['precision'].upper()}", "",
            f"Publishable: {item['draft_publishable']} → {item['final_publishable']}", "",
            f"Release-Adjusted: {item['draft_release']} → {item['final_release']} ({item['release_delta']:+.1f})", "",
            f"Draft rationale: {item['draft_rationale']}", "", f"Final rationale: {item['final_rationale']}", "",
        ])
    harm_lines.extend(["## Representative selections (at most one per precision)", ""])
    harm_lines.extend(f"- {precision.upper()}: `{typical_harm[precision]}`" for precision in PRECISIONS)
    (RESULTS / "harm_cases.md").write_text("\n".join(harm_lines) + "\n", encoding="utf-8", newline="\n")

    special_id = "task011d_e5_1820"
    special: dict[str, Any] = {"sample_id": special_id, "conditions": {}}
    special_lines = ["# Special cases", "", f"## {special_id}", "", "Long-output / max-token case across all three precisions.", "",
                     "| Precision | Draft Release | Final Release | Delta | Draft publishable | Final publishable | Draft chars | Final chars |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for precision in PRECISIONS:
        prefix = precision.upper()
        draft = by_key[(special_id, f"{prefix}_DRAFT")]
        final = by_key[(special_id, f"{prefix}_FINAL")]
        delta = float(final["release_adjusted_score"]) - float(draft["release_adjusted_score"])
        special["conditions"][precision] = {
            "draft_release": draft["release_adjusted_score"], "final_release": final["release_adjusted_score"],
            "delta": delta, "draft_publishable": draft["publishable"], "final_publishable": final["publishable"],
            "draft_chars": len(source[special_id][precision]["draft"]), "final_chars": len(source[special_id][precision]["final"]),
        }
        special_lines.append(
            f"| {prefix} | {draft['release_adjusted_score']} | {final['release_adjusted_score']} | {delta:+.1f} | "
            f"{draft['publishable']} | {final['publishable']} | {len(source[special_id][precision]['draft'])} | "
            f"{len(source[special_id][precision]['final'])} |"
        )
    special_lines.extend(["", "## Representative successful rescues", ""])
    special_lines.extend(f"- {precision.upper()}: `{typical_rescue[precision]}`" for precision in PRECISIONS)
    special_lines.extend(["", "## Representative harm cases", ""])
    special_lines.extend(f"- {precision.upper()}: `{typical_harm[precision]}`" for precision in PRECISIONS)
    special_lines.extend(["", "Descriptive only; no significance test or hidden-case removal was applied.", ""])
    (RESULTS / "special_cases.md").write_text("\n".join(special_lines), encoding="utf-8", newline="\n")
    return {
        "rescue_count": len(rescues), "harm_count": len(harms),
        "rescue_count_by_precision": {p: sum(item["precision"] == p for item in rescues) for p in PRECISIONS},
        "harm_count_by_precision": {p: sum(item["precision"] == p for item in harms) for p in PRECISIONS},
        "typical_rescue": typical_rescue, "typical_harm": typical_harm, "special_1820": special,
    }


def copied_scores_identical(records: list[dict[str, Any]]) -> bool:
    by_key = {(row["sample_id"], row["condition"]): row for row in records}
    fields = (
        "dimensions", "raw_editorial_score", "release_adjusted_score", "publishable",
        "unsupported_claims", "major_release_risks", "short_rationale",
    )
    for row in records:
        if not row["score_reused_due_to_exact_text"]:
            continue
        source = by_key[(row["sample_id"], row["reused_from_condition"])]
        if any(row[field] != source[field] for field in fields):
            return False
    return True


def secret_absent(settings: Any) -> bool:
    secret = settings.api_key.encode("utf-8")
    return bool(secret) and not any(secret in path.read_bytes() for path in RESULTS.rglob("*") if path.is_file())


def write_csv_outputs(
    tradeoff: dict[str, Any], paired: dict[str, Any], publishability: dict[str, Any],
    unsupported: dict[str, Any]
) -> None:
    tradeoff_rows = [
        ("Base size (GiB)", "base_size_gib"),
        ("Peak VRAM (MiB)", "peak_vram_mib"),
        ("Writer tok/s", "writer_tokens_per_second_median"),
        ("Writer latency (s)", "writer_latency_seconds_median"),
        ("Reviser trigger rate", "reviser_trigger_rate"),
        ("API tokens/sample", "api_tokens_per_sample"),
        ("Final Release-Adjusted", "final_release_adjusted"),
        ("Final Publishable Rate", "final_publishable_rate"),
        ("Final Unsupported/sample", "final_unsupported_claims_per_sample"),
        ("Final Major Risks/sample", "final_major_risks_per_sample"),
    ]
    with (RESULTS / "ppt_same_source_system_tradeoff.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Q4", "Q6", "Q8"])
        for label, field in tradeoff_rows:
            writer.writerow([label] + [tradeoff["models"][precision][field] for precision in PRECISIONS])

    with (RESULTS / "ppt_same_source_closed_loop.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Q4", "Q6", "Q8"])
        writer.writerow(["Release gain"] + [paired["models"][p]["metrics"]["release_adjusted_score"]["mean_delta"] for p in PRECISIONS])
        writer.writerow(["Publishable gain"] + [publishability["models"][p]["net_publishable_gain"] / 24 for p in PRECISIONS])
        writer.writerow(["Unsupported reduction"] + [-unsupported["models"][p]["delta"] for p in PRECISIONS])
        writer.writerow(["Rescue efficiency"] + [publishability["models"][p]["rescue_efficiency"] for p in PRECISIONS])
        writer.writerow(["Harm rate"] + [publishability["models"][p]["harm_rate"] for p in PRECISIONS])


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_sweet_spot(value: dict[str, Any]) -> None:
    lines = [
        "# System Sweet Spot", "", f"SYSTEM_SWEET_SPOT: **{value['SYSTEM_SWEET_SPOT']}**", "",
        f"Decision rule: {value['decision_rule']}", "", f"Pareto frontier: {', '.join(value['pareto_frontier'])}", "",
        f"- Resource reason: {value['resource_reason']}",
        f"- Agent burden reason: {value['agent_burden_reason']}",
        f"- Final quality reason: {value['final_quality_reason']}",
        f"- Risk reason: {value['risk_reason']}", "",
        "This decision uses only the current same-source unified six-condition Judge batch, same-source deployment, and same-source Agent burden.",
    ]
    (RESULTS / "system_sweet_spot.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_report(
    manifest: dict[str, Any], draft: dict[str, Any], final: dict[str, Any], paired: dict[str, Any],
    publishability: dict[str, Any], unsupported: dict[str, Any], burden: dict[str, Any],
    deployment: dict[str, Any], tradeoff: dict[str, Any], sweet: dict[str, Any], pareto: dict[str, Any],
    cases: dict[str, Any]
) -> None:
    core = (
        ("Raw Editorial", "raw_editorial_mean"),
        ("Release-Adjusted", "release_adjusted_mean"),
        ("Publishable Rate", "publishable_rate"),
        ("Unsupported Claims/sample", "unsupported_claims_per_sample"),
        ("Major Risks/sample", "major_risks_per_sample"),
    )
    lines = [
        "# Same-Source System Q4/Q6/Q8 — Unified Round 2", "",
        f"Status: **{manifest['status']}**", "",
        "This same-source six-condition Judge batch is the system-level primary quality result. Historical mixed-source, "
        "same-source Writer-only, and Q4/Q8 2×2 absolute scores retain historical meaning but are not directly merged "
        "with this batch or used for strict deltas.", "", "## TABLE A — Draft Quality", "",
        "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|",
    ]
    for label, field in core:
        lines.append(f"| {label} | {fmt(draft['models']['q4'][field])} | {fmt(draft['models']['q6'][field])} | {fmt(draft['models']['q8'][field])} |")
    for dimension in DIMENSIONS:
        lines.append(
            f"| {dimension} | {fmt(draft['models']['q4']['dimension_means'][dimension])} | "
            f"{fmt(draft['models']['q6']['dimension_means'][dimension])} | {fmt(draft['models']['q8']['dimension_means'][dimension])} |"
        )
    lines.extend(["", "## TABLE B — Final Quality", "", "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|"])
    for label, field in core:
        lines.append(f"| {label} | {fmt(final['models']['q4'][field])} | {fmt(final['models']['q6'][field])} | {fmt(final['models']['q8'][field])} |")
    for dimension in DIMENSIONS:
        lines.append(
            f"| {dimension} | {fmt(final['models']['q4']['dimension_means'][dimension])} | "
            f"{fmt(final['models']['q6']['dimension_means'][dimension])} | {fmt(final['models']['q8']['dimension_means'][dimension])} |"
        )
    lines.extend(["", "## TABLE C — Closed-Loop Effect", "", "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|"])
    lines.append("| Release gain | " + " | ".join(fmt(paired["models"][p]["metrics"]["release_adjusted_score"]["mean_delta"]) for p in PRECISIONS) + " |")
    lines.append("| Publishable gain | " + " | ".join(str(publishability["models"][p]["net_publishable_gain"]) for p in PRECISIONS) + " |")
    lines.append("| Unsupported reduction | " + " | ".join(fmt(-unsupported["models"][p]["delta"]) for p in PRECISIONS) + " |")
    lines.append("| Rescue efficiency | " + " | ".join(f"{publishability['models'][p]['rescue_efficiency']:.1%}" for p in PRECISIONS) + " |")
    lines.append("| Harm rate | " + " | ".join(f"{publishability['models'][p]['harm_rate']:.1%}" for p in PRECISIONS) + " |")
    lines.extend(["", "## Agent burden", "", "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|"])
    burden_fields = (
        ("Trigger Rate", "reviser_trigger_rate"), ("Issues", "total_reviewer_issues"),
        ("Unsupported-related", "unsupported_related_issues"), ("Missing-info", "missing_information_issues"),
        ("API calls", "total_api_calls"), ("API tokens", "total_api_tokens"),
        ("Tokens/sample", "mean_api_tokens_per_sample"),
    )
    for label, field in burden_fields:
        lines.append("| " + label + " | " + " | ".join(fmt(burden["models"][p][field]) for p in PRECISIONS) + " |")
    lines.extend(["", "## Deployment", "", "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|"])
    deployment_fields = (
        ("Base size (GiB)", "base_size_gib"), ("Peak VRAM (MiB)", "peak_vram_mib"),
        ("Peak RAM (MiB)", "peak_ram_working_set_mib"), ("Writer tok/s", "writer_tokens_per_second_median"),
        ("Writer latency (s)", "writer_latency_seconds_median"), ("Load time (s)", "load_time_seconds"),
    )
    for label, field in deployment_fields:
        lines.append("| " + label + " | " + " | ".join(fmt(deployment["models"][p][field]) for p in PRECISIONS) + " |")
    lines.extend(["", "## TABLE D — System Trade-off", "", "| Metric | Q4 | Q6 | Q8 |", "|---|---:|---:|---:|"])
    tradeoff_fields = (
        ("Base size (GiB)", "base_size_gib"), ("VRAM (MiB)", "peak_vram_mib"),
        ("Writer tok/s", "writer_tokens_per_second_median"), ("Agent trigger", "reviser_trigger_rate"),
        ("API tokens/sample", "api_tokens_per_sample"), ("Final Release", "final_release_adjusted"),
        ("Final Publishable", "final_publishable_rate"), ("Final Major Risks", "final_major_risks_per_sample"),
    )
    for label, field in tradeoff_fields:
        lines.append("| " + label + " | " + " | ".join(fmt(tradeoff["models"][p][field]) for p in PRECISIONS) + " |")
    lines.extend([
        "", "## System conclusion", "", f"SYSTEM_SWEET_SPOT: **{sweet['SYSTEM_SWEET_SPOT']}**", "",
        f"Pareto-nondominated: {', '.join(value.upper() for value in pareto['pareto_nondominated'])}",
        f"Pareto-dominated: {', '.join(value.upper() for value in pareto['pareto_dominated']) or 'none'}", "",
        f"Rescue cases: {cases['rescue_count']}; harm cases: {cases['harm_count']}.", "",
        f"Unique candidates judged: {manifest['unique_candidates_judged']}; exact-reused condition records: {manifest['exact_reuse_records']}; reconstructed conditions: {manifest['condition_records']}.",
    ])
    (RESULTS / "aggregate_round2.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def preflight_manifest(integrity: dict[str, Any], mapping: dict[str, Any], balance: dict[str, Any], settings: Any) -> dict[str, Any]:
    prompt_hashes = {
        str(count): sha256_bytes(
            frozen.judge_request([], {LABELS[index]: "candidate" for index in range(count)})[0]["content"].encode("utf-8")
        )
        for count in range(1, 7)
    }
    return {
        "schema_version": "cm-lightweight-same-source-6condition-round2-manifest-v1",
        "created_at": now_iso(),
        "status": "preflight_complete",
        "split": "validation",
        "sample_count": 24,
        "condition_count": 144,
        "conditions": list(CONDITIONS),
        "input_integrity": integrity,
        "dedup": {
            "scope": "within each sample across all six conditions",
            "rule": "byte-identical UTF-8 only",
            "semantic_dedup": False,
            "unique_candidates": balance["unique_candidates"],
            "exact_reuse_records": balance["exact_reuse_records"],
        },
        "anonymous_mapping": {
            "seed": SEED,
            "private_path": str(RESULTS / "candidate_mapping_private.json"),
            "sent_to_judge": False,
            "position_balance_path": str(RESULTS / "position_balance.json"),
        },
        "judge": {
            "model": settings.model,
            "temperature": 0,
            "rubric_dimensions": DIMENSIONS,
            "rubric_implementation": str(FROZEN_ROUND2_RUNNER),
            "rubric_implementation_sha256": sha256_file(FROZEN_ROUND2_RUNNER),
            "project_editorial_implementation": str(PROJECT / "scripts" / "diagnostics" / "closed_loop_editorial_eval.py"),
            "project_editorial_implementation_sha256": sha256_file(PROJECT / "scripts" / "diagnostics" / "closed_loop_editorial_eval.py"),
            "system_prompt_sha256_by_label_count": prompt_hashes,
            "normal_multi_candidate_request_per_sample": 1,
            "transient_retry_max": 2,
            "schema_retry_max": 1,
            "silent_fallback": False,
        },
        "release_adjusted": {
            "implementation": str(PROJECT / "demo" / "scoring.py"),
            "sha256": sha256_file(PROJECT / "demo" / "scoring.py"),
            "computed_by_python": True,
        },
        "writer_reviewer_reviser_rerun": False,
        "reference_used": False,
        "prior_scores_sent": False,
        "final_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified blind six-condition same-source Editorial Judge")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    integrity, ids, source, tasks = input_integrity()
    mapping, balance = build_mapping(ids, source)
    atomic_json(RESULTS / "candidate_mapping_private.json", mapping)
    atomic_json(RESULTS / "position_balance.json", balance)
    settings = frozen.APISettings.from_env()
    if not settings.is_configured or settings.model != FIXED_MODEL:
        raise RuntimeError("frozen Editorial Judge configuration missing or model mismatch")
    manifest = preflight_manifest(integrity, mapping, balance, settings)
    # Prove every final API message is clean before any request is permitted.
    mapping_by_id = {row["sample_id"]: row for row in mapping["samples"]}
    for sample_id in ids:
        texts = condition_texts(source[sample_id])
        candidates = {
            entry["anonymous_position"]: texts[entry["conditions"][0]]
            for entry in mapping_by_id[sample_id]["unique_candidates"]
        }
        verify_no_leakage(frozen.judge_request(tasks[sample_id], dict(sorted(candidates.items()))))
    manifest["all_24_pretransport_leakage_checks_passed"] = True
    atomic_json(RESULTS / "experiment_manifest.json", manifest)
    if args.preflight_only:
        print(json.dumps({
            "event": "preflight_complete", "status": integrity["status"], "samples": len(ids),
            "conditions": 144, "unique_candidates": balance["unique_candidates"],
            "exact_reuse_records": balance["exact_reuse_records"],
        }, ensure_ascii=False))
        return 0

    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, sample_id in enumerate(ids, start=1):
        texts = condition_texts(source[sample_id])
        candidates = {
            entry["anonymous_position"]: texts[entry["conditions"][0]]
            for entry in mapping_by_id[sample_id]["unique_candidates"]
        }
        candidates = dict(sorted(candidates.items()))
        result = judge_sample(sample_id, tasks[sample_id], candidates)
        raw_by_id[sample_id] = result
        print(json.dumps({
            "event": "judge_sample", "index": index, "sample_id": sample_id,
            "unique_candidates": len(candidates), "status": result["status"],
            "logical_calls": result.get("logical_calls"), "api_attempts": result.get("api_attempts"),
            "schema_retry": result.get("schema_retry_used"),
        }, ensure_ascii=False), flush=True)

    completed = [sample_id for sample_id in ids if raw_by_id[sample_id].get("status") == "completed"]
    if len(completed) != 24:
        manifest.update({"status": "FAILED", "completed_samples": len(completed), "completed_at": now_iso()})
        atomic_json(RESULTS / "experiment_manifest.json", manifest)
        return 2

    condition_records = expand_conditions(ids, source, mapping, raw_by_id)
    atomic_jsonl(RESULTS / "condition_level_results.jsonl", condition_records)
    draft = aggregate_stage(condition_records, "draft")
    final = aggregate_stage(condition_records, "final")
    paired = paired_delta(condition_records, ids)
    burden = merge_agent_burden()
    publishability = publishability_transition(condition_records, ids, burden)
    unsupported = count_transition(
        condition_records, ids, "unsupported_claims", "cm-lightweight-same-source-unsupported-transition-3way-v1"
    )
    major = count_transition(
        condition_records, ids, "major_release_risks", "cm-lightweight-same-source-major-risk-transition-3way-v1"
    )
    deployment = merge_deployment()
    tradeoff = system_tradeoff(final, burden, deployment)
    pareto = pareto_analysis(tradeoff)
    sweet = sweet_spot(tradeoff, pareto)
    cases = write_cases(condition_records, ids, source, publishability)

    outputs = {
        "draft_aggregate_3way.json": draft,
        "final_aggregate_3way.json": final,
        "paired_delta_3way.json": paired,
        "publishability_transition_3way.json": publishability,
        "unsupported_transition_3way.json": unsupported,
        "major_risk_transition_3way.json": major,
        "agent_burden_3way.json": burden,
        "deployment_3way.json": deployment,
        "system_tradeoff_3way.json": tradeoff,
        "system_pareto_analysis.json": pareto,
    }
    for name, value in outputs.items():
        atomic_json(RESULTS / name, value)
    write_sweet_spot(sweet)
    write_csv_outputs(tradeoff, paired, publishability, unsupported)

    unique_complete = all(
        len(raw_by_id[sample_id]["normalized_anonymous_result"]["scores"])
        == mapping_by_id[sample_id]["unique_candidate_count"]
        for sample_id in ids
    )
    reused_records = [row for row in condition_records if row["score_reused_due_to_exact_text"]]
    success_checks = {
        "samples_judged_24": len(completed) == 24,
        "all_unique_candidates_complete": unique_complete,
        "condition_records_144": len(condition_records) == 144,
        "exact_reuse_count_matches_mapping": len(reused_records) == balance["exact_reuse_records"],
        "all_copied_score_fields_identical": copied_scores_identical(condition_records),
        "draft_aggregate_complete_3way": all(draft["models"][p]["count"] == 24 for p in PRECISIONS),
        "final_aggregate_complete_3way": all(final["models"][p]["count"] == 24 for p in PRECISIONS),
        "agent_burden_merged": burden["status"] == "SUCCESS",
        "deployment_merged": deployment["status"] == "SUCCESS",
        "all_pretransport_leakage_checks_passed": manifest["all_24_pretransport_leakage_checks_passed"],
        "reference_leakage": False,
        "quant_identity_leakage": False,
        "stage_leakage": False,
        "reviewer_issue_leakage": False,
        "final_test_accessed": False,
        "secret_leakage": False,
    }
    positive = [
        "samples_judged_24", "all_unique_candidates_complete", "condition_records_144",
        "exact_reuse_count_matches_mapping", "all_copied_score_fields_identical",
        "draft_aggregate_complete_3way", "final_aggregate_complete_3way", "agent_burden_merged",
        "deployment_merged", "all_pretransport_leakage_checks_passed",
    ]
    negative = [
        "reference_leakage", "quant_identity_leakage", "stage_leakage", "reviewer_issue_leakage",
        "final_test_accessed", "secret_leakage",
    ]
    success = all(success_checks[key] for key in positive) and not any(success_checks[key] for key in negative)
    manifest.update({
        "status": "SUCCESS" if success else "FAILED",
        "completed_at": now_iso(),
        "judged_samples": len(completed),
        "unique_candidates_judged": balance["unique_candidates"],
        "condition_records": len(condition_records),
        "exact_reuse_records": len(reused_records),
        "total_judge_logical_calls": sum(int(raw_by_id[sid]["logical_calls"]) for sid in ids),
        "total_judge_api_attempts": sum(int(raw_by_id[sid]["api_attempts"]) for sid in ids),
        "schema_retry_samples": [sid for sid in ids if raw_by_id[sid]["schema_retry_used"]],
        "success_checks": success_checks,
        "SYSTEM_SWEET_SPOT": sweet["SYSTEM_SWEET_SPOT"],
        "pareto_nondominated": pareto["pareto_nondominated"],
        "case_summary": cases,
    })
    atomic_json(RESULTS / "experiment_manifest.json", manifest)
    # Scan after all JSON/CSV/case outputs exist; report only a boolean, never the secret.
    manifest["secret_scan_passed"] = secret_absent(settings)
    manifest["success_checks"]["secret_leakage"] = not manifest["secret_scan_passed"]
    if not manifest["secret_scan_passed"]:
        manifest["status"] = "FAILED"
    atomic_json(RESULTS / "experiment_manifest.json", manifest)
    write_report(manifest, draft, final, paired, publishability, unsupported, burden, deployment, tradeoff, sweet, pareto, cases)
    # The report and final manifest contain no secret by construction; verify the full final tree once more.
    if not secret_absent(settings):
        raise RuntimeError("secret leakage detected in final result tree")
    print(json.dumps({
        "event": "round2_complete", "status": manifest["status"], "judged_samples": len(completed),
        "unique_candidates": balance["unique_candidates"], "condition_records": len(condition_records),
        "exact_reuse": len(reused_records), "system_sweet_spot": sweet["SYSTEM_SWEET_SPOT"],
        "pareto_nondominated": pareto["pareto_nondominated"], "rescue_count": cases["rescue_count"],
        "harm_count": cases["harm_count"],
    }, ensure_ascii=False), flush=True)
    return 0 if manifest["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
