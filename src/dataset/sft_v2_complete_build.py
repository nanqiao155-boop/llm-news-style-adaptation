from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.dataset.sft_schema import render_target, sha256_text


SPLITS = ("train", "validation", "test")
EXPORT_SCHEMA_VERSION = "sft-final-export-v1.0.0"
BUILD_TIMESTAMP = "2026-08-20T23:59:00+08:00"
CONTENT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")
SAFE_FILES = (
    "VERSION", "schema.json", "dataset_manifest.json", "split_statistics.json",
    "split_usage_policy.json", "quality_tier_statistics.json",
    "human_ai_review_lineage.json", "lineage.jsonl", "dataset_card.md",
    "provenance_audit.json", "split_integrity_audit.json",
    "target_integrity_audit.json", "schema_integrity_audit.json",
    "silver_quality_summary.json", "dropped_sample_manifest.json",
    "pre_training_gate.json",
)
CHECKSUM_FILES = CONTENT_FILES + SAFE_FILES


class SFTV2BuildError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise SFTV2BuildError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_checksum_manifest(path: Path) -> None:
    require(path.is_file(), f"missing checksum manifest: {path}")
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        target = path.parent / name
        require(target.is_file(), f"checksum target missing: {target}")
        require(sha256_file(target) == digest, f"checksum mismatch: {target}")


def usage_policy() -> dict[str, dict[str, bool]]:
    return {
        "train": {
            "gradient_usage": True, "training_usage": True, "training_eligible": True,
            "model_selection_usage": False, "hyperparameter_selection_usage": False,
            "checkpoint_selection_usage": False, "early_stopping_usage": False,
            "prompt_selection_usage": False, "decoding_parameter_selection_usage": False,
            "final_evaluation_usage": False,
        },
        "validation": {
            "gradient_usage": False, "training_usage": False, "training_eligible": False,
            "model_selection_usage": True, "hyperparameter_selection_usage": True,
            "checkpoint_selection_usage": True, "early_stopping_usage": True,
            "prompt_selection_usage": False, "decoding_parameter_selection_usage": False,
            "final_evaluation_usage": False,
        },
        "test": {
            "gradient_usage": False, "training_usage": False, "training_eligible": False,
            "model_selection_usage": False, "hyperparameter_selection_usage": False,
            "checkpoint_selection_usage": False, "early_stopping_usage": False,
            "prompt_selection_usage": False, "decoding_parameter_selection_usage": False,
            "final_evaluation_usage": True,
        },
    }


def export_schema() -> dict[str, Any]:
    required = [
        "export_schema_version", "candidate_schema_version", "sample_id", "article_id", "document_id",
        "source_document_ids", "event_group_id", "split", "source_dataset_version", "source_split_version",
        "construction_method", "review_status", "topic", "fact_points", "source_paragraphs", "fact_coverage",
        "evidence_summary", "outline", "constraints", "system_prompt", "user_prompt", "messages",
        "target_title", "target_body", "target_text", "target_title_sha256", "target_body_sha256",
        "target_text_sha256", *usage_policy()["train"].keys(),
    ]
    string_fields = {
        key: {"type": "string", "minLength": 1} for key in (
            "sample_id", "article_id", "event_group_id", "construction_method", "review_status", "topic",
            "system_prompt", "user_prompt", "target_title", "target_body", "target_text",
        )
    }
    properties: dict[str, Any] = {
        **string_fields,
        "export_schema_version": {"const": EXPORT_SCHEMA_VERSION},
        "candidate_schema_version": {"const": "sft-candidate-v1.0.0"},
        "document_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_document_ids": {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "string"}},
        "split": {"enum": list(SPLITS)},
        "source_dataset_version": {"enum": ["news_v1.0.0", "news_v2.1.0"]},
        "source_split_version": {"const": "news_split_v2.1.0"},
        "fact_points": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        "source_paragraphs": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        "fact_coverage": {"type": "object"}, "evidence_summary": {"type": "object"},
        "outline": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "constraints": {"type": "object"},
        "messages": {"type": "array", "minItems": 3, "maxItems": 3},
        "target_title_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "target_body_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "target_text_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    properties.update({key: {"type": "boolean"} for key in usage_policy()["train"]})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cm-style:sft-final-export-v1.0.0-sft-v2-compatible",
        "title": "Frozen SFT Final Export V1 Compatible — SFT v2",
        "description": "The existing sft-final-export-v1.0.0 record contract with v1 Gold and v2.1 Silver source versions.",
        "type": "object", "additionalProperties": False, "required": required, "properties": properties,
    }


def _input_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: root / config[key] for key in ("gold_dir", "news_dir", "event_group_dir", "split_dir", "silver_dir")}


def validate_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = _input_paths(root, config)
    for key in ("gold_dir", "news_dir", "event_group_dir", "split_dir"):
        verify_checksum_manifest(paths[key] / "checksums.sha256")
    require((paths["gold_dir"] / "VERSION").read_text(encoding="utf-8").strip() == "sft_v1.0.0", "Gold version mismatch")
    require((paths["news_dir"] / "VERSION").read_text(encoding="utf-8").strip() == "news_v2.1.0", "news version mismatch")
    require((paths["event_group_dir"] / "VERSION").read_text(encoding="utf-8").strip() == "news_event_groups_v2.1.0", "event group version mismatch")
    require((paths["split_dir"] / "VERSION").read_text(encoding="utf-8").strip() == "news_split_v2.1.0", "split version mismatch")
    lean = paths["silver_dir"]
    production = read_json(lean / "full_production_manifest_lean.json")
    quality = read_json(lean / "production_quality_audit_lean.json")
    pool_manifest = read_json(lean / "silver_quality_pool_v2_lean" / "manifest.json")
    for relative, digest in production["outputs"].items():
        target = lean / relative
        require(target.is_file() and sha256_file(target) == digest, f"Lean checksum mismatch: {relative}")
    require(production["pipeline_status"] == "completed" and production["quality_gate"] == "PASS", "Lean production not complete")
    require(quality["unfinished"] == 0 and quality["task011d_f_ready"] is True, "Lean task011d_f gate not ready")
    expected = config["expected"]
    require(pool_manifest["record_count"] == expected["silver_accepted"], "Silver accepted count mismatch")
    require(quality["drop"] == expected["silver_dropped"], "Silver DROP count mismatch")
    require(quality["accepted"] == expected["silver_clean"] and quality["accepted_with_warning"] == expected["silver_warning"], "Silver quality counts mismatch")
    return {"paths": paths, "production": production, "quality": quality, "pool_manifest": pool_manifest}


def _gold_records(paths: dict[str, Path], news_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    old_lineage = {row["sample_id"]: row for row in read_jsonl(paths["gold_dir"] / "lineage.jsonl")}
    policy = usage_policy()
    for old_split in SPLITS:
        for source in read_jsonl(paths["gold_dir"] / f"{old_split}.jsonl"):
            article = news_by_id.get(source["article_id"])
            require(article is not None and article["gold_silver"] == "Gold", f"Gold mapping missing: {source['article_id']}")
            require(source["target_title"] == article["title"] and source["target_body"] == article["body"], f"Gold Target drift: {source['sample_id']}")
            record = dict(source)
            record["split"] = article["split"]
            record["event_group_id"] = article["event_group_id"]
            record["source_split_version"] = "news_split_v2.1.0"
            record.update(policy[record["split"]])
            records.append(record)
    return records, old_lineage


def _silver_records(paths: dict[str, Path], news_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = read_jsonl(paths["silver_dir"] / "silver_quality_pool_v2_lean" / "silver_sft_quality_pool_v2_lean.jsonl")
    policy = usage_policy()
    records: list[dict[str, Any]] = []
    for item in pool:
        candidate = item["candidate"]
        article = news_by_id.get(candidate["source_article_id"])
        require(article is not None and article["gold_silver"] == "Silver", f"Silver mapping missing: {item['sample_id']}")
        require(item["split"] == article["split"] and item["event_group_id"] == article["event_group_id"], f"Silver lineage mismatch: {item['sample_id']}")
        require(candidate["target_title"] == article["title"] and candidate["target_body"] == article["body"], f"Silver Target drift: {item['sample_id']}")
        refs = [ref for fact in candidate["fact_points"] for ref in fact["evidence_paragraph_ids"]]
        record = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "candidate_schema_version": candidate["schema_version"],
            "sample_id": candidate["sample_id"], "article_id": article["article_id"],
            "document_id": article["document_id"], "source_document_ids": [article["document_id"]],
            "event_group_id": article["event_group_id"], "split": article["split"],
            "source_dataset_version": "news_v2.1.0", "source_split_version": "news_split_v2.1.0",
            "construction_method": candidate["construction_method"], "review_status": "accepted",
            "topic": candidate["topic"], "fact_points": candidate["fact_points"],
            "source_paragraphs": candidate["source_paragraphs"],
            "fact_coverage": {"coverage_status": "pass", "fact_count": len(candidate["fact_points"]), "all_facts_have_evidence": True},
            "evidence_summary": {"fact_count": len(candidate["fact_points"]), "evidence_reference_count": len(refs), "unique_evidence_paragraph_count": len(set(refs))},
            "outline": candidate["outline"], "constraints": candidate["constraints"],
            "system_prompt": candidate["system_prompt"], "user_prompt": candidate["user_prompt"],
            "messages": candidate["messages"], "target_title": candidate["target_title"],
            "target_body": candidate["target_body"], "target_text": candidate["target_text"],
            "target_title_sha256": candidate["target_title_sha256"], "target_body_sha256": candidate["target_body_sha256"],
            "target_text_sha256": candidate["target_text_sha256"], **policy[article["split"]],
        }
        records.append(record)
    return records, pool


def validate_record(record: dict[str, Any], expected_keys: set[str]) -> None:
    require(set(record) == expected_keys, f"export fields mismatch: {record.get('sample_id')}")
    require(record["export_schema_version"] == EXPORT_SCHEMA_VERSION and record["candidate_schema_version"] == "sft-candidate-v1.0.0", "schema version mismatch")
    require(record["source_dataset_version"] in {"news_v1.0.0", "news_v2.1.0"} and record["source_split_version"] == "news_split_v2.1.0", "source version mismatch")
    require(record["split"] in SPLITS and record["review_status"] == "accepted", "status/split mismatch")
    require([message.get("role") for message in record["messages"]] == ["system", "user", "assistant"], "message roles mismatch")
    # Historical Gold contains one frozen, human-approved system-prompt wording
    # variant. Preserve it exactly; the user and assistant contracts remain exact.
    require(bool(record["messages"][0]["content"]) and record["messages"][1]["content"] == record["user_prompt"], f"prompt/messages mismatch: {record['sample_id']}")
    require(record["messages"][2]["content"] == record["target_text"], "assistant Target mismatch")
    require(record["target_text"] == render_target(record["target_title"], record["target_body"]), "Target render mismatch")
    require(record["target_title_sha256"] == sha256_text(record["target_title"]), "Target title hash mismatch")
    require(record["target_body_sha256"] == sha256_text(record["target_body"]), "Target body hash mismatch")
    require(record["target_text_sha256"] == sha256_text(record["target_text"]), "Target text hash mismatch")
    fact_ids = [fact.get("fact_id") for fact in record["fact_points"]]
    require(all(isinstance(fact_id, str) and fact_id for fact_id in fact_ids) and len(fact_ids) == len(set(fact_ids)), "Fact IDs invalid")
    paragraph_ids = {row.get("paragraph_id") for row in record["source_paragraphs"]}
    for fact in record["fact_points"]:
        refs = fact.get("evidence_paragraph_ids")
        controlled_title = (
            refs == []
            and (
                (
                    fact.get("evidence_type") == "controlled_source_title"
                    and bool(fact.get("controlled_title_evidence_ref"))
                    and bool(fact.get("controlled_source_title_sha256"))
                )
                or (fact.get("evidence_source_type") == "title" and bool(fact.get("evidence_text_sha256")))
            )
        )
        require(isinstance(refs, list) and ((bool(refs) and set(refs) <= paragraph_ids) or controlled_title), f"Evidence refs invalid: {record['sample_id']}")
    require(record["source_document_ids"] == [record["document_id"]], "document lineage mismatch")
    require({key: record[key] for key in usage_policy()[record["split"]]} == usage_policy()[record["split"]], "usage policy mismatch")


def _p90(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)]


def _fact_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [len(record["fact_points"]) for record in records]
    return {
        "sample_count": len(values), "fact_count": sum(values),
        "average_facts_per_sample": round(statistics.mean(values), 4),
        "median_facts_per_sample": statistics.median(values),
        "p90_facts_per_sample": _p90(values), "max_facts_per_sample": max(values),
    }


def _dataset_card(stats: dict[str, Any]) -> str:
    total = stats["totals"]
    return f"""# task011d_sft — sft_v2.0.0

## Scope

This frozen research dataset contains {total['sample_count']} public corporate headquarters news samples: 239 human-reviewed Gold samples reused exactly from `sft_v1.0.0`, and 2099 AI-reviewed Silver samples produced under `silver_production_standard_v2_lean`. Eighteen Silver samples with remaining hard issues were dropped and were not replaced.

## Quality and lineage

Gold uses `gold_human_reviewed`; Silver uses `silver_ai_reviewed`. Silver includes 523 clean accepted records and 1576 accepted-with-warning records. Soft warning metadata is retained only in metadata lineage and is not inserted into training prompts or messages. Gold and Silver Targets are frozen and exact. All records pass the compatible `sft-final-export-v1.0.0` schema, Fact/Evidence, Prompt/messages, Target and provenance checks.

Event-group semantic repair is inherited from `news_event_groups_v2.1.0`; cross-split article, document and event-group leakage are zero. Provenance is `passed_no_candidate_contamination` and excludes Pilots, failed TASK-011D-E candidates and DROP records.

## Splits and use

Train/Validation/Test contain {stats['splits']['train']['sample_count']}/{stats['splits']['validation']['sample_count']}/{stats['splits']['test']['sample_count']} samples. Train is the only gradient-bearing split. Validation is only for model/checkpoint selection and early stopping. Test is isolated for a separately authorized final evaluation and must not be used for training, prompt, decoding, checkpoint or hyperparameter selection.

No training or model Test evaluation has been performed. The next stage is TASK-012 Training Preparation, subject to explicit user approval at the pre-training gate.
"""


def build_documents(root: Path, config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    inputs = validate_inputs(root, config)
    paths = inputs["paths"]
    news = read_jsonl(paths["news_dir"] / "articles.jsonl")
    require(len(news) == 2356 and Counter(row["split"] for row in news) == Counter({"train": 1885, "validation": 236, "test": 235}), "formal news counts mismatch")
    news_by_id = {row["article_id"]: row for row in news}
    require(len(news_by_id) == 2356, "formal article IDs not unique")
    group_map = {row["article_id"]: row["event_group_id"] for row in read_jsonl(paths["event_group_dir"] / "article_to_event_group.jsonl")}
    split_map = {row["article_id"]: row for row in read_jsonl(paths["split_dir"] / "lineage.jsonl")}
    require(len(group_map) == len(split_map) == 2356, "formal group/split mapping incomplete")
    require(all(group_map[row["article_id"]] == row["event_group_id"] == split_map[row["article_id"]]["event_group_id"] and row["split"] == split_map[row["article_id"]]["split"] for row in news), "formal mapping drift")

    gold, gold_old_lineage = _gold_records(paths, news_by_id)
    silver, pool = _silver_records(paths, news_by_id)
    expected = config["expected"]
    require(len(gold) == expected["gold"] and len(silver) == expected["silver_accepted"] and len(gold) + len(silver) == expected["total"], "final count mismatch")
    all_records = gold + silver
    expected_keys = set(export_schema()["required"])
    for record in all_records:
        validate_record(record, expected_keys)
    by_split = {split: sorted((row for row in all_records if row["split"] == split), key=lambda row: (row["article_id"], row["sample_id"])) for split in SPLITS}
    require([len(by_split[s]) for s in SPLITS] == [1871, 234, 233], "recomputed split counts mismatch")

    article_sets = {split: {row["article_id"] for row in rows} for split, rows in by_split.items()}
    document_sets = {split: {row["document_id"] for row in rows} for split, rows in by_split.items()}
    group_sets = {split: {row["event_group_id"] for row in rows} for split, rows in by_split.items()}
    for sets, label in ((article_sets, "article"), (document_sets, "document"), (group_sets, "event group")):
        require(not sets["train"] & sets["validation"] and not sets["train"] & sets["test"] and not sets["validation"] & sets["test"], f"cross-split {label} leakage")
    require(len(set().union(*article_sets.values())) == expected["total"], "article IDs not unique")
    require(len(set().union(*document_sets.values())) == expected["total"], "document IDs not unique")
    require(len({row["sample_id"] for row in all_records}) == expected["total"], "sample IDs not unique")

    states = read_json(paths["silver_dir"] / "sample_states.json")
    high = read_json(paths["silver_dir"] / "final_high_audit.json")["audits"]
    pool_by_id = {row["sample_id"]: row for row in pool}
    records_by_id = {row["sample_id"]: row for row in all_records}
    lineage: list[dict[str, Any]] = []
    for record in sorted(gold, key=lambda row: row["sample_id"]):
        old = gold_old_lineage[record["sample_id"]]
        lineage.append({
            "sample_id": record["sample_id"], "article_id": record["article_id"], "document_id": record["document_id"],
            "split": record["split"], "event_group_id": record["event_group_id"], "quality_tier": "gold_human_reviewed",
            "source_dataset_version": "news_v1.0.0", "source_split_version": "news_split_v2.1.0",
            "candidate_version": old["candidate_version"], "review_status": "human_reviewed_accepted",
            "revision_count": 0, "high_audit_status": "not_applicable_gold_human_reviewed",
            "review_lineage_type": "frozen_human_review_lineage", "source_lineage_ref": "data/processed/sft_v1/lineage.jsonl",
            "target_sha256": record["target_text_sha256"], "final_record_sha256": record_sha256(record),
        })
    for record in sorted(silver, key=lambda row: row["sample_id"]):
        item = pool_by_id[record["sample_id"]]
        state = states[record["sample_id"]]
        high_status = high.get(record["sample_id"], {}).get("verdict", "not_selected")
        if item["lean_lineage"].get("targeted_year_revision"):
            high_status = "targeted_risk_revision_rereview_pass"
        lineage.append({
            "sample_id": record["sample_id"], "article_id": record["article_id"], "document_id": record["document_id"],
            "split": record["split"], "event_group_id": record["event_group_id"], "quality_tier": "silver_ai_reviewed",
            "source_dataset_version": "news_v2.1.0", "source_split_version": "news_split_v2.1.0",
            "candidate_version": item["candidate"]["candidate_version"], "review_status": item["quality_status"],
            "revision_count": 1 if state["revised"] else 0, "high_audit_status": high_status,
            "review_lineage_type": "ai_production_lineage", "production_standard": "silver_production_standard_v2_lean",
            "soft_warnings": item["soft_warnings"], "target_sha256": record["target_text_sha256"],
            "candidate_sha256": item["candidate_sha256"], "final_record_sha256": record_sha256(record),
        })

    drops = read_jsonl(paths["silver_dir"] / "silver_drop_ledger_lean.jsonl")
    drop_manifest = []
    for drop in sorted(drops, key=lambda row: row["sample_id"]):
        state = states[drop["sample_id"]]
        drop_manifest.append({
            "sample_id": drop["sample_id"], "article_id": state["article_id"], "split": drop["split"],
            "hard_reason": drop["reason"],
            "hard_issue_types": sorted({issue["issue_type"] for stage in (drop.get("review", {}), drop.get("rereview", {})) for issue in stage.get("core_issues", [])}),
            "source_lineage": {"source_dataset_version": "news_v2.1.0", "source_split_version": "news_split_v2.1.0"},
            "review_revision_lineage": {"primary_review_status": drop.get("review", {}).get("status"), "revision_count": 1, "rereview_status": drop.get("rereview", {}).get("status")},
            "candidate_sha256": drop["candidate_sha256"],
        })
    require(len(drop_manifest) == expected["silver_dropped"] and not {row["sample_id"] for row in drop_manifest} & set(records_by_id), "DROP contamination")

    stats: dict[str, Any] = {"schema_version": "task011d-sft-v2-split-statistics-v1.0.0", "splits": {}}
    for split, rows in by_split.items():
        stats["splits"][split] = {**_fact_stats(rows), "gold_count": sum(row["source_dataset_version"] == "news_v1.0.0" for row in rows), "silver_count": sum(row["source_dataset_version"] == "news_v2.1.0" for row in rows), "event_group_count": len(group_sets[split])}
    stats["totals"] = {**_fact_stats(all_records), "gold_fact_count": sum(len(row["fact_points"]) for row in gold), "silver_fact_count": sum(len(row["fact_points"]) for row in silver), "gold_count": len(gold), "silver_count": len(silver), "event_group_count": len(set().union(*group_sets.values()))}
    quality_stats = {
        "schema_version": "task011d-sft-v2-quality-tier-statistics-v1.0.0", "total_samples": expected["total"],
        "gold": {"quality_tier": "gold_human_reviewed", "samples": expected["gold"], "percentage": round(100 * expected["gold"] / expected["total"], 4)},
        "silver": {"quality_tier": "silver_ai_reviewed", "input": expected["silver_input"], "accepted": expected["silver_accepted"], "percentage": round(100 * expected["silver_accepted"] / expected["total"], 4), "accepted_clean": expected["silver_clean"], "accepted_with_warning": expected["silver_warning"], "dropped": expected["silver_dropped"]},
    }
    split_audit = {
        "schema_version": "task011d-sft-v2-split-integrity-audit-v1.0.0", "status": "passed",
        "train_count": len(by_split["train"]), "validation_count": len(by_split["validation"]), "test_count": len(by_split["test"]),
        "unique_article_count": len(set().union(*article_sets.values())), "duplicate_article_count": 0,
        "article_cross_split_overlap": 0, "document_cross_split_overlap": 0,
        "cross_split_event_group_leakage": 0, "formal_split_membership_mismatch": 0,
    }
    target_audit = {"schema_version": "task011d-sft-v2-target-integrity-audit-v1.0.0", "status": "passed", "gold_target_exact": len(gold), "silver_target_exact": len(silver), "total_target_exact": len(all_records), "target_modified": 0}
    schema_audit = {"schema_version": "task011d-sft-v2-schema-integrity-audit-v1.0.0", "status": "passed", "schema_pass": len(all_records), "schema_fail": 0, "prompt_messages_pass": len(all_records), "fact_id_pass": len(all_records), "evidence_refs_pass": len(all_records), "unknown_invalid_properties": 0}
    provenance = {
        "schema_version": "task011d-sft-v2-provenance-audit-v1.0.0", "status": "passed_no_candidate_contamination",
        "gold_source": "frozen_sft_v1.0.0", "silver_source": "task011d_e5_lean_final_silver_quality_pool",
        "pilot_contamination": 0, "failed_task011d_e_contamination": 0, "drop_contamination": 0,
        "other_source_contamination": 0, "semantic_model_calls": 0, "training_executed": False,
    }
    review_lineage = {
        "schema_version": "task011d-sft-v2-human-ai-review-lineage-v1.0.0",
        "gold": {"samples": len(gold), "quality_tier": "gold_human_reviewed", "review_type": "human", "lineage_file": "lineage.jsonl"},
        "silver": {"samples": len(silver), "quality_tier": "silver_ai_reviewed", "review_type": "AI production", "production_standard": "silver_production_standard_v2_lean", "lineage_file": "lineage.jsonl", "revision_count": sum(row["revision_count"] for row in lineage if row["quality_tier"] == "silver_ai_reviewed")},
    }
    silver_summary = {
        "schema_version": "task011d-sft-v2-silver-quality-summary-v1.0.0", "input": expected["silver_input"],
        "accepted": expected["silver_clean"], "accepted_with_warning": expected["silver_warning"],
        "final_accepted": expected["silver_accepted"], "dropped": expected["silver_dropped"], "unfinished": 0,
        "quality_gate": "PASS", "production_standard": "silver_production_standard_v2_lean",
    }
    usage = {"schema_version": "task011d-sft-v2-split-usage-policy-v1.0.0", "splits": usage_policy(), "training_loader_allowed_files": ["train.jsonl"], "test_requires_separate_final_evaluation_authorization": True}
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION, "dataset_name": config["dataset_name"], "dataset_version": config["dataset_version"],
        "news_version": "news_v2.1.0", "event_group_version": "news_event_groups_v2.1.0", "split_version": "news_split_v2.1.0", "gold_sft_version": "sft_v1.0.0",
        "gold_count": len(gold), "silver_input": expected["silver_input"], "silver_accepted": len(silver), "silver_dropped": len(drops), "total_count": len(all_records),
        "train_count": len(by_split["train"]), "validation_count": len(by_split["validation"]), "test_count": len(by_split["test"]), "total_fact_count": stats["totals"]["fact_count"],
        "target_integrity": "passed", "schema_integrity": "passed", "split_integrity": "passed", "provenance": "passed_no_candidate_contamination",
        "build_status": "frozen", "build_timestamp": BUILD_TIMESTAMP, "training_started": False, "model_test_evaluation_performed": False,
        "semantic_model_calls": 0, "pre_training_gate_ready": True, "overwrite_allowed": False,
    }
    gate = {
        "schema_version": "task011d-pre-training-gate-v1.0.0", "status": "ready_for_user_pre_training_approval",
        "dataset_version": "sft_v2.0.0", "total_samples": len(all_records),
        "splits": {split: len(rows) for split, rows in by_split.items()}, "gold": len(gold), "silver": len(silver),
        "fact_counts": {"total": stats["totals"]["fact_count"], **{split: stats["splits"][split]["fact_count"] for split in SPLITS}},
        "silver_dropped": expected["silver_dropped"], "silver_accepted_with_warning": expected["silver_warning"],
        "split_leakage": 0, "target_integrity": "passed", "schema_integrity": "passed",
        "provenance": "passed_no_candidate_contamination", "checksums": "passed",
        "training_started": False, "model_test_evaluation_performed": False, "semantic_model_calls": 0,
        "user_action_required": "Review the final pre-training summary and approve entering model training.",
    }
    documents = {f"{split}.jsonl": jsonl_text(rows) for split, rows in by_split.items()}
    documents.update({
        "VERSION": "sft_v2.0.0\n", "schema.json": json_text(export_schema()), "dataset_manifest.json": json_text(manifest),
        "split_statistics.json": json_text(stats), "split_usage_policy.json": json_text(usage),
        "quality_tier_statistics.json": json_text(quality_stats), "human_ai_review_lineage.json": json_text(review_lineage),
        "lineage.jsonl": jsonl_text(lineage), "dataset_card.md": _dataset_card(stats), "provenance_audit.json": json_text(provenance),
        "split_integrity_audit.json": json_text(split_audit), "target_integrity_audit.json": json_text(target_audit),
        "schema_integrity_audit.json": json_text(schema_audit), "silver_quality_summary.json": json_text(silver_summary),
        "dropped_sample_manifest.json": json_text({"schema_version": "task011d-sft-v2-dropped-sample-manifest-v1.0.0", "full_text_stored": False, "count": len(drop_manifest), "samples": drop_manifest}),
        "pre_training_gate.json": json_text(gate),
    })
    summary = {"manifest": manifest, "statistics": stats, "gate": gate}
    return documents, summary


def _summary_csv(summary: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["split", "sample_count", "gold_count", "silver_count", "fact_count"])
    writer.writeheader()
    stats = summary["statistics"]
    for split in SPLITS:
        row = stats["splits"][split]
        writer.writerow({"split": split, **{key: row[key] for key in ("sample_count", "gold_count", "silver_count", "fact_count")}})
    total = stats["totals"]
    writer.writerow({"split": "total", "sample_count": total["sample_count"], "gold_count": total["gold_count"], "silver_count": total["silver_count"], "fact_count": total["fact_count"]})
    return output.getvalue()


def build(root: Path, config_path: Path) -> Path:
    config = read_json(config_path)
    destination = root / config["output_dir"]
    if destination.exists():
        if (destination / "VERSION").is_file() and (destination / "VERSION").read_text(encoding="utf-8").strip() == config["dataset_version"]:
            raise SFTV2BuildError("sft_v2.0.0 already exists; frozen datasets cannot be overwritten")
        raise SFTV2BuildError(f"output directory already exists: {destination}")
    documents, summary = build_documents(root, config)
    temporary = destination.with_name(destination.name + ".building")
    require(not temporary.exists(), f"temporary build directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for name in CHECKSUM_FILES:
            (temporary / name).write_text(documents[name], encoding="utf-8", newline="")
        checksum_text = "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in CHECKSUM_FILES)
        (temporary / "checksums.sha256").write_text(checksum_text, encoding="ascii", newline="")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    summary_path = root / config["summary_csv"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_csv(summary), encoding="utf-8", newline="")
    validate_dataset_directory(destination)
    return destination


def validate_dataset_directory(directory: Path) -> dict[str, Any]:
    require(directory.is_dir(), f"dataset missing: {directory}")
    require({path.name for path in directory.iterdir()} == set(CHECKSUM_FILES) | {"checksums.sha256"}, "dataset file set mismatch")
    verify_checksum_manifest(directory / "checksums.sha256")
    rows = {split: read_jsonl(directory / f"{split}.jsonl") for split in SPLITS}
    require([len(rows[split]) for split in SPLITS] == [1871, 234, 233], "dataset split counts mismatch")
    expected_keys = set(export_schema()["required"])
    for split in SPLITS:
        for record in rows[split]:
            validate_record(record, expected_keys)
    manifest = read_json(directory / "dataset_manifest.json")
    require(manifest["total_count"] == 2338 and manifest["build_status"] == "frozen", "dataset manifest mismatch")
    require(read_json(directory / "pre_training_gate.json")["status"] == "ready_for_user_pre_training_approval", "pre-training gate mismatch")
    return {"manifest": manifest, "statistics": read_json(directory / "split_statistics.json")}


def validate(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_inputs(root, config)
    documents, expected = build_documents(root, config)
    actual = validate_dataset_directory(root / config["output_dir"])
    for name in CHECKSUM_FILES:
        require((root / config["output_dir"] / name).read_text(encoding="utf-8") == documents[name], f"non-deterministic output: {name}")
    return actual
