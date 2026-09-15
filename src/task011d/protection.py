from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROTECTED_ROOTS = (
    "data/processed/news_v1",
    "data/processed/news_event_groups_v1",
    "data/processed/news_split_v1_0_1",
    "data/processed/sft_batches_v1",
    "data/processed/sft_validation_v1",
    "data/processed/sft_test_v1",
    "data/processed/sft_v1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_files(repo_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative_root in PROTECTED_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(relative_root)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def build_manifest(repo_root: Path, source_git_commit: str) -> dict[str, Any]:
    entries = inventory_files(repo_root)
    return {
        "schema_version": "task011d-v1-protection-manifest-v1.0.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_git_commit": source_git_commit,
        "protection_policy": "immutable_checksum_guard",
        "protected_versions": [
            "news_v1.0.0",
            "news_event_groups_v1.0.0",
            "news_split_v1.0.1",
            "sft_v1.0.0",
            "train_batch_001_frozen_v1.0.0..train_batch_008_frozen_v1.0.0",
            "validation_batch_001_frozen_v1.0.0",
            "test_batch_001_frozen_v1.0.0",
        ],
        "protected_roots": list(PROTECTED_ROOTS),
        "file_count": len(entries),
        "files": entries,
        "validation_status": "passed",
    }


def validate_manifest(repo_root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    recorded = {entry["path"]: entry for entry in manifest.get("files", [])}
    current = {entry["path"]: entry for entry in inventory_files(repo_root)}
    if recorded.keys() != current.keys():
        errors.append("protected_file_set_changed")
    for path in sorted(recorded.keys() & current.keys()):
        if recorded[path]["sha256"] != current[path]["sha256"]:
            errors.append(f"checksum_mismatch:{path}")
        if recorded[path]["bytes"] != current[path]["bytes"]:
            errors.append(f"size_mismatch:{path}")
    return errors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
