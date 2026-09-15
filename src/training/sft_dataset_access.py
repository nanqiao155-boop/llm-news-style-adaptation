"""Stage-gated access to the frozen SFT split files.

The training stack must ask for a purpose instead of discovering JSONL files by
directory glob.  This makes validation gradient use and all pre-final Test use
fail closed.
"""

from __future__ import annotations

from pathlib import Path


class DatasetAccessError(ValueError):
    """Raised when a caller requests a split for an unauthorized purpose."""


PURPOSE_TO_FILE = {
    "gradient_training": "train.jsonl",
    "model_selection_evaluation": "validation.jsonl",
    "final_evaluation": "test.jsonl",
}


def resolve_sft_file(dataset_dir: Path, purpose: str, *, final_evaluation_authorized: bool = False) -> Path:
    """Return the sole file allowed for *purpose* and reject implicit discovery."""

    if purpose not in PURPOSE_TO_FILE:
        raise DatasetAccessError(f"unsupported dataset access purpose: {purpose}")
    if purpose == "final_evaluation" and not final_evaluation_authorized:
        raise DatasetAccessError("Test access requires explicit final-evaluation authorization")
    path = dataset_dir / PURPOSE_TO_FILE[purpose]
    if not path.is_file():
        raise DatasetAccessError(f"frozen split file does not exist: {path}")
    return path


def gradient_training_files(dataset_dir: Path) -> tuple[Path, ...]:
    """Return exactly train.jsonl; never Validation or Test."""

    return (resolve_sft_file(dataset_dir, "gradient_training"),)
