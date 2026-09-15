#!/usr/bin/env python3
"""Independent production decoding for frozen LoRA V2.1 runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import lora_v2 as protocol


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CHAT_TEMPLATE = "qwen3_nothink"
PRODUCTION_SCHEMA = "lora-v2.1-production-decoding-v1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sample_ids(path: Path | None) -> tuple[list[str] | None, str | None]:
    if path is None:
        return None, None
    if not path.is_file():
        raise protocol.GuardError(f"sample IDs file does not exist: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise protocol.GuardError(f"invalid sample IDs JSON: {path}") from exc
        if not isinstance(value, list):
            raise protocol.GuardError("sample IDs JSON must be an array")
        values = value
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not values or any(not isinstance(item, (str, int)) or not str(item).strip() for item in values):
        raise protocol.GuardError("sample IDs file must contain non-empty string or integer IDs")
    sample_ids = [str(item).strip() for item in values]
    if len(set(sample_ids)) != len(sample_ids):
        raise protocol.GuardError("sample IDs file contains duplicate IDs")
    return sample_ids, protocol.sha256(path)


def _select_rows(source: Path, requested_ids: list[str] | None) -> list[dict[str, Any]]:
    rows = list(protocol.iter_jsonl(source))
    seen: set[str] = set()
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, (str, int)) or not str(sample_id):
            raise protocol.GuardError("inference source contains a missing sample_id")
        key = str(sample_id)
        if key in seen:
            raise protocol.GuardError(f"inference source contains duplicate sample_id: {key}")
        seen.add(key)
        indexed[key] = row
    if requested_ids is None:
        return rows
    missing = [sample_id for sample_id in requested_ids if sample_id not in indexed]
    if missing:
        raise protocol.GuardError(f"requested sample IDs are absent from the selected split: {missing}")
    requested = set(requested_ids)
    return [row for row in rows if str(row["sample_id"]) in requested]


def _validate_run_manifest(manifest: dict[str, Any]) -> None:
    experiment_id = manifest.get("experiment_id")
    if experiment_id not in {f"E{index:02d}" for index in range(1, 9)}:
        raise protocol.GuardError("production decoding requires an E01-E08 LoRA run manifest")
    if manifest.get("training_enabled") is not True or manifest.get("finetuning_type") != "lora":
        raise protocol.GuardError("production decoding requires a completed LoRA training run")
    if manifest.get("status") != "completed":
        raise protocol.GuardError("production decoding requires run status=completed")
    if manifest.get("model_name") != MODEL_NAME:
        raise protocol.GuardError(f"model_name must remain {MODEL_NAME}")
    if manifest.get("chat_template") != CHAT_TEMPLATE:
        raise protocol.GuardError(f"chat_template must remain {CHAT_TEMPLATE}")
    if not isinstance(manifest.get("model_revision"), str) or not manifest["model_revision"]:
        raise protocol.GuardError("run manifest must record model_revision")
    if not isinstance(manifest.get("best_checkpoint"), str) or not manifest["best_checkpoint"]:
        raise protocol.GuardError("production decoding requires the recorded best_checkpoint")


def build_plan(
    run_manifest: Path,
    output_dir: Path,
    *,
    split: str,
    authorization_path: Path | None,
    sample_ids_file: Path | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if output_dir.exists():
        raise protocol.GuardError(f"refusing to overwrite production decoding output: {output_dir}")
    if split not in {"validation", "test"}:
        raise protocol.GuardError("split must be validation or test")
    if split == "test" and authorization_path is None:
        raise protocol.GuardError("--authorization is required for Test decoding")
    if isinstance(repetition_penalty, bool) or not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise protocol.GuardError("repetition_penalty must be a finite number greater than zero")
    if isinstance(no_repeat_ngram_size, bool) or no_repeat_ngram_size < 0:
        raise protocol.GuardError("no_repeat_ngram_size must be zero or greater")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise protocol.GuardError("max_new_tokens must be greater than zero")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise protocol.GuardError("temperature must be a finite number greater than zero")
    if isinstance(top_p, bool) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise protocol.GuardError("top_p must be a finite number in (0, 1]")
    if isinstance(top_k, bool) or top_k < 0:
        raise protocol.GuardError("top_k must be zero or greater")
    if isinstance(seed, bool) or seed < 0:
        raise protocol.GuardError("seed must be zero or greater")
    if not run_manifest.is_file():
        raise protocol.GuardError(f"run manifest does not exist: {run_manifest}")

    manifest = protocol.read_json(run_manifest)
    _validate_run_manifest(manifest)
    purpose = "validation_inference" if split == "validation" else "final_evaluation"
    source = protocol.split_path(
        manifest["dataset_version"],
        split,
        purpose=purpose,
        authorization_path=authorization_path,
    )
    if not source.is_file():
        raise protocol.GuardError(f"selected split file does not exist: {source}")
    requested_ids, sample_ids_sha256 = _load_sample_ids(sample_ids_file)
    rows = _select_rows(source, requested_ids)
    model_source, _ = protocol._inference_model_location(manifest)

    decoding = {
        "do_sample": do_sample,
        "num_beams": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    authorization = None
    if split == "test":
        authorization = {
            "path": str(authorization_path.resolve()),
            "sha256": protocol.sha256(authorization_path),
        }
    plan = {
        "schema_version": PRODUCTION_SCHEMA,
        "status": "prepared_not_executed",
        "created_at": _utc_now(),
        "split": split,
        "sample_count": len(rows),
        "sample_selection": "explicit_ids" if requested_ids is not None else "full_split",
        "sample_ids_file": str(sample_ids_file.resolve()) if sample_ids_file else None,
        "sample_ids_file_sha256": sample_ids_sha256,
        "dataset_version": manifest["dataset_version"],
        "dataset_file": str(source.resolve()),
        "dataset_file_sha256": protocol.sha256(source),
        "source_run_manifest": str(run_manifest.resolve()),
        "source_run_manifest_sha256": protocol.sha256(run_manifest),
        "source_run_id": manifest["run_id"],
        "source_experiment_id": manifest["experiment_id"],
        "checkpoint": manifest["best_checkpoint"],
        "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"],
        "local_model_path": manifest.get("local_model_path"),
        "model_verification_artifact": manifest.get("model_verification_artifact"),
        "model_load_source": (
            "verified_local_snapshot" if model_source != manifest["model_name"] else "official_identifier_revision"
        ),
        "chat_template": CHAT_TEMPLATE,
        "enable_thinking": False,
        "decoding": decoding,
        "authorization": authorization,
        "predictions": str((output_dir / "predictions.jsonl").resolve()),
        "repetition_diagnostics": str((output_dir / "repetition_diagnostics.json").resolve()),
    }
    return plan, rows


def _line_repeat_flags(text: str) -> tuple[bool, bool, bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    any_ge_3 = any(count >= 3 for count in counts.values())
    severe_ge_5 = any(count >= 5 for count in counts.values())
    consecutive_ge_3 = False
    previous: str | None = None
    run_length = 0
    for line in lines:
        if line == previous:
            run_length += 1
        else:
            previous = line
            run_length = 1
        if run_length >= 3:
            consecutive_ge_3 = True
            break
    return any_ge_3, consecutive_ge_3, severe_ge_5


def repetition_diagnostics(predictions: Iterable[dict[str, Any]]) -> dict[str, int | str]:
    rows = list(predictions)
    any_count = consecutive_count = severe_count = 0
    for row in rows:
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise protocol.GuardError("prediction record has no string prediction")
        any_ge_3, consecutive_ge_3, severe_ge_5 = _line_repeat_flags(prediction)
        any_count += int(any_ge_3)
        consecutive_count += int(consecutive_ge_3)
        severe_count += int(severe_ge_5)
    return {
        "schema_version": "line-level-repetition-diagnostics-v1.0.0",
        "sample_count": len(rows),
        "any_line_repeat_ge_3": any_count,
        "consecutive_repeat_ge_3": consecutive_count,
        "severe_repeat_ge_5": severe_count,
    }


def execute_plan(
    run_manifest: Path,
    output_dir: Path,
    *,
    split: str,
    authorization_path: Path | None,
    sample_ids_file: Path | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    plan, rows = build_plan(
        run_manifest,
        output_dir,
        split=split,
        authorization_path=authorization_path,
        sample_ids_file=sample_ids_file,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    manifest = protocol.read_json(run_manifest)
    output_dir.mkdir(parents=True)
    protocol.write_json(output_dir / "inference_manifest.json", {**plan, "status": "running"})
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_source, load_kwargs = protocol._inference_model_location(manifest)
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            **load_kwargs,
        )
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, manifest["best_checkpoint"])
        random.seed(seed)
        manual_seed = getattr(torch, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        generation_kwargs = {
            "do_sample": plan["decoding"]["do_sample"],
            "num_beams": plan["decoding"]["num_beams"],
            "temperature": plan["decoding"]["temperature"],
            "top_p": plan["decoding"]["top_p"],
            "max_new_tokens": plan["decoding"]["max_new_tokens"],
            "repetition_penalty": plan["decoding"]["repetition_penalty"],
        }
        if plan["decoding"]["do_sample"]:
            generation_kwargs["top_k"] = plan["decoding"]["top_k"]
        if plan["decoding"]["no_repeat_ngram_size"] > 0:
            generation_kwargs["no_repeat_ngram_size"] = plan["decoding"]["no_repeat_ngram_size"]
        predictions_path = output_dir / "predictions.jsonl"
        with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                prompt = tokenizer.apply_chat_template(
                    row["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_kwargs)
                continuation = generated[0, inputs["input_ids"].shape[1] :]
                prediction = tokenizer.decode(continuation, skip_special_tokens=True)
                record = {
                    "sample_id": row["sample_id"],
                    "reference": row["target_text"],
                    "prediction": prediction,
                    "model_name": manifest["model_name"],
                    "model_revision": manifest["model_revision"],
                    "run_id": manifest["run_id"],
                    "checkpoint": manifest["best_checkpoint"],
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        prediction_rows = list(protocol.iter_jsonl(predictions_path))
        diagnostics = repetition_diagnostics(prediction_rows)
        protocol.write_json(output_dir / "repetition_diagnostics.json", diagnostics)
        finished = {
            **plan,
            "status": "completed",
            "completed_at": _utc_now(),
            "prediction_count": len(prediction_rows),
            "predictions_sha256": protocol.sha256(predictions_path),
            "repetition_diagnostics_sha256": protocol.sha256(output_dir / "repetition_diagnostics.json"),
        }
        protocol.write_json(output_dir / "inference_manifest.json", finished)
        print(json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
        return finished
    except Exception as exc:
        protocol.write_json(
            output_dir / "inference_manifest.json",
            {**plan, "status": "failed", "failed_at": _utc_now(), "error_type": type(exc).__name__},
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--no-repeat-ngram-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        kwargs = {
            "split": args.split,
            "authorization_path": args.authorization,
            "sample_ids_file": args.sample_ids_file,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        }
        if args.execute:
            result = execute_plan(args.run_manifest, args.output_dir, **kwargs)
        else:
            result, _ = build_plan(args.run_manifest, args.output_dir, **kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except protocol.GuardError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
