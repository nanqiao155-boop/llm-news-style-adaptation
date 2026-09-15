"""Command line entry point for LoRA Experiment Protocol V2.1 infrastructure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.lora_v2 import (  # noqa: E402
    GuardError,
    aggregate_results,
    build_comparison_safe_mask,
    environment_manifest,
    evaluate_predictions,
    finalize_run,
    freeze_cutoff,
    launch_training,
    parse_trainer_log,
    plot_results,
    plot_loss_curves,
    prepare_training_view,
    prepare_smoke_dataset,
    prepare_cloud_smoke,
    resolve_experiment,
    run_validation_inference,
    run_cloud_smoke,
    token_statistics,
    token_statistics_verified,
    validate_dataset,
    validate_experiment_matrix,
    validate_smoke_manifest,
    validation_inference_plan,
    verify_model_snapshot,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data"); validate.add_argument("version", choices=("sft_v1.0.0", "sft_v2.0.0"))
    sub.add_parser("validate-matrix")
    resolve = sub.add_parser("resolve"); resolve.add_argument("experiment_id"); resolve.add_argument("--selection", type=Path)
    tokens = sub.add_parser("token-stats"); tokens.add_argument("--model-path", required=True); tokens.add_argument("--output", type=Path, required=True)
    tokens.add_argument("--model-verification", type=Path)
    verify_model = sub.add_parser("verify-model"); verify_model.add_argument("--model-path", type=Path, required=True)
    verify_model.add_argument("--repo-id", required=True); verify_model.add_argument("--revision", required=True)
    verify_model.add_argument("--tokenizer-revision", required=True); verify_model.add_argument("--output", type=Path, required=True)
    cutoff = sub.add_parser("freeze-cutoff"); cutoff.add_argument("--statistics", type=Path, required=True)
    cutoff.add_argument("--model-verification", type=Path, required=True); cutoff.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("launch"); launch.add_argument("experiment_id"); launch.add_argument("--run-id", required=True)
    launch.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "lora_v2"); launch.add_argument("--selection", type=Path)
    launch.add_argument("--execute", action="store_true"); launch.add_argument("--cli", default="llamafactory-cli")
    launch.add_argument("--training-view-root", type=Path)
    training_view = sub.add_parser("prepare-training-view")
    training_view.add_argument("version", choices=("sft_v1.0.0", "sft_v2.0.0"))
    training_view.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "lora_v2" / "training_views")
    infer = sub.add_parser("infer"); infer.add_argument("--run-manifest", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True); infer.add_argument("--split", choices=("validation", "test"), default="validation")
    infer.add_argument("--authorization", type=Path); infer.add_argument("--execute", action="store_true")
    metrics = sub.add_parser("evaluate"); metrics.add_argument("--predictions", type=Path, required=True); metrics.add_argument("--output", type=Path, required=True)
    mask = sub.add_parser("safe-mask"); mask.add_argument("--output", type=Path, required=True); mask.add_argument("--judge-count", type=int, default=50)
    aggregate = sub.add_parser("aggregate"); aggregate.add_argument("--run-root", type=Path, default=ROOT / "outputs" / "lora_v2")
    aggregate.add_argument("--output", type=Path, default=ROOT / "outputs" / "lora_v2" / "experiments.csv")
    logs = sub.add_parser("parse-log"); logs.add_argument("path", type=Path); logs.add_argument("--output", type=Path, required=True)
    plots = sub.add_parser("plot"); plots.add_argument("--experiments", type=Path, required=True); plots.add_argument("--output-dir", type=Path, required=True)
    plot_log = sub.add_parser("plot-loss"); plot_log.add_argument("--trainer-state", type=Path, required=True)
    plot_log.add_argument("--effective-batch", type=int, required=True); plot_log.add_argument("--output", type=Path, required=True)
    env = sub.add_parser("environment"); env.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("validate-smoke"); smoke.add_argument("manifest", type=Path)
    smoke_data = sub.add_parser("prepare-smoke-data"); smoke_data.add_argument("--output-dir", type=Path, required=True)
    smoke_data.add_argument("--train-count", type=int, default=32); smoke_data.add_argument("--validation-count", type=int, default=8)
    finalize = sub.add_parser("finalize-run"); finalize.add_argument("run_dir", type=Path)
    cloud_smoke = sub.add_parser("cloud-smoke"); cloud_smoke.add_argument("--model-path", type=Path, required=True)
    cloud_smoke.add_argument("--model-verification", type=Path, required=True); cloud_smoke.add_argument("--cutoff-freeze", type=Path, required=True)
    cloud_smoke.add_argument("--output-dir", type=Path, required=True); cloud_smoke.add_argument("--cli", default="llamafactory-cli")
    cloud_smoke.add_argument("--execute", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "validate-data": result = validate_dataset(args.version)
        elif args.command == "validate-matrix": result = validate_experiment_matrix()
        elif args.command == "resolve": result = resolve_experiment(args.experiment_id, args.selection)
        elif args.command == "token-stats":
            result = (token_statistics_verified(args.model_path, args.model_verification, args.output)
                      if args.model_verification else token_statistics(args.model_path, args.output))
        elif args.command == "verify-model": result = verify_model_snapshot(args.model_path, args.repo_id, args.revision,
                                                                             args.tokenizer_revision, args.output)
        elif args.command == "freeze-cutoff": result = freeze_cutoff(args.statistics, args.model_verification, args.output)
        elif args.command == "launch": result = launch_training(args.experiment_id, args.run_id, args.output_root,
                                                                  execute=args.execute, selection_path=args.selection, cli=args.cli,
                                                                  training_view_root=args.training_view_root)
        elif args.command == "prepare-training-view": result = prepare_training_view(args.version, args.output_root)
        elif args.command == "infer":
            result = (run_validation_inference(args.run_manifest, args.output_dir, split=args.split,
                                               authorization_path=args.authorization) if args.execute else
                      validation_inference_plan(args.run_manifest, args.output_dir, split=args.split,
                                                authorization_path=args.authorization))
        elif args.command == "evaluate": result = evaluate_predictions(args.predictions, args.output)
        elif args.command == "safe-mask": result = build_comparison_safe_mask(args.output, judge_count=args.judge_count)
        elif args.command == "aggregate": result = {"status": "completed", "rows": len(aggregate_results(args.run_root, args.output)), "output": str(args.output)}
        elif args.command == "parse-log":
            records = parse_trainer_log(args.path); write_json(args.output, {"records": records}); result = {"status": "completed", "records": len(records)}
        elif args.command == "plot": result = {"generated": [str(path) for path in plot_results(args.experiments, args.output_dir)]}
        elif args.command == "plot-loss": result = {"generated": str(plot_loss_curves(args.trainer_state, args.output,
                                                                                        effective_batch=args.effective_batch))}
        elif args.command == "environment": result = environment_manifest(); write_json(args.output, result)
        elif args.command == "validate-smoke": result = validate_smoke_manifest(args.manifest)
        elif args.command == "prepare-smoke-data": result = prepare_smoke_dataset(args.output_dir, train_count=args.train_count,
                                                                                   validation_count=args.validation_count)
        elif args.command == "finalize-run": result = finalize_run(args.run_dir)
        elif args.command == "cloud-smoke":
            if args.execute:
                result = run_cloud_smoke(args.model_path, args.model_verification, args.cutoff_freeze,
                                         args.output_dir, cli=args.cli)
            else:
                _, result = prepare_cloud_smoke(args.model_path, args.model_verification, args.cutoff_freeze,
                                                args.output_dir)
        else: raise AssertionError(args.command)
    except GuardError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
