# Methodology

## Data and adaptation

The source pipeline keeps source evidence, event grouping, split isolation, quality
review and deterministic SFT targets separate. Public fixtures contain no source text.
Use independently authorized inputs for new studies; the frozen historical dataset
cannot be reconstructed from this public repository alone.

Qwen3-4B-Instruct-2507 is adapted using standard BF16 LoRA in LLaMA-Factory.
The formal runtime YAMLs under `configs/reproduction/` preserve LR/rank/batch sweep
settings and cadence. Dataset/output paths alone are rebased for staging.
LR sweep: 5e-5, 1e-4, 2e-4 at rank 16, batch 8.
Rank sweep: 8, 16, 32 at LR 2e-4, batch 8.
Batch sweep: 4, 8, 16 at LR 2e-4, rank 32 (micro batch 1).
Alpha 32, dropout .05, three epochs, seed 42, cutoff 4096 stay fixed.
The Batch-16 curve has only five recorded evaluation points; do not invent an Epoch-3 point.

## Stability repair

Teacher-forcing loss did not reliably predict free-generation repetition.
The selected LR/rank/batch combination was 2e-4/32/8. Original severe repetition
was 57/234; repaired output was 1/234. The runner contains the original diagnostics,
checkpoint and adapter-strength experiments. `checkpoint-117` was selected after
examining multiple checkpoints, rather than defining the original run as 0.5 epoch.
The fixed-24 checkpoint comparison (1 vs 5 severe cases) differs from full Validation.

Effective adapter strength .50 scales each active LoRA alpha/r contribution once.
With r=32 and original alpha=32, alpha=16 in a deployment-only config is equivalent
in scaling. Do not also halve it at runtime. Model runtime equivalence requires
separate verification; weights are absent here.

## Agents

Writer produces title/body from confirmed facts. Reviewer checks facts, retained
information, title, formality, structure, unsupported additions and repetition.
PASS sends the identical Draft to Editorial Judge, skipping Reviser. FAIL invokes
one targeted Reviser with a no-new-facts instruction, then Judge. Schema retry is
limited to one; errors fail closed. Human confirmation remains outside the scoring rule.

## Deployment

Route A: same-source Qwen Q4/Q6/Q8 with the repaired adapter. Preserve measured
size, peak GPU memory, writer latency/throughput, Draft/Final quality and agent burden.
Route B: LFM2.5-1.2B F16/PTQ/QAD on CPU. The published QAD model is evaluated here;
this repository does not claim to implement or rerun Liquid AI's QAD training.
Independent fact audit is essential to interpreting the QAD closed loop.

The portable public demo uses original agent/workflow implementations with synthetic
responses. Historical runners under `experiments/lightweight/reference/` expose the
research algorithms but require excluded local artifacts and dependency wiring.
They are not executable reproduction entry points in the public quickstart.
