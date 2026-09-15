# Models and reproduction boundaries

Weights are not included in the public repository. No adapter, base model, GGUF or
checkpoint is bundled, and this repository makes no commitment to publish weights.

| Dependency | Public repository | Recorded revision |
|---|---|---|
| Base Qwen | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| LFM CPU models | [LiquidAI LFM2.5 GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) | `6767265158422fb8a19c62ceb45f16f05363615b` |

Revisions come from the local frozen model configuration/integrity records; no model
download was performed during release preparation. Independently check the model
provider's terms and compatibility before downloading. LFM files were F16, Q4_0 PTQ,
and QAD-Q4_0; Qwen Route A uses same-source Q4_K_M, Q6_K and Q8_0.

## Public execution

1. Use Python 3.11+ and install `requirements.txt` in an isolated environment.
2. Run `python -m pytest -q -p no:cacheprovider` and `python -m demo.app`.
3. Recreate aggregate plots with `python scripts/plot_public_results.py`.
4. For optional online operation, configure the variables in `demo/.env.example`
   locally; use your own permitted Writer and Judge deployment. The client does not
   download models or establish the actual adapter identity.

## A new training reproduction

Obtain independently permitted data; create messages-only `formal_train` and
`formal_validation` JSONL plus LLaMA-Factory `dataset_info.json` under
`data/processed/private/runtime_data/`. Point each dataset entry to the corresponding
file, use ShareGPT format, map `messages` to conversations, with role/content tags
and user/assistant/system role tags. Test must not appear in training config.
Install a compatible GPU environment separately (recorded LLaMA-Factory 0.9.4;
PEFT 0.17.1 in adapter audit), pin the model snapshot above, then run e.g.
`llamafactory-cli train configs/reproduction/E05_rank32_run1.yaml`.
Runtime dependency compatibility has not been tested in this release.

The original configs have `save_total_limit: 3`; preserve a copy of checkpoint-117
before rolling checkpoint cleanup if repeating the retrospective selection study.
The historical launcher under `configs/training/lora_v2/` retains unresolved preflight
guards by design. It needs excluded frozen manifests and is not the public training
shortcut. Do not interpret its historical planned state as the final selected settings:
use `configs/repaired_writer.json` and formal runtime YAMLs for those.

Exact historical end-to-end reproduction is limited by excluded dataset, trained
weights, service outputs and hardware. Public smoke tests and aggregate plots are
reproducible without them; full training, quantization and live inference are untested here.

Route A quantized source: [bartowski same-source Qwen GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF). Revision TODO: not extracted into the public manifest; do not assume the current main branch is identical.
