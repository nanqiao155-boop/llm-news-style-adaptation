# Final deployment routes

Route A is the frozen same-source Q4/Q6/Q8 study. Route B is LFM F16/PTQ/QAD CPU-only.
Aggregates live under `results/lightweight/`; all row-level news and responses are excluded.

`reference/` preserves selected original generation, judge, aggregation and anchor-audit
source for inspection. Filesystem literals were rebased, but their private workspace,
shared historical runners, weights, server executable and frozen candidate mapping
are deliberately absent. Do not run these as standalone public commands.
TODO for a later authorized round: replace those artifact dependencies with a public CLI.
Use `scripts/plot_public_results.py` for reproducible aggregate-only execution today.
