# Data boundary and schema

Full dataset is not included. No original articles, raw HTML, training splits,
annotation records, real replay text or generated news corpus are redistributed.

Frozen research dataset: 2,338 SFT samples; Train 1,871 / Validation 234 / Test 233;
18,437 facts. Gold: 239; accepted Silver: 2,099; excluded Silver: 18.
These are historical study counts, not counts of the public fixtures.

Pipeline: permitted public news → parsing / normalization → duplicate and event grouping
→ source/event-disjoint splits → fact evidence and target checks → Gold/Silver quality
review → versioned SFT messages. Split by source and event before constructing SFT
examples; never let paraphrases from one event cross splits. Keep Test isolated from
training and validation selection. Collection requires checking the target robots.txt,
permission, a reasonable delay and low concurrency before any request.

The fully synthetic [example](samples/example_input.jsonl) contains `id`, `synthetic`,
`topic`, `category`, `fact_points` and `outline`. It uses an invented company and place.
It demonstrates public input only. Historical SFT exports also require provenance,
review lineage and exact source targets; this fixture is not a substitute for them.
`src/dataset/sft_schema.py` and `sft_v2_complete_build.py` document those contracts.

Generated public smoke outputs belong in `data/interim/`. Privately obtained,
permitted reproduction inputs belong in `data/processed/private/` and stay ignored.
