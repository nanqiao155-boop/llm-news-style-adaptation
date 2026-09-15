# Evaluation

## Text overlap

The retained implementation in `src/training/lora_v2.py` tokenizes each CJK character
and lower-case contiguous ASCII alphanumeric word, dropping punctuation and spaces.
BLEU-4 is corpus modified precision with uniform weights and add-one smoothing.
ROUGE-1/2 are macro F1 multiset overlap; ROUGE-L is macro F1 based on longest common
subsequence. Scores are fractions. These metrics measure overlap with a reference,
not factual truth or publication readiness.

## Editorial score

The six caps are factual grounding 30, key information retention 15, title quality 15,
formality/objectivity 15, structure/formatting 10, conciseness/naturalness 15.
Python sums the six validated integer scores. Reviewer/Editor receives confirmed facts,
not the reference answer. Blind judge batches conceal model and stage identity;
aggregation restores labels separately. Exact-text reuse is documented in the frozen
same-source batch: 128 unique candidates, 144 condition records, 16 reuses.

## Release-Adjusted v1

An internal engineering metric, not an industry standard. The unchanged implementation
is `demo/scoring.py`. Let U be unsupported claim count and R major release risk count.
The cap is 59 if R≥2; 69 if R=1; 79 if R=0 and publishable is false; otherwise 100.
`score = max(0, min(raw_total, cap) - 3*U)`. Aggregate scores are the mean of
per-sample adjusted scores, not the formula applied to aggregate means.
Publishable rate is the fraction marked publishable. Unsupported claims/sample is
mean count of unsupported additions. Publishability never authorizes actual release.

## Cohorts and audit

Full Validation n=234 supports the stability comparison. The recovery and deployment
subsets use n=24. Final Test n=50 supports the Base/Final System table and is separate
from calibration/replay. Do not treat these as one common cohort.

Qwen deployment uses the frozen same-source Q4/Q6/Q8 batch. All three precisions were
Pareto-nondominated under the original seven dimensions; sweet spot is INCONCLUSIVE.
LFM is a CPU-only F16/PTQ/QAD route, not a GPU throughput comparison with Qwen.
The original QAD Final Judge score (99.917) and 100% publishability are NOT an
independent factual guarantee: independent audit found 15/24 publishable (62.5%),
unsupported issues 181→18 (7.542→0.750/sample), with remaining risks in 9 Finals.

The audit aggregate preserves the original anchor counts and its v1.1 erratum:
new exact anchors Draft 23→22 and Final 7→6. Prefer corrected counts when discussing
anchors. Paired removal events (184) are not a subset ledger of the 181 frozen Draft
issues and must not be algebraically reconciled with the 18 Final issues.
Transition sensitivity is retained: 19 improved/5 mixed vs alternative 18/6.

Single seed, small deployment subset and LLM-as-a-Judge bias limit inference.
No paired significance test is supplied and no statistical significance is claimed.
