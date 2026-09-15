from __future__ import annotations

from copy import deepcopy
from typing import Any


SEVERITIES = ("none", "warning", "minor", "major", "blocking")
SEVERITY_RANK = {value: index for index, value in enumerate(SEVERITIES)}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


SEVERITY_SCHEMA = {"type": "string", "enum": list(SEVERITIES)}


FACT_DIMENSION_SCHEMA = _object(
    {
        "fact_id": {"type": "string"},
        "support": _object(
            {
                "fact_supported": {"type": "boolean"},
                "core_fact_unsupported": {"type": "boolean"},
                "unsupported_components": {"type": "array", "items": {"type": "string"}},
                "support_severity": SEVERITY_SCHEMA,
                "support_rationale": {"type": "string"},
            }
        ),
        "evidence": _object(
            {
                "evidence_complete": {"type": "boolean"},
                "evidence_direct": {"type": "boolean"},
                "missing_or_indirect_evidence": {"type": "array", "items": {"type": "string"}},
                "evidence_issue_severity": SEVERITY_SCHEMA,
                "evidence_rationale": {"type": "string"},
            }
        ),
        "number_date": _object(
            {
                "number_correct": {"type": "boolean"},
                "date_correct": {"type": "boolean"},
                "incorrect_numbers_or_dates": {"type": "array", "items": {"type": "string"}},
                "number_date_severity": SEVERITY_SCHEMA,
                "number_date_rationale": {"type": "string"},
            }
        ),
        "entity": _object(
            {
                "entity_correct": {"type": "boolean"},
                "incorrect_entities": {"type": "array", "items": {"type": "string"}},
                "entity_severity": SEVERITY_SCHEMA,
                "entity_rationale": {"type": "string"},
            }
        ),
        "entity_relation": _object(
            {
                "entity_relation_correct": {"type": "boolean"},
                "incorrect_relation_description": {"type": "string"},
                "entity_relation_severity": SEVERITY_SCHEMA,
                "entity_relation_rationale": {"type": "string"},
            }
        ),
        "atomicity": _object(
            {
                "atomicity_pass": {"type": "boolean"},
                "independent_proposition_count": {"type": "integer", "minimum": 1},
                "contains_multiple_independently_assertable_claims": {"type": "boolean"},
                "should_split": {"type": "boolean"},
                "should_merge_with_adjacent_fact": {"type": "boolean"},
                "atomicity_severity": SEVERITY_SCHEMA,
                "atomicity_rationale": {"type": "string"},
            }
        ),
        "micro_fragmentation": _object(
            {
                "independently_interpretable": {"type": "boolean"},
                "subject_explicit": {"type": "boolean"},
                "predicate_complete": {"type": "boolean"},
                "depends_on_neighbor_fact": {"type": "boolean"},
                "micro_fragmentation_severity": SEVERITY_SCHEMA,
                "micro_fragmentation_rationale": {"type": "string"},
            }
        ),
        "duplicate": _object(
            {
                "duplicate_status": {
                    "type": "string",
                    "enum": ["none", "exact_duplicate", "semantic_duplicate", "contained_by_other", "contains_other"],
                },
                "duplicate_with_fact_ids": {"type": "array", "items": {"type": "string"}},
                "duplicate_severity": SEVERITY_SCHEMA,
                "duplicate_rationale": {"type": "string"},
            }
        ),
        "unsupported_inference": _object(
            {
                "every_claim_component_directly_supported": {"type": "boolean"},
                "inferred_components": {"type": "array", "items": {"type": "string"}},
                "causal_inference_present": {"type": "boolean"},
                "temporal_inference_present": {"type": "boolean"},
                "attribution_inference_present": {"type": "boolean"},
                "relation_inference_present": {"type": "boolean"},
                "unsupported_inference_present": {"type": "boolean"},
                "material_inference_present": {"type": "boolean"},
                "changes_core_meaning": {"type": "boolean"},
                "unsupported_inference_severity": SEVERITY_SCHEMA,
                "unsupported_inference_rationale": {"type": "string"},
            }
        ),
        "coverage_contribution": _object(
            {
                "contributes_unique_coverage": {"type": "boolean"},
                "covered_source_propositions": {"type": "array", "items": {"type": "string"}},
                "coverage_contribution_rationale": {"type": "string"},
            }
        ),
    }
)


SEMANTIC_ISSUE_CANDIDATE_SCHEMA = _object(
    {
        "issue_type": {"type": "string"},
        "dimension": {
            "type": "string",
            "enum": [
                "support",
                "evidence",
                "number_date",
                "entity",
                "entity_relation",
                "atomicity",
                "micro_fragmentation",
                "duplicate",
                "unsupported_inference",
                "coverage",
                "target_policy",
                "title_body_consistency",
                "input_target_boundary",
                "cross_fact_duplicate",
                "input_target_copy_risk",
                "global_coverage",
                "policy_compatibility",
                "other",
            ],
        },
        "severity": SEVERITY_SCHEMA,
        "fact_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "recommended_action": {"type": "string"},
    }
)


DUPLICATE_PAIR_SCHEMA = _object(
    {
        "left_fact_id": {"type": "string"},
        "right_fact_id": {"type": "string"},
        "duplicate_type": {
            "type": "string",
            "enum": [
                "exact_duplicate",
                "normalized_duplicate",
                "containment",
                "paraphrase_duplicate",
                "summary_detail_redundancy",
                "repeated_semantic_clause",
            ],
        },
        "duplicate_severity": SEVERITY_SCHEMA,
        "independent_increment_present": {"type": "boolean"},
        "canonical_duplicate_issue": {"type": "boolean"},
        "rationale": {"type": "string"},
    }
)


SAMPLE_LEVEL_REVIEW_SCHEMA = _object(
    {
        "title_body_semantic_consistency": _object(
            {
                "title_body_consistency_pass": {"type": "boolean"},
                "title_only_information_present": {"type": "boolean"},
                "title_only_information": {"type": "array", "items": {"type": "string"}},
                "body_supports_same_semantics": {"type": "boolean"},
                "candidate_uses_title_only_information": {"type": "boolean"},
                "target_uses_title_only_information": {"type": "boolean"},
                "input_target_gap_present": {"type": "boolean"},
                "controlled_source_title_required": {"type": "boolean"},
                "controlled_source_title_valid": {"type": "boolean"},
                "title_body_gap_severity": SEVERITY_SCHEMA,
                "rationale": {"type": "string"},
            }
        ),
        "input_target_information_boundary": _object(
            {
                "boundary_pass": {"type": "boolean"},
                "target_information_missing_from_input": {"type": "array", "items": {"type": "string"}},
                "input_information_not_needed_by_target": {"type": "array", "items": {"type": "string"}},
                "boundary_severity": SEVERITY_SCHEMA,
                "rationale": {"type": "string"},
            }
        ),
        "cross_fact_duplicate_redundancy": _object(
            {
                "all_fact_pairs_compared": {"type": "boolean"},
                "most_relevant_comparison_by_fact": {
                    "type": "array",
                    "items": _object(
                        {
                            "fact_id": {"type": "string"},
                            "compared_fact_id": {"type": "string"},
                            "relationship": {"type": "string"},
                        }
                    ),
                },
                "duplicate_pairs": {"type": "array", "items": DUPLICATE_PAIR_SCHEMA},
                "duplicate_dimension_pass": {"type": "boolean"},
                "rationale": {"type": "string"},
            }
        ),
        "input_target_copy_risk": _object(
            {
                "input_target_copy_risk_present": {"type": "boolean"},
                "copy_risk_fact_ids": {"type": "array", "items": {"type": "string"}},
                "copy_risk_source": {"type": "string"},
                "target_answer_exposure_present": {"type": "boolean"},
                "copy_risk_necessary": {"type": "boolean"},
                "unavoidable_factual_expression": {"type": "boolean"},
                "more_abstract_expression_possible": {"type": "boolean"},
                "copy_risk_semantic_severity": SEVERITY_SCHEMA,
                "copy_risk_recommended_action": {"type": "string"},
                "deterministic_metric_name": {"type": "string"},
                "deterministic_metric_value": {"type": "number", "minimum": 0, "maximum": 1},
                "deterministic_signal_level": {
                    "type": "string",
                    "enum": ["pass", "warning", "blocking"],
                },
                "rationale": {"type": "string"},
            }
        ),
        "global_semantic_coverage": _object(
            {
                "global_semantic_coverage_pass": {"type": "boolean"},
                "source_to_fact_coverage_pass": {"type": "boolean"},
                "fact_to_target_coverage_pass": {"type": "boolean"},
                "coverage_severity": SEVERITY_SCHEMA,
                "rationale": {"type": "string"},
            }
        ),
        "policy_compatibility": _object(
            {
                "policy_compatibility_pass": {"type": "boolean"},
                "unresolved_policy_conflicts": {"type": "array", "items": {"type": "string"}},
                "policy_compatibility_severity": SEVERITY_SCHEMA,
                "rationale": {"type": "string"},
            }
        ),
    }
)


DIMENSION_REVIEW_SCHEMA = _object(
    {
        "review_summary": {"type": "string"},
        "fact_reviews": {"type": "array", "minItems": 1, "items": FACT_DIMENSION_SCHEMA},
        "coverage": _object(
            {
                "core_event_covered": {"type": "boolean"},
                "critical_entities_covered": {"type": "boolean"},
                "critical_relations_covered": {"type": "boolean"},
                "critical_numbers_covered": {"type": "boolean"},
                "critical_dates_covered": {"type": "boolean"},
                "major_actions_covered": {"type": "boolean"},
                "major_results_covered": {"type": "boolean"},
                "future_plans_covered": {"type": "boolean"},
                "missing_critical_information": {"type": "array", "items": {"type": "string"}},
                "coverage_severity": SEVERITY_SCHEMA,
                "coverage_rationale": {"type": "string"},
            }
        ),
        "target_policy": _object(
            {
                "target_supported_by_candidate_facts": {"type": "boolean"},
                "target_mismatch_description": {"type": "string"},
                "target_integrity_severity": SEVERITY_SCHEMA,
                "grammatical_first_person_present": {"type": "boolean"},
                "first_person_policy_violation": {"type": "boolean"},
                "first_person_evidence": {"type": "array", "items": {"type": "string"}},
                "policy_severity": SEVERITY_SCHEMA,
                "title_body_evidence_gap_present": {"type": "boolean"},
                "title_body_evidence_gap_description": {"type": "string"},
                "title_body_evidence_gap_severity": SEVERITY_SCHEMA,
                "controlled_source_title_valid": {"type": "boolean"},
                "prompt_constraints_satisfied": {"type": "boolean"},
                "target_policy_rationale": {"type": "string"},
            }
        ),
        "sample_level_review": SAMPLE_LEVEL_REVIEW_SCHEMA,
        "semantic_issue_candidates": {
            "type": "array",
            "items": SEMANTIC_ISSUE_CANDIDATE_SCHEMA,
        },
    }
)


TAXONOMY_ALIAS_MAP: dict[str, list[str]] = {
    "fact_not_atomic": [
        "compound_fact",
        "multiple_claims",
        "multi_proposition_fact",
        "non_atomic_fact",
        "fact_fragmented",
    ],
    "fact_micro_fragmentation": [
        "micro_fragment",
        "dependent_fragment",
        "subjectless_fragment",
        "incomplete_predicate",
        "over_fragmentation",
    ],
    "fact_duplicate": ["duplicate_fact", "semantic_duplicate", "fact_containment", "redundant_fact"],
    "fact_unsupported": ["unsupported_fact", "unsupported_claim", "fact_support_failure"],
    "unsupported_inference": [
        "fact_unsupported",
        "unsupported_fact",
        "unsupported_relation",
        "inferred_relation",
        "causal_claim_not_supported",
        "unsupported_causality",
        "unsupported_temporal_claim",
        "unsupported_attribution",
        "inferred_conclusion",
    ],
    "entity_relation_error": ["false_entity_relation", "actor_object_error", "ownership_error"],
    "entity_error": ["incorrect_entity", "entity_mismatch"],
    "evidence_incorrect": ["evidence_incomplete", "evidence_indirect", "evidence_mismatch"],
    "number_date_error": ["number_error", "date_error", "numeric_scope_error"],
    "missing_critical_fact": ["critical_fact_missing", "major_omission"],
    "semantic_coverage_gap": ["coverage_gap", "incomplete_coverage"],
    "target_mismatch": ["target_integrity_error", "target_fact_mismatch"],
    "policy_compatibility_needs_resolution": ["first_person_policy", "prompt_policy_conflict"],
    "title_body_evidence_gap": ["title_evidence_gap", "title_body_support_gap"],
    "input_target_copy_risk": [
        "copy_risk",
        "target_answer_exposure",
        "input_target_overlap_risk",
        "verbatim_target_exposure",
    ],
}


TAXONOMY_ALIAS_RULES = {
    "canonicalization_source": "dimension_results_first",
    "raw_issue_role": "supplementary_alias_only",
    "contextual_disambiguation": {
        "fact_unsupported": "map to fact_unsupported when support.fact_supported=false; also map to unsupported_inference when unsupported_inference.unsupported_inference_present=true",
        "unsupported_relation": "map to unsupported_inference only when the unsupported-inference dimension is active; entity_relation_error may be emitted independently when the entity-relation dimension is also active",
        "compound_fact": "map to fact_not_atomic only when multiple independently assertable claims and should_split are both true",
    },
    "multi_label_allowed": True,
}


SEVERITY_POLICY: dict[str, Any] = {
    "rank_order": list(SEVERITIES),
    "fact_not_atomic": {
        "minimum": "major",
        "when": "contains_multiple_independently_assertable_claims and should_split",
        "exclusions": [
            "necessary parallel qualifiers under one subject and action",
            "minor verbosity that remains one proposition",
            "length alone",
        ],
    },
    "unsupported_inference": {
        "minimum": "major",
        "when": "material inference changes actor relation, causality, result, technology effect, time/status, plan/completion, or core meaning",
        "minor_allowed_only_when": "explicitly non-material wording risk that does not change the core proposition",
    },
    "fact_unsupported": {
        "minimum": "major",
        "when": "the core proposition lacks source support",
    },
    "entity_relation_error": {
        "minimum": "major",
        "when": "actor-action-object, ownership, cooperation, target, or state relation materially changes",
    },
    "missing_critical_fact": {
        "minimum": "major",
        "when": "material source information needed for the training target is absent",
    },
    "fact_duplicate": {
        "minimum": "major",
        "when": "two Facts carry the same independent training proposition, or one contains the other's full training value without necessary incremental information",
        "exclusions": ["shared subject or product name alone", "overlap with independent necessary information"],
    },
    "title_body_evidence_gap": {
        "minimum": "blocking",
        "when": "the immutable Target requires title-only information, the body lacks equivalent semantics, and the candidate has no valid fact-scoped controlled_source_title Evidence",
    },
    "input_target_copy_risk": {
        "minimum": "major",
        "when": "the input unnecessarily exposes a long or near-verbatim Target expression and a more abstract faithful Fact is possible",
        "metric_role": "signal_only; neither a high ratio automatically fails nor a low ratio automatically passes",
        "warning_threshold": 0.88,
        "blocking_threshold": 0.96,
    },
    "policy_compatibility_needs_resolution": {
        "minimum": "blocking",
        "when": "a genuine grammatical first-person target conflicts with an explicit avoid-first-person constraint and no valid sample-scoped compatibility resolution exists",
        "substring_exclusions": ["我国", "自我", "product-name substrings"],
    },
    "blocking": {
        "conditions": [
            "the sample is unusable without correction",
            "immutable target has a material title/body evidence gap",
            "unresolved policy conflict prevents a valid training example",
        ]
    },
}


DIMENSION_REVIEWER_SYSTEM = """You are an isolated blind dimension-first semantic reviewer for Chinese-news SFT annotations.
You receive only the complete source, candidate, all Facts, expanded Evidence, and this general rubric. You do not receive
historical labels, expected answers, Issue IDs, prior reviews, or Revision results. Return only the required JSON schema.

For EVERY Fact, independently complete EVERY required dimension. Never skip a dimension because you did not create a free
Issue. First decompose the Fact into independently assertable claim components, then compare every component against its
cited Evidence and the complete source. The semantic_issue_candidates list is supplementary; the dimension fields are the
primary semantic judgment.

Atomicity: atomic does not mean shortest. A Fact fails atomicity when it contains two or more independently true core
propositions with separate training value and should be split. A necessary multi-object list under one subject/action, shared
qualifiers, or mere length does not fail. If multiple independently assertable claims require splitting, atomicity severity is
at least major. Minor verbosity without a split requirement may be minor/warning.

Unsupported Fact and Unsupported Inference are not mutually exclusive. fact_unsupported means the core proposition itself
lacks source support. unsupported_inference means source entities or facts are reused but the Fact adds an unstated causal,
temporal, attribution, result/effect, necessity, actor, ownership, or relation conclusion. A Fact may have both. A material
inference changing actor relation, causality, action result, technology effect, time/status, plan/completion state, or core
meaning is at least major. Explicitly mark temporal_inference_present when a present source description is changed into a
future plan, or a plan into a completed/current state.

Entity Relation is independent from entity string presence. Verify actor, action, object, partners, ownership, technology
target, project/plan subject, and published/built/planned/completed status. Parallel relations must not become a common-subject
or causal relation. If one wording defect is both an incorrect relation and an unsupported inferred relation/status, mark both
dimensions; multi-label findings are allowed.

Micro-fragmentation: a Fact is problematic when it lacks an explicit/recoverable subject or complete predicate, depends on a
neighbor for interpretation, or should merge with an adjacent Fact. Do not flag concise but independently meaningful Facts.
Coverage is sample-level and must inventory the complete source. Do not treat 我国, 自我, or product-name substrings as
grammatical first person.

After every Fact review, independently complete all six sample-level dimensions. Passing every individual Fact never implies
that the sample passes.

Title / Body Semantic Consistency and Input / Target Boundary: decompose the source title into minimal factual components
(especially year, edition, place, event identity, actor, action, object, status, and superlative or ordinal attributes). For
each component, decide whether the body states the same semantics, not merely a related token or a product proper name.
Separately decide whether the immutable Target uses the title-only component and whether Topic, Facts, Outline, Constraints,
or User Prompt legally supply it. Seeing source.title in this audit payload does not itself put that information into the
training input. A valid title-only training component requires a fact-scoped Evidence type controlled_source_title; never
pretend a body paragraph contains it. When a necessary Target title component is absent from the body and training input,
set input_target_gap_present=true, controlled_source_title_required=true, controlled_source_title_valid=false, and severity
blocking. If the body already supports equivalent semantics, controlled_source_title_required=false. A year inside a named
plan/product does not automatically establish that the event, strategy, or article itself occurs in that year.

Cross-Fact Duplicate / Redundancy: compare every Fact against its most relevant other Fact and report pairwise exact,
normalized, containment, paraphrase, summary/detail, and repeated-clause duplicates. Shared actor or product names alone are
not duplicates. A pair is a major canonical duplicate when both primarily train the same independent proposition, or one
Fact fully contains the other's training value and the narrower Fact adds no necessary independent information.

Input-Target Copy Risk: use any supplied deterministic similarity only as a signal. Semantically inspect Topic, Facts,
Outline, and User Prompt for unnecessary exposure of a Target sentence or long distinctive clause. Low similarity never
automatically passes and high similarity never automatically fails. Distinguish unavoidable names, numbers, dates, and
necessary factual phrasing from a near-answer Fact that can be expressed more abstractly without losing fidelity. Unnecessary
near-verbatim answer exposure is normally major; choose blocking only when the policy threshold and semantic unusability both
justify it. Explicitly echo the supplied metric name/value/signal in the structured result.

Global Semantic Coverage and Policy Compatibility remain independent sample judgments and must be answered explicitly.
Use severity none for a passing dimension and provide concise evidence-based rationales."""


SAMPLE_LEVEL_REVIEWER_PROMPTS = {
    "task011d-e2-sample-level-reviewer-v1": DIMENSION_REVIEWER_SYSTEM,
    "task011d-e2-sample-level-reviewer-v2": DIMENSION_REVIEWER_SYSTEM
    + """

Calibration discipline: before returning JSON, explicitly build three internal matrices from the supplied data only:
(1) every minimal title component versus body support, Target use, and candidate controlled-title Evidence; (2) every Fact
versus its most semantically similar Fact; and (3) every Fact versus the closest Target sentence, separating necessary
factual tokens from avoidable copied syntax. Populate the required sample-level fields from those matrices. Do not disclose
the matrices outside the required schema and do not infer expected labels.""",
    "task011d-e2-sample-level-reviewer-v3": DIMENSION_REVIEWER_SYSTEM
    + """

Adversarial calibration discipline: attempt to falsify a sample-level pass independently for title/body consistency,
input/Target boundary, pairwise redundancy, and semantic copy risk. For a title component, require equivalent body meaning,
not token coincidence. For duplicate containment, ask whether deleting the narrower Fact loses any independent training
proposition. Judge the proposition duplicated inside the pair, not whether the broader Fact also contains other propositions:
a broad compound Fact's extra material does not make a narrower Fact non-duplicate. A generic context qualifier or a
verify/implement paraphrase is not an independent increment when both Facts still train the same outcome.

For copy risk, ask whether a shorter abstract Fact could preserve the same entities, date, place, action, and relation without
reproducing Target sentence structure. Necessary entity/date/place tokens do not make the surrounding clause order and syntax
unavoidable. Treat a ratio close to the warning line on either side as a review signal, never as a pass/fail rule. Cross-check
atomicity: when a long Fact closely follows one or adjacent Target sentences and should be split into independent propositions,
its combined answer-like syntax is not necessary merely because each factual token is necessary. If faithful splitting and
paraphrase are possible, mark semantic copy risk major even when the deterministic signal level is pass. Record a defect only
when this evidence-based challenge succeeds; never use sample identity or hidden history.""",
    "task011d-e2-sample-level-reviewer-v4": DIMENSION_REVIEWER_SYSTEM
    + """

Duplicate Taxonomy v2 calibration: duplicate status is controlled by a delete-one independent-information test, not text
similarity or shared-event identity. Compare subject, action, object, result, number, time, relationship, and status. Mark
exact/paraphrase/containment/summary-detail redundancy only when deleting one Fact loses no independently necessary trainable
proposition. If both Facts preserve non-substitutable information, classify the relationship as related but independent and
do not emit a duplicate. Do not use any sample identity, fixture wording, or expected answer.""",
    "task011d-e2-sample-level-reviewer-v5": DIMENSION_REVIEWER_SYSTEM
    + """

Duplicate Taxonomy v2 strict calibration: construct the shared proposition and each Fact's unique propositions. A broader
Fact containing additional material does not rescue a narrower Fact when the narrower Fact itself adds nothing; conversely,
a detail is independent only when its actor/action/object/result/number/time/relationship/status cannot be recovered from the
other Fact. Apply the delete-one test symmetrically. Emit a canonical major duplicate only when one deletion is information-
preserving; otherwise record related-but-independent comparison text without a duplicate issue. Never key on a sample ID or
historical label.""",
}

# Duplicate governance calibration must preserve every v3 capability and add only the
# general independent-information policy.  These post-definitions deliberately avoid
# copying any fixture identifier or wording into the prompt.
SAMPLE_LEVEL_REVIEWER_PROMPTS["task011d-e2-sample-level-reviewer-v4"] = (
    SAMPLE_LEVEL_REVIEWER_PROMPTS["task011d-e2-sample-level-reviewer-v3"]
    + """

Duplicate Taxonomy v2 calibration: apply a symmetric delete-one independent-information test. Compare subject, action,
object, result, number, time, relationship, and status. Exact, paraphrase, containment, or summary/detail duplication exists
only when deleting one Fact loses no independently necessary trainable proposition. Shared event identity or lexical overlap
is never sufficient. When both Facts retain non-substitutable information, record related-but-independent comparison text and
do not emit a duplicate. Never use sample identity, fixture wording, or expected labels."""
)
SAMPLE_LEVEL_REVIEWER_PROMPTS["task011d-e2-sample-level-reviewer-v5"] = (
    SAMPLE_LEVEL_REVIEWER_PROMPTS["task011d-e2-sample-level-reviewer-v3"]
    + """

Duplicate Taxonomy v2 strict calibration: explicitly identify the shared proposition and each Fact's unique propositions,
then apply the delete-one test symmetrically. A broader Fact's extra material does not rescue a narrower Fact that adds no
information; a detail is independent only if its subject/action/object/result/number/time/relationship/status cannot be
recovered from the other Fact. Emit a canonical major duplicate only when one deletion preserves all independent information.
Otherwise classify the pair as related but independent. Never use sample identity, fixture wording, or expected labels."""
)


def _at_least(value: str, minimum: str) -> str:
    if value not in SEVERITY_RANK:
        value = "none"
    return minimum if SEVERITY_RANK[value] < SEVERITY_RANK[minimum] else value


def _candidate_aliases(review: dict[str, Any], canonical_type: str, fact_ids: list[str]) -> list[str]:
    accepted = {canonical_type, *TAXONOMY_ALIAS_MAP.get(canonical_type, [])}
    fact_set = set(fact_ids)
    aliases: list[str] = []
    for issue in review.get("semantic_issue_candidates", []):
        raw = issue.get("issue_type", "").lower().replace("-", "_").replace(" ", "_")
        raw_fact_ids = set(issue.get("fact_ids", []))
        if raw in accepted and (not fact_set or not raw_fact_ids or fact_set & raw_fact_ids):
            aliases.append(raw)
    return sorted(set(aliases))


def _issue(
    review: dict[str, Any],
    *,
    issue_type: str,
    dimension: str,
    fact_ids: list[str],
    severity: str,
    rationale: str,
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "issue_type": issue_type,
        "dimension": dimension,
        "fact_ids": fact_ids,
        "severity": severity,
        "rationale": rationale,
        "recommended_action": recommended_action,
        "semantic_issue_aliases": _candidate_aliases(review, issue_type, fact_ids),
        "derived_from_dimensions": True,
    }


def canonical_issue_builder(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert model semantic dimensions into project taxonomy without inventing semantic judgments."""

    issues: list[dict[str, Any]] = []
    for fact in review["fact_reviews"]:
        fact_ids = [fact["fact_id"]]
        support = fact["support"]
        if not support["fact_supported"]:
            severity = support["support_severity"]
            if support["core_fact_unsupported"]:
                severity = _at_least(severity, "major")
            issues.append(
                _issue(
                    review,
                    issue_type="fact_unsupported",
                    dimension="support",
                    fact_ids=fact_ids,
                    severity=severity,
                    rationale=support["support_rationale"],
                    recommended_action="Remove or rewrite unsupported claim components using direct source support.",
                )
            )
        evidence = fact["evidence"]
        if not evidence["evidence_complete"] or not evidence["evidence_direct"]:
            issues.append(
                _issue(
                    review,
                    issue_type="evidence_incorrect",
                    dimension="evidence",
                    fact_ids=fact_ids,
                    severity=evidence["evidence_issue_severity"],
                    rationale=evidence["evidence_rationale"],
                    recommended_action="Replace or complete Evidence with paragraphs that directly support every claim component.",
                )
            )
        number_date = fact["number_date"]
        if not number_date["number_correct"] or not number_date["date_correct"]:
            issues.append(
                _issue(
                    review,
                    issue_type="number_date_error",
                    dimension="number_date",
                    fact_ids=fact_ids,
                    severity=number_date["number_date_severity"],
                    rationale=number_date["number_date_rationale"],
                    recommended_action="Restore the exact source-supported number, date, scope, and status qualifiers.",
                )
            )
        entity = fact["entity"]
        if not entity["entity_correct"]:
            issues.append(
                _issue(
                    review,
                    issue_type="entity_error",
                    dimension="entity",
                    fact_ids=fact_ids,
                    severity=entity["entity_severity"],
                    rationale=entity["entity_rationale"],
                    recommended_action="Restore the source-supported entity identity and scope.",
                )
            )
        relation = fact["entity_relation"]
        if not relation["entity_relation_correct"]:
            issues.append(
                _issue(
                    review,
                    issue_type="entity_relation_error",
                    dimension="entity_relation",
                    fact_ids=fact_ids,
                    severity=_at_least(relation["entity_relation_severity"], "major"),
                    rationale=relation["entity_relation_rationale"],
                    recommended_action="Rewrite actor-action-object, ownership, cooperation, target, and state relations exactly.",
                )
            )
        atomicity = fact["atomicity"]
        if atomicity["contains_multiple_independently_assertable_claims"] and atomicity["should_split"]:
            issues.append(
                _issue(
                    review,
                    issue_type="fact_not_atomic",
                    dimension="atomicity",
                    fact_ids=fact_ids,
                    severity=_at_least(atomicity["atomicity_severity"], "major"),
                    rationale=atomicity["atomicity_rationale"],
                    recommended_action="Split the independently assertable propositions into separately evidenced Facts.",
                )
            )
        micro = fact["micro_fragmentation"]
        micro_failed = (
            not micro["independently_interpretable"]
            or not micro["subject_explicit"]
            or not micro["predicate_complete"]
            or micro["depends_on_neighbor_fact"]
            or atomicity["should_merge_with_adjacent_fact"]
        )
        if micro_failed:
            issues.append(
                _issue(
                    review,
                    issue_type="fact_micro_fragmentation",
                    dimension="micro_fragmentation",
                    fact_ids=fact_ids,
                    severity=micro["micro_fragmentation_severity"],
                    rationale=micro["micro_fragmentation_rationale"],
                    recommended_action="Merge with the necessary adjacent Fact or restore an explicit subject and complete predicate.",
                )
            )
        duplicate = fact["duplicate"]
        if duplicate["duplicate_status"] != "none":
            duplicate_ids = sorted(set(fact_ids + duplicate["duplicate_with_fact_ids"]))
            issues.append(
                _issue(
                    review,
                    issue_type="fact_duplicate",
                    dimension="duplicate",
                    fact_ids=duplicate_ids,
                    severity=duplicate["duplicate_severity"],
                    rationale=duplicate["duplicate_rationale"],
                    recommended_action="Remove the duplicate or consolidate contained semantics without losing Evidence.",
                )
            )
        inference = fact["unsupported_inference"]
        if inference["unsupported_inference_present"]:
            severity = inference["unsupported_inference_severity"]
            if inference["material_inference_present"] or inference["changes_core_meaning"]:
                severity = _at_least(severity, "major")
            issues.append(
                _issue(
                    review,
                    issue_type="unsupported_inference",
                    dimension="unsupported_inference",
                    fact_ids=fact_ids,
                    severity=severity,
                    rationale=inference["unsupported_inference_rationale"],
                    recommended_action="Remove the inferred causal, temporal, attribution, effect, or relation component.",
                )
            )

    coverage = review["coverage"]
    if coverage["missing_critical_information"]:
        issues.append(
            _issue(
                review,
                issue_type="missing_critical_fact",
                dimension="coverage",
                fact_ids=[],
                severity=_at_least(coverage["coverage_severity"], "major"),
                rationale=coverage["coverage_rationale"],
                recommended_action="Add source-supported Facts for the listed critical information.",
            )
        )
    coverage_checks = (
        "core_event_covered",
        "critical_entities_covered",
        "critical_relations_covered",
        "critical_numbers_covered",
        "critical_dates_covered",
        "major_actions_covered",
        "major_results_covered",
        "future_plans_covered",
    )
    if any(not coverage[name] for name in coverage_checks):
        issues.append(
            _issue(
                review,
                issue_type="semantic_coverage_gap",
                dimension="coverage",
                fact_ids=[],
                severity=coverage["coverage_severity"],
                rationale=coverage["coverage_rationale"],
                recommended_action="Restore material source coverage while avoiding redundant or synthetic Facts.",
            )
        )

    target = review["target_policy"]
    if not target["target_supported_by_candidate_facts"]:
        issues.append(
            _issue(
                review,
                issue_type="target_mismatch",
                dimension="target_policy",
                fact_ids=[],
                severity=target["target_integrity_severity"],
                rationale=target["target_mismatch_description"],
                recommended_action="Align the candidate Facts and prompt with the immutable target using source-supported content.",
            )
        )
    if target["first_person_policy_violation"] or (
        not target["prompt_constraints_satisfied"] and target["policy_severity"] != "none"
    ):
        policy_severity = target["policy_severity"]
        if target["first_person_policy_violation"] and target["grammatical_first_person_present"]:
            policy_severity = _at_least(policy_severity, "blocking")
        issues.append(
            _issue(
                review,
                issue_type="policy_compatibility_needs_resolution",
                dimension="target_policy",
                fact_ids=[],
                severity=policy_severity,
                rationale=target["target_policy_rationale"],
                recommended_action="Resolve the genuine prompt/target policy conflict without lexical-substring false positives.",
            )
        )
    if target["title_body_evidence_gap_present"]:
        issues.append(
            _issue(
                review,
                issue_type="title_body_evidence_gap",
                dimension="target_policy",
                fact_ids=[],
                severity=target["title_body_evidence_gap_severity"],
                rationale=target["title_body_evidence_gap_description"],
                recommended_action="Use a valid controlled title source or remove the unsupported title-only training claim.",
            )
        )

    sample_level = review.get("sample_level_review", {})
    title_body = sample_level.get("title_body_semantic_consistency", {})
    if title_body.get("input_target_gap_present"):
        severity = title_body.get("title_body_gap_severity", "none")
        if title_body.get("controlled_source_title_required") and not title_body.get(
            "controlled_source_title_valid"
        ):
            severity = _at_least(severity, "blocking")
        issues.append(
            _issue(
                review,
                issue_type="title_body_evidence_gap",
                dimension="title_body_consistency",
                fact_ids=[],
                severity=severity,
                rationale=title_body.get("rationale", "Title-only Target information is absent from the input."),
                recommended_action="Add fact-scoped controlled_source_title Evidence when policy permits, without fabricating body support or modifying the immutable Target.",
            )
        )

    duplicate_review = sample_level.get("cross_fact_duplicate_redundancy", {})
    for pair in duplicate_review.get("duplicate_pairs", []):
        if not pair.get("canonical_duplicate_issue"):
            continue
        severity = pair.get("duplicate_severity", "none")
        if not pair.get("independent_increment_present"):
            severity = _at_least(severity, "major")
        issues.append(
            _issue(
                review,
                issue_type="fact_duplicate",
                dimension="cross_fact_duplicate",
                fact_ids=sorted({pair["left_fact_id"], pair["right_fact_id"]}),
                severity=severity,
                rationale=pair.get("rationale", "The pair repeats one independent training proposition."),
                recommended_action="Remove the redundant Fact or rewrite the pair into non-overlapping propositions while preserving necessary Evidence.",
            )
        )

    copy_risk = sample_level.get("input_target_copy_risk", {})
    if copy_risk.get("input_target_copy_risk_present"):
        severity = copy_risk.get("copy_risk_semantic_severity", "none")
        if copy_risk.get("target_answer_exposure_present") and not copy_risk.get("copy_risk_necessary"):
            severity = _at_least(severity, "major")
        issues.append(
            _issue(
                review,
                issue_type="input_target_copy_risk",
                dimension="input_target_copy_risk",
                fact_ids=sorted(set(copy_risk.get("copy_risk_fact_ids", []))),
                severity=severity,
                rationale=copy_risk.get("rationale", "The input unnecessarily exposes Target wording."),
                recommended_action=copy_risk.get(
                    "copy_risk_recommended_action",
                    "Paraphrase the input at the Fact level while retaining exact necessary entities, numbers, dates, and relations.",
                ),
            )
        )

    # Fact-level and sample-level dimensions may independently identify the same defect.
    # Emit one canonical issue per taxonomy/fact scope and preserve the stricter severity.
    merged: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    order: list[tuple[str, tuple[str, ...]]] = []
    for issue in issues:
        key = (issue["issue_type"], tuple(sorted(issue["fact_ids"])))
        if key not in merged:
            merged[key] = issue
            order.append(key)
            continue
        current = merged[key]
        if SEVERITY_RANK.get(issue["severity"], 0) > SEVERITY_RANK.get(current["severity"], 0):
            merged[key] = issue
    return [merged[key] for key in order]


def validate_dimension_review(review: dict[str, Any], expected_fact_ids: list[str]) -> dict[str, Any]:
    actual_fact_ids = [row["fact_id"] for row in review.get("fact_reviews", [])]
    required_fact_dimensions = set(FACT_DIMENSION_SCHEMA["required"])
    dimension_completeness = [
        required_fact_dimensions <= set(row) for row in review.get("fact_reviews", [])
    ]
    return {
        "expected_fact_ids": expected_fact_ids,
        "actual_fact_ids": actual_fact_ids,
        "fact_ids_exact": actual_fact_ids == expected_fact_ids,
        "fact_ids_unique": len(actual_fact_ids) == len(set(actual_fact_ids)),
        "all_fact_dimensions_required": all(dimension_completeness) and bool(dimension_completeness),
        "sample_coverage_present": "coverage" in review,
        "target_policy_present": "target_policy" in review,
        "sample_level_review_present": "sample_level_review" in review,
        "sample_level_dimensions_complete": set(SAMPLE_LEVEL_REVIEW_SCHEMA["required"])
        <= set(review.get("sample_level_review", {})),
        "semantic_issue_candidates_present": "semantic_issue_candidates" in review,
        "passed": (
            actual_fact_ids == expected_fact_ids
            and len(actual_fact_ids) == len(set(actual_fact_ids))
            and all(dimension_completeness)
            and bool(dimension_completeness)
            and "coverage" in review
            and "target_policy" in review
            and "sample_level_review" in review
            and set(SAMPLE_LEVEL_REVIEW_SCHEMA["required"])
            <= set(review.get("sample_level_review", {}))
            and "semantic_issue_candidates" in review
        ),
    }


def _historical_fact_ids(value: str | list[str]) -> set[str]:
    if isinstance(value, list):
        return {str(row) for row in value if row}
    normalized = value.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return {token.strip() for token in normalized.replace("|", ",").split(",") if token.strip()}


def severity_satisfies(actual: str, expected: str) -> bool:
    return actual in SEVERITY_RANK and expected in SEVERITY_RANK and SEVERITY_RANK[actual] >= SEVERITY_RANK[expected]


def canonical_taxonomy_type(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if normalized in TAXONOMY_ALIAS_MAP:
        return normalized
    for canonical_type, aliases in TAXONOMY_ALIAS_MAP.items():
        if normalized in aliases:
            return canonical_type
    return normalized


def core_issue_evaluation(historical: dict[str, Any], canonical_issues: list[dict[str, Any]]) -> dict[str, Any]:
    expected_type = historical["issue_type"]
    expected_canonical_type = canonical_taxonomy_type(expected_type)
    expected_severity = historical["severity"]
    expected_fact_ids = _historical_fact_ids(historical.get("fact_ids", historical.get("fact_id", "")))
    matches: list[dict[str, Any]] = []
    for issue in canonical_issues:
        if canonical_taxonomy_type(issue["issue_type"]) != expected_canonical_type:
            continue
        actual_fact_ids = set(issue["fact_ids"])
        fact_scope_matches = not expected_fact_ids or not actual_fact_ids or bool(expected_fact_ids & actual_fact_ids)
        if fact_scope_matches:
            matches.append(issue)
    severity_correct = any(severity_satisfies(row["severity"], expected_severity) for row in matches)
    recommended_action_reasonable = any(bool(row["recommended_action"].strip()) for row in matches)
    return {
        "expected_issue_type": expected_type,
        "expected_canonical_issue_type": expected_canonical_type,
        "expected_severity": expected_severity,
        "expected_fact_ids": sorted(expected_fact_ids),
        "matching_canonical_issue_count": len(matches),
        "matching_canonical_issues": deepcopy(matches),
        "canonical_issue_correct_or_semantically_equivalent": bool(matches),
        "severity_correct": severity_correct,
        "recommended_action_reasonable": recommended_action_reasonable,
        "core_issue_detected": bool(matches) and severity_correct and recommended_action_reasonable,
        "raw_issue_label_exact_match_required": False,
    }
