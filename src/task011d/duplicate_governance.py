from __future__ import annotations

from typing import Any


DUPLICATE_TAXONOMY_V2: dict[str, Any] = {
    "schema_version": "task011d-e2-duplicate-taxonomy-v2.0.0",
    "decision_criterion": (
        "Delete-one independent-information test: a duplicate exists only when removing one Fact "
        "loses no independently necessary trainable proposition. Text similarity and shared event identity are signals only."
    ),
    "comparison_fields": ["subject", "action", "object", "result", "number", "time", "relationship", "status"],
    "classes": {
        "exact_duplicate": "The core trainable proposition and independently necessary information are identical.",
        "paraphrase_duplicate": "Wording differs, but independently necessary factual content is identical.",
        "containment_duplicate": "One Fact contains all independent information of the other, which adds no independent value.",
        "summary_detail_redundancy": "A summary merely compresses a detail Fact and adds no independent proposition value.",
        "related_but_independent": "Facts share an event or subject but each preserves information not replaceable by the other.",
        "ambiguous": "The available source and Fact boundaries do not support a stable independent-information decision.",
    },
    "shared_event_is_sufficient": False,
    "text_similarity_is_sufficient": False,
    "detail_is_automatically_independent": False,
    "summary_is_automatically_duplicate": False,
}


DUPLICATE_GOVERNANCE_REVIEWER_SYSTEM = """
Apply the supplied general Duplicate Taxonomy only. Compare Fact A and Fact B proposition-by-proposition using subject,
action, object, result, number, time, relationship, and status. The controlling test is whether deleting either Fact
loses independently necessary trainable information. A shared event, actor, product, or semantic overlap is not enough.
Detail is not automatically valuable and summary is not automatically redundant. Report compact conclusions only;
do not provide hidden chain-of-thought. Do not infer or seek historical labels, expected answers, sample history, or
other reviewers. Use only the supplied source, the two Facts and their Evidence, and the general policy.
""".strip()


DUPLICATE_GOVERNANCE_JUDGE_SYSTEM = """
Act as an independent governance Judge. Use the supplied source, two Facts and Evidence, general Duplicate Taxonomy,
and three structured blind reviews. Do not infer or seek any historical label or expected answer. Decide whether the
delete-one independent-information test establishes a true duplicate, independent Facts, or a genuine taxonomy
boundary. If true duplicate, select exactly one subtype. Return compact conclusions only, without hidden reasoning.
""".strip()


DUPLICATE_GOVERNANCE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "classification", "shared_information", "unique_information_fact_a", "unique_information_fact_b",
        "removing_a_loses_information", "removing_b_loses_information", "recommended_keep", "confidence",
        "short_rationale",
    ],
    "properties": {
        "classification": {"type": "string", "enum": list(DUPLICATE_TAXONOMY_V2["classes"])},
        "shared_information": {"type": "array", "items": {"type": "string"}},
        "unique_information_fact_a": {"type": "array", "items": {"type": "string"}},
        "unique_information_fact_b": {"type": "array", "items": {"type": "string"}},
        "removing_a_loses_information": {"type": "boolean"},
        "removing_b_loses_information": {"type": "boolean"},
        "recommended_keep": {"type": "string", "enum": ["fact_a", "fact_b", "both", "either", "undetermined"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "short_rationale": {"type": "string"},
    },
}


DUPLICATE_GOVERNANCE_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision", "duplicate_subtype", "shared_information", "unique_information_fact_a",
        "unique_information_fact_b", "removing_a_loses_information", "removing_b_loses_information",
        "recommended_keep", "confidence", "short_rationale",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["true_duplicate", "not_duplicate_independent", "taxonomy_boundary_ambiguous"],
        },
        "duplicate_subtype": {
            "type": "string",
            "enum": ["exact_duplicate", "paraphrase_duplicate", "containment_duplicate", "summary_detail_redundancy", "not_applicable"],
        },
        "shared_information": {"type": "array", "items": {"type": "string"}},
        "unique_information_fact_a": {"type": "array", "items": {"type": "string"}},
        "unique_information_fact_b": {"type": "array", "items": {"type": "string"}},
        "removing_a_loses_information": {"type": "boolean"},
        "removing_b_loses_information": {"type": "boolean"},
        "recommended_keep": {"type": "string", "enum": ["fact_a", "fact_b", "both", "either", "undetermined"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "short_rationale": {"type": "string"},
    },
}
