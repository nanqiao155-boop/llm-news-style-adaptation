from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.task011d.duplicate_governance import (
    DUPLICATE_GOVERNANCE_JUDGE_SCHEMA,
    DUPLICATE_GOVERNANCE_JUDGE_SYSTEM,
    DUPLICATE_GOVERNANCE_REVIEW_SCHEMA,
    DUPLICATE_GOVERNANCE_REVIEWER_SYSTEM,
)
from src.task011d.reviewer_taxonomy import (
    DIMENSION_REVIEW_SCHEMA,
    DIMENSION_REVIEWER_SYSTEM,
    SAMPLE_LEVEL_REVIEWER_PROMPTS,
)


class SemanticProviderError(RuntimeError):
    """Raised when a real semantic model invocation cannot be completed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _extract_output_text(response: dict[str, Any]) -> str:
    values: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                values.append(content["text"])
    if not values:
        raise SemanticProviderError("Responses API returned no output_text item")
    return "".join(values)


class InvocationLogger:
    """Append-only JSONL logger for semantic inference calls."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = _canonical_json(record) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)


class SemanticModelProvider(ABC):
    """The semantic boundary; deterministic validators must never implement it."""

    @abstractmethod
    def generate_sft(self, *, sample_id: str, split: str, source: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def review_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviewer: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def revise_sft(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        issues: list[dict[str, Any]],
        revision_round: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def re_review_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def judge_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviews: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError


GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "topic": {"type": "string"},
        "facts": {
            "type": "array",
            "minItems": 3,
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fact": {"type": "string"},
                    "evidence_paragraph_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "fact_type": {
                        "type": "string",
                        "enum": ["event", "cooperation", "plan", "result", "number", "other"],
                    },
                },
                "required": ["fact", "evidence_paragraph_ids", "fact_type"],
            },
        },
        "outline": {"type": "array", "minItems": 3, "maxItems": 7, "items": {"type": "string"}},
        "target_first_person_compatibility": {"type": "boolean"},
        "first_person_rationale": {"type": "string"},
        "controlled_title_fact": {"type": ["string", "null"]},
        "controlled_title_rationale": {"type": "string"},
    },
    "required": [
        "topic",
        "facts",
        "outline",
        "target_first_person_compatibility",
        "first_person_rationale",
        "controlled_title_fact",
        "controlled_title_rationale",
    ],
}


ISSUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "issue_type": {
            "type": "string",
            "enum": [
                "fact_not_atomic",
                "fact_micro_fragmentation",
                "fact_fragmented",
                "over_fragmentation",
                "fact_duplicate",
                "unsupported_inference",
                "entity_relation_error",
                "entity_error",
                "fact_unsupported",
                "evidence_incorrect",
                "number_date_error",
                "missing_critical_fact",
                "semantic_coverage_gap",
                "topic_error",
                "outline_error",
                "prompt_error",
                "target_mismatch",
                "policy_compatibility_needs_resolution",
                "title_body_evidence_gap",
                "provenance_error",
                "input_target_copy_risk",
                "other",
            ],
        },
        "severity": {"type": "string", "enum": ["blocking", "major", "minor", "warning"]},
        "field_path": {"type": "string"},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "suggested_fix": {"type": "string"},
    },
    "required": ["issue_type", "severity", "field_path", "fact_ids", "rationale", "suggested_fix"],
}


CHECK_NAMES = (
    "topic_correctness",
    "fact_support",
    "evidence_correctness",
    "number_date",
    "entity",
    "entity_relation",
    "atomicity",
    "micro_fragmentation",
    "over_fragmentation",
    "duplicate",
    "unsupported_inference",
    "missing_critical_fact",
    "semantic_coverage",
    "outline",
    "prompt_sync",
    "target_integrity",
    "policy_compatibility",
    "provenance",
)


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "accepted",
                "accepted_with_warning",
                "revision_required",
                "judge_required",
                "quarantine_recommended",
            ],
        },
        "issues": {"type": "array", "items": ISSUE_SCHEMA},
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "properties": {name: {"type": "string", "enum": ["pass", "warning", "fail"]} for name in CHECK_NAMES},
            "required": list(CHECK_NAMES),
        },
        "rationale": {"type": "string"},
        "fact_granularity_review_required": {"type": "boolean"},
        "high_risk_fact_granularity_confirmed_necessary": {"type": "boolean"},
        "target_first_person_compatibility_valid": {"type": "boolean"},
        "controlled_source_title_valid": {"type": "boolean"},
    },
    "required": [
        "verdict",
        "issues",
        "checks",
        "rationale",
        "fact_granularity_review_required",
        "high_risk_fact_granularity_confirmed_necessary",
        "target_first_person_compatibility_valid",
        "controlled_source_title_valid",
    ],
}


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["accept_a", "accept_b", "revision_required", "quarantine_recommended"],
        },
        "confirmed_issues": {"type": "array", "items": ISSUE_SCHEMA},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "confirmed_issues", "rationale"],
}


EVENT_PAIR_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["same_event", "different_event", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "core_event_identity": {"type": "string"},
        "decisive_evidence": {"type": "string"},
    },
    "required": ["verdict", "confidence", "core_event_identity", "decisive_evidence"],
}


EVENT_PAIR_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["same_event", "different_event", "still_uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "core_event_identity": {"type": "string"},
        "decisive_evidence": {"type": "string"},
    },
    "required": ["verdict", "confidence", "core_event_identity", "decisive_evidence"],
}


EVENT_PAIR_REVIEWER_SYSTEM = """You are AI_EVENT_GROUP_AUDITOR, an isolated event-identity reviewer.
Decide whether Article A and Article B describe the same real-world event, not merely the same topic,
organization, product, conference series, or recurring activity. Use only the supplied title, publish date,
and body for each article. Same event requires a shared concrete occurrence with compatible actors, action,
object, place, time, and event state; follow-up or preview coverage may be the same event when the concrete
occurrence is clearly identical. Prefer uncertain when evidence cannot safely distinguish same event from a
related event. Return only the structured result. You are not told and must not infer dataset split or current
event-group assignments."""


EVENT_PAIR_JUDGE_SYSTEM = """You are AI_EVENT_GROUP_JUDGE, an independent event-identity judge.
From only Article A and Article B, decide whether they describe the same concrete real-world event. You receive
no prior reviewer verdict or hidden reasoning and no dataset split or event-group information. Same organization,
topic, product, conference series, or recurring campaign is insufficient without a shared concrete occurrence.
Return still_uncertain whenever the supplied evidence cannot safely resolve identity. Return only the structured
result."""


EVENT_PAIR_REPAIR_JUDGE_SYSTEM = """You are AI_EVENT_GROUP_JUDGE, an independent event-identity judge.
Use only Article A, Article B, and the supplied structured reviewer result. The reviewer result contains no hidden
reasoning. You are not told dataset split, current event-group membership, historical audit labels, or an expected
answer. Decide whether the two articles describe the same concrete real-world event. Same organization, topic,
product, conference series, or recurring campaign is insufficient without a shared concrete occurrence. Return
still_uncertain whenever the supplied evidence cannot safely resolve identity. Return only the structured result."""


GENERATOR_SYSTEM = """You are the isolated Semantic SFT Generator for Chinese news annotation.
Extract training-useful propositions from only the supplied title and numbered body paragraphs.
Atomic does not mean shortest. Every fact needs a clear subject, complete predicate, necessary object,
and qualifiers needed for independent understanding. Consolidate clauses sharing one subject/action/result
when splitting removes independent training value. Keep genuinely distinct events separate.
Recommended fact counts: short 3-6, medium 6-10, long 8-14, very_long/complex 10-16.
Never mechanically create separate subject/date/number/action/object facts. Never invent evidence.
Evidence IDs must be body paragraph IDs. A controlled title fact is allowed only when the title alone
supports a minimal fact, the body does not support equivalent semantics, the target needs it, and omission
would create an input-target gap. First-person compatibility is true only for real grammatical first-person
language in the rendered target, including direct quotations; exclude 我国, 自我, product names, and substrings.
Return only the required structured result."""


REVISION_SYSTEM = """You are the isolated AI SFT Revision Agent for Chinese-news annotation.
Use only the supplied source, current candidate, and confirmed issues. Make the smallest semantic change that
resolves those issues. Do not rewrite unaffected Facts, broaden scope, alter the immutable Target, invent Evidence,
or use title Evidence unless the formal controlled-title rule is satisfied. Every Fact must remain independently
useful with a complete subject, predicate, object, and necessary qualifiers. Preserve supported names, numbers,
dates, actor-action-object relations, and status. Evidence paragraph IDs must refer to supplied body paragraphs.
Return the complete revised annotation fields required by the schema, with no self-evaluation or commentary."""


REVIEWER_SYSTEMS = {
    "REVIEW_A": """You are isolated Semantic Reviewer A. Independently assess source and candidate.
You cannot see generator rationale/self-evaluation or any peer review. Do semantic judgment, not rule mapping.
Check every requested dimension. Unsupported facts are usually major or blocking; target mismatch is blocking;
missing critical facts and false entity relations are major; fragmentation can be minor or major. A high fact
count is a review signal, never an automatic warning-only acceptance. For every Fact separately check support,
Evidence, number/date, entity, entity relation, atomicity, micro-fragmentation, duplicate containment,
unsupported inference, and coverage contribution. Return concrete issues and concise rationale.""",
    "REVIEW_B": """You are isolated Semantic Reviewer B. Independently assess source and candidate from scratch.
You cannot see Reviewer A's verdict, rationale, or issues, and cannot see generator self-evaluation.
Look especially for subtle relation errors, unsupported inference, missing critical facts, duplicate containment,
and over-fragmentation. For every Fact separately check support, Evidence, number/date, entity, entity relation,
atomicity, micro-fragmentation, duplicate containment, unsupported inference, and coverage contribution.
Apply blocking/major/minor/warning severities by semantic impact. Return only structured output.""",
    "RE_REVIEWER": """You are an isolated Semantic Re-Reviewer. Verify the revised candidate against the source
and supplied previously confirmed issues. You cannot see any revision-agent self-evaluation. Re-run all semantic
checks and state whether each substantive defect remains. Return only structured output.""",
    "PILOT_QUALITY_AUDITOR": """You are the independent Pilot Quality Auditor. You receive no A/B verdicts,
rationales, or issue lists. Audit the candidate directly against the source for fact granularity/support/evidence,
entity relations, unsupported inference, missing facts, coverage, and policy. Be strict and return only structured output.""",
    "GOLD_REPLAY_REVIEWER": """You are the calibrated Gold regression Semantic Reviewer (calibration v2). Independently review
the pre-revision candidate against its source. You do not receive or read historical labels. Detect and classify
substantive defects with blocking/major/minor/warning severity. Use these canonical issue_type names whenever
applicable: fact_not_atomic, fact_micro_fragmentation, fact_fragmented, fact_duplicate, unsupported_inference,
entity_relation_error, fact_unsupported, missing_critical_fact, semantic_coverage_gap,
policy_compatibility_needs_resolution, title_body_evidence_gap. Genuine first-person target/constraint conflict
without a valid sample-scoped target-only exception is blocking. A title-only fact/evidence gap that makes the
immutable target unsupported is blocking. Unsupported facts, false entity relations, missing critical facts,
and material coverage gaps are major unless they make the whole sample unusable.
Perform this checklist before choosing a verdict: compare every Fact with its cited paragraphs; identify Facts that
combine multiple independently useful propositions; identify subjectless/micro fragments; compare Facts pairwise
for duplicate or containment; verify actor-action-object relations; list critical source propositions absent from
Facts; compare title-only semantics with body support; and inspect real grammatical first-person target language
against constraints and policy scope. A polished writing style is not evidence that the annotation is correct.
When any blocking/major defect exists, verdict cannot be accepted or accepted_with_warning. Every failing check
must have a canonical Issue. Return only structured output.""",
}


BLIND_REVIEWER_PROMPTS = {
    "task011d-e2-blind-reviewer-v3": """You are an isolated blind semantic reviewer for Chinese-news SFT annotations.
You receive only the source, candidate, Evidence mappings, and general policy. You have no historical labels,
expected verdict, revision outcome, or other review. Review from scratch and return only the required JSON.

For every Fact, explicitly assess: (A) Fact Support, (B) Evidence correctness and completeness,
(C) Number/Date, (D) Entity, (E) Entity Relation, (F) Atomicity, (G) Micro Fragmentation,
(H) Duplicate or containment, (I) Unsupported Inference, and (J) Coverage contribution.

Entity Relation is independent from entity presence. Even when every entity string occurs in the source, report
an entity_relation_error when the candidate changes who performs an action, the action object, cooperation parties,
product/platform ownership, the object affected by a technology, the subject of a project or plan, or the
published/built/planned/completed state. Also detect when parallel source relations become a false causal relation.
A material false actor-action-object or ownership relation is normally major.

Compare every Fact with all cited Evidence and with the full source. Build a whole-source proposition inventory
before deciding missing_critical_fact or semantic_coverage_gap. Atomic does not mean shortest: do not flag a
self-contained proposition merely for length, shared qualifiers, or a complete multi-object list. Do not treat
substrings such as 我国 or 自我 as grammatical first person. When blocking/major defects exist, the verdict cannot
be accepted or accepted_with_warning. Every failing check must have a concrete Issue.""",
    "task011d-e2-blind-reviewer-v4": """You are an isolated blind semantic reviewer for Chinese-news SFT annotations.
Use two passes and return only the required JSON. Pass 1: create a claim-to-Evidence matrix for every Fact and test
Fact Support, Evidence, Number/Date, Entity, Entity Relation, Atomicity, Micro Fragmentation, Duplicate/Containment,
Unsupported Inference, and Coverage contribution. Pass 2: inventory material source propositions and reconcile
them against the candidate to find omissions, false joins, false ownership, state changes, and over-splitting.

Entity presence does not prove relation correctness. Independently verify actor, action, object, cooperation party,
product/platform owner, technology target, project/plan subject, and published/built/planned/completed status.
Parallel source relations must not become causation. Material relation changes are normally major.
Atomicity requires independently useful propositions, not the shortest possible strings. Do not activate first-person
policy for lexical substrings such as 我国 or 自我. You have no historical labels, expected answer, revision result,
or peer review. Any blocking/major defect requires a non-accept verdict and a concrete Issue.""",
    "task011d-e2-blind-reviewer-v5": """You are an isolated blind semantic reviewer. Return only the required JSON.
Audit each Fact against both its cited Evidence and the complete source, then audit the complete source against all
Facts. For each Fact decide support, Evidence, numbers/dates, entities, actor-action-object relations, ownership,
cooperation, status/tense, atomicity, micro-fragmentation, duplicate containment, unsupported inference, and unique
coverage value. Explicitly test counterfactual readings: would the Fact incorrectly attribute another product's,
party's, project's, or technology's action or capability to this subject? Entity strings alone never establish a
correct relation. Material relation errors, unsupported facts, and critical omissions are normally major.

Do not equate detail with non-atomicity, and do not equate short text with fragmentation. Do not treat 我国, 自我,
or product-name substrings as grammatical first person. You receive no historical issue labels, expected severity,
expected verdict, revision result, or prior reviewer output. Blocking/major defects prohibit acceptance.""",
}


JUDGE_SYSTEM = """You are the isolated Semantic Judge. You may read A/B structured verdicts and issues but no
hidden reasoning. Resolve only material disagreements by checking the source and candidate. Confirm the defensible
issues and require revision or quarantine where appropriate. Return only structured output."""


class OpenAIResponsesSemanticModelProvider(SemanticModelProvider):
    provider_id = "openai_responses_api"

    def __init__(
        self,
        *,
        model: str,
        invocation_log: Path,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = "https://api.example.invalid/v1/responses",
        reasoning_effort: str = "medium",
        timeout_seconds: int = 180,
        max_retries: int = 2,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SemanticProviderError(f"required environment variable is absent: {api_key_env}")
        self._api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = InvocationLogger(invocation_log)

    def _invoke(
        self,
        *,
        role: str,
        sample_id: str,
        split: str,
        prompt_version: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        invocation_id = str(uuid.uuid4())
        started_at = _timestamp()
        request_payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _canonical_json(user_payload)},
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }
        input_hash = _sha256({"system": system_prompt, "payload": user_payload, "schema": schema})
        retry_count = 0
        response: dict[str, Any] | None = None
        last_error: Exception | None = None
        while retry_count <= self.max_retries:
            request = urllib.request.Request(
                self.endpoint,
                data=_canonical_json(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or retry_count >= self.max_retries:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if retry_count >= self.max_retries:
                    break
            retry_count += 1
            time.sleep(min(8, 2 ** retry_count))
        if response is None:
            self.logger.append(
                {
                    "invocation_id": invocation_id,
                    "role": role,
                    "sample_id": sample_id,
                    "split": split,
                    "model_provider_identifier": self.provider_id,
                    "model": self.model,
                    "prompt_version": prompt_version,
                    "input_hash": input_hash,
                    "output_hash": None,
                    "started_at": started_at,
                    "finished_at": _timestamp(),
                    "status": "failed",
                    "retry_count": retry_count,
                    "inference_type": "semantic_model_inference",
                    "error_type": type(last_error).__name__ if last_error else "UnknownError",
                }
            )
            if isinstance(last_error, urllib.error.HTTPError):
                raise SemanticProviderError(f"semantic provider HTTP failure: {last_error.code}") from last_error
            raise SemanticProviderError("semantic provider invocation failed") from last_error
        try:
            output = json.loads(_extract_output_text(response))
        except (json.JSONDecodeError, SemanticProviderError) as exc:
            self.logger.append(
                {
                    "invocation_id": invocation_id,
                    "role": role,
                    "sample_id": sample_id,
                    "split": split,
                    "model_provider_identifier": self.provider_id,
                    "model": response.get("model", self.model),
                    "prompt_version": prompt_version,
                    "input_hash": input_hash,
                    "output_hash": None,
                    "started_at": started_at,
                    "finished_at": _timestamp(),
                    "status": "failed_invalid_output",
                    "retry_count": retry_count,
                    "inference_type": "semantic_model_inference",
                    "response_id": response.get("id"),
                }
            )
            raise SemanticProviderError("semantic provider returned invalid structured output") from exc
        self.logger.append(
            {
                "invocation_id": invocation_id,
                "role": role,
                "sample_id": sample_id,
                "split": split,
                "model_provider_identifier": self.provider_id,
                "model": response.get("model", self.model),
                "prompt_version": prompt_version,
                "input_hash": input_hash,
                "output_hash": _sha256(output),
                "started_at": started_at,
                "finished_at": _timestamp(),
                "status": "succeeded",
                "retry_count": retry_count,
                "inference_type": "semantic_model_inference",
                "response_id": response.get("id"),
                "usage": response.get("usage", {}),
            }
        )
        return output

    def generate_sft(self, *, sample_id: str, split: str, source: dict[str, Any]) -> dict[str, Any]:
        return self._invoke(
            role="AI_SFT_GENERATOR",
            sample_id=sample_id,
            split=split,
            prompt_version="task011d-e2-generator-v1",
            system_prompt=GENERATOR_SYSTEM,
            user_payload={"source": source},
            schema_name="semantic_sft_generation",
            schema=GENERATION_SCHEMA,
        )

    def revise_sft(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        issues: list[dict[str, Any]],
        revision_round: int,
    ) -> dict[str, Any]:
        return self._invoke(
            role="AI_REVISION_AGENT",
            sample_id=sample_id,
            split=split,
            prompt_version=f"task011d-e2-revision-v{revision_round}",
            system_prompt=REVISION_SYSTEM,
            user_payload={"source": source, "candidate": candidate, "confirmed_issues": issues},
            schema_name="semantic_sft_revision",
            schema=GENERATION_SCHEMA,
        )

    def review_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviewer: str,
    ) -> dict[str, Any]:
        if reviewer not in REVIEWER_SYSTEMS:
            raise SemanticProviderError(f"unsupported reviewer role: {reviewer}")
        return self._invoke(
            role=reviewer,
            sample_id=sample_id,
            split=split,
            prompt_version=f"task011d-e2-{reviewer.lower()}-v2",
            system_prompt=REVIEWER_SYSTEMS[reviewer],
            user_payload={"source": source, "candidate": candidate},
            schema_name="semantic_sft_review",
            schema=REVIEW_SCHEMA,
        )

    def re_review_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._invoke(
            role="RE_REVIEWER",
            sample_id=sample_id,
            split=split,
            prompt_version="task011d-e2-re-reviewer-v1",
            system_prompt=REVIEWER_SYSTEMS["RE_REVIEWER"],
            user_payload={"source": source, "candidate": candidate, "confirmed_issues": issues},
            schema_name="semantic_sft_re_review",
            schema=REVIEW_SCHEMA,
        )

    def judge_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviews: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._invoke(
            role="AI_SFT_JUDGE",
            sample_id=sample_id,
            split=split,
            prompt_version="task011d-e2-judge-v1",
            system_prompt=JUDGE_SYSTEM,
            user_payload={"source": source, "candidate": candidate, "structured_reviews": reviews},
            schema_name="semantic_sft_judge",
            schema=JUDGE_SCHEMA,
        )


class CodexExecSemanticProvider(OpenAIResponsesSemanticModelProvider):
    """Isolated non-interactive Codex CLI semantic provider."""

    provider_id = "codex_exec"

    def __init__(
        self,
        *,
        model: str,
        invocation_log: Path,
        codex_cli_path: str,
        reasoning_effort: str = "high",
        reviewer_prompt_version: str = "task011d-e2-blind-reviewer-v3",
        cli_version: str = "unknown",
        authentication_status: str = "unknown",
        timeout_seconds: int = 600,
        parent_task: str = "TASK-011D-E2",
    ):
        if (
            reviewer_prompt_version not in BLIND_REVIEWER_PROMPTS
            and reviewer_prompt_version != "task011d-e2-dimension-reviewer-v1"
            and reviewer_prompt_version not in SAMPLE_LEVEL_REVIEWER_PROMPTS
        ):
            raise SemanticProviderError(f"unsupported blind reviewer prompt: {reviewer_prompt_version}")
        cli_path = Path(codex_cli_path)
        if not cli_path.is_file():
            raise SemanticProviderError(f"Codex CLI is unavailable: {cli_path}")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reviewer_prompt_version = reviewer_prompt_version
        self.codex_cli_path = cli_path
        self.cli_version = cli_version
        self.authentication_status = authentication_status
        self.timeout_seconds = timeout_seconds
        self.parent_task = parent_task
        self.max_retries = 0
        self.logger = InvocationLogger(invocation_log)

    def _safe_command_summary(self) -> list[str]:
        return [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "-C <isolated-temp-dir>",
            f"--model {self.model}",
            f'-c model_reasoning_effort="{self.reasoning_effort}"',
            "-c tools.web_search=false",
            '-c approval_policy="never"',
            "--sandbox read-only",
            "--strict-config",
            "--output-schema <temp-schema>",
            "--output-last-message <temp-output>",
            "--json",
            "-",
        ]

    def _invoke(
        self,
        *,
        role: str,
        sample_id: str,
        split: str,
        prompt_version: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        del schema_name
        fallback_invocation_id = str(uuid.uuid4())
        started_at = _timestamp()
        started_monotonic = time.monotonic()
        prompt = (
            "You are a pure isolated semantic JSON worker. Do not call or use tools. Do not inspect files, "
            "repositories, environment variables, network resources, web/search, prior conversations, or other "
            "samples. Do not modify any file. Use only the policy and task JSON below.\n\n"
            f"GENERAL REVIEW POLICY:\n{system_prompt}\n\n"
            f"TASK JSON:\n{_canonical_json(user_payload)}\n\n"
            "Return only the JSON object required by the supplied output schema."
        )
        input_hash = _sha256({"system": system_prompt, "payload": user_payload, "schema": schema})
        output: dict[str, Any] | None = None
        exit_code: int | None = None
        parse_status = "not_attempted"
        structured_output_status = "not_attempted"
        tool_call_count = 0
        invocation_id = fallback_invocation_id
        error_type: str | None = None
        provider_failure_class: str | None = None
        event_usage: dict[str, Any] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="task011d-codexexec-") as temp_value:
                temp_root = Path(temp_value)
                schema_path = temp_root / "output-schema.json"
                output_path = temp_root / "last-message.json"
                schema_path.write_text(_canonical_json(schema), encoding="utf-8", newline="")
                args = [
                    str(self.codex_cli_path),
                    "exec",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--skip-git-repo-check",
                    "-C",
                    str(temp_root),
                    "-m",
                    self.model,
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-c",
                    "tools.web_search=false",
                    "-c",
                    'approval_policy="never"',
                    "-s",
                    "read-only",
                    "--strict-config",
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    "--json",
                    "-",
                ]
                completed = subprocess.run(
                    args,
                    input=prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                exit_code = completed.returncode
                events: list[dict[str, Any]] = []
                for line in completed.stdout.splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(event)
                    if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                        invocation_id = event["thread_id"]
                    if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                        event_usage = event["usage"]
                    if event.get("type") in {"item.started", "item.completed"}:
                        item_type = (event.get("item") or {}).get("type")
                        if item_type in {
                            "command_execution",
                            "mcp_tool_call",
                            "web_search",
                            "file_search",
                            "computer_tool_call",
                            "tool_call",
                        }:
                            tool_call_count += 1
                if exit_code != 0:
                    error_type = "CodexExecNonZeroExit"
                    safe_error_text = (completed.stdout + "\n" + completed.stderr).lower()
                    if "rate limit" in safe_error_text or "rate_limit" in safe_error_text or "429" in safe_error_text:
                        provider_failure_class = "explicit_rate_limit"
                    elif "quota" in safe_error_text or "credit exhausted" in safe_error_text or "credits exhausted" in safe_error_text:
                        provider_failure_class = "quota"
                    elif "capacity" in safe_error_text or "overloaded" in safe_error_text:
                        provider_failure_class = "explicit_capacity"
                    elif "authentication" in safe_error_text or "unauthorized" in safe_error_text or "login required" in safe_error_text:
                        provider_failure_class = "authentication"
                    else:
                        provider_failure_class = "child_process_error"
                    raise SemanticProviderError(f"Codex CLI semantic invocation failed: {provider_failure_class}")
                if tool_call_count:
                    error_type = "ChildToolUseDetected"
                    provider_failure_class = "child_process_error"
                    raise SemanticProviderError("Codex semantic child used a prohibited tool")
                if not output_path.is_file():
                    error_type = "MissingStructuredOutput"
                    provider_failure_class = "parse"
                    raise SemanticProviderError("Codex CLI did not write the structured output")
                try:
                    parsed = json.loads(output_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    parse_status = "failed"
                    error_type = "InvalidJSON"
                    provider_failure_class = "parse"
                    raise SemanticProviderError("Codex CLI returned invalid structured JSON") from exc
                if not isinstance(parsed, dict):
                    parse_status = "failed"
                    error_type = "InvalidStructuredOutputType"
                    provider_failure_class = "parse"
                    raise SemanticProviderError("Codex CLI structured output was not an object")
                output = parsed
                parse_status = "passed"
                structured_output_status = "passed"
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            error_type = "TimeoutExpired"
            provider_failure_class = "timeout"
            raise SemanticProviderError("Codex CLI semantic invocation timed out") from exc
        finally:
            finished_at = _timestamp()
            record = {
                "invocation_id": invocation_id,
                "parent_task": self.parent_task,
                "semantic_role": role,
                "role": role,
                "sample_ids": [sample_id],
                "sample_id": sample_id,
                "split": split,
                "model_provider_identifier": self.provider_id,
                "model_requested": self.model,
                "model_effective": None,
                "effective_model_attestation": "not_reported_by_current_cli",
                "reasoning_effort_requested": self.reasoning_effort,
                "reasoning_effort_effective": None,
                "effective_effort_attestation": "not_reported_by_current_cli",
                "prompt_version": prompt_version,
                "input_sha256": input_hash,
                "input_hash": input_hash,
                "output_sha256": _sha256(output) if output is not None else None,
                "output_hash": _sha256(output) if output is not None else None,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(time.monotonic() - started_monotonic, 6),
                "exit_code": exit_code,
                "retry_count": 0,
                "parse_status": parse_status,
                "structured_output_status": structured_output_status,
                "semantic_status": "succeeded" if output is not None else "failed",
                "status": "succeeded" if output is not None else "failed",
                "inference_type": "semantic_model_inference",
                "session_mode": "ephemeral_independent",
                "conversation_reused": False,
                "sandbox_mode": "read-only",
                "tool_call_count": tool_call_count,
                "web_search_enabled": False,
                "fallback_used": False,
                "cli_version": self.cli_version,
                "authentication_status": self.authentication_status,
                "safe_command_arguments": self._safe_command_summary(),
                "usage": event_usage,
            }
            if error_type:
                record["error_type"] = error_type
            if provider_failure_class:
                record["provider_failure_class"] = provider_failure_class
            self.logger.append(record)
        if output is None:
            raise SemanticProviderError("Codex CLI semantic invocation produced no result")
        return output

    def review_candidate(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviewer: str,
    ) -> dict[str, Any]:
        if reviewer not in {"GOLD_REPLAY_REVIEWER", "FALSE_POSITIVE_REVIEWER", "REVIEW_A", "REVIEW_B"}:
            raise SemanticProviderError(f"unsupported CodexExec reviewer role: {reviewer}")
        system_prompt = BLIND_REVIEWER_PROMPTS[self.reviewer_prompt_version]
        if reviewer == "REVIEW_A":
            system_prompt += "\nAct as Reviewer A. Do not assume or seek any Reviewer B result."
        elif reviewer == "REVIEW_B":
            system_prompt += "\nAct as Reviewer B. Do not assume or seek any Reviewer A result."
        return self._invoke(
            role=reviewer,
            sample_id=sample_id,
            split=split,
            prompt_version=f"{self.reviewer_prompt_version}-{reviewer.lower()}",
            system_prompt=system_prompt,
            user_payload={"source": source, "candidate": candidate},
            schema_name="semantic_sft_review",
            schema=REVIEW_SCHEMA,
        )

    def review_dimensions(
        self,
        *,
        sample_id: str,
        split: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        reviewer: str = "DIMENSION_REVIEWER",
    ) -> dict[str, Any]:
        if self.reviewer_prompt_version == "task011d-e2-dimension-reviewer-v1":
            system_prompt = DIMENSION_REVIEWER_SYSTEM
        elif self.reviewer_prompt_version in SAMPLE_LEVEL_REVIEWER_PROMPTS:
            system_prompt = SAMPLE_LEVEL_REVIEWER_PROMPTS[self.reviewer_prompt_version]
        else:
            raise SemanticProviderError("CodexExec dimension review requires a dimension-first reviewer prompt")
        return self._invoke(
            role=reviewer,
            sample_id=sample_id,
            split=split,
            prompt_version=self.reviewer_prompt_version,
            system_prompt=system_prompt,
            user_payload={"source": source, "candidate": candidate},
            schema_name="semantic_sft_dimension_review",
            schema=DIMENSION_REVIEW_SCHEMA,
        )

    def review_duplicate_governance(
        self,
        *,
        fixture_key: str,
        governance_input: dict[str, Any],
        reviewer: str,
    ) -> dict[str, Any]:
        if reviewer not in {
            "DUPLICATE_GOVERNANCE_REVIEWER_A",
            "DUPLICATE_GOVERNANCE_REVIEWER_B",
            "DUPLICATE_GOVERNANCE_REVIEWER_C",
        }:
            raise SemanticProviderError(f"unsupported duplicate governance reviewer role: {reviewer}")
        return self._invoke(
            role=reviewer,
            sample_id=fixture_key,
            split="hidden_from_governance_reviewer",
            prompt_version="task011d-e2-duplicate-governance-reviewer-v1",
            system_prompt=DUPLICATE_GOVERNANCE_REVIEWER_SYSTEM,
            user_payload=governance_input,
            schema_name="duplicate_governance_review",
            schema=DUPLICATE_GOVERNANCE_REVIEW_SCHEMA,
        )

    def judge_duplicate_governance(
        self,
        *,
        fixture_key: str,
        governance_input: dict[str, Any],
        reviewer_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._invoke(
            role="DUPLICATE_GOVERNANCE_JUDGE",
            sample_id=fixture_key,
            split="hidden_from_governance_judge",
            prompt_version="task011d-e2-duplicate-governance-judge-v1",
            system_prompt=DUPLICATE_GOVERNANCE_JUDGE_SYSTEM,
            user_payload={"governance_input": governance_input, "structured_reviewer_results": reviewer_results},
            schema_name="duplicate_governance_judge",
            schema=DUPLICATE_GOVERNANCE_JUDGE_SCHEMA,
        )

    def review_event_pair(self, *, pair_id: str, article_a: dict[str, Any], article_b: dict[str, Any]) -> dict[str, Any]:
        """Run one blind, isolated same-event review without split/group metadata."""
        return self._invoke(
            role="AI_EVENT_GROUP_AUDITOR",
            sample_id=pair_id,
            split="hidden_from_reviewer",
            prompt_version="task011d-d-audit-event-reviewer-v1",
            system_prompt=EVENT_PAIR_REVIEWER_SYSTEM,
            user_payload={"article_a": article_a, "article_b": article_b},
            schema_name="event_pair_review",
            schema=EVENT_PAIR_REVIEW_SCHEMA,
        )

    def judge_event_pair(self, *, pair_id: str, article_a: dict[str, Any], article_b: dict[str, Any]) -> dict[str, Any]:
        """Run an independent judge; the first review is deliberately not part of the payload."""
        return self._invoke(
            role="AI_EVENT_GROUP_JUDGE",
            sample_id=pair_id,
            split="hidden_from_judge",
            prompt_version="task011d-d-audit-event-judge-v1",
            system_prompt=EVENT_PAIR_JUDGE_SYSTEM,
            user_payload={"article_a": article_a, "article_b": article_b},
            schema_name="event_pair_judge",
            schema=EVENT_PAIR_JUDGE_SCHEMA,
        )

    def judge_event_pair_with_review(
        self,
        *,
        pair_id: str,
        article_a: dict[str, Any],
        article_b: dict[str, Any],
        reviewer_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Judge an uncertain repair pair from sources plus the structured reviewer output only."""
        return self._invoke(
            role="AI_EVENT_GROUP_JUDGE",
            sample_id=pair_id,
            split="hidden_from_judge",
            prompt_version="task011d-d-repair-event-judge-v1",
            system_prompt=EVENT_PAIR_REPAIR_JUDGE_SYSTEM,
            user_payload={
                "article_a": article_a,
                "article_b": article_b,
                "structured_reviewer_result": reviewer_result,
            },
            schema_name="event_pair_judge",
            schema=EVENT_PAIR_JUDGE_SCHEMA,
        )


class OllamaSemanticModelProvider(OpenAIResponsesSemanticModelProvider):
    """Local Ollama semantic provider using the native structured chat API."""

    provider_id = "ollama_local_chat_api"

    def __init__(
        self,
        *,
        model: str,
        invocation_log: Path,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: int = 900,
        max_retries: int = 1,
    ):
        self.model = model
        self.endpoint = endpoint
        self.reasoning_effort = "model_default_thinking_disabled_for_structured_output"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = InvocationLogger(invocation_log)

    def _invoke(
        self,
        *,
        role: str,
        sample_id: str,
        split: str,
        prompt_version: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        del schema_name
        invocation_id = str(uuid.uuid4())
        started_at = _timestamp()
        grounded_user = {
            "task": user_payload,
            "required_output_json_schema": schema,
        }
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _canonical_json(grounded_user)},
            ],
            "stream": False,
            "think": role != "GENERATOR",
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 32768},
            "keep_alive": "30m",
        }
        input_hash = _sha256({"system": system_prompt, "payload": grounded_user})
        retry_count = 0
        response: dict[str, Any] | None = None
        last_error: Exception | None = None
        while retry_count <= self.max_retries:
            request = urllib.request.Request(
                self.endpoint,
                data=_canonical_json(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if retry_count >= self.max_retries:
                    break
                retry_count += 1
                time.sleep(2 ** retry_count)
        base_record = {
            "invocation_id": invocation_id,
            "role": role,
            "sample_id": sample_id,
            "split": split,
            "model_provider_identifier": self.provider_id,
            "model": self.model,
            "prompt_version": prompt_version,
            "input_hash": input_hash,
            "started_at": started_at,
            "finished_at": _timestamp(),
            "retry_count": retry_count,
            "inference_type": "semantic_model_inference",
        }
        if response is None:
            self.logger.append(
                base_record
                | {
                    "output_hash": None,
                    "status": "failed",
                    "error_type": type(last_error).__name__ if last_error else "UnknownError",
                }
            )
            raise SemanticProviderError("local Ollama semantic provider invocation failed") from last_error
        try:
            output = json.loads(response["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            self.logger.append(base_record | {"output_hash": None, "status": "failed_invalid_output"})
            raise SemanticProviderError("local Ollama provider returned invalid structured output") from exc
        self.logger.append(
            base_record
            | {
                "output_hash": _sha256(output),
                "status": "succeeded",
                "provider_created_at": response.get("created_at"),
                "usage": {
                    "input_tokens": response.get("prompt_eval_count", 0),
                    "output_tokens": response.get("eval_count", 0),
                    "total_duration_ns": response.get("total_duration", 0),
                },
            }
        )
        return output


def create_semantic_provider(config: dict[str, Any], *, root: Path) -> SemanticModelProvider:
    common = {
        "model": config["semantic_model"],
        "invocation_log": root / config["invocation_log"],
    }
    if config["semantic_provider"] == "ollama_local_chat_api":
        return OllamaSemanticModelProvider(**common)
    if config["semantic_provider"] == "openai_responses_api":
        return OpenAIResponsesSemanticModelProvider(
            **common,
            reasoning_effort=config.get("reasoning_effort", "medium"),
        )
    if config["semantic_provider"] == "codex_exec":
        return CodexExecSemanticProvider(
            **common,
            codex_cli_path=config["codex_cli_path"],
            reasoning_effort=config.get("reasoning_effort", "high"),
            reviewer_prompt_version=config.get("reviewer_prompt_version", "task011d-e2-blind-reviewer-v3"),
            cli_version=config.get("codex_cli_version", "unknown"),
            authentication_status=config.get("authentication_status", "unknown"),
            timeout_seconds=int(config.get("semantic_timeout_seconds", 600)),
            parent_task=config.get("task_id", "TASK-011D-E2"),
        )
    raise SemanticProviderError(f"unsupported semantic provider: {config['semantic_provider']}")
