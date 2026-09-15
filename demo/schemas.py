from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    from .config import EDITORIAL_DIMENSIONS
except ImportError:
    from config import EDITORIAL_DIMENSIONS


ISSUE_TYPES = {
    "factual_grounding",
    "key_information_retention",
    "title_quality",
    "formality_objectivity",
    "structure_formatting",
    "conciseness_naturalness",
    "unsupported_additions",
    "repetition",
}
SEVERITIES = {"minor", "major"}


class SchemaValidationError(ValueError):
    """Raised when an API response does not match the frozen Phase 2A contract."""


class DimensionValidationError(SchemaValidationError):
    """Raised when an Editorial Judge dimension is missing or out of range."""


class InputValidationError(ValueError):
    """Raised for a user-correctable workbench input problem."""


def _mapping(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"{name} must be a JSON object")
    return payload


def _text(payload: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SchemaValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{key} must be a boolean")
    return value


def _string_list(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaValidationError(f"{key} must be a string list")
    return tuple(item.strip() for item in value if item.strip())


@dataclass(frozen=True)
class WritingRequest:
    topic: str
    category: str
    facts: str
    outline: str = ""

    def validate(self) -> None:
        if not self.topic.strip():
            raise InputValidationError("新闻主题不能为空。")
        if not self.facts.strip():
            raise InputValidationError("事实材料不能为空。")
        if len(self.facts.strip()) < 20:
            raise InputValidationError("事实材料过短，请补充可核验的机构、事件、时间、地点或数据后再运行。")

    def as_prompt_payload(self) -> dict[str, str]:
        return {
            "topic": self.topic.strip(),
            "category": self.category.strip(),
            "confirmed_facts": self.facts.strip(),
            "optional_outline": self.outline.strip(),
        }


@dataclass(frozen=True)
class Draft:
    title: str
    body: str
    raw_output: str | None = None
    usage: Mapping[str, int] | None = None

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "body": self.body}


@dataclass(frozen=True)
class ReviewIssue:
    type: str
    severity: str
    evidence: str
    reason: str
    revision_instruction: str

    def as_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "severity": self.severity,
            "evidence": self.evidence,
            "reason": self.reason,
            "revision_instruction": self.revision_instruction,
        }


@dataclass(frozen=True)
class ReviewResult:
    passed: bool
    issues: tuple[ReviewIssue, ...]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {"pass": self.passed, "issues": [item.as_dict() for item in self.issues], "summary": self.summary}


@dataclass(frozen=True)
class JudgeResult:
    dimensions: Mapping[str, float]
    publishable: bool
    unsupported_claims: tuple[str, ...]
    major_release_risks: tuple[str, ...]
    rationale: str

    @property
    def raw_total(self) -> float:
        return float(sum(self.dimensions.values()))


def parse_draft(payload: Any) -> Draft:
    obj = _mapping(payload, "draft")
    return Draft(title=_text(obj, "title"), body=_text(obj, "body"))


def parse_review(payload: Any) -> ReviewResult:
    obj = _mapping(payload, "review")
    raw_issues = obj.get("issues")
    if not isinstance(raw_issues, list):
        raise SchemaValidationError("issues must be a list")
    issues = []
    for raw_issue in raw_issues:
        issue = _mapping(raw_issue, "issue")
        issue_type = _text(issue, "type")
        severity = _text(issue, "severity")
        if issue_type not in ISSUE_TYPES:
            raise SchemaValidationError("unknown issue type")
        if severity not in SEVERITIES:
            raise SchemaValidationError("unknown issue severity")
        issues.append(
            ReviewIssue(
                type=issue_type,
                severity=severity,
                evidence=_text(issue, "evidence"),
                reason=_text(issue, "reason"),
                revision_instruction=_text(issue, "revision_instruction"),
            )
        )
    passed = _bool(obj, "pass")
    if passed and issues:
        raise SchemaValidationError("pass=true requires an empty issues list")
    if not passed and not issues:
        raise SchemaValidationError("pass=false requires at least one issue")
    return ReviewResult(passed=passed, issues=tuple(issues), summary=_text(obj, "summary"))


def parse_judge(payload: Any) -> JudgeResult:
    obj = _mapping(payload, "judge")
    raw_dimensions = _mapping(obj.get("dimensions"), "dimensions")
    if set(raw_dimensions) != set(EDITORIAL_DIMENSIONS):
        raise DimensionValidationError("Judge dimensions are incomplete or contain unknown keys")
    dimensions: dict[str, float] = {}
    for name, maximum in EDITORIAL_DIMENSIONS.items():
        value = raw_dimensions[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise DimensionValidationError(f"{name} must be an integer")
        numeric = float(value)
        if not 0 <= numeric <= maximum:
            raise DimensionValidationError(f"{name} must be between 0 and {maximum}")
        dimensions[name] = numeric
    return JudgeResult(
        dimensions=dimensions,
        publishable=_bool(obj, "publishable"),
        unsupported_claims=_string_list(obj, "unsupported_claims"),
        major_release_risks=_string_list(obj, "major_release_risks"),
        rationale=_text(obj, "rationale"),
    )
