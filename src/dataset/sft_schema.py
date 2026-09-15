from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sft-candidate-v1.0.0"
PILOT_VERSION = "task011a-pilot-v1"
PILOT_REVISION_VERSION = "task011a-pilot-v1.1"
PILOT_VERSIONS = {PILOT_VERSION, PILOT_REVISION_VERSION}
SYSTEM_PROMPT_VERSION = "enterprise-news-system-v1"
USER_PROMPT_VERSION = "enterprise-news-user-v1"
ANNOTATION_PROMPT_VERSION = "task011a-v1"

SYSTEM_PROMPT = (
    "你是一名企业新闻通稿撰写助手。请仅依据用户提供的主题、事实要点、大纲和写作要求生成新闻标题与正文。"
    "不得虚构、补充或修改未提供的人名、机构、时间、地点、数字、合作关系、成果和结论；信息不足时应保持克制，"
    "不得自行编造。语言应正式、客观、严谨、简洁、结构清晰，避免第一人称、口语化表达、夸张营销措辞、"
    "Markdown格式以及未经事实支持的评价。"
)

STYLE = ["正式", "客观", "严谨", "简洁", "结构化"]
EXAGGERATION_TERMS = ("重磅", "震撼", "划时代", "彻底改变")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|％)?")
DATE_PATTERNS = (
    re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
    re.compile(r"\d{4}年\d{1,2}月"),
    re.compile(r"\d{4}年"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def normalize_evidence(paragraphs: list[str]) -> str:
    return "\n".join(re.sub(r"\s+", " ", item).strip() for item in paragraphs)


def body_length_bucket(length: int) -> str:
    if length <= 600:
        return "short"
    if length <= 1200:
        return "medium"
    if length <= 2000:
        return "long"
    return "very_long"


def extract_number_tokens(text: str) -> list[str]:
    return sorted(set(NUMBER_PATTERN.findall(text)))


def extract_date_tokens(text: str) -> list[str]:
    values: set[str] = set()
    for pattern in DATE_PATTERNS:
        values.update(pattern.findall(text))
    return sorted(values)


def render_target(title: str, body: str) -> str:
    return f"标题：{title}\n\n正文：{body}"


def fixed_constraints(length_bucket: str) -> dict[str, Any]:
    return {
        "style": STYLE.copy(),
        "do_not_invent_facts": True,
        "preserve_names_dates_numbers": True,
        "preserve_given_entities": True,
        "avoid_first_person": True,
        "avoid_markdown": True,
        "avoid_exaggeration": True,
        "output_format": "title_and_body",
        "length_bucket": length_bucket,
    }


def source_paragraph_records(paragraphs: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "paragraph_id": index,
            "text": paragraph,
            "text_sha256": sha256_text(paragraph),
        }
        for index, paragraph in enumerate(paragraphs, 1)
    ]


REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "pilot_version",
    "sample_id",
    "source_document_id",
    "source_article_id",
    "source_dataset_version",
    "split",
    "construction_method",
    "annotation_provider",
    "annotation_model",
    "annotation_prompt_version",
    "created_at",
    "review_status",
    "reviewer_role",
    "quality_issues",
    "validation_warnings",
    "system_prompt",
    "system_prompt_version",
    "topic",
    "fact_points",
    "outline",
    "constraints",
    "user_prompt",
    "user_prompt_version",
    "messages_preview",
    "target_title",
    "target_body",
    "target_text",
    "target_title_sha256",
    "target_body_sha256",
    "target_text_sha256",
    "source_paragraphs",
    "fact_coverage",
    "evidence_summary",
    "source_body_sha256",
}


def validate_candidate_shape(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - candidate.keys())
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    if candidate.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if candidate.get("pilot_version") not in PILOT_VERSIONS:
        errors.append("pilot_version")
    if candidate.get("split") is not None:
        errors.append("split_must_be_null")
    if candidate.get("review_status") != "pending":
        errors.append("review_status_must_be_pending")
    if candidate.get("reviewer_role") is not None:
        errors.append("reviewer_role_must_be_null")
    if candidate.get("annotation_model") is not None:
        errors.append("annotation_model_must_be_null")
    if candidate.get("construction_method") != "codex_assisted_pilot":
        errors.append("construction_method")
    if candidate.get("annotation_provider") != "codex_local_agent":
        errors.append("annotation_provider")
    if candidate.get("system_prompt") != SYSTEM_PROMPT:
        errors.append("system_prompt")
    if candidate.get("system_prompt_version") != SYSTEM_PROMPT_VERSION:
        errors.append("system_prompt_version")
    if candidate.get("user_prompt_version") != USER_PROMPT_VERSION:
        errors.append("user_prompt_version")
    if not isinstance(candidate.get("fact_points"), list) or not candidate.get("fact_points"):
        errors.append("fact_points")
    if not isinstance(candidate.get("outline"), list) or not candidate.get("outline"):
        errors.append("outline")
    if not isinstance(candidate.get("source_paragraphs"), list) or not candidate.get("source_paragraphs"):
        errors.append("source_paragraphs")
    return errors


def validate_json_schema(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Validate the JSON-Schema subset used by sft-candidate-v1 without adding a dependency."""

    errors: list[str] = []

    def resolve(reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise ValueError(f"unsupported schema reference: {reference}")
        value: Any = schema
        for part in reference[2:].split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
        return value

    def matches_type(value: Any, expected: str) -> bool:
        checks = {
            "null": value is None,
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        }
        return checks[expected]

    def visit(value: Any, rule: dict[str, Any], path: str) -> None:
        if "$ref" in rule:
            visit(value, resolve(rule["$ref"]), path)
            return
        if "const" in rule and value != rule["const"]:
            errors.append(f"{path}:const")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{path}:enum")
        expected_type = rule.get("type")
        if expected_type and not matches_type(value, expected_type):
            errors.append(f"{path}:type")
            return
        if isinstance(value, dict):
            for name in rule.get("required", []):
                if name not in value:
                    errors.append(f"{path}.{name}:required")
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False:
                for name in value.keys() - properties.keys():
                    errors.append(f"{path}.{name}:additional")
            for name, child_rule in properties.items():
                if name in value:
                    visit(value[name], child_rule, f"{path}.{name}")
        elif isinstance(value, list):
            if len(value) < rule.get("minItems", 0):
                errors.append(f"{path}:minItems")
            if "maxItems" in rule and len(value) > rule["maxItems"]:
                errors.append(f"{path}:maxItems")
            if rule.get("uniqueItems"):
                rendered = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value]
                if len(rendered) != len(set(rendered)):
                    errors.append(f"{path}:uniqueItems")
            for index, child_rule in enumerate(rule.get("prefixItems", [])):
                if index < len(value):
                    visit(value[index], child_rule, f"{path}[{index}]")
            if "items" in rule:
                for index, item in enumerate(value):
                    visit(item, rule["items"], f"{path}[{index}]")
        elif isinstance(value, str):
            if len(value) < rule.get("minLength", 0):
                errors.append(f"{path}:minLength")
            if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
                errors.append(f"{path}:pattern")
            if rule.get("format") == "date-time":
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    errors.append(f"{path}:date-time")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                errors.append(f"{path}:minimum")
            if "maximum" in rule and value > rule["maximum"]:
                errors.append(f"{path}:maximum")

    visit(instance, schema, "$")
    return errors


def load_and_validate_json_schema(candidate: dict[str, Any], schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return validate_json_schema(candidate, schema)
