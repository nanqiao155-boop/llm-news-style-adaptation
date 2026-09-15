from __future__ import annotations

import re

try:
    from .schemas import Draft
except ImportError:
    from schemas import Draft


class LoRAOutputParseError(ValueError):
    """Raised when plain LoRA output cannot be split into a title and body."""


_TITLE_PREFIX = re.compile(r"^标题\s*[:：]\s*")
_BODY_PREFIX = re.compile(r"^正文\s*[:：]\s*")


def _remove_outer_markdown(value: str) -> str:
    text = value.strip()
    while len(text) >= 4 and (
        (text.startswith("**") and text.endswith("**"))
        or (text.startswith("__") and text.endswith("__"))
    ):
        text = text[2:-2].strip()
    return re.sub(r"^#{1,6}\s+", "", text).strip()


def _remove_code_fence(lines: list[str]) -> list[str]:
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    last = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()), None)
    if first is not None and last is not None and lines[first].strip().startswith("```"):
        lines = lines[:first] + lines[first + 1 :]
        last = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()), None)
        if last is not None and lines[last].strip() == "```":
            lines = lines[:last] + lines[last + 1 :]
    return lines


def parse_lora_writer_output(raw_output: str, usage: dict[str, int] | None = None) -> Draft:
    """Parse presentation structure without rewriting any factual content."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise LoRAOutputParseError("LoRA Writer returned empty output")

    lines = _remove_code_fence(raw_output.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        raise LoRAOutputParseError("LoRA Writer returned empty output")

    title_line = _remove_outer_markdown(lines[first])
    title = _remove_outer_markdown(_TITLE_PREFIX.sub("", title_line, count=1))

    body_lines = lines[first + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    if body_lines:
        first_body = _remove_outer_markdown(body_lines[0])
        if _BODY_PREFIX.match(first_body):
            body_lines[0] = _BODY_PREFIX.sub("", first_body, count=1)
    body = "\n".join(body_lines).strip()

    if not title or not body:
        raise LoRAOutputParseError("LoRA Writer output does not contain both title and body")
    return Draft(title=title, body=body, raw_output=raw_output, usage=dict(usage or {}))
