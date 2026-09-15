from __future__ import annotations

from typing import Any

from src.dataset.sft_schema import SYSTEM_PROMPT, USER_PROMPT_VERSION


def render_user_prompt(
    topic: str,
    fact_points: list[dict[str, Any]],
    outline: list[str],
    constraints: dict[str, Any],
) -> str:
    facts = "\n".join(f"{index}. {item['fact']}" for index, item in enumerate(fact_points, 1))
    outline_text = "\n".join(f"{index}. {item}" for index, item in enumerate(outline, 1))
    return (
        f"主题：\n{topic}\n\n"
        f"事实要点：\n{facts}\n\n"
        f"建议大纲：\n{outline_text}\n\n"
        "写作要求：\n"
        "- 使用正式、客观、严谨、简洁、结构化的企业新闻通稿风格；\n"
        "- 严格依据上述事实，不得新增或修改人名、机构、时间、地点、数字、合作关系、成果和结论；\n"
        "- 保留给定的关键实体与数字；\n"
        "- 避免第一人称、口语化、夸张营销表达和Markdown格式；\n"
        f"- 篇幅类型：{constraints['length_bucket']}；\n"
        "- 输出完整标题与正文。\n\n"
        "输出格式：\n标题：新闻标题\n\n正文：新闻正文"
    )


def render_messages(user_prompt: str, target_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_text},
    ]


def prompt_version() -> str:
    return USER_PROMPT_VERSION
