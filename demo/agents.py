from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

try:
    from .api_client import TextCompletion
    from .lora_parser import LoRAOutputParseError, parse_lora_writer_output
    from .schemas import (
        Draft,
        JudgeResult,
        ReviewResult,
        WritingRequest,
        parse_draft,
        parse_judge,
        parse_review,
    )
except ImportError:
    from api_client import TextCompletion
    from lora_parser import LoRAOutputParseError, parse_lora_writer_output
    from schemas import Draft, JudgeResult, ReviewResult, WritingRequest, parse_draft, parse_judge, parse_review


T = TypeVar("T")


class JSONClient(Protocol):
    def request_json(
        self,
        messages: Sequence[Mapping[str, str]],
        parser: Callable[[Any], T],
        schema_name: str,
    ) -> T: ...


class TextClient(Protocol):
    def request_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> TextCompletion: ...


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


class OnlineQwenWriter:
    def __init__(self, client: JSONClient):
        self.client = client

    def run(self, request: WritingRequest) -> Draft:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Online Qwen Writer，负责根据已确认材料撰写正式企业新闻通稿。"
                    "不得补充材料未提供的事实，不得推断效果、价值、行业地位或未来结果；"
                    "必须保留关键机构、人物、地点、时间和数字；机构简称首次使用前须规范定义；"
                    "标题应凝练并体现新闻事件，避免‘情况介绍’‘全面展现’等空泛表达。"
                    "只输出 JSON：{\"title\":\"...\",\"body\":\"...\"}，不要输出分析过程。"
                ),
            },
            {"role": "user", "content": _json(request.as_prompt_payload())},
        ]
        return self.client.request_json(messages, parse_draft, "Writer")


# Backward-compatible name for the retained, non-default Phase 2A emergency implementation.
WriterAgent = OnlineQwenWriter


class RepairedLoRAWriter:
    def __init__(self, client: TextClient):
        self.client = client
        self.last_raw_outputs: tuple[str, ...] = ()

    @staticmethod
    def _messages(request: WritingRequest) -> list[dict[str, str]]:
        system = (
            "你是 Repaired LoRA Writer。依据给定事实材料撰写正式、客观、严谨、简洁的企业新闻通稿。"
            "不得补充材料中未提供的事实、效果、行业地位、经济价值或社会效益；"
            "保留机构、人物、时间、地点、数字和后续安排；"
            "首次出现机构全称且后文需要简称时，规范定义简称；标题应具有新闻事件导向；不要重复生成。"
            "第一行只输出标题，空一行后输出正文；不要输出分析、说明、Markdown 或 JSON。"
        )
        payload = request.as_prompt_payload()
        user = (
            f"新闻主题：{payload['topic']}\n"
            f"新闻类别：{payload['category']}\n"
            f"已确认事实材料：\n{payload['confirmed_facts']}\n"
            f"可选结构要求：{payload['optional_outline'] or '无'}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def run(self, request: WritingRequest) -> Draft:
        messages = self._messages(request)
        raw_outputs: list[str] = []
        for attempt in range(2):
            completion = self.client.request_text(messages, max_tokens=2048, enable_thinking=False)
            raw_outputs.append(completion.content)
            try:
                draft = parse_lora_writer_output(completion.content, dict(completion.usage))
            except LoRAOutputParseError:
                if attempt == 1:
                    self.last_raw_outputs = tuple(raw_outputs)
                    raise
                messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "上一次输出无法可靠拆分标题和正文。请基于完全相同的事实材料重新输出："
                            "第一行仅标题，空一行后仅正文。不要改变事实内容要求，不要输出标签、解释、Markdown 或 JSON。"
                        ),
                    },
                ]
                continue
            self.last_raw_outputs = tuple(raw_outputs)
            return draft
        raise AssertionError("unreachable")


class ReviewerAgent:
    def __init__(self, client: JSONClient):
        self.client = client

    def run(self, request: WritingRequest, draft: Draft) -> ReviewResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是独立 Reviewer。你只能依据用户的已确认事实材料和 Writer draft 审核，不存在也不得索取 reference answer。"
                    "检查事实依据、关键信息保留、标题新闻性、正式客观性、结构格式、简洁自然性、材料外扩写和重复。"
                    "issue type 只能是 factual_grounding、key_information_retention、title_quality、"
                    "formality_objectivity、structure_formatting、conciseness_naturalness、unsupported_additions、repetition；"
                    "severity 只能是 minor 或 major。必须逐项核对明确事实锚点：完整日期、具体数字、机构名称、地点、人物和后续安排"
                    "不得被遗漏或替换成更模糊的表达。若‘2026年8月28日’被写成‘近日’、‘12处’被写成‘多处’，"
                    "或明确机构全称被泛化，应标记 key_information_retention，通常为 minor；只有影响事件真实性或关键判断时才 major，"
                    "不能把语义大致一致视为完整保留。key_information_retention 专用于输入信息的缺失或模糊化；"
                    "候选新增任何输入未明确提供的词语、行动、结果、效果、计划、程度判断、行业判断、价值判断或宣传性结论，"
                    "即使听起来合理，也应标记 unsupported_additions。例如输入仅有‘持续开展网络质量监测’，候选新增"
                    "‘根据园区实际使用情况及时进行优化调整’，reason 必须说明‘该后续行动未由输入事实材料支持’，"
                    "revision_instruction 应要求删除该新增行动、保留原有监测安排。若输入为‘园区重点区域5G网络覆盖得到改善’，候选写成"
                    "‘有效改善了园区重点区域的5G网络覆盖情况’，‘有效’属于材料未支持的程度强化，只标记一个 unsupported_additions，"
                    "不得再对同一 evidence 同时标记 key_information_retention。生成 issues 前必须按实际问题和 evidence span 去重；"
                    "同一证据理论上可归入多个类别时，只选择最主要、最直接的一个 issue type，不得为了覆盖维度重复报告；"
                    "不同证据、不同问题仍应分别报告。只有所有八类检查均不存在需要修改的问题时才允许"
                    "pass=true 且 issues=[]；存在日期模糊化或未支持新增行动时必须 pass=false。不得为了展示闭环而故意判失败，"
                    "也不得因文章整体通顺而忽略具体问题。"
                    "只输出 JSON：{\"pass\":true,\"issues\":[{\"type\":\"...\",\"severity\":\"minor|major\","
                    "\"evidence\":\"...\",\"reason\":\"...\",\"revision_instruction\":\"...\"}],\"summary\":\"...\"}。"
                ),
            },
            {
                "role": "user",
                "content": _json({"writing_request": request.as_prompt_payload(), "writer_draft": draft.as_dict()}),
            },
        ]
        return self.client.request_json(messages, parse_review, "Reviewer")


class ReviserAgent:
    def __init__(self, client: JSONClient):
        self.client = client

    def run(self, request: WritingRequest, draft: Draft, review: ReviewResult) -> Draft:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Reviser。你只能看到用户事实材料、Writer draft 与 Reviewer issues/instructions，不存在 reference answer。"
                    "只修订 Reviewer 指出的问题；修复材料外扩写、标题、客观性、结构、冗余或重复，同时保留正确关键事实。"
                    "不得为了优化加入新的宣传性结论或任何材料外事实。"
                    "只输出 JSON：{\"title\":\"...\",\"body\":\"...\"}，不要输出解释。"
                ),
            },
            {
                "role": "user",
                "content": _json(
                    {
                        "writing_request": request.as_prompt_payload(),
                        "writer_draft": draft.as_dict(),
                        "reviewer_result": review.as_dict(),
                    }
                ),
            },
        ]
        return self.client.request_json(messages, parse_draft, "Reviser")


class EditorialJudgeAgent:
    def __init__(self, client: JSONClient):
        self.client = client

    def run(self, request: WritingRequest, final_draft: Draft) -> JudgeResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是冻结六维 Editorial Judge。只依据用户事实材料/写作任务和最终候选文本评分，不使用 reference answer。"
                    "六维满分固定为 factual_grounding 30、key_information_retention 15、title_quality 15、"
                    "formality_objectivity 15、structure_formatting 10、conciseness_naturalness 15。"
                    "key_information_retention 必须逐项检查完整日期、具体数字、主体、地点、人物和后续安排；"
                    "明确的‘2026年8月28日’被改成‘近日’时不得给满分 15，其他明确锚点被模糊化或遗漏也必须扣分。"
                    "factual_grounding 与 unsupported_claims 必须检查候选新增的行动、结果、效果、计划、行业判断、价值判断和宣传性结论。"
                    "例如输入仅写‘持续开展网络质量监测’，候选新增‘根据园区实际使用情况及时进行优化调整’，"
                    "该新增行动属于输入未明确支持的 unsupported claim 候选，不能因其听起来合理而忽略。"
                    "不得仅因文章通顺、正式、短小就给 100 分；满分要求所有明确事实准确保留、没有 unsupported addition，"
                    "且标题、文风、结构和简洁性均达到各维满分标准。识别 unsupported_claims 与 major_release_risks；"
                    "不要输出 total score，Python 将严格求和，Release-Adjusted 规则不由 Judge 改写。"
                    "只输出 JSON：{\"dimensions\":{六个固定键及数值},\"publishable\":true,"
                    "\"unsupported_claims\":[],\"major_release_risks\":[],\"rationale\":\"...\"}。"
                ),
            },
            {
                "role": "user",
                "content": _json(
                    {"writing_request": request.as_prompt_payload(), "final_candidate": final_draft.as_dict()}
                ),
            },
        ]
        return self.client.request_json(messages, parse_judge, "Editorial Judge")
