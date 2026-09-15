from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

try:
    from .agents import EditorialJudgeAgent, RepairedLoRAWriter, ReviewerAgent, ReviserAgent
    from .api_client import (
        APIConfigurationError,
        APINetworkError,
        APIResponseError,
        SchemaParseError,
    )
    from .lora_parser import LoRAOutputParseError
    from .scoring import release_adjusted_score
    from .schemas import (
        DimensionValidationError,
        Draft,
        InputValidationError,
        JudgeResult,
        ReviewResult,
        SchemaValidationError,
        WritingRequest,
    )
except ImportError:
    from agents import EditorialJudgeAgent, RepairedLoRAWriter, ReviewerAgent, ReviserAgent
    from api_client import APIConfigurationError, APINetworkError, APIResponseError, SchemaParseError
    from lora_parser import LoRAOutputParseError
    from scoring import release_adjusted_score
    from schemas import (
        DimensionValidationError,
        Draft,
        InputValidationError,
        JudgeResult,
        ReviewResult,
        SchemaValidationError,
        WritingRequest,
    )


STAGES = ("material", "writer", "reviewer", "reviser", "quality")


@dataclass(frozen=True)
class WorkflowEvent:
    stage: str
    states: Mapping[str, str]
    progress: int
    status_text: str
    writer_draft: Draft | None = None
    review: ReviewResult | None = None
    final_draft: Draft | None = None
    judge: JudgeResult | None = None
    release_adjusted: float | None = None
    error: str | None = None


@dataclass
class _RunState:
    states: dict[str, str] = field(default_factory=lambda: {name: "pending" for name in STAGES})
    writer_draft: Draft | None = None
    review: ReviewResult | None = None
    final_draft: Draft | None = None
    judge: JudgeResult | None = None
    release_adjusted: float | None = None

    def event(self, stage: str, progress: int, status_text: str, error: str | None = None) -> WorkflowEvent:
        return WorkflowEvent(
            stage=stage,
            states=dict(self.states),
            progress=progress,
            status_text=status_text,
            writer_draft=self.writer_draft,
            review=self.review,
            final_draft=self.final_draft,
            judge=self.judge,
            release_adjusted=self.release_adjusted,
            error=error,
        )


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, InputValidationError):
        return str(exc)
    if isinstance(exc, APIConfigurationError):
        return "API 未配置，请在启动进程中设置 LLM_JUDGE_API_BASE、LLM_JUDGE_API_KEY 和 LLM_JUDGE_MODEL。"
    if isinstance(exc, APINetworkError):
        return "API 网络连接或请求超时，请检查网络后重试。"
    if isinstance(exc, APIResponseError):
        return "API 返回失败或响应结构不完整，请稍后重试。"
    if isinstance(exc, SchemaParseError):
        return "模型连续两次未返回有效 JSON，当前流程已安全停止。"
    if isinstance(exc, DimensionValidationError):
        return "Editorial Judge 六维分数缺失或超出固定范围，当前流程已安全停止。"
    if isinstance(exc, SchemaValidationError):
        return "模型结构化返回未通过 schema 校验，当前流程已安全停止。"
    return "工作流发生未预期错误，当前流程已安全停止。"


class EditorialWorkflow:
    def __init__(
        self,
        writer: RepairedLoRAWriter,
        reviewer: ReviewerAgent,
        reviser: ReviserAgent,
        judge: EditorialJudgeAgent,
    ):
        self.writer = writer
        self.reviewer = reviewer
        self.reviser = reviser
        self.judge = judge

    def run(self, request: WritingRequest) -> Iterator[WorkflowEvent]:
        run = _RunState()
        run.states["material"] = "active"
        yield run.event("material", 10, "正在校验新闻主题与已确认事实材料……")
        try:
            request.validate()
        except InputValidationError as exc:
            run.states["material"] = "failed"
            yield run.event("material", 10, "素材校验未通过。", _safe_error(exc))
            return

        run.states["material"] = "completed"
        run.states["writer"] = "active"
        yield run.event("writer", 30, "正在依据事实材料生成正式、客观的新闻初稿……")
        try:
            run.writer_draft = self.writer.run(request)
        except LoRAOutputParseError:
            run.states["writer"] = "failed"
            yield run.event(
                "writer",
                30,
                "Writer 阶段失败。",
                "Repaired LoRA Writer 连续两次返回无法解析的标题和正文，流程已停止。",
            )
            return
        except Exception:
            run.states["writer"] = "failed"
            yield run.event("writer", 30, "Writer 阶段失败。", "Repaired LoRA Writer 服务暂时不可用")
            return

        run.states["writer"] = "completed"
        run.states["reviewer"] = "active"
        yield run.event("reviewer", 55, "正在检查事实依据、关键信息、标题、文风、结构与重复问题……")
        try:
            run.review = self.reviewer.run(request, run.writer_draft)
        except Exception as exc:
            run.states["reviewer"] = "failed"
            yield run.event("reviewer", 55, "Reviewer 阶段失败。", _safe_error(exc))
            return

        run.states["reviewer"] = "completed"
        if run.review.passed:
            run.states["reviser"] = "skipped"
            run.final_draft = run.writer_draft
        else:
            run.states["reviser"] = "active"
            yield run.event("reviser", 80, "正在根据审核意见修订新闻初稿……")
            try:
                run.final_draft = self.reviser.run(request, run.writer_draft, run.review)
            except Exception as exc:
                run.states["reviser"] = "failed"
                yield run.event("reviser", 80, "Reviser 阶段失败。", _safe_error(exc))
                return
            run.states["reviser"] = "completed"

        run.states["quality"] = "active"
        yield run.event("quality", 100, "正在计算六维 Editorial Score 与发布风险校正分……")
        try:
            run.judge = self.judge.run(request, run.final_draft)
            run.release_adjusted = release_adjusted_score(
                raw_total=run.judge.raw_total,
                major_release_risks=len(run.judge.major_release_risks),
                publishable=run.judge.publishable,
                unsupported_claim_count=len(run.judge.unsupported_claims),
            )
        except Exception as exc:
            run.states["quality"] = "failed"
            yield run.event("quality", 100, "质量评测阶段失败。", _safe_error(exc))
            return

        run.states["quality"] = "completed"
        yield run.event("quality", 100, "写审改与质量评测已完成。")
