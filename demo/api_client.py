from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Mapping, Sequence, TypeVar
from urllib.parse import urlparse

try:
    from .config import APISettings, WriterAPISettings
    from .schemas import SchemaValidationError
except ImportError:
    from config import APISettings, WriterAPISettings
    from schemas import SchemaValidationError


T = TypeVar("T")
Transport = Callable[[str, Mapping[str, str], bytes, float], Mapping[str, Any]]


class APIClientError(RuntimeError):
    """Base class for safe, user-displayable API errors."""


class APIConfigurationError(APIClientError):
    pass


class APINetworkError(APIClientError):
    pass


class APIResponseError(APIClientError):
    pass


class SchemaParseError(APIClientError):
    pass


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> Mapping[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        raise APIConfigurationError("缺少 openai SDK，请安装 demo/requirements.txt。") from None
    payload = json.loads(body.decode("utf-8"))
    base_url = url.rsplit("/chat/completions", 1)[0]
    api_key = headers.get("Authorization", "").removeprefix("Bearer ")
    enable_thinking = payload.pop("enable_thinking", None)
    if enable_thinking is not None:
        payload["extra_body"] = {"enable_thinking": enable_thinking}
    try:
        response = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout).chat.completions.create(**payload)
        return response.model_dump()
    except Exception:
        # Never surface SDK exceptions because they may include request metadata.
        raise APINetworkError("API 网络连接或请求超时，请稍后重试。") from None


@dataclass(frozen=True)
class TextCompletion:
    content: str
    usage: Mapping[str, int]


class OpenAICompatibleClient:
    """Shared OpenAI-compatible client with JSON and plain-text completion modes."""

    def __init__(self, settings: APISettings | WriterAPISettings, transport: Transport | None = None):
        if not settings.is_configured:
            raise APIConfigurationError("API 未配置，请设置三个 LLM_JUDGE 环境变量。")
        parsed = urlparse(settings.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise APIConfigurationError("LLM_JUDGE_API_BASE 不是有效的 HTTP(S) 地址。")
        self._settings = settings
        self._transport = transport or _default_transport
        base = settings.api_base.rstrip("/")
        self._endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    @property
    def model(self) -> str:
        return self._settings.model

    def _completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Mapping[str, str] | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> TextCompletion:
        payload = {
            "model": self._settings.model,
            "messages": list(messages),
            "temperature": 0,
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._transport(
                self._endpoint,
                headers,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                self._settings.timeout_seconds,
            )
        except APIClientError:
            raise
        except Exception:
            # Never surface arbitrary transport text: it may contain headers or secrets.
            raise APINetworkError("API 网络连接或请求超时，请稍后重试。") from None
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise APIResponseError("API 返回缺少结构化内容。") from None
        if not isinstance(content, str) or not content.strip():
            raise APIResponseError("API 返回内容为空。")
        raw_usage = response.get("usage", {})
        usage = {
            str(key): int(value)
            for key, value in raw_usage.items()
            if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        } if isinstance(raw_usage, Mapping) else {}
        return TextCompletion(content=content, usage=usage)

    def request_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> TextCompletion:
        return self._completion(
            messages,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    def request_json(
        self,
        messages: Sequence[Mapping[str, str]],
        parser: Callable[[Any], T],
        schema_name: str,
    ) -> T:
        """Request and validate JSON, with exactly one schema-only retry."""
        retry_messages = list(messages)
        for attempt in range(2):
            completion = self._completion(retry_messages, response_format={"type": "json_object"})
            try:
                payload = json.loads(completion.content)
            except JSONDecodeError:
                if attempt == 1:
                    raise SchemaParseError(f"{schema_name} 连续两次返回非有效 JSON，流程已停止。") from None
            else:
                try:
                    return parser(payload)
                except SchemaValidationError:
                    if attempt == 1:
                        raise
            retry_messages = [
                *messages,
                {
                    "role": "system",
                    "content": f"上一次 {schema_name} 输出未通过严格 JSON schema 校验。请仅返回符合既定 schema 的 JSON 对象，不要输出解释或 Markdown。",
                },
            ]
        raise AssertionError("unreachable")
