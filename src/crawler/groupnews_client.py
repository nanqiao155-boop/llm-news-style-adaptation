from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

import requests

from .groupnews_urls import APPROVED_DOMAIN
from .models import ErrorCode, GroupNewsError, HttpAttempt


class StopValidation(GroupNewsError):
    """Fatal condition that stops the whole fixed-sample validation run."""


class GroupNewsClient:
    def __init__(
        self,
        *,
        user_agent: str,
        minimum_interval_seconds: float = 5.0,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        maximum_retries: int = 2,
        maximum_requests: int = 25,
    ) -> None:
        if minimum_interval_seconds < 5.0:
            raise ValueError("minimum_interval_seconds must be at least 5")
        if not 0 <= maximum_retries <= 2:
            raise ValueError("maximum_retries must be between 0 and 2")
        self.minimum_interval_seconds = minimum_interval_seconds
        self.timeout = (connect_timeout_seconds, read_timeout_seconds)
        self.maximum_retries = maximum_retries
        self.maximum_requests = maximum_requests
        self.request_events: list[dict[str, object]] = []
        self._last_started_monotonic: float | None = None
        self._consecutive_5xx = 0
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"User-Agent": user_agent})

    def close(self) -> None:
        self._session.cookies.clear()
        self._session.close()

    def _wait_for_slot(self) -> float | None:
        if len(self.request_events) >= self.maximum_requests:
            raise StopValidation(ErrorCode.HTTP_ERROR, "Request budget exhausted")
        now = time.monotonic()
        if self._last_started_monotonic is not None:
            remaining = self.minimum_interval_seconds - (now - self._last_started_monotonic)
            if remaining > 0:
                time.sleep(remaining)
        started = time.monotonic()
        gap = None if self._last_started_monotonic is None else started - self._last_started_monotonic
        self._last_started_monotonic = started
        return gap

    def fetch(
        self,
        url: str,
        expected_resource: Literal["page_html", "content_json", "listing_html", "listing_json"],
        on_attempt: Callable[[HttpAttempt], None] | None = None,
    ) -> HttpAttempt:
        for attempt_number in range(1, self.maximum_retries + 2):
            gap = self._wait_for_slot()
            started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self._session.cookies.clear()
            try:
                response = self._session.get(url, timeout=self.timeout, allow_redirects=True)
                body = response.content
                content_type = response.headers.get("Content-Type")
                raw_length = response.headers.get("Content-Length")
                content_length = int(raw_length) if raw_length and raw_length.isdigit() else None
                result = HttpAttempt(
                    requested_url=url,
                    final_url=response.url,
                    started_at=started_at,
                    start_gap_seconds=gap,
                    status_code=response.status_code,
                    content_type=content_type,
                    content_length=content_length,
                    body=body,
                    redirected=response.url != url,
                    attempt_number=attempt_number,
                )
            except requests.RequestException as exc:
                result = HttpAttempt(
                    requested_url=url,
                    final_url=None,
                    started_at=started_at,
                    start_gap_seconds=gap,
                    status_code=None,
                    content_type=None,
                    content_length=None,
                    body=b"",
                    redirected=False,
                    attempt_number=attempt_number,
                    error_type=type(exc).__name__,
                )
            finally:
                self._session.cookies.clear()

            self.request_events.append(
                {
                    "started_at": started_at,
                    "gap_seconds": gap,
                    "url": url,
                    "status": result.status_code,
                    "attempt_number": attempt_number,
                }
            )
            if on_attempt is not None:
                on_attempt(result)

            if result.error_type:
                if attempt_number <= self.maximum_retries:
                    time.sleep(min(2**attempt_number, 4))
                    continue
                raise GroupNewsError(ErrorCode.HTTP_ERROR, "Network request failed")

            if result.final_url and urlsplit(result.final_url).hostname != APPROVED_DOMAIN:
                raise StopValidation(ErrorCode.UNSUPPORTED_DOMAIN, "Redirect left the approved domain")
            if result.status_code == 429:
                raise StopValidation(ErrorCode.HTTP_ERROR, "HTTP 429; validation stopped")
            if result.status_code and 500 <= result.status_code <= 599:
                self._consecutive_5xx += 1
                if self._consecutive_5xx >= 2:
                    raise StopValidation(ErrorCode.HTTP_ERROR, "Consecutive 5xx responses; validation stopped")
                if attempt_number <= self.maximum_retries:
                    time.sleep(min(2**attempt_number, 4))
                    continue
            else:
                self._consecutive_5xx = 0
            if result.status_code != 200:
                raise GroupNewsError(ErrorCode.HTTP_ERROR, f"Unexpected HTTP status {result.status_code}")

            media_type = (result.content_type or "").split(";", 1)[0].strip().lower()
            valid_type = media_type == "text/html" if expected_resource in {"page_html", "listing_html"} else (
                media_type == "application/json" or media_type.endswith("+json")
            )
            if not valid_type:
                raise StopValidation(
                    ErrorCode.UNSUPPORTED_CONTENT_TYPE,
                    f"Unexpected Content-Type for {expected_resource}",
                )
            if expected_resource in {"page_html", "listing_html"}:
                probe = result.body[:8192].decode("utf-8", errors="ignore").lower()
                if any(marker in probe for marker in ("captcha", "验证码", "access denied")):
                    raise StopValidation(ErrorCode.HTTP_ERROR, "Access-control page detected")
            return result
        raise AssertionError("unreachable")
