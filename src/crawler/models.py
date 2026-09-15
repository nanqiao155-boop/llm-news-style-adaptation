from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ErrorCode:
    INVALID_URL = "invalid_url"
    UNSUPPORTED_DOMAIN = "unsupported_domain"
    UNSUPPORTED_PATH = "unsupported_path"
    MISSING_ARTICLE_ID = "missing_article_id"
    HTTP_ERROR = "http_error"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    INVALID_JSON = "invalid_json"
    MISSING_CDATA = "missing_cdata"
    MISSING_CONTENT_OBJECT = "missing_content_object"
    MISSING_TITLE = "missing_title"
    MISSING_BODY = "missing_body"
    MISSING_PUBLISH_TIME = "missing_publish_time"
    MISSING_SOURCE = "missing_source"
    TITLE_MISMATCH = "title_mismatch"
    DATE_MISMATCH = "date_mismatch"
    PARAGRAPH_STRUCTURE_INCOMPLETE = "paragraph_structure_incomplete"
    PARAGRAPHS_RECOVERED_WITH_FALLBACK = "paragraphs_recovered_with_fallback"
    VIDEO_FALLBACK_REMOVED = "video_fallback_removed"
    LIST_DETAIL_TITLE_MISMATCH = "list_detail_title_mismatch"
    LIST_DETAIL_DATE_MISMATCH = "list_detail_date_mismatch"
    EMPTY_BODY = "empty_body"
    SHORT_BODY_CANDIDATE = "short_body_candidate"
    UNSUPPORTED_PAGE_TEMPLATE = "unsupported_page_template"
    DUPLICATE_EXACT = "duplicate_exact"
    DUPLICATE_NEAR = "duplicate_near"
    PARSE_FAILURE = "parse_failure"
    UNEXPECTED_CONTENT_TYPE = "unexpected_content_type"
    BLOCKED_OR_RATE_LIMITED = "blocked_or_rate_limited"


class GroupNewsError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class HttpAttempt:
    requested_url: str
    final_url: str | None
    started_at: str
    start_gap_seconds: float | None
    status_code: int | None
    content_type: str | None
    content_length: int | None
    body: bytes
    redirected: bool
    attempt_number: int
    error_type: str | None = None


@dataclass
class ParsedArticle:
    document_id: str
    page_snapshot_id: str
    payload_snapshot_id: str
    source_url: str
    content_api_url: str
    page_template: str
    title_raw: str
    title_normalized: str
    publish_time_raw: str | None
    publish_date: str | None
    source: str | None
    category: str
    category_derivation: str
    body_html: str
    body_text: str
    paragraphs: list[str]
    paragraph_count: int
    paragraph_content_consistent: bool
    paragraphs_text_length_chars: int
    body_comparison_length_chars: int
    body_length_chars: int
    image_count: int
    image_urls: list[str]
    heading_count: int
    visual_subheadings: list[str]
    visual_subheading_count: int
    visual_subheading_detection_version: str
    video_fallback_removed_count: int = 0
    source_level: str = "group"
    region: None = None
    document_genre: str = "news_release"
    training_eligible: None = None
    parser_version: str = "groupnews-parser-v3-video-fallback"
    parsed_at: str = ""
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
