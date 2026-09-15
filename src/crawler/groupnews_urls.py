from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import ErrorCode, GroupNewsError

APPROVED_DOMAIN = "www.10086.cn"
DETAIL_PATH_RE = re.compile(
    r"^/aboutus/news/groupnews/index_detail_(?P<article_id>[0-9]+)\.html$"
)
CONTENT_FILENAME_PREFIX = "5018449_5585_11769_detail_"
LISTING_PATH = "/aboutus/news/groupnews/"


def normalize_page_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise GroupNewsError(ErrorCode.INVALID_URL, "URL cannot be parsed") from exc
    if parts.scheme not in {"http", "https"}:
        raise GroupNewsError(ErrorCode.INVALID_URL, "Only HTTP(S) URLs are supported")
    if parts.hostname != APPROVED_DOMAIN:
        raise GroupNewsError(ErrorCode.UNSUPPORTED_DOMAIN, "Domain is not approved")
    if parts.port not in {None, 80, 443}:
        raise GroupNewsError(ErrorCode.INVALID_URL, "Non-standard ports are not approved")
    if parts.query:
        raise GroupNewsError(ErrorCode.INVALID_URL, "Detail-page query strings are not approved")
    return urlunsplit((parts.scheme.lower(), APPROVED_DOMAIN, parts.path, "", ""))


def extract_article_id(url: str, approved_urls: set[str]) -> str:
    normalized, article_id = validate_discovered_detail_url(url)
    approved_normalized = {normalize_page_url(item) for item in approved_urls}
    if normalized not in approved_normalized:
        raise GroupNewsError(ErrorCode.INVALID_URL, "URL is not present in the approved sample CSV")
    return article_id


def validate_discovered_detail_url(url: str) -> tuple[str, str]:
    """Validate a URL emitted by an approved listing response without deriving an ID."""
    normalized = normalize_page_url(url)
    match = DETAIL_PATH_RE.fullmatch(urlsplit(normalized).path)
    if not match:
        path = urlsplit(normalized).path
        if path.endswith("index_detail_.html"):
            raise GroupNewsError(ErrorCode.MISSING_ARTICLE_ID, "Article ID is missing")
        raise GroupNewsError(ErrorCode.UNSUPPORTED_PATH, "Path is not an approved group-news detail path")
    return normalized, match.group("article_id")


def validate_listing_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname != APPROVED_DOMAIN:
        raise GroupNewsError(ErrorCode.UNSUPPORTED_DOMAIN, "Listing URL must use the approved domain")
    if parts.port not in {None, 80, 443} or not parts.path.startswith(LISTING_PATH):
        raise GroupNewsError(ErrorCode.UNSUPPORTED_PATH, "Listing URL is outside the approved group-news path")
    return urlunsplit((parts.scheme.lower(), APPROVED_DOMAIN, parts.path, parts.query, ""))


def build_content_json_url(
    page_url: str,
    approved_urls: set[str],
    cachebuster_ms: int | None = None,
) -> str:
    article_id = extract_article_id(page_url, approved_urls)
    base = (
        f"https://{APPROVED_DOMAIN}/aboutus/news/groupnews/"
        f"{CONTENT_FILENAME_PREFIX}{article_id}.json"
    )
    if cachebuster_ms is None:
        return base
    return f"{base}?{urlencode({'nowtime': str(cachebuster_ms)})}"


def normalize_content_url(url: str) -> str:
    parts = urlsplit(url)
    filtered = [(key, value) for key, value in parse_qsl(parts.query) if key != "nowtime"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(filtered), ""))
