from __future__ import annotations

import json
import re
from datetime import date, datetime

from bs4 import BeautifulSoup, NavigableString, Tag

from .models import ErrorCode, GroupNewsError, ParsedArticle

SEMANTIC_BLOCK_TAGS = {"p", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6"}
HTML_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {"script", "style", "nav", "footer", "noscript"}
MEDIA_TAGS = {"video", "source", "track"}
SKIP_MARKERS = {"recommend", "related", "footer", "navigation"}
VISUAL_SUBHEADING_DETECTION_VERSION = "visual-subheading-v1"
VIDEO_FALLBACK_PATTERN = re.compile(
    r"(?:您的\s*)?浏览器\s*不支持\s*video\s*标签\s*[。.!！]?",
    flags=re.IGNORECASE,
)


def normalize_for_comparison(value: str) -> str:
    """Normalize Unicode whitespace without changing visible text or punctuation."""
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def normalize_visible_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with(" ")
    return normalize_for_comparison(soup.get_text(" ", strip=True))


def count_video_fallback_occurrences(value: str) -> int:
    """Count controlled video fallback phrases without matching ordinary browser prose."""
    return len(VIDEO_FALLBACK_PATTERN.findall(value))


def normalize_publish_date(value: object, warnings: list[str]) -> tuple[str | None, str | None]:
    if value is None or not isinstance(value, str) or not value.strip():
        warnings.append(ErrorCode.MISSING_PUBLISH_TIME)
        return None, None
    raw = value.strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(raw).date()
        except ValueError:
            warnings.append(ErrorCode.MISSING_PUBLISH_TIME)
            return raw, None
    return raw, parsed.isoformat()


def detect_page_template(page_html: bytes) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    if soup.select_one(".newsArea") or "aboutusnewsdetail.js" in page_html.decode("utf-8", "ignore"):
        return "new"
    if soup.select_one("#newsbody, .newsbody") or "newsfunction_detail.js" in page_html.decode("utf-8", "ignore"):
        return "old"
    return "unknown"


def _is_hidden_or_noncontent(tag: Tag) -> bool:
    if tag.name is None:
        return True
    attrs = tag.attrs or {}
    if tag.name in SKIP_TAGS or tag.name in MEDIA_TAGS or "hidden" in attrs:
        return True
    if str(attrs.get("aria-hidden", "")).lower() == "true":
        return True
    style = str(attrs.get("style", "")).replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    markers = " ".join([str(attrs.get("id", "")), *[str(value) for value in attrs.get("class", [])]]).lower()
    return any(marker in markers for marker in SKIP_MARKERS)


def _is_standalone_span(tag: Tag) -> bool:
    if tag.name != "span":
        return False
    ancestor = tag.parent
    while isinstance(ancestor, Tag) and ancestor.name not in {"[document]", "html", "body"}:
        if ancestor.name in SEMANTIC_BLOCK_TAGS or ancestor.name == "span":
            return False
        ancestor = ancestor.parent
    return True


def _extract_dom_text_blocks(soup: BeautifulSoup) -> list[str]:
    """Walk DOM boundaries in order and emit non-overlapping visible text blocks."""
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        normalized = normalize_for_comparison(" ".join(buffer))
        buffer.clear()
        if normalized and (not blocks or blocks[-1] != normalized):
            blocks.append(normalized)

    def visit(node: Tag | NavigableString) -> None:
        if isinstance(node, NavigableString):
            text = normalize_for_comparison(str(node))
            if text:
                buffer.append(text)
            return
        if not isinstance(node, Tag) or _is_hidden_or_noncontent(node):
            return
        if node.name == "br":
            flush()
            return
        if node.name == "img":
            return
        is_boundary = node.name in SEMANTIC_BLOCK_TAGS or _is_standalone_span(node)
        if is_boundary:
            flush()
        for child in node.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child)
        if is_boundary:
            flush()

    for child in soup.children:
        if isinstance(child, (Tag, NavigableString)):
            visit(child)
    flush()
    return blocks


def ensure_paragraph_completeness(
    body_text: str,
    paragraphs: list[str],
    fallback_paragraphs: list[str],
) -> tuple[list[str], list[str], bool]:
    expected = normalize_for_comparison(body_text)
    actual = normalize_for_comparison("\n".join(paragraphs))
    if actual == expected:
        return paragraphs, [], True
    fallback_actual = normalize_for_comparison("\n".join(fallback_paragraphs))
    if fallback_actual == expected:
        return fallback_paragraphs, [ErrorCode.PARAGRAPHS_RECOVERED_WITH_FALLBACK], True
    return paragraphs, [ErrorCode.PARAGRAPH_STRUCTURE_INCOMPLETE], False


def _meaningful_sibling(tag: Tag, direction: str) -> Tag | NavigableString | None:
    sibling = tag.previous_sibling if direction == "previous" else tag.next_sibling
    while sibling is not None:
        if isinstance(sibling, NavigableString) and not normalize_for_comparison(str(sibling)):
            sibling = sibling.previous_sibling if direction == "previous" else sibling.next_sibling
            continue
        return sibling if isinstance(sibling, (Tag, NavigableString)) else None
    return None


def _is_isolated_emphasis(tag: Tag) -> bool:
    if tag.name not in {"strong", "b"} or tag.find_parent(["strong", "b"]):
        return False
    text = normalize_visible_text(str(tag))
    if not 2 <= len(text) <= 50:
        return False
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if parent and parent.name in SEMANTIC_BLOCK_TAGS:
        return normalize_visible_text(str(parent)) == text
    previous = _meaningful_sibling(tag, "previous")
    following = _meaningful_sibling(tag, "next")
    previous_boundary = previous is None or (isinstance(previous, Tag) and previous.name in {"br", *SEMANTIC_BLOCK_TAGS})
    following_boundary = following is None or (isinstance(following, Tag) and following.name in {"br", *SEMANTIC_BLOCK_TAGS})
    return previous_boundary and following_boundary


def _extract_visual_subheadings(soup: BeautifulSoup) -> list[str]:
    candidates: list[str] = []
    for tag in soup.find_all([*HTML_HEADING_TAGS, "strong", "b"]):
        if _is_hidden_or_noncontent(tag):
            continue
        if tag.name not in HTML_HEADING_TAGS and not _is_isolated_emphasis(tag):
            continue
        text = normalize_visible_text(str(tag))
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _remove_video_fallbacks(soup: BeautifulSoup) -> int:
    """Remove media DOM and standalone fallback-only text nodes from a parsed fragment."""
    removed_count = 0
    for video in list(soup.find_all("video")):
        removed_count += count_video_fallback_occurrences(video.get_text(" ", strip=True))
        video.decompose()
    for media in list(soup.find_all(["source", "track"])):
        media.decompose()
    for node in list(soup.find_all(string=True)):
        raw = str(node)
        matches = list(VIDEO_FALLBACK_PATTERN.finditer(raw))
        if not matches:
            continue
        remainder = VIDEO_FALLBACK_PATTERN.sub("", raw)
        if normalize_for_comparison(remainder):
            continue
        removed_count += len(matches)
        node.extract()
    return removed_count


def _extract_body(body_html: str) -> dict[str, object]:
    soup = BeautifulSoup(body_html, "html.parser")
    video_fallback_removed_count = _remove_video_fallbacks(soup)
    for unwanted in list(soup.find_all(True)):
        if unwanted.name is not None and _is_hidden_or_noncontent(unwanted):
            unwanted.decompose()
    image_urls = [str(img.get("src")) for img in soup.find_all("img") if img.get("src")]
    heading_count = len(soup.find_all([*HTML_HEADING_TAGS]))
    visual_subheadings = _extract_visual_subheadings(soup)
    fallback_paragraphs = [
        normalize_for_comparison(line)
        for line in soup.get_text("\n", strip=True).splitlines()
        if normalize_for_comparison(line)
    ]
    body_text = "\n".join(fallback_paragraphs)
    semantic_paragraphs = _extract_dom_text_blocks(soup)
    paragraphs, warnings, consistent = ensure_paragraph_completeness(
        body_text,
        semantic_paragraphs,
        fallback_paragraphs,
    )
    return {
        "body_html": str(soup),
        "body_text": body_text,
        "paragraphs": paragraphs,
        "paragraph_content_consistent": consistent,
        "paragraphs_text_length_chars": len(normalize_for_comparison("\n".join(paragraphs))),
        "body_comparison_length_chars": len(normalize_for_comparison(body_text)),
        "image_urls": image_urls,
        "image_count": len(soup.find_all("img")),
        "heading_count": heading_count,
        "visual_subheadings": visual_subheadings,
        "video_fallback_removed_count": video_fallback_removed_count,
        "warnings": warnings,
    }


def parse_groupnews_article(
    *,
    payload_bytes: bytes,
    page_html_bytes: bytes,
    document_id: str,
    page_snapshot_id: str,
    payload_snapshot_id: str,
    source_url: str,
    content_api_url: str,
    parsed_at: str,
) -> ParsedArticle:
    try:
        root = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroupNewsError(ErrorCode.INVALID_JSON, "Payload is not valid UTF-8 JSON") from exc
    if not isinstance(root, dict):
        raise GroupNewsError(ErrorCode.INVALID_JSON, "JSON root must be an object")
    cdata = root.get("cData")
    if not isinstance(cdata, dict):
        raise GroupNewsError(ErrorCode.MISSING_CDATA, "cData object is required")
    content = cdata.get("content")
    if not isinstance(content, dict):
        raise GroupNewsError(ErrorCode.MISSING_CONTENT_OBJECT, "cData.content object is required")
    title = content.get("mainTitle")
    if not isinstance(title, str) or not title.strip():
        raise GroupNewsError(ErrorCode.MISSING_TITLE, "mainTitle must be a non-empty string")
    body_html = content.get("mainBody")
    if not isinstance(body_html, str) or not body_html.strip():
        raise GroupNewsError(ErrorCode.MISSING_BODY, "mainBody must be a non-empty string")

    warnings: list[str] = []
    publish_time_raw, publish_date = normalize_publish_date(content.get("publishTime"), warnings)
    source_value = content.get("source")
    source = normalize_visible_text(source_value) if isinstance(source_value, str) and source_value.strip() else None
    if source is None:
        warnings.append(ErrorCode.MISSING_SOURCE)
    body = _extract_body(body_html)
    warnings.extend(body["warnings"])
    if int(body["video_fallback_removed_count"]) > 0:
        warnings.append(ErrorCode.VIDEO_FALLBACK_REMOVED)
    paragraphs = list(body["paragraphs"])
    visual_subheadings = list(body["visual_subheadings"])
    body_text = str(body["body_text"])
    image_urls = list(body["image_urls"])
    return ParsedArticle(
        document_id=document_id,
        page_snapshot_id=page_snapshot_id,
        payload_snapshot_id=payload_snapshot_id,
        source_url=source_url,
        content_api_url=content_api_url,
        page_template=detect_page_template(page_html_bytes),
        title_raw=title,
        title_normalized=normalize_visible_text(title),
        publish_time_raw=publish_time_raw,
        publish_date=publish_date,
        source=source,
        category="集团新闻",
        category_derivation="url_path",
        body_html=str(body["body_html"]),
        body_text=body_text,
        paragraphs=paragraphs,
        paragraph_count=len(paragraphs),
        paragraph_content_consistent=bool(body["paragraph_content_consistent"]),
        paragraphs_text_length_chars=int(body["paragraphs_text_length_chars"]),
        body_comparison_length_chars=int(body["body_comparison_length_chars"]),
        body_length_chars=len(body_text),
        image_count=int(body["image_count"]),
        image_urls=image_urls,
        heading_count=int(body["heading_count"]),
        visual_subheadings=visual_subheadings,
        visual_subheading_count=len(visual_subheadings),
        visual_subheading_detection_version=VISUAL_SUBHEADING_DETECTION_VERSION,
        video_fallback_removed_count=int(body["video_fallback_removed_count"]),
        parsed_at=parsed_at,
        parse_warnings=warnings,
    )
