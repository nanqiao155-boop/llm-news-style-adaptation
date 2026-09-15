from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable


NUMBER_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:%|％|万|亿|兆|个|项|家|次|年|月|日|天|小时|分钟|秒|公里|米|元|G|GB|TB|P|bps)?",
    re.IGNORECASE,
)
QUOTED_PATTERN = re.compile(r"[“\"《]([^”\"》]{2,40})[”\"》]")
LATIN_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+._-]{1,24}(?![A-Za-z0-9])")
ORG_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·]{2,24}(?:公司|集团|研究院|实验室|中心|委员会|协会|大学|政府|研究所|联盟)"
)
EVENT_VERBS = {
    "发布", "签约", "启动", "开幕", "召开", "完成", "保障", "成立", "上线", "投产",
    "获批", "中标", "揭牌", "交付", "发射", "入轨", "合作", "举行", "举办", "闭幕",
}
GENERIC_ENTITIES = {
    "企业", "企业集团", "企业通信集团公司", "有限公司", "集团公司",
    "相关部门", "有关单位", "大会", "会议", "活动", "项目", "平台", "系统",
}
PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
        "（": "(", "）": ")", "【": "[", "】": "]", "、": ",", "—": "-",
        "－": "-", "～": "~", "“": '"', "”": '"', "‘": "'", "’": "'",
    }
)


def normalize_event_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").translate(PUNCT_TRANSLATION).lower()
    normalized = re.sub(r"[\u200b\ufeff]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def compact_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalize_event_text(value))


def char_ngrams(value: str, minimum: int, maximum: int) -> list[str]:
    text = compact_text(value)
    result: list[str] = []
    for size in range(minimum, maximum + 1):
        result.extend(text[index : index + size] for index in range(max(0, len(text) - size + 1)))
    return result


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def extract_numbers(value: str) -> set[str]:
    return {item.lower() for item in NUMBER_PATTERN.findall(normalize_event_text(value))}


def extract_event_verbs(value: str) -> set[str]:
    normalized = normalize_event_text(value)
    return {verb for verb in EVENT_VERBS if verb in normalized}


def extract_entities(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value or "")
    values = {item.strip() for item in QUOTED_PATTERN.findall(normalized)}
    values.update(item.strip() for item in LATIN_PATTERN.findall(normalized))
    values.update(item.strip() for item in ORG_PATTERN.findall(normalized))
    return {
        normalize_event_text(item)
        for item in values
        if 2 <= len(compact_text(item)) <= 40 and normalize_event_text(item) not in GENERIC_ENTITIES
    }


def extract_keyphrases(article: dict[str, Any]) -> set[str]:
    values = set(extract_entities(article["title_normalized"]))
    values.update(normalize_event_text(item) for item in article.get("visual_subheadings", []) if 3 <= len(compact_text(item)) <= 30)
    values.update(normalize_event_text(item) for item in QUOTED_PATTERN.findall(article["body_text"]) if 3 <= len(compact_text(item)) <= 30)
    return {item for item in values if item and item not in GENERIC_ENTITIES}


def parse_publish_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def date_gap_days(left: str | None, right: str | None) -> int | None:
    left_date = parse_publish_date(left)
    right_date = parse_publish_date(right)
    if left_date is None or right_date is None:
        return None
    return abs((left_date - right_date).days)


def body_length_bucket(length: int) -> str:
    if length <= 600:
        return "short"
    if length <= 1200:
        return "medium"
    if length <= 2000:
        return "long"
    return "very_long"


@dataclass(frozen=True)
class TfidfResult:
    vectors: list[dict[str, float]]
    vocabulary_size: int


def build_tfidf_vectors(
    texts: list[str],
    ngram_range: tuple[int, int],
    minimum_document_frequency: int,
    maximum_document_frequency_ratio: float,
    max_features: int,
) -> TfidfResult:
    counters: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    collection_frequency: Counter[str] = Counter()
    for text in texts:
        counter = Counter(char_ngrams(text, *ngram_range))
        counters.append(counter)
        document_frequency.update(counter.keys())
        collection_frequency.update(counter)
    maximum_document_frequency = max(1, int(len(texts) * maximum_document_frequency_ratio))
    eligible = [
        term
        for term, frequency in document_frequency.items()
        if minimum_document_frequency <= frequency <= maximum_document_frequency
    ]
    eligible.sort(
        key=lambda term: (
            -(math.log((1 + len(texts)) / (1 + document_frequency[term])) + 1) * math.log1p(collection_frequency[term]),
            term,
        )
    )
    vocabulary = set(eligible[:max_features])
    vectors: list[dict[str, float]] = []
    for counter in counters:
        raw = {
            term: (1 + math.log(count)) * (math.log((1 + len(texts)) / (1 + document_frequency[term])) + 1)
            for term, count in counter.items()
            if term in vocabulary
        }
        norm = math.sqrt(sum(value * value for value in raw.values()))
        vectors.append({term: value / norm for term, value in raw.items()} if norm else {})
    return TfidfResult(vectors=vectors, vocabulary_size=len(vocabulary))


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def stable_intersection(values: Iterable[str]) -> list[str]:
    return sorted(set(values))
