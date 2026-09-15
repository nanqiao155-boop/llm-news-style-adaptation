from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .groupnews_parser import normalize_for_comparison


def normalized_body_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_comparison(text).encode("utf-8")).hexdigest()


def character_ngrams(text: str, size: int = 5) -> set[str]:
    normalized = normalize_for_comparison(text)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def ngram_jaccard(left: str, right: str, size: int = 5) -> float:
    left_set = character_ngrams(left, size)
    right_set = character_ngrams(right, size)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


@dataclass(frozen=True)
class DuplicateMatch:
    document_id: str
    other_document_id: str
    similarity: float
    kind: str


def diagnose_duplicates(records: list[dict[str, object]], ngram_size: int = 5) -> list[DuplicateMatch]:
    matches: list[DuplicateMatch] = []
    hashes: dict[str, str] = {}
    for index, record in enumerate(records):
        document_id = str(record["document_id"])
        body = str(record.get("body_text") or "")
        digest = normalized_body_hash(body)
        if digest in hashes:
            matches.append(DuplicateMatch(document_id, hashes[digest], 1.0, "duplicate_exact"))
        else:
            hashes[digest] = document_id
        for other in records[:index]:
            other_id = str(other["document_id"])
            if any(
                match.document_id == document_id
                and match.other_document_id == other_id
                and match.kind == "duplicate_exact"
                for match in matches
            ):
                continue
            similarity = ngram_jaccard(body, str(other.get("body_text") or ""), ngram_size)
            if similarity >= 0.90:
                matches.append(DuplicateMatch(document_id, other_id, similarity, "duplicate_near"))
    return matches
