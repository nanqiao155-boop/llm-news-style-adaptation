from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date
from typing import Any


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def stable_event_group_id(article_ids: list[str], version: str) -> str:
    payload = f"{version}|{'|'.join(sorted(article_ids))}".encode("utf-8")
    return "event_" + hashlib.sha256(payload).hexdigest()[:16]


def _date_span_days(members: list[dict[str, Any]]) -> int:
    dates = sorted(date.fromisoformat(item["publish_date"]) for item in members)
    return (dates[-1] - dates[0]).days if dates else 0


def _compatible_pair(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    settings = config["grouping_settings"]
    gap = metrics.get("date_gap_days")
    if gap is None or gap > settings["maximum_auto_date_span_days"]:
        return False
    return (
        metrics["body_similarity"] >= settings["compatibility_minimum_body_similarity"]
        and metrics["title_similarity"] >= settings["compatibility_minimum_title_similarity"]
    ) or (
        metrics["body_similarity"] >= settings["compatibility_high_body_similarity"]
        and (
            metrics["keyphrase_overlap"] >= settings["compatibility_minimum_keyphrase_overlap"]
            or bool(metrics["shared_rare_title_ngrams"])
        )
    )


def _can_merge(
    left_ids: set[str],
    right_ids: set[str],
    article_by_id: dict[str, dict[str, Any]],
    all_pair_metrics: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    settings = config["grouping_settings"]
    union = left_ids | right_ids
    if len(union) > settings["maximum_auto_group_size"]:
        return False, "maximum_auto_group_size_exceeded"
    members = [article_by_id[item] for item in union]
    if _date_span_days(members) > settings["maximum_auto_date_span_days"]:
        return False, "maximum_auto_date_span_exceeded"
    for left_id in left_ids:
        for right_id in right_ids:
            if not _compatible_pair(all_pair_metrics[pair_key(left_id, right_id)], config):
                return False, "pairwise_compatibility_failed"
    return True, None


def build_preliminary_event_groups(
    articles: list[dict[str, Any]],
    candidate_pairs: list[dict[str, Any]],
    all_pair_metrics: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build conservative groups without unconditional connected components.

    Only strong edges may initiate a merge. Every proposed union must pass a
    complete cross-pair compatibility check, the size limit and the date-span
    limit. Review edges never merge articles automatically.
    """

    article_by_id = {item["article_id"]: item for item in articles}
    parent = {item["article_id"]: item["article_id"] for item in articles}
    members = {item["article_id"]: {item["article_id"]} for item in articles}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    blocked_edges: list[dict[str, Any]] = []
    strong_edges = [
        item for item in candidate_pairs if item["automatic_relation"] == "strong_same_event_candidate"
    ]
    strong_edges.sort(key=lambda item: (-item["combined_score"], -item["body_similarity"], item["pair_id"]))
    for edge in strong_edges:
        left_root = find(edge["article_a_id"])
        right_root = find(edge["article_b_id"])
        if left_root == right_root:
            continue
        can_merge, reason = _can_merge(
            members[left_root], members[right_root], article_by_id, all_pair_metrics, config
        )
        if not can_merge:
            blocked_edges.append({"pair_id": edge["pair_id"], "reason": reason})
            continue
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root] |= members.pop(right_root)

    grouped_ids: dict[str, list[str]] = {}
    for article_id in article_by_id:
        grouped_ids.setdefault(find(article_id), []).append(article_id)

    candidate_by_article: dict[str, list[dict[str, Any]]] = {item: [] for item in article_by_id}
    for pair in candidate_pairs:
        candidate_by_article[pair["article_a_id"]].append(pair)
        candidate_by_article[pair["article_b_id"]].append(pair)
    blocked_pair_ids = {item["pair_id"] for item in blocked_edges}

    groups: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for article_ids in sorted(grouped_ids.values(), key=lambda values: sorted(values)):
        article_ids = sorted(article_ids)
        group_members = [article_by_id[item] for item in article_ids]
        group_id = stable_event_group_id(article_ids, config["grouping_version"])
        internal_candidates = [
            pair
            for pair in candidate_pairs
            if pair["article_a_id"] in article_ids and pair["article_b_id"] in article_ids
        ]
        touching_candidates = {
            pair["pair_id"]: pair
            for article_id in article_ids
            for pair in candidate_by_article[article_id]
        }
        external_candidates = [
            pair
            for pair in touching_candidates.values()
            if not (pair["article_a_id"] in article_ids and pair["article_b_id"] in article_ids)
        ]
        risk_reasons: list[str] = []
        if any(pair["pair_id"] in blocked_pair_ids for pair in external_candidates):
            risk_reasons.append("blocked_strong_merge_requires_review")
        if len(article_ids) > config["grouping_settings"]["maximum_auto_group_size"]:
            risk_reasons.append("group_size_exceeds_auto_limit")
        date_span = _date_span_days(group_members)
        if date_span > config["grouping_settings"]["maximum_auto_date_span_days"]:
            risk_reasons.append("date_span_exceeds_auto_limit")
        if len(article_ids) > 1:
            status = "needs_group_review" if risk_reasons else "preliminary_auto_group"
        elif touching_candidates:
            status = "needs_pair_review"
        else:
            status = "singleton"
        themes = Counter(item["content_theme"] for item in group_members)
        years = Counter(str(item["publish_year"]) for item in group_members)
        lengths = Counter(item.get("length_bucket") or "unknown" for item in group_members)
        geographies = Counter(item["geographic_scope"] for item in group_members)
        group = {
            "event_group_id": group_id,
            "grouping_version": config["grouping_version"],
            "member_article_ids": article_ids,
            "member_document_ids": [article_by_id[item]["document_id"] for item in article_ids],
            "member_count": len(article_ids),
            "group_size": len(article_ids),
            "canonical_title": min((item["title_normalized"] for item in group_members), key=lambda value: (len(value), value)),
            "date_start": min(item["publish_date"] for item in group_members),
            "date_end": max(item["publish_date"] for item in group_members),
            "date_span_days": date_span,
            "main_theme": themes.most_common(1)[0][0],
            "geographic_scope": geographies.most_common(1)[0][0],
            "creation_method": "pairwise_compatible_strong_edges" if len(article_ids) > 1 else "default_singleton",
            "strong_edges": sorted(pair["pair_id"] for pair in internal_candidates if pair["automatic_relation"] == "strong_same_event_candidate"),
            "review_edges": sorted(pair["pair_id"] for pair in touching_candidates.values() if pair["automatic_relation"] == "human_review_candidate"),
            "content_theme_distribution": dict(sorted(themes.items())),
            "publish_year_distribution": dict(sorted(years.items())),
            "length_bucket_distribution": dict(sorted(lengths.items())),
            "geographic_scope_distribution": dict(sorted(geographies.items())),
            "internal_candidate_pair_ids": sorted(pair["pair_id"] for pair in internal_candidates),
            "external_candidate_pair_ids": sorted(pair["pair_id"] for pair in external_candidates),
            "risk_reasons": risk_reasons,
            "review_status": status,
            "group_status": status,
            "review_reasons": risk_reasons,
            "human_event_label": None,
            "human_review_notes": None,
        }
        groups.append(group)
        for item in group_members:
            assignments.append(
                {
                    "article_id": item["article_id"],
                    "document_id": item["document_id"],
                    "event_group_id": group_id,
                    "grouping_version": config["grouping_version"],
                    "assignment_status": status,
                }
            )
    groups.sort(key=lambda item: item["event_group_id"])
    assignments.sort(key=lambda item: item["article_id"])
    return groups, assignments, blocked_edges
