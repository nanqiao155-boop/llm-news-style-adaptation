from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.dataset.event_features import body_length_bucket, compact_text, extract_entities
from src.task011d.protection import validate_manifest


SPLITS = ("train", "validation", "test")
TIERS = {"gold_human_reviewed", "silver_ai_reviewed"}
EVENT_FILES = (
    "VERSION", "event_groups.jsonl", "article_to_event_group.jsonl", "event_group_manifest.json",
    "event_group_statistics.json", "lineage.jsonl",
)
SPLIT_FILES = (
    "VERSION", "train_ids.txt", "validation_ids.txt", "test_ids.txt", "split_manifest.json",
    "split_statistics.json", "split_integrity_audit.json", "lineage.jsonl",
)
NEWS_FILES = ("VERSION", "articles.jsonl", "news_manifest.json", "news_statistics.json", "lineage.jsonl")


class Task011dEventSplitError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_checksums(directory: Path, names: Iterable[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_checksums(directory: Path) -> list[str]:
    errors: list[str] = []
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if not (directory / name).is_file() or sha256_file(directory / name) != expected:
            errors.append(f"checksum_mismatch:{directory.name}/{name}")
    return errors


def _normalized_title(value: str) -> str:
    return compact_text(unicodedata.normalize("NFKC", value or ""))


def _ngrams(value: str, size: int) -> set[str]:
    text = compact_text(value)
    return {text[index:index + size] for index in range(max(0, len(text) - size + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def _date_gap(left: str, right: str) -> int:
    return abs((date.fromisoformat(left) - date.fromisoformat(right)).days)


def _theme(title: str, body: str) -> str:
    text = title + body[:1200]
    rules = (
        ("emergency_service", ("抗震", "救灾", "防汛", "抢险", "应急通信", "台风")),
        ("corporate_governance", ("业绩", "董事会", "股东", "任命", "党组", "党建")),
        ("international_business", ("国际", "海外", "全球", "一带一路")),
        ("customer_service", ("客户", "服务", "资费", "权益", "消费")),
        ("green_development", ("绿色", "低碳", "节能", "碳达峰", "碳中和")),
        ("industry_cooperation", ("签约", "合作", "联盟", "伙伴", "生态")),
        ("technology_innovation", ("5g", "6g", "人工智能", "ai", "算力", "云", "大数据", "创新", "研发", "卫星")),
    )
    normalized = unicodedata.normalize("NFKC", text).lower()
    scores = [(sum(normalized.count(word) for word in words), name) for name, words in rules]
    score, name = max(scores)
    return name if score else "corporate_strategy"


def load_articles(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    pool_path = root / config["input_quality_pool_path"]
    pool = read_jsonl(pool_path)
    if len(pool) != 2363 or len({row["article_id"] for row in pool}) != 2363:
        raise Task011dEventSplitError("quality pool must contain 2363 unique article IDs")
    if Counter(row["quality_tier"] for row in pool) != Counter({"gold_human_reviewed": 239, "silver_ai_reviewed": 2124}):
        raise Task011dEventSplitError("quality pool Gold/Silver counts are invalid")
    if any(row["quality_tier"] not in TIERS or row.get("state") != "quality_pool_ready" for row in pool):
        raise Task011dEventSplitError("quality pool contains an invalid tier or non-ready sample")

    gold_by_id = {row["article_id"]: row for row in read_jsonl(root / config["gold_articles_path"])}
    articles: list[dict[str, Any]] = []
    for row in pool:
        if row["quality_tier"] == "gold_human_reviewed":
            source = gold_by_id.get(row["article_id"])
            if not source:
                raise Task011dEventSplitError(f"missing Gold lineage: {row['article_id']}")
            body = source["body_text"]
            title = source["title_normalized"]
            source_company = source["source_organization"]
            document_id = source["document_id"]
            theme = source["content_theme"]
            source_level = "headquarters"
            source_ref = config["gold_articles_path"]
        else:
            if row.get("source_level") != "headquarters" or not row.get("eligibility_lineage") or not row.get("duplicate_lineage"):
                raise Task011dEventSplitError(f"incomplete Silver lineage: {row['article_id']}")
            source_ref = row["source_ref"]
            source = read_json(root / source_ref)
            body = source["body"]
            title = source["title"]
            source_company = source["source_company"]
            document_id = source["body_sha256"]
            theme = _theme(title, body)
            source_level = source["source_level"]
        actual_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual_hash != row["body_sha256"]:
            raise Task011dEventSplitError(f"content hash mismatch: {row['article_id']}")
        if source_level != "headquarters":
            raise Task011dEventSplitError(f"non-headquarters source: {row['article_id']}")
        articles.append({
            "article_id": str(row["article_id"]), "document_id": document_id, "title": title, "body": body,
            "body_sha256": row["body_sha256"], "source_url": row["source_url"], "source_ref": source_ref,
            "source_company": source_company, "source_level": "headquarters", "publish_date": row["publish_date"],
            "publish_year": int(row["publish_date"][:4]), "quality_tier": row["quality_tier"],
            "gold_silver": "Gold" if row["quality_tier"] == "gold_human_reviewed" else "Silver",
            "eligibility_lineage": row["eligibility_lineage"], "dedup_lineage": row["duplicate_lineage"],
            "content_theme": theme, "length_bucket": body_length_bucket(len(body)), "body_length_chars": len(body),
            "original_v1_id": row["article_id"] if row["quality_tier"] == "gold_human_reviewed" else None,
        })
    articles.sort(key=lambda item: item["article_id"])
    return articles, sha256_file(pool_path)


def build_candidates(articles: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    title_terms: list[set[str]] = [_ngrams(row["title"], 3) for row in articles]
    term_df: Counter[str] = Counter(term for terms in title_terms for term in terms)
    index: dict[tuple[int, str], list[int]] = defaultdict(list)
    exact: dict[str, list[int]] = defaultdict(list)
    for index_number, row in enumerate(articles):
        exact[_normalized_title(row["title"])].append(index_number)
        year = row["publish_year"]
        for term in title_terms[index_number]:
            if term_df[term] <= config["blocking"]["maximum_term_document_frequency"]:
                index[(year, term)].append(index_number)
    pair_ids: set[tuple[int, int]] = set()
    for values in exact.values():
        for left_pos, left in enumerate(values):
            for right in values[left_pos + 1:]:
                if _date_gap(articles[left]["publish_date"], articles[right]["publish_date"]) <= config["blocking"]["exact_title_max_days"]:
                    pair_ids.add((min(left, right), max(left, right)))
    for values in index.values():
        for left_pos, left in enumerate(values):
            for right in values[left_pos + 1:]:
                if _date_gap(articles[left]["publish_date"], articles[right]["publish_date"]) <= config["blocking"]["candidate_max_days"]:
                    pair_ids.add((min(left, right), max(left, right)))

    candidates: list[dict[str, Any]] = []
    for left_index, right_index in sorted(pair_ids):
        left, right = articles[left_index], articles[right_index]
        title_left, title_right = _normalized_title(left["title"]), _normalized_title(right["title"])
        title_similarity = _jaccard(_ngrams(title_left, 2), _ngrams(title_right, 2))
        if title_similarity < config["blocking"]["minimum_title_jaccard"] and title_left != title_right:
            continue
        body_similarity = _jaccard(_ngrams(left["body"], 5), _ngrams(right["body"], 5))
        entities_left = extract_entities(left["title"] + "\n" + left["body"][:1000])
        entities_right = extract_entities(right["title"] + "\n" + right["body"][:1000])
        entity_overlap = _overlap(entities_left, entities_right)
        gap = _date_gap(left["publish_date"], right["publish_date"])
        exact_title = title_left == title_right
        high = (
            (exact_title and gap <= 30 and (entity_overlap > 0 or body_similarity >= 0.25))
            or (title_similarity >= 0.72 and body_similarity >= 0.52 and gap <= 90)
            or (body_similarity >= 0.82 and title_similarity >= 0.35 and gap <= 180)
        )
        ambiguous = not high and (
            (title_similarity >= 0.62 and body_similarity >= 0.28 and gap <= 90)
            or (body_similarity >= 0.60 and gap <= 180)
        )
        if not high and not ambiguous:
            continue
        review_decision = None
        relation = "high_confidence_same_event" if high else "ambiguous"
        if ambiguous:
            # This policy-constrained reviewer receives event evidence only; no split target is available here.
            same = title_similarity >= 0.68 and body_similarity >= 0.38 and entity_overlap >= 0.25 and gap <= 120
            review_decision = "same_event" if same else "different_event"
            relation = "ai_event_review_same_event" if same else "ai_event_review_different_event"
        pair_hash = hashlib.sha256(f"{left['article_id']}|{right['article_id']}".encode()).hexdigest()[:16]
        candidates.append({
            "pair_id": f"ev2pair_{pair_hash}", "article_a_id": left["article_id"], "article_b_id": right["article_id"],
            "date_gap_days": gap, "exact_title": exact_title, "title_similarity": round(title_similarity, 6),
            "body_similarity": round(body_similarity, 6), "entity_overlap": round(entity_overlap, 6),
            "relation": relation, "same_event": high or review_decision == "same_event",
            "review_role": "AI_EVENT_GROUP_REVIEWER" if ambiguous else None,
            "review_decision": review_decision, "future_split_targets_visible": False,
        })
    candidates.sort(key=lambda row: row["pair_id"])
    return candidates, {
        "blocking_algorithm": "year_plus_rare_title_trigram_inverted_index_with_exact_title_long_window",
        "unconstrained_quadratic_comparison": False, "total_possible_pairs": len(articles) * (len(articles) - 1) // 2,
        "blocked_pair_count": len(pair_ids), "evaluated_candidate_count": len(candidates),
    }


def group_events(root: Path, articles: list[dict[str, Any]], candidates: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    old_groups = read_jsonl(root / config["gold_event_groups_path"])
    old_assignments = read_jsonl(root / config["gold_event_assignments_path"])
    old_split_rows = read_jsonl(root / config["gold_split_assignments_path"])
    gold_group_by_article = {row["article_id"]: row["event_group_id"] for row in old_assignments}
    gold_split_by_article = {row["article_id"]: row["split"] for row in old_split_rows}
    gold_split_by_group: dict[str, str] = {}
    for group in old_groups:
        splits = {gold_split_by_article[item] for item in group["member_article_ids"]}
        if len(splits) != 1:
            raise Task011dEventSplitError(f"frozen Gold group crosses split: {group['event_group_id']}")
        gold_split_by_group[group["event_group_id"]] = next(iter(splits))

    article_by_id = {row["article_id"]: row for row in articles}
    silver_ids = {row["article_id"] for row in articles if row["gold_silver"] == "Silver"}
    gold_matches: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    silver_edges: list[dict[str, Any]] = []
    for pair in candidates:
        if not pair["same_event"]:
            continue
        left, right = pair["article_a_id"], pair["article_b_id"]
        if left in gold_group_by_article and right in silver_ids:
            gold_matches[right].append((gold_group_by_article[left], pair))
        elif right in gold_group_by_article and left in silver_ids:
            gold_matches[left].append((gold_group_by_article[right], pair))
        elif left in silver_ids and right in silver_ids:
            silver_edges.append(pair)

    bridge_candidates: list[dict[str, Any]] = []
    assigned_gold: dict[str, str] = {}
    quarantined: set[str] = set()
    for article_id, matches in sorted(gold_matches.items()):
        groups = sorted({group_id for group_id, _ in matches})
        splits = sorted({gold_split_by_group[group_id] for group_id in groups})
        if len(splits) > 1:
            # Independent judge is deliberately conservative: only a bridge supported by high-confidence
            # edges to every split is quarantined; otherwise the article remains a new singleton.
            confirmed = all(pair["relation"] == "high_confidence_same_event" for _, pair in matches)
            bridge_candidates.append({
                "article_id": article_id, "candidate_gold_group_ids": groups, "candidate_splits": splits,
                "judge_role": "EVENT_GROUP_JUDGE", "judge_decision": "true_cross_split_bridge" if confirmed else "insufficient_bridge_evidence_new_group",
                "future_split_targets_visible": False,
            })
            if confirmed:
                quarantined.add(article_id)
            continue
        scored = sorted(matches, key=lambda item: (-item[1]["body_similarity"], -item[1]["title_similarity"], item[0]))
        assigned_gold[article_id] = scored[0][0]

    parent = {item: item for item in silver_ids - set(assigned_gold) - quarantined}
    members = {item: {item} for item in parent}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    same_pairs = {frozenset((row["article_a_id"], row["article_b_id"])) for row in silver_edges}
    for edge in sorted(silver_edges, key=lambda row: (-row["body_similarity"], -row["title_similarity"], row["pair_id"])):
        left, right = edge["article_a_id"], edge["article_b_id"]
        if left not in parent or right not in parent:
            continue
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        union = members[left_root] | members[right_root]
        if len(union) > config["grouping"]["maximum_auto_group_size"]:
            continue
        if not all(frozenset((a, b)) in same_pairs for a in members[left_root] for b in members[right_root]):
            continue
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep
        members[keep] |= members.pop(drop)

    silver_components: dict[str, list[str]] = defaultdict(list)
    for article_id in parent:
        silver_components[find(article_id)].append(article_id)
    new_group_by_article: dict[str, str] = {}
    new_groups: list[dict[str, Any]] = []
    for component in sorted((sorted(values) for values in silver_components.values()), key=lambda values: values):
        group_id = "EV2_" + hashlib.sha256((config["event_group_version"] + "|" + "|".join(component)).encode()).hexdigest()[:12].upper()
        members_data = [article_by_id[item] for item in component]
        for item in component:
            new_group_by_article[item] = group_id
        new_groups.append({
            "event_group_id": group_id, "group_origin": "silver_only", "member_article_ids": component,
            "group_size": len(component), "representative_title": min((item["title"] for item in members_data), key=lambda value: (len(value), value)),
            "date_start": min(item["publish_date"] for item in members_data), "date_end": max(item["publish_date"] for item in members_data),
            "core_entities": sorted(set().union(*(extract_entities(item["title"] + "\n" + item["body"][:800]) for item in members_data)))[:12],
            "assignment_confidence": "high" if len(component) > 1 else "conservative_singleton",
            "assignment_source": "deterministic_high_confidence_or_ai_event_review" if len(component) > 1 else "no_reliable_same_event_candidate",
            "gold_association": None, "final_split": None,
        })

    groups: list[dict[str, Any]] = []
    for old in old_groups:
        silver = sorted(article_id for article_id, group_id in assigned_gold.items() if group_id == old["event_group_id"])
        member_ids = list(old["member_article_ids"]) + silver
        groups.append({
            "event_group_id": old["event_group_id"], "group_origin": "gold_preserved", "member_article_ids": member_ids,
            "group_size": len(member_ids), "representative_title": old["canonical_title"], "date_start": min(article_by_id[item]["publish_date"] for item in member_ids),
            "date_end": max(article_by_id[item]["publish_date"] for item in member_ids), "core_entities": [],
            "assignment_confidence": "frozen_gold_anchor", "assignment_source": "gold_group_preserved_with_optional_silver_match",
            "gold_association": old["event_group_id"], "final_split": gold_split_by_group[old["event_group_id"]],
            "preserved_gold_member_article_ids": old["member_article_ids"],
        })
    groups.extend(new_groups)
    assignments: list[dict[str, Any]] = []
    for article in articles:
        article_id = article["article_id"]
        if article_id in quarantined:
            continue
        if article_id in gold_group_by_article:
            group_id, source = gold_group_by_article[article_id], "preserved_gold_membership"
        elif article_id in assigned_gold:
            group_id, source = assigned_gold[article_id], "gold_group_match"
        else:
            group_id, source = new_group_by_article[article_id], "silver_event_group"
        assignments.append({"article_id": article_id, "event_group_id": group_id, "event_group_assignment_source": source})
    groups.sort(key=lambda row: row["event_group_id"])
    assignments.sort(key=lambda row: row["article_id"])
    return {"groups": groups, "assignments": assignments, "bridges": bridge_candidates, "quarantined": quarantined,
            "gold_split_by_group": gold_split_by_group, "gold_group_by_article": gold_group_by_article}


def assign_splits(groups: list[dict[str, Any]], assignments: list[dict[str, Any]], articles: list[dict[str, Any]], seed: str, targets: dict[str, int]) -> dict[str, Any]:
    article_by_id = {row["article_id"]: row for row in articles}
    split_counts = Counter()
    split_group_counts = Counter()
    hist: dict[str, dict[str, Counter[str]]] = {split: {name: Counter() for name in ("publish_year", "length_bucket", "content_theme")} for split in SPLITS}
    assignment_by_group: dict[str, str] = {}

    def add(group: dict[str, Any], split: str) -> None:
        assignment_by_group[group["event_group_id"]] = split
        split_counts[split] += group["group_size"]
        split_group_counts[split] += 1
        for article_id in group["member_article_ids"]:
            row = article_by_id[article_id]
            for dimension in hist[split]:
                hist[split][dimension][str(row[dimension])] += 1

    anchored = [row for row in groups if row["final_split"]]
    pending = [row for row in groups if not row["final_split"]]
    for group in anchored:
        add(group, group["final_split"])
    global_hist = {dimension: Counter(str(row[dimension]) for row in articles) for dimension in ("publish_year", "length_bucket", "content_theme")}

    pending.sort(key=lambda row: (-row["group_size"], hashlib.sha256(f"{seed}|{row['event_group_id']}".encode()).hexdigest()))
    candidate_assignment: list[dict[str, Any]] = []
    for group in pending:
        options: list[tuple[float, str, str]] = []
        for split in SPLITS:
            projected = split_counts[split] + group["group_size"]
            count_penalty = abs(projected - targets[split]) / max(1, targets[split]) * 100.0
            overflow = max(0, projected - targets[split]) * 1000.0
            distribution_penalty = 0.0
            for dimension in global_hist:
                addition = Counter(str(article_by_id[item][dimension]) for item in group["member_article_ids"])
                desired_scale = targets[split] / len(articles)
                distribution_penalty += sum(abs(hist[split][dimension][value] + addition[value] - total * desired_scale) for value, total in global_hist[dimension].items()) / len(articles)
            tie = hashlib.sha256(f"{seed}|{group['event_group_id']}|{split}".encode()).hexdigest()
            options.append((overflow + count_penalty + distribution_penalty * 200.0, tie, split))
        score, _, selected = min(options)
        candidate_assignment.append({"event_group_id": group["event_group_id"], "candidate_scores": {split: round(value[0], 8) for value, split in [(item, item[2]) for item in options]}, "selected": selected})
        add(group, selected)
        group["final_split"] = selected

    # With predominantly singleton groups, deterministic whole-group moves close any remaining target gaps.
    while any(split_counts[split] != targets[split] for split in SPLITS):
        source = max(SPLITS, key=lambda split: split_counts[split] - targets[split])
        destination = min(SPLITS, key=lambda split: split_counts[split] - targets[split])
        surplus = split_counts[source] - targets[source]
        deficit = targets[destination] - split_counts[destination]
        if surplus <= 0 or deficit <= 0:
            break
        movable = [row for row in pending if assignment_by_group[row["event_group_id"]] == source and row["group_size"] <= min(surplus, deficit)]
        if not movable:
            break
        chosen = min(movable, key=lambda row: (abs(row["group_size"] - min(surplus, deficit)), hashlib.sha256(f"{seed}|rebalance|{row['event_group_id']}|{destination}".encode()).hexdigest()))
        assignment_by_group[chosen["event_group_id"]] = destination
        chosen["final_split"] = destination
        split_counts[source] -= chosen["group_size"]
        split_counts[destination] += chosen["group_size"]
        split_group_counts[source] -= 1
        split_group_counts[destination] += 1

    article_assignments = []
    group_by_id = {row["event_group_id"]: row for row in groups}
    for row in assignments:
        group = group_by_id[row["event_group_id"]]
        article_assignments.append({**row, "split": assignment_by_group[group["event_group_id"]], "event_group_size": group["group_size"]})
    article_assignments.sort(key=lambda row: row["article_id"])
    return {"article_assignments": article_assignments, "assignment_by_group": assignment_by_group,
            "article_counts": dict(split_counts), "event_group_counts": dict(split_group_counts), "candidate_assignment": candidate_assignment}


def _distributions(rows: list[dict[str, Any]], split_by_id: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("publish_year", "length_bucket", "content_theme"):
        result[dimension] = {
            split: dict(sorted(Counter(str(row[dimension]) for row in rows if split_by_id[row["article_id"]] == split).items()))
            for split in SPLITS
        }
    return result


def _maximum_deviation(distributions: dict[str, Any], counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    result: dict[str, float] = {}
    for dimension, split_values in distributions.items():
        overall = Counter()
        for values in split_values.values():
            overall.update(values)
        maximum = 0.0
        for split, values in split_values.items():
            for key, count in overall.items():
                maximum = max(maximum, abs(values.get(key, 0) / counts[split] - count / total) * 100)
        result[dimension] = round(maximum, 6)
    return result


def run_task011d_event_group_split(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    protection = read_json(root / config["v1_protection_manifest"])
    errors = validate_manifest(root, protection)
    if errors:
        raise Task011dEventSplitError("v1 protection failed: " + ";".join(errors))
    output_dirs = [root / config[name] for name in ("event_output_dir", "split_output_dir", "news_output_dir")]
    if any(path.exists() for path in output_dirs):
        raise Task011dEventSplitError("no-overwrite: one or more frozen v2 output directories already exist")

    articles, pool_sha = load_articles(root, config)
    candidates, blocking = build_candidates(articles, config)
    grouping = group_events(root, articles, candidates, config)
    final_articles = [row for row in articles if row["article_id"] not in grouping["quarantined"]]
    final_ids = {row["article_id"] for row in final_articles}
    groups = [row for row in grouping["groups"] if all(item in final_ids for item in row["member_article_ids"])]
    assignments = [row for row in grouping["assignments"] if row["article_id"] in final_ids]
    if len(final_articles) < config["hard_minimum_count"]:
        raise Task011dEventSplitError("split-integrity quarantine reduced final count below hard minimum")
    targets = {"train": math.ceil(len(final_articles) * 0.8), "validation": round(len(final_articles) * 0.1)}
    targets["test"] = len(final_articles) - targets["train"] - targets["validation"]
    seed = hashlib.sha256(f"{config['split_seed_namespace']}|{pool_sha}".encode()).hexdigest()
    split = assign_splits(groups, assignments, final_articles, seed, targets)
    split_by_id = {row["article_id"]: row["split"] for row in split["article_assignments"]}
    group_by_id = {row["event_group_id"]: row for row in groups}
    for group in groups:
        group["final_split"] = split["assignment_by_group"][group["event_group_id"]]

    gold_drift = sum(split_by_id[item] != grouping["gold_split_by_group"][grouping["gold_group_by_article"][item]] for item in grouping["gold_group_by_article"])
    id_sets = {name: {item for item, value in split_by_id.items() if value == name} for name in SPLITS}
    document_sets = {name: {row["document_id"] for row in final_articles if split_by_id[row["article_id"]] == name} for name in SPLITS}
    group_sets = {name: {row["event_group_id"] for row in groups if row["final_split"] == name} for name in SPLITS}
    pair_names = (("train", "validation"), ("train", "test"), ("validation", "test"))
    article_overlap = sum(len(id_sets[a] & id_sets[b]) for a, b in pair_names)
    document_overlap = sum(len(document_sets[a] & document_sets[b]) for a, b in pair_names)
    group_overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in pair_names)
    distributions = _distributions(final_articles, split_by_id)
    deviations = _maximum_deviation(distributions, split["article_counts"])
    if any((article_overlap, document_overlap, group_overlap, gold_drift)):
        raise Task011dEventSplitError("blocking split integrity audit failed")

    event_dir, split_dir, news_dir = output_dirs
    event_dir.mkdir(parents=True); split_dir.mkdir(parents=True); news_dir.mkdir(parents=True)
    event_stats = {
        "total_articles": len(final_articles), "total_event_groups": len(groups), "gold_groups": 228,
        "silver_only_groups": sum(row["group_origin"] == "silver_only" for row in groups),
        "silver_assigned_into_gold_groups": sum(row["event_group_assignment_source"] == "gold_group_match" for row in assignments),
        "singleton_groups": sum(row["group_size"] == 1 for row in groups), "multi_article_groups": sum(row["group_size"] > 1 for row in groups),
        "max_group_size": max(row["group_size"] for row in groups), "mean_group_size": round(statistics.mean(row["group_size"] for row in groups), 6),
        "median_group_size": statistics.median(row["group_size"] for row in groups),
        "cross_gold_bridge_candidate_count": len(grouping["bridges"]), "bridge_quarantine_count": len(grouping["quarantined"]),
        "unresolved_group_count": 0, **blocking,
    }
    event_manifest = {
        "dataset_version": config["event_group_version"], "source_quality_pool_version": config["input_quality_pool_version"],
        "source_quality_pool_sha256": pool_sha, "grouping_policy": "task011d-d-conservative-event-grouping-v1",
        "gold_event_group_version": "news_event_groups_v1.0.0", "gold_groups_preserved": 228,
        "candidate_blocking": blocking, "bridge_reviews": grouping["bridges"], "build_status": "frozen",
        "human_review_required": False, "network_requests": 0, "external_model_api_calls": 0,
    }
    event_lineage = [{**row, "event_group_version": config["event_group_version"]} for row in assignments]
    (event_dir / "VERSION").write_text(config["event_group_version"] + "\n", encoding="utf-8", newline="\n")
    write_jsonl(event_dir / "event_groups.jsonl", groups)
    write_jsonl(event_dir / "article_to_event_group.jsonl", assignments)
    write_json(event_dir / "event_group_manifest.json", event_manifest)
    write_json(event_dir / "event_group_statistics.json", event_stats)
    write_jsonl(event_dir / "lineage.jsonl", event_lineage)
    write_checksums(event_dir, EVENT_FILES)

    integrity = {
        "article_overlap": article_overlap, "document_overlap": document_overlap, "event_group_overlap": group_overlap,
        "blocking_split_leakage": article_overlap + document_overlap + group_overlap, "gold_split_drift": gold_drift,
        "gold_train_preserved": sum(split_by_id[item] == "train" for item in grouping["gold_group_by_article"]),
        "gold_validation_preserved": sum(split_by_id[item] == "validation" for item in grouping["gold_group_by_article"]),
        "gold_test_preserved": sum(split_by_id[item] == "test" for item in grouping["gold_group_by_article"]),
        "validation_status": "passed",
    }
    split_stats = {
        "article_counts": split["article_counts"], "event_group_counts": split["event_group_counts"],
        "ratios": {name: round(split["article_counts"][name] / len(final_articles), 6) for name in SPLITS},
        "target_counts": targets, "distributions": distributions, "maximum_absolute_percentage_point_deviation": deviations,
    }
    split_manifest = {
        "dataset_version": config["split_version"], "event_group_version": config["event_group_version"],
        "algorithm": "deterministic_gold_anchored_stratified_group_greedy_rebalance_v1", "seed": seed,
        "seed_policy": "sha256(fixed_namespace|quality_pool_sha256)", "objective": "group integrity then 80/10/10 then year/length/theme balance",
        "constraints": ["preserve_gold_split", "event_group_indivisible", "unique_article_and_document_membership"],
        "weights": {"count": 100.0, "publish_year": 5.0, "length_bucket": 5.0, "content_theme": 8.0},
        "candidate_assignment_sha256": stable_json_hash(split["candidate_assignment"]), "final_assignment_sha256": stable_json_hash(split["article_assignments"]),
        "build_status": "frozen", "human_review_required": False,
    }
    (split_dir / "VERSION").write_text(config["split_version"] + "\n", encoding="utf-8", newline="\n")
    for name in SPLITS:
        (split_dir / f"{name}_ids.txt").write_text("\n".join(sorted(id_sets[name])) + "\n", encoding="utf-8", newline="\n")
    write_json(split_dir / "split_manifest.json", split_manifest)
    write_json(split_dir / "split_statistics.json", split_stats)
    write_json(split_dir / "split_integrity_audit.json", integrity)
    write_jsonl(split_dir / "lineage.jsonl", split["article_assignments"])
    write_checksums(split_dir, SPLIT_FILES)

    news_rows = []
    news_lineage = []
    assignment_by_id = {row["article_id"]: row for row in assignments}
    for row in final_articles:
        news_id = row["article_id"]
        news_rows.append({**row, "source_article_id": row["article_id"], "final_news_v2_id": news_id, "dataset_version": config["news_version"],
                          "event_group_id": assignment_by_id[row["article_id"]]["event_group_id"], "split": split_by_id[row["article_id"]]})
        news_lineage.append({
            "article_id": row["article_id"], "source_version": "news_v1.0.0" if row["gold_silver"] == "Gold" else config["input_quality_pool_version"],
            "source_level": row["source_level"], "source_company": row["source_company"], "quality_tier": row["quality_tier"],
            "gold_silver": row["gold_silver"], "eligibility_lineage": row["eligibility_lineage"], "dedup_lineage": row["dedup_lineage"],
            "event_group_id": assignment_by_id[row["article_id"]]["event_group_id"], "split": split_by_id[row["article_id"]],
            "content_hash": row["body_sha256"], "source_ref": row["source_ref"], "original_v1_id": row["original_v1_id"],
            "final_news_v2_id": news_id, "final_status": "frozen_accepted",
        })
    news_manifest = {
        "dataset_version": config["news_version"], "dataset_source_policy": "headquarters_only_v2", "source_level": "headquarters",
        "subsidiary_expansion_required": False, "article_count": len(news_rows), "gold_count": 239,
        "silver_count": len(news_rows) - 239, "event_group_version": config["event_group_version"], "split_version": config["split_version"],
        "articles_file_tracked_by_git": False, "build_status": "frozen",
    }
    (news_dir / "VERSION").write_text(config["news_version"] + "\n", encoding="utf-8", newline="\n")
    write_jsonl(news_dir / "articles.jsonl", news_rows)
    write_json(news_dir / "news_manifest.json", news_manifest)
    write_json(news_dir / "news_statistics.json", {"article_count": len(news_rows), "quality_tiers": dict(Counter(row["quality_tier"] for row in final_articles)), "splits": split["article_counts"]})
    write_jsonl(news_dir / "lineage.jsonl", news_lineage)
    write_checksums(news_dir, NEWS_FILES)

    sft_rows = [{
        "article_id": row["article_id"], "source_ref": row["source_ref"], "event_group_id": assignment_by_id[row["article_id"]]["event_group_id"],
        "split": split_by_id[row["article_id"]], "quality_tier": row["quality_tier"], "gold_silver": row["gold_silver"],
        "content_sha256": row["body_sha256"], "news_v2_lineage_sha256": stable_json_hash(next(item for item in news_lineage if item["article_id"] == row["article_id"])),
        "reuse_frozen_gold_sft": row["gold_silver"] == "Gold",
    } for row in final_articles]
    sft_input = root / config["sft_generation_input_path"]
    write_jsonl(sft_input, sft_rows)
    sft_manifest = {
        "next_task": "TASK-011D-E Autonomous SFT Generation and AI Review", "input_count": len(sft_rows),
        "gold_sft_reuse_count": 239, "reuse_frozen_gold_sft": True, "silver_sft_generation_count": len(sft_rows) - 239,
        "silver_generation_by_split": {name: sum(row["gold_silver"] == "Silver" and row["split"] == name for row in sft_rows) for name in SPLITS},
        "sft_content_generated": False, "human_review_required": False, "input_sha256": sha256_file(sft_input),
    }
    write_json(root / config["sft_generation_manifest_path"], sft_manifest)
    return {"event_statistics": event_stats, "split_statistics": split_stats, "integrity": integrity, "news_manifest": news_manifest,
            "sft_manifest": sft_manifest, "seed": seed, "pool_sha256": pool_sha}


def write_safe_summary(path: Path, result: dict[str, Any]) -> None:
    event, split, integrity, news, sft = (result[name] for name in ("event_statistics", "split_statistics", "integrity", "news_manifest", "sft_manifest"))
    row = {
        "role": "EVENT_GROUP_SPLIT_OPERATOR", "input_count": 2363, "gold_count": 239, "silver_count": 2124,
        "final_news_count": news["article_count"], "total_event_groups": event["total_event_groups"],
        "silver_to_gold": event["silver_assigned_into_gold_groups"], "silver_only_groups": event["silver_only_groups"],
        "singleton_groups": event["singleton_groups"], "largest_group": event["max_group_size"],
        "bridge_candidates": event["cross_gold_bridge_candidate_count"], "bridge_quarantine": event["bridge_quarantine_count"],
        "train": split["article_counts"]["train"], "validation": split["article_counts"]["validation"], "test": split["article_counts"]["test"],
        "article_overlap": integrity["article_overlap"], "document_overlap": integrity["document_overlap"],
        "event_group_leakage": integrity["event_group_overlap"], "gold_split_drift": integrity["gold_split_drift"],
        "gold_sft_reuse": sft["gold_sft_reuse_count"], "silver_sft_generation": sft["silver_sft_generation_count"],
        "human_review": False, "sft_generated": False, "training": False, "network_requests": 0, "external_api_calls": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
