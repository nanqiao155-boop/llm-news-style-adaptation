from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from .protection import validate_manifest


AI_DECISIONS = {"eligible", "borderline", "rejected"}
SECOND_DECISIONS = {"eligible", "rejected", "uncertain"}
JUDGE_DECISIONS = {"accept", "reject", "quarantine"}
QUALITY_TIERS = {"gold_human_reviewed", "silver_ai_reviewed", "quarantine", "rejected"}
RULE_VERSION = "task011d-c-deterministic-v1.0.0"
REVIEWER_VERSION = "task011d-c-char-ngram-nb-v1.0.0"
SECOND_REVIEWER_VERSION = "task011d-c-independent-token-nb-v1.0.0"
JUDGE_VERSION = "task011d-c-ensemble-judge-v1.0.0"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.translate(str.maketrans({"，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?", "“": '"', "”": '"', "‘": "'", "’": "'"}))
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", value)
    value = re.sub(r"(?:责任编辑|编辑)[:：]?\s*[^\n]{0,30}$", "", value)
    return re.sub(r"\s+", "", value).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except (TypeError, ValueError):
        return False


def _repeat_ratio(paragraphs: list[str]) -> float:
    normalized = [normalize_text(item) for item in paragraphs if normalize_text(item)]
    if not normalized:
        return 1.0
    return 1.0 - len(set(normalized)) / len(normalized)


def deterministic_review(row: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    title, body = str(article.get("title") or "").strip(), str(article.get("body") or "").strip()
    paragraphs = article.get("paragraphs") if isinstance(article.get("paragraphs"), list) else []
    reject: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {
        "title_present": bool(title), "body_present": bool(body),
        "publish_date_valid": _valid_date(str(article.get("publish_date") or "")),
        "character_count": len(body), "paragraph_count": len(paragraphs),
        "repeat_paragraph_ratio": round(_repeat_ratio(paragraphs), 6),
    }
    if not title:
        reject.append("missing_title")
    if not body:
        reject.append("missing_body")
    if not evidence["publish_date_valid"]:
        reject.append("invalid_publish_date")
    if "\ufffd" in title + body:
        reject.append("encoding_replacement_character")
    if urlsplit(str(row.get("canonical_url") or "")).hostname != "www.10086.cn":
        reject.append("source_domain_out_of_scope")
    if row.get("source_level") != "headquarters" or row.get("source_section") != "groupnews":
        reject.append("source_section_out_of_scope")
    normalized_title, normalized_body = normalize_text(title), normalize_text(body)
    title_only = normalized_body == normalized_title
    evidence["body_equals_title"] = title_only
    if title_only and len(normalized_body) <= 80:
        reject.append("title_only_no_news_body")
    if re.fullmatch(r"[（(]?点击播放视频[^）)]*[）)]?", normalized_body):
        reject.append("video_only_page")
    if len(normalized_body) <= 40 and len(paragraphs) <= 1 and not reject:
        warnings.append("extremely_short_single_block")
    elif len(normalized_body) < 180:
        warnings.append("short_body_requires_semantic_review")
    if "详见附件" in body and len(normalized_body) < 220:
        warnings.append("attachment_dependent_body")
    if re.search(r"(?:公告|通知|公示|招聘|采购|招标|征集|报名|名单|启事)$", title):
        warnings.append("notice_like_title")
    if evidence["repeat_paragraph_ratio"] >= 0.5 and len(paragraphs) >= 4:
        warnings.append("abnormal_repeated_paragraphs")
    if re.search(r"<(?:div|p|span|script|style)\b|&nbsp;|javascript:", body, re.I):
        warnings.append("html_or_script_residue")
    if row.get("technical_status") == "technical_warning":
        warnings.append("upstream_technical_warning")
    if reject:
        status = "deterministic_reject"
    elif warnings:
        status = "deterministic_warning"
    else:
        status = "deterministic_pass"
    return {
        "article_id": row["article_id"], "status": status,
        "reject_reason_codes": sorted(set(reject)), "warning_codes": sorted(set(warnings)),
        "rule_version": RULE_VERSION, "evidence": evidence,
        "body_sha256": row["body_sha256"], "normalized_body_sha256": sha256_text(normalized_body),
        "state_history": ["parsed", status],
    }


def _char_ngrams(value: str, sizes: tuple[int, ...]) -> list[str]:
    value = normalize_text(value)
    return [value[i:i+n] for n in sizes for i in range(max(0, len(value) - n + 1))]


def _tokens(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value or "")
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", value)
    output: list[str] = []
    for chunk in chunks:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            output.extend(chunk[i:i+2] for i in range(max(1, len(chunk) - 1)))
        else:
            output.append(chunk.lower())
    return output


@dataclass
class NaiveBayesTextModel:
    feature_mode: str
    positive: Counter[str]
    negative: Counter[str]
    positive_total: int
    negative_total: int
    positive_docs: int
    negative_docs: int
    vocabulary: set[str]

    @classmethod
    def train(cls, examples: list[tuple[str, bool]], feature_mode: str) -> "NaiveBayesTextModel":
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        p_docs = n_docs = 0
        for text, label in examples:
            features = _char_ngrams(text, (3, 4)) if feature_mode == "char" else _tokens(text)
            # Document-frequency features avoid making a long, otherwise valid
            # release look more negative merely because common terms repeat.
            counts = Counter(set(features))
            if label:
                positive.update(counts); p_docs += 1
            else:
                negative.update(counts); n_docs += 1
        if not p_docs or not n_docs:
            raise ValueError("eligibility training data requires both accepted and rejected examples")
        vocabulary = set(positive) | set(negative)
        return cls(feature_mode, positive, negative, sum(positive.values()), sum(negative.values()), p_docs, n_docs, vocabulary)

    def probability(self, text: str) -> float:
        raw = _char_ngrams(text, (3, 4)) if self.feature_mode == "char" else _tokens(text)
        features = set(raw) & self.vocabulary
        prior_odds = math.log(self.positive_docs / self.negative_docs)
        if features:
            alpha = 0.01
            evidence = sum(
                math.log((self.positive[feature] + alpha * self.positive_docs) / (self.positive_docs * (1 + 2 * alpha)))
                - math.log((self.negative[feature] + alpha * self.negative_docs) / (self.negative_docs * (1 + 2 * alpha)))
                for feature in features
            ) / len(features)
        else:
            evidence = 0.0
        log_odds = max(-40.0, min(40.0, prior_odds + 12.0 * evidence))
        return 1.0 / (1.0 + math.exp(-log_odds))


def _historical_examples(root: Path) -> list[tuple[str, bool]]:
    quality_path = root / "data/interim/quality/task008_quality_final_20260718T163002+0800.jsonl"
    parsed_path = root / "data/interim/parsed/task008_parser_v3_video_fallback_20260718T152336+0800.jsonl"
    quality = {row["document_id"]: bool(row["training_eligible"]) for row in read_jsonl(quality_path)}
    examples: list[tuple[str, bool]] = []
    for row in read_jsonl(parsed_path):
        if row["document_id"] in quality:
            examples.append((f"{row.get('title_normalized','')}\n{row.get('body_text','')}", quality[row["document_id"]]))
    return examples


def _policy_flags(title: str, body: str) -> list[str]:
    flags: list[str] = []
    normalized = normalize_text(body)
    if normalize_text(title) == normalized:
        flags.append("no_independent_body_content")
    if "详见附件" in body and len(normalized) < 220:
        flags.append("attachment_dependent")
    if re.search(r"(?:公告|通知|公示|招聘|采购|招标|征集|报名|名单|启事)$", title):
        flags.append("notice_or_list_risk")
    if len(normalized) < 120:
        flags.append("fact_density_or_completeness_risk")
    return flags


def primary_review(model: NaiveBayesTextModel, row: dict[str, Any], article: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    title, body = str(article.get("title") or ""), str(article.get("body") or "")
    probability = model.probability(f"{title}\n{body}")
    flags = _policy_flags(title, body)
    if "no_independent_body_content" in flags:
        decision = "rejected"
    elif "attachment_dependent" in flags:
        decision = "borderline"
    elif probability >= 0.72 and not flags:
        decision = "eligible"
    elif probability < 0.18 and (flags or len(normalize_text(body)) < 180):
        decision = "rejected"
    else:
        decision = "borderline"
    confidence = probability if decision == "eligible" else (1 - probability if decision == "rejected" else 1 - abs(probability - 0.5))
    reason_codes = flags or (["formal_news_release_likely"] if decision == "eligible" else ["semantic_eligibility_uncertain"])
    rationale = {
        "eligible": "Current article is consistent with a complete official news release.",
        "borderline": "A second isolated review is required for a genuine genre or completeness boundary.",
        "rejected": "Current article lacks a complete target-news body or matches an excluded content form.",
    }[decision]
    return {
        "article_id": row["article_id"], "decision": decision,
        "reason_codes": reason_codes, "confidence": round(float(confidence), 6),
        "short_rationale": rationale, "reviewer_version": REVIEWER_VERSION,
        "input_fields": ["title", "publish_date", "source_metadata", "body"],
        "model_probability_eligible": round(probability, 6),
        "state_history": deterministic["state_history"] + ["eligibility_pending", "eligibility_ai_reviewed"],
    }


def second_review(model: NaiveBayesTextModel, row: dict[str, Any], article: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    probability = model.probability(f"{article.get('title','')}\n{article.get('body','')}")
    flags = _policy_flags(str(article.get("title") or ""), str(article.get("body") or ""))
    if "no_independent_body_content" in flags or probability < 0.16:
        decision = "rejected"
    elif probability >= 0.67 and "attachment_dependent" not in flags:
        decision = "eligible"
    else:
        decision = "uncertain"
    return {
        "article_id": row["article_id"], "decision": decision,
        "reason_codes": flags or ["independent_news_release_assessment"],
        "confidence": round(max(probability, 1 - probability), 6),
        "reviewer_version": SECOND_REVIEWER_VERSION,
        "first_reviewer_rationale_visible": False,
        "model_probability_eligible": round(probability, 6),
        "state_history": primary["state_history"] + ["eligibility_second_review"],
    }


def judge_review(row: dict[str, Any], article: dict[str, Any], primary: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    p1 = float(primary["model_probability_eligible"])
    p2 = float(second["model_probability_eligible"])
    mean = (p1 + p2) / 2
    flags = set(_policy_flags(str(article.get("title") or ""), str(article.get("body") or "")))
    if "no_independent_body_content" in flags:
        decision = "reject"
    elif "attachment_dependent" in flags and len(normalize_text(str(article.get("body") or ""))) < 120:
        decision = "reject"
    elif mean >= 0.52 and second["decision"] != "rejected":
        decision = "accept"
    elif mean <= 0.30 and second["decision"] == "rejected":
        decision = "reject"
    else:
        decision = "quarantine"
    return {
        "article_id": row["article_id"], "decision": decision,
        "reason_codes": sorted(flags) or ["ensemble_resolution"],
        "confidence": round(max(mean, 1 - mean), 6), "judge_version": JUDGE_VERSION,
        "state_history": second["state_history"] + ["eligibility_judged"],
    }


def _simhash(value: str) -> int:
    features = set(_char_ngrams(value, (5,)))
    vector = [0] * 64
    for feature in features:
        number = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if number & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(vector) if score >= 0)


def _shingle_jaccard(left: str, right: str) -> float:
    a, b = set(_char_ngrams(left, (5,))), set(_char_ngrams(right, (5,)))
    return len(a & b) / len(a | b) if a or b else 1.0


def _survivor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda item: (-len(normalize_text(item["body"])), -int(bool(item.get("publish_date"))), item.get("publish_date") or "9999-99-99", str(item["article_id"])))[0]


def deduplicate(accepted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    groups: list[dict[str, Any]] = []
    rejected: set[str] = set()
    gold_by_exact = {sha256_text(str(item["body_text"])): item for item in gold}
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted:
        exact[sha256_text(str(item["body"]))].append(item)
    for digest, items in sorted(exact.items()):
        gold_item = gold_by_exact.get(digest)
        if len(items) < 2 and gold_item is None:
            continue
        survivor_id = gold_item["article_id"] if gold_item else _survivor(items)["article_id"]
        losers = [item["article_id"] for item in items if item["article_id"] != survivor_id]
        rejected.update(losers)
        groups.append({"duplicate_type": "exact", "confidence": 1.0, "survivor": survivor_id, "rejected_articles": losers, "decision_source": "sha256", "evidence_summary": "identical body bytes"})
    remaining = [item for item in accepted if item["article_id"] not in rejected]
    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_normalized = {sha256_text(normalize_text(str(item["body_text"]))): item for item in gold}
    for item in remaining:
        normalized[sha256_text(normalize_text(str(item["body"])))].append(item)
    for digest, items in sorted(normalized.items()):
        gold_item = gold_normalized.get(digest)
        if len(items) < 2 and gold_item is None:
            continue
        survivor_id = gold_item["article_id"] if gold_item else _survivor(items)["article_id"]
        losers = [item["article_id"] for item in items if item["article_id"] != survivor_id]
        rejected.update(losers)
        groups.append({"duplicate_type": "normalized", "confidence": 1.0, "survivor": survivor_id, "rejected_articles": losers, "decision_source": "normalized_sha256", "evidence_summary": "identical after approved normalization"})
    remaining = [item for item in accepted if item["article_id"] not in rejected]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes = [_simhash(str(item["body"])) for item in remaining]
    candidates: set[tuple[int, int]] = set()
    for index, number in enumerate(hashes):
        for band in range(4):
            key = (band, (number >> (band * 16)) & 0xFFFF)
            for other in buckets[key]:
                candidates.add((other, index))
            buckets[key].append(index)
    for left_index, right_index in sorted(candidates):
        left, right = remaining[left_index], remaining[right_index]
        if left["article_id"] in rejected or right["article_id"] in rejected:
            continue
        body_score = _shingle_jaccard(str(left["body"]), str(right["body"]))
        title_score = SequenceMatcher(None, normalize_text(str(left["title"])), normalize_text(str(right["title"]))).ratio()
        if body_score >= 0.97 or (body_score >= 0.92 and title_score >= 0.85):
            survivor = _survivor([left, right])
            loser = right if survivor is left else left
            rejected.add(str(loser["article_id"]))
            groups.append({"duplicate_type": "near", "confidence": round(body_score, 6), "survivor": survivor["article_id"], "rejected_articles": [loser["article_id"]], "decision_source": "high_confidence_text_overlap", "evidence_summary": f"5-gram Jaccard={body_score:.4f}; title similarity={title_score:.4f}"})
    counts = Counter(group["duplicate_type"] for group in groups for _ in group["rejected_articles"])
    return {
        "schema_version": "task011d-c-dedup-v1.0.0", "groups": groups,
        "exact_duplicate_rejected": counts["exact"],
        "normalized_duplicate_rejected": counts["normalized"],
        "near_duplicate_rejected": counts["near"],
        "same_event_not_duplicate_policy": "title_or_event_similarity_alone_never_rejects",
        "gold_core_preference": True,
    }, rejected


def _ledger_row(row: dict[str, Any], stage: str, reasons: list[str], primary: dict[str, Any] | None, second: dict[str, Any] | None, judge: dict[str, Any] | None, final_status: str) -> dict[str, Any]:
    return {
        "article_id": row["article_id"], "url": row["canonical_url"], "source": "corporate Headquarters/groupnews",
        "stage": stage, "reason_codes": "|".join(reasons),
        "reviewer_decision": primary["decision"] if primary else "not_run",
        "second_reviewer_decision": second["decision"] if second else "not_run",
        "judge_decision": judge["decision"] if judge else "not_run", "final_status": final_status,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def target_policy(total: int, hard: int = 2000, preferred: int = 2200) -> dict[str, Any]:
    if total >= preferred:
        status = "preferred_target_met"
    elif total >= hard:
        status = "hard_min_met_preferred_not_met"
    else:
        status = "hard_min_not_met"
    return {
        "headquarters_target_status": status,
        "subsidiary_expansion_required": total < hard,
        "accepted_gap": max(0, hard - total),
    }


def run_headquarters_quality(root: Path, config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    protection = read_json(root / config["v1_protection_manifest"])
    errors = validate_manifest(root, protection)
    if errors:
        raise RuntimeError(f"v1 protection failed: {errors}")
    input_path = root / config["cleaning_input_path"]
    input_rows = read_jsonl(input_path)
    if len(input_rows) != 2170 or len({row["article_id"] for row in input_rows}) != 2170:
        raise ValueError("formal cleaning input count or article_id uniqueness mismatch")
    article_rows: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        if row.get("source_level") != "headquarters" or row.get("source_section") != "groupnews":
            raise ValueError(f"out-of-scope input: {row['article_id']}")
        ref = root / row["parsed_article_ref"]
        article = read_json(ref)
        if article["article_id"] != row["article_id"] or sha256_text(article["body"]) != row["body_sha256"]:
            raise ValueError(f"invalid article reference: {row['article_id']}")
        article_rows[row["article_id"]] = article
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_paths = [output_dir / name for name in config["detailed_output_files"]]
    summary_path = output_dir / "quality_summary.json"
    pool_path = root / config["quality_pool_path"]
    checkpoint_dir = root / config["checkpoint_dir"]
    if resume and summary_path.is_file() and pool_path.is_file() and all(path.is_file() for path in detailed_paths):
        summary = read_json(summary_path)
        if summary.get("parsed_hq_input") != len(input_rows):
            raise ValueError("resume summary input count mismatch")
        if len(read_jsonl(pool_path)) != summary.get("hq_final_accepted_total"):
            raise ValueError("resume quality pool count mismatch")
        return {**summary, "resumed": True, "resume_cache_hits": len(input_rows)}
    if not resume and (any(path.exists() for path in detailed_paths) or (checkpoint_dir.is_dir() and any(checkpoint_dir.glob("batch_*.json")))):
        raise FileExistsError("no-overwrite: TASK-011D-C output already exists; use --resume")
    examples = _historical_examples(root)
    primary_model = NaiveBayesTextModel.train(examples, "char")
    second_model = NaiveBayesTextModel.train(examples, "token")
    deterministic_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    second_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    quarantines: list[dict[str, Any]] = []
    accepted_articles: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_count = 0
    if resume:
        checkpoints = sorted(checkpoint_dir.glob("batch_*.json"))
        if checkpoints:
            checkpoint = read_json(checkpoints[-1])
            if checkpoint.get("reviewer_version") != REVIEWER_VERSION:
                raise ValueError("resume reviewer version mismatch")
            completed_count = int(checkpoint["completed_count"])
            deterministic_rows.extend(checkpoint["deterministic_rows"])
            primary_rows.extend(checkpoint["primary_rows"])
            second_rows.extend(checkpoint["second_rows"])
            judge_rows.extend(checkpoint["judge_rows"])
            rejections.extend(checkpoint["rejections"])
            quarantines.extend(checkpoint["quarantines"])
            accepted_articles.extend(article_rows[article_id] for article_id in checkpoint["accepted_article_ids"])
    for index, row in enumerate(input_rows, start=1):
        if index <= completed_count:
            continue
        article = article_rows[row["article_id"]]
        deterministic = deterministic_review(row, article)
        deterministic_rows.append(deterministic)
        if deterministic["status"] == "deterministic_reject":
            rejections.append(_ledger_row(row, "deterministic_cleaning", deterministic["reject_reason_codes"], None, None, None, "rejected"))
        else:
            primary = primary_review(primary_model, row, article, deterministic)
            primary_rows.append(primary)
            second = judge = None
            if primary["decision"] == "eligible":
                primary["state_history"].append("eligible")
                accepted_articles.append(article)
            elif primary["decision"] == "rejected":
                primary["state_history"].append("rejected")
                rejections.append(_ledger_row(row, "ai_eligibility", primary["reason_codes"], primary, None, None, "rejected"))
            else:
                second = second_review(second_model, row, article, primary)
                second_rows.append(second)
                judge = judge_review(row, article, primary, second)
                judge_rows.append(judge)
                if judge["decision"] == "accept":
                    judge["state_history"].append("eligible")
                    accepted_articles.append(article)
                elif judge["decision"] == "reject":
                    judge["state_history"].append("rejected")
                    rejections.append(_ledger_row(row, "eligibility_judge", judge["reason_codes"], primary, second, judge, "rejected"))
                else:
                    judge["state_history"].append("quarantined")
                    quarantines.append(_ledger_row(row, "eligibility_judge", judge["reason_codes"], primary, second, judge, "quarantined"))
        if index % batch_size == 0 or index == len(input_rows):
            write_json(checkpoint_dir / f"batch_{math.ceil(index / batch_size):03d}.json", {
                "completed_count": index, "last_article_id": row["article_id"], "status": "completed",
                "reviewer_version": REVIEWER_VERSION,
                "deterministic_rows": deterministic_rows, "primary_rows": primary_rows,
                "second_rows": second_rows, "judge_rows": judge_rows,
                "rejections": rejections, "quarantines": quarantines,
                "accepted_article_ids": [item["article_id"] for item in accepted_articles],
            })
    gold = read_jsonl(root / config["gold_articles_path"])
    if len(gold) != 239:
        raise ValueError("Gold Core count mismatch")
    gold_predictions = []
    for item in gold:
        probability = primary_model.probability(f"{item.get('title_normalized','')}\n{item.get('body_text','')}")
        decision = "eligible" if probability >= 0.52 else ("borderline" if probability >= 0.18 else "rejected")
        gold_predictions.append({"article_id": item["article_id"], "predicted_decision": decision, "probability_eligible": round(probability, 6)})
    gold_counts = Counter(row["predicted_decision"] for row in gold_predictions)
    systematic_failure = gold_counts["rejected"] > int(config["gold_regression_max_rejected"])
    gold_regression = {
        "schema_version": "task011d-c-gold-regression-v1.0.0", "sample_count": 239,
        "gold_predicted_eligible": gold_counts["eligible"], "gold_predicted_borderline": gold_counts["borderline"],
        "gold_predicted_rejected": gold_counts["rejected"],
        "mismatch_cases": [row for row in gold_predictions if row["predicted_decision"] != "eligible"],
        "gold_regression_systematic_failure": systematic_failure, "gold_status_modified": False,
    }
    if systematic_failure:
        write_json(output_dir / "gold_eligibility_regression.json", gold_regression)
        raise RuntimeError("Gold Eligibility regression systematic failure")
    dedup_manifest, duplicate_rejected = deduplicate(accepted_articles, gold)
    accepted_final = [item for item in accepted_articles if item["article_id"] not in duplicate_rejected]
    group_by_rejected = {rid: group for group in dedup_manifest["groups"] for rid in group["rejected_articles"]}
    input_by_id = {row["article_id"]: row for row in input_rows}
    for article_id in sorted(duplicate_rejected):
        group = group_by_rejected[article_id]
        rejections.append(_ledger_row(input_by_id[article_id], f"dedup_{group['duplicate_type']}", [f"duplicate_{group['duplicate_type']}_rejected"], next((x for x in primary_rows if x["article_id"] == article_id), None), next((x for x in second_rows if x["article_id"] == article_id), None), next((x for x in judge_rows if x["article_id"] == article_id), None), "rejected"))
    quality_pool: list[dict[str, Any]] = []
    for item in gold:
        quality_pool.append({
            "article_id": item["article_id"], "source_ref": "data/processed/news_v1/articles.jsonl",
            "source_url": item["source_url"], "publish_date": item["publish_date"],
            "quality_tier": "gold_human_reviewed", "eligibility_status": "accepted_gold",
            "eligibility_lineage": "data/meta/task011d_gold_core_manifest.json",
            "duplicate_lineage": "gold_survivor_preference", "body_sha256": item["body_text_sha256"],
            "state": "quality_pool_ready",
        })
    for item in accepted_final:
        source = input_by_id[item["article_id"]]
        primary_lineage = next(row for row in primary_rows if row["article_id"] == item["article_id"])
        judge_lineage = next((row for row in judge_rows if row["article_id"] == item["article_id"]), None)
        state_history = list((judge_lineage or primary_lineage)["state_history"]) + ["deduplicated", "quality_pool_ready"]
        quality_pool.append({
            "article_id": item["article_id"], "source_ref": source["parsed_article_ref"],
            "source_url": source["canonical_url"], "publish_date": source["publish_date"],
            "source_level": "headquarters", "source_section": "groupnews",
            "quality_tier": "silver_ai_reviewed", "eligibility_status": "eligible",
            "eligibility_lineage": "ai_eligibility_results.jsonl", "duplicate_lineage": "dedup_manifest.json",
            "body_sha256": source["body_sha256"], "normalized_body_sha256": sha256_text(normalize_text(item["body"])),
            "state": "quality_pool_ready", "state_history": state_history,
        })
    deterministic_counts = Counter(row["status"] for row in deterministic_rows)
    ai_counts = Counter(row["decision"] for row in primary_rows)
    judge_counts = Counter(row["decision"] for row in judge_rows)
    total = len(quality_pool)
    hard, preferred, stretch = (int(config[key]) for key in ("hard_min_final_accepted_count", "preferred_final_accepted_count", "stretch_target_count"))
    target = target_policy(total, hard, preferred)
    target_status = target["headquarters_target_status"]
    gap = target["accepted_gap"]
    summary = {
        "task_id": "TASK-011D-C", "role": "AUTOMATED_DATA_QUALITY_OPERATOR",
        "parsed_hq_input": len(input_rows), "gold_core_count": len(gold),
        "deterministic_pass": deterministic_counts["deterministic_pass"],
        "deterministic_warning": deterministic_counts["deterministic_warning"],
        "deterministic_reject": deterministic_counts["deterministic_reject"],
        "ai_direct_eligible": ai_counts["eligible"], "ai_borderline": ai_counts["borderline"], "ai_rejected": ai_counts["rejected"],
        "second_review_count": len(second_rows), "judge_count": len(judge_rows),
        "judge_accepted": judge_counts["accept"], "judge_rejected": judge_counts["reject"],
        "quarantine_count": len(quarantines),
        "exact_duplicate_rejected": dedup_manifest["exact_duplicate_rejected"],
        "normalized_duplicate_rejected": dedup_manifest["normalized_duplicate_rejected"],
        "near_duplicate_rejected": dedup_manifest["near_duplicate_rejected"],
        "new_hq_final_accepted": len(accepted_final), "hq_final_accepted_total": total,
        "acceptance_rate": round(len(accepted_final) / len(input_rows), 6),
        "gold_regression": "passed", "gold_predicted_rejected": gold_counts["rejected"],
        "hard_min_final_accepted_count": hard, "preferred_final_accepted_count": preferred, "stretch_target_count": stretch,
        "headquarters_target_status": target_status, "subsidiary_expansion_required": target["subsidiary_expansion_required"],
        "accepted_gap": gap, "subsidiary_candidate_buffer_if_required": {"minimum": math.ceil(gap * 1.25), "maximum": math.ceil(gap * 1.40)} if gap else None,
        "human_review_required": False, "network_requests": 0, "external_model_api_calls": 0,
        "news_v2_frozen": False, "sft_v2_created": False, "event_group_executed": False,
        "split_executed": False, "sft_generated": False, "training_executed": False,
        "v1_protection": "passed", "quality_pool_path": config["quality_pool_path"],
        "review_strategy": "two isolated locally trained text classifiers plus policy-constrained ensemble judge",
    }
    write_jsonl(output_dir / "deterministic_cleaning_results.jsonl", deterministic_rows)
    write_jsonl(output_dir / "ai_eligibility_results.jsonl", primary_rows)
    write_jsonl(output_dir / "eligibility_second_review_results.jsonl", second_rows)
    write_jsonl(output_dir / "eligibility_judge_results.jsonl", judge_rows)
    write_json(output_dir / "dedup_manifest.json", dedup_manifest)
    write_jsonl(root / config["quality_pool_path"], quality_pool)
    ledger_fields = ["article_id", "url", "source", "stage", "reason_codes", "reviewer_decision", "second_reviewer_decision", "judge_decision", "final_status"]
    _write_csv(output_dir / "eligibility_rejection_ledger.csv", rejections, ledger_fields)
    _write_csv(output_dir / "eligibility_quarantine_ledger.csv", quarantines, ledger_fields)
    write_json(output_dir / "quality_summary.json", summary)
    write_json(output_dir / "gold_eligibility_regression.json", gold_regression)
    safe_path = root / config["safe_summary_path"]
    _write_csv(safe_path, [{key: (json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value) for key, value in summary.items()}], list(summary))
    return summary
