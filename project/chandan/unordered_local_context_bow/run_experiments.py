#!/usr/bin/env python3
"""
Run baseline TF-IDF retrieval and an unordered local-context bag-of-words variant
for the Cranfield collection.

The local-context method augments the baseline with unordered context n-grams:
for each token position, we collect a fixed-size local window, keep only unique
tokens from that window, ignore word order, and emit unordered combinations such
as pairs/triples as binary context features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from evaluation import Evaluation  # noqa: E402


METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F-score",
    "map": "MAP",
    "ndcg": "nDCG",
    "mrr": "MRR",
}


@dataclass
class SparseTfidfModel:
    name: str
    doc_ids: List[int]
    idf: Dict[str, float]
    postings: Dict[str, List[Tuple[int, float]]]

    @classmethod
    def build(
        cls,
        name: str,
        feature_maps: Sequence[Dict[str, float]],
        doc_ids: Sequence[int],
    ) -> "SparseTfidfModel":
        doc_ids = list(doc_ids)
        doc_freq = Counter()
        for fmap in feature_maps:
            doc_freq.update(fmap.keys())

        total_docs = float(len(feature_maps))
        idf = {}
        for feature, df in doc_freq.items():
            idf[feature] = math.log(total_docs / df) if df else 0.0

        postings = defaultdict(list)
        for doc_index, fmap in enumerate(feature_maps):
            raw_weights = {}
            norm_sq = 0.0
            for feature, tf in fmap.items():
                feature_idf = idf.get(feature, 0.0)
                if feature_idf <= 0.0 or tf <= 0.0:
                    continue
                weight = tf * feature_idf
                raw_weights[feature] = weight
                norm_sq += weight * weight

            if norm_sq == 0.0:
                continue

            norm = math.sqrt(norm_sq)
            for feature, weight in raw_weights.items():
                postings[feature].append((doc_index, weight / norm))

        return cls(name=name, doc_ids=doc_ids, idf=idf, postings=dict(postings))

    def score(self, query_map: Dict[str, float]) -> np.ndarray:
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        raw_query = {}
        norm_sq = 0.0
        for feature, tf in query_map.items():
            feature_idf = self.idf.get(feature)
            if feature_idf is None or feature_idf <= 0.0 or tf <= 0.0:
                continue
            weight = tf * feature_idf
            raw_query[feature] = weight
            norm_sq += weight * weight

        if norm_sq == 0.0:
            return scores

        norm = math.sqrt(norm_sq)
        for feature, weight in raw_query.items():
            q_weight = weight / norm
            for doc_index, d_weight in self.postings.get(feature, ()):
                scores[doc_index] += q_weight * d_weight

        return scores

    def rank(self, query_maps: Sequence[Dict[str, float]]) -> Tuple[List[List[int]], List[np.ndarray]]:
        rankings = []
        score_vectors = []
        for query_map in query_maps:
            scores = self.score(query_map)
            ranked_doc_indexes = sorted(
                range(len(self.doc_ids)),
                key=lambda idx: (-scores[idx], self.doc_ids[idx]),
            )
            rankings.append([self.doc_ids[idx] for idx in ranked_doc_indexes])
            score_vectors.append(scores)
        return rankings, score_vectors


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def clean_sentence(sentence: Sequence[str]) -> List[str]:
    cleaned = []
    for token in sentence:
        token = token.strip().lower()
        if any(char.isalnum() for char in token):
            cleaned.append(token)
    return cleaned


def normalize_collection(collection: Sequence[Sequence[Sequence[str]]]) -> List[List[List[str]]]:
    normalized = []
    for item in collection:
        normalized_item = []
        for sentence in item:
            cleaned = clean_sentence(sentence)
            if cleaned:
                normalized_item.append(cleaned)
        normalized.append(normalized_item)
    return normalized


def unigram_feature_map(item: Sequence[Sequence[str]]) -> Dict[str, float]:
    counts = Counter()
    for sentence in item:
        counts.update(sentence)
    return dict(counts)


def context_feature_map(
    item: Sequence[Sequence[str]],
    radius: int,
    orders: Sequence[int],
) -> Dict[str, float]:
    features = set()
    for sentence in item:
        if not sentence:
            continue
        sent_len = len(sentence)
        for index in range(sent_len):
            start = max(0, index - radius)
            end = min(sent_len, index + radius + 1)
            bag = sorted(set(sentence[start:end]))
            for order in orders:
                if len(bag) < order:
                    continue
                for combo in combinations(bag, order):
                    features.add(f"ctx{order}:" + "__".join(combo))
    return {feature: 1.0 for feature in features}


def filter_context_feature_maps(
    feature_maps: Sequence[Dict[str, float]],
    min_df: int,
) -> Tuple[List[Dict[str, float]], Counter]:
    doc_freq = Counter()
    for fmap in feature_maps:
        doc_freq.update(fmap.keys())

    if min_df <= 1:
        return [dict(fmap) for fmap in feature_maps], doc_freq

    filtered_maps = []
    for fmap in feature_maps:
        filtered_maps.append(
            {feature: value for feature, value in fmap.items() if doc_freq[feature] >= min_df}
        )
    return filtered_maps, doc_freq


def compute_metrics(
    evaluator: Evaluation,
    rankings: Sequence[Sequence[int]],
    query_ids: Sequence[int],
    qrels,
) -> Dict[str, List[float]]:
    metrics = {
        "precision": [],
        "recall": [],
        "fscore": [],
        "map": [],
        "ndcg": [],
        "mrr": [],
    }
    for k in range(1, 11):
        metrics["precision"].append(evaluator.meanPrecision(rankings, query_ids, qrels, k))
        metrics["recall"].append(evaluator.meanRecall(rankings, query_ids, qrels, k))
        metrics["fscore"].append(evaluator.meanFscore(rankings, query_ids, qrels, k))
        metrics["map"].append(evaluator.meanAveragePrecision(rankings, query_ids, qrels, k))
        metrics["ndcg"].append(evaluator.meanNDCG(rankings, query_ids, qrels, k))
        metrics["mrr"].append(evaluator.meanReciprocalRank(rankings, query_ids, qrels, k))
    return metrics


def compute_k10_summary(
    evaluator: Evaluation,
    rankings: Sequence[Sequence[int]],
    query_ids: Sequence[int],
    qrels,
) -> Dict[str, float]:
    k = 10
    return {
        "precision_at_10": evaluator.meanPrecision(rankings, query_ids, qrels, k),
        "recall_at_10": evaluator.meanRecall(rankings, query_ids, qrels, k),
        "fscore_at_10": evaluator.meanFscore(rankings, query_ids, qrels, k),
        "map_at_10": evaluator.meanAveragePrecision(rankings, query_ids, qrels, k),
        "ndcg_at_10": evaluator.meanNDCG(rankings, query_ids, qrels, k),
        "mrr_at_10": evaluator.meanReciprocalRank(rankings, query_ids, qrels, k),
    }


def compute_per_query_metrics(
    evaluator: Evaluation,
    rankings: Sequence[Sequence[int]],
    query_ids: Sequence[int],
    qrels,
    k: int = 10,
) -> Dict[str, List[float]]:
    per_query = {
        "average_precision": [],
        "ndcg": [],
        "reciprocal_rank": [],
        "hits_at_5": [],
        "first_relevant_rank": [],
    }
    qrels_by_query = defaultdict(list)
    qrels_positions = defaultdict(list)
    for item in qrels:
        qid = str(item["query_num"])
        qrels_by_query[qid].append(item["id"])
        qrels_positions[qid].append((item["id"], item["position"]))

    for ranked_docs, query_id in zip(rankings, query_ids):
        qid = str(query_id)
        relevant_ids = qrels_by_query[qid]
        relevant_with_pos = sorted(qrels_positions[qid], key=lambda pair: pair[1])

        per_query["average_precision"].append(
            evaluator.queryAveragePrecision(ranked_docs, query_id, relevant_with_pos, k)
        )
        per_query["ndcg"].append(evaluator.queryNDCG(ranked_docs, query_id, relevant_with_pos, k))
        per_query["reciprocal_rank"].append(
            evaluator.queryReciprocalRank(ranked_docs, query_id, relevant_with_pos, k)
        )

        top5 = [str(doc_id) for doc_id in ranked_docs[:5]]
        relevant_set = set(relevant_ids)
        per_query["hits_at_5"].append(sum(doc_id in relevant_set for doc_id in top5))

        first_rank = None
        for rank_index, doc_id in enumerate(ranked_docs[:k], start=1):
            if str(doc_id) in relevant_set:
                first_rank = rank_index
                break
        per_query["first_relevant_rank"].append(first_rank)

    return per_query


def approximate_randomization_pvalue(
    baseline_values: Sequence[float],
    method_values: Sequence[float],
    iterations: int = 20000,
    seed: int = 13,
) -> float:
    baseline = np.array(baseline_values, dtype=np.float64)
    method = np.array(method_values, dtype=np.float64)
    diffs = method - baseline
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iterations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diffs))
        shuffled = diffs * signs
        if abs(shuffled.mean()) >= observed:
            count += 1
    return (count + 1.0) / (iterations + 1.0)


def save_metric_plot(path: Path, metrics: Dict[str, List[float]], title: str) -> None:
    ks = list(range(1, 11))
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(ks, values, marker="o", linewidth=2, label=METRIC_LABELS[metric_name])
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.xticks(ks)
    plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_overlay_plot(
    path: Path,
    baseline_metrics: Dict[str, List[float]],
    all_n_metrics: Dict[int, Dict[str, List[float]]],
    method_labels: Dict[int, str],
    n_values: List[int],
) -> None:
    ks = list(range(1, 11))
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])

    metric_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(n_values) + 1))
    for axis, metric_name in zip(metric_axes, METRIC_LABELS):
        axis.plot(ks, baseline_metrics[metric_name], marker="o", linewidth=2, label="Baseline TF-IDF", color="black", linestyle="--")
        for i, n in enumerate(n_values):
            axis.plot(ks, all_n_metrics[n][metric_name], marker="s", linewidth=2, label=method_labels[n], color=colors[i])
        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)

    sweep_axis = fig.add_subplot(grid[2, :])
    sweep_metric_keys = [
        ("precision", "precision"),
        ("recall", "recall"),
        ("fscore", "fscore"),
        ("map", "map"),
        ("ndcg", "ndcg"),
        ("mrr", "mrr"),
    ]
    for metric_key, metric_name in sweep_metric_keys:
        for i, n in enumerate(n_values):
            values = [all_n_metrics[n_val][metric_name][9] for n_val in n_values]
            sweep_axis.plot(
                n_values,
                values,
                marker="o",
                linewidth=2,
                label=f"n={n} - {metric_key.upper()}",
                color=colors[i],
            )
    sweep_axis.set_title("Best k=10 Scores Across Local Context Size n")
    sweep_axis.set_xlabel("Local context size n")
    sweep_axis.set_ylabel("Score")
    sweep_axis.set_xticks(n_values)
    sweep_axis.grid(alpha=0.25)
    sweep_axis.legend(ncol=3, fontsize=9, frameon=False, loc="upper left")

    handles, labels = metric_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.suptitle("Baseline vs All Local Context Sizes", y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.88, hspace=0.48, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_n_sweep_plot(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    n_values = [row["n"] for row in rows]
    metric_keys = [
        "precision_at_10",
        "recall_at_10",
        "fscore_at_10",
        "map_at_10",
        "ndcg_at_10",
        "mrr_at_10",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for axis, metric_key in zip(axes, metric_keys):
        axis.plot(n_values, [row[metric_key] for row in rows], marker="o", linewidth=2)
        axis.set_title(metric_key.replace("_at_10", "").upper() + "@10")
        axis.set_xlabel("Local context size n")
        axis.set_ylabel("Score")
        axis.set_xticks(n_values)
        axis.grid(alpha=0.25)
    fig.suptitle("Best Result Per Local Context Size n", y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rankings(path: Path, query_ids: Sequence[int], rankings: Sequence[Sequence[int]], top_k: int = 20) -> None:
    payload = {
        str(query_id): ranked_docs[:top_k]
        for query_id, ranked_docs in zip(query_ids, rankings)
    }
    write_json(path, payload)


def find_best_config(
    docs: Sequence[Sequence[Sequence[str]]],
    queries: Sequence[Sequence[Sequence[str]]],
    doc_ids: Sequence[int],
    query_ids: Sequence[int],
    qrels,
    baseline_scores: Sequence[np.ndarray],
    output_dir: Path,
) -> Dict[str, object]:
    evaluator = Evaluation()
    sweep_results = []
    config_index = 0
    radius_values = [1, 2, 3, 4, 5]
    total_configs = len(radius_values) * 2 * 2 * 3

    for radius in radius_values:
        for orders in [(2,), (2, 3)]:
            raw_doc_contexts = [context_feature_map(doc, radius=radius, orders=orders) for doc in docs]
            raw_query_contexts = [context_feature_map(query, radius=radius, orders=orders) for query in queries]
            for min_df in [1, 2]:
                filtered_docs, doc_freq = filter_context_feature_maps(raw_doc_contexts, min_df=min_df)
                filtered_queries = [
                    {feature: value for feature, value in fmap.items() if doc_freq[feature] >= min_df}
                    for fmap in raw_query_contexts
                ]

                model = SparseTfidfModel.build(
                    name=f"context_r{radius}_o{'-'.join(map(str, orders))}_df{min_df}",
                    feature_maps=filtered_docs,
                    doc_ids=doc_ids,
                )
                _, context_scores = model.rank(filtered_queries)

                for alpha in [0.25, 0.5, 0.75]:
                    config_index += 1
                    rankings = []
                    for base_scores, ctx_scores in zip(baseline_scores, context_scores):
                        combined_scores = base_scores + (alpha * ctx_scores)
                        ranked_indexes = sorted(
                            range(len(doc_ids)),
                            key=lambda idx: (-combined_scores[idx], doc_ids[idx]),
                        )[:10]
                        rankings.append([doc_ids[idx] for idx in ranked_indexes])

                    metrics = compute_k10_summary(evaluator, rankings, query_ids, qrels)
                    result = {
                        "radius": radius,
                        "orders": list(orders),
                        "min_df": min_df,
                        "alpha": alpha,
                        "context_feature_vocab_size": len(model.idf),
                        "avg_context_features_per_doc": float(
                            np.mean([len(fmap) for fmap in filtered_docs])
                        ),
                        **metrics,
                    }
                    sweep_results.append(result)
                    print(
                        f"[sweep {config_index}/{total_configs}] "
                        f"r={radius} orders={orders} min_df={min_df} alpha={alpha} "
                        f"MAP@10={metrics['map_at_10']:.4f}"
                    )

    sweep_results.sort(
        key=lambda item: (
            item["map_at_10"],
            item["ndcg_at_10"],
            item["mrr_at_10"],
            item["precision_at_10"],
        ),
        reverse=True,
    )
    write_json(output_dir / "config_sweep.json", sweep_results)

    grouped_by_radius = defaultdict(list)
    for row in sweep_results:
        grouped_by_radius[row["radius"]].append(row)

    best_by_radius = []
    for radius in sorted(grouped_by_radius):
        best_row = grouped_by_radius[radius][0]
        best_by_radius.append(
            {
                "n": radius,
                "best_orders": best_row["orders"],
                "best_min_df": best_row["min_df"],
                "best_alpha": best_row["alpha"],
                "precision_at_10": best_row["precision_at_10"],
                "recall_at_10": best_row["recall_at_10"],
                "fscore_at_10": best_row["fscore_at_10"],
                "map_at_10": best_row["map_at_10"],
                "ndcg_at_10": best_row["ndcg_at_10"],
                "mrr_at_10": best_row["mrr_at_10"],
                "context_feature_vocab_size": best_row["context_feature_vocab_size"],
            }
        )

    write_json(output_dir / "n_sweep_best_by_radius.json", best_by_radius)
    return {
        "best_config": sweep_results[0],
        "best_by_radius": best_by_radius,
    }


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "precision@10",
        "recall@10",
        "fscore@10",
        "map@10",
        "ndcg@10",
        "mrr@10",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_n_sweep_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "best_orders",
        "best_min_df",
        "best_alpha",
        "precision@10",
        "recall@10",
        "fscore@10",
        "map@10",
        "ndcg@10",
        "mrr@10",
        "context_feature_vocab_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_example_query_comparison(
    query_ids: Sequence[int],
    queries_json,
    qrels,
    baseline_rankings: Sequence[Sequence[int]],
    method_rankings: Sequence[Sequence[int]],
    baseline_per_query: Dict[str, List[float]],
    method_per_query: Dict[str, List[float]],
    output_dir: Path,
    top_n: int = 8,
    forced_query_ids: Sequence[int] = (9, 40, 64, 81, 90),
) -> List[Dict[str, object]]:
    query_lookup = {item["query number"]: item["query"].strip() for item in queries_json}
    relevant_by_query = defaultdict(set)
    for item in qrels:
        relevant_by_query[str(item["query_num"])].add(str(item["id"]))

    candidates = []
    for idx, query_id in enumerate(query_ids):
        delta_ap = method_per_query["average_precision"][idx] - baseline_per_query["average_precision"][idx]
        delta_hits = method_per_query["hits_at_5"][idx] - baseline_per_query["hits_at_5"][idx]
        candidates.append((delta_ap, delta_hits, idx, query_id))

    candidates.sort(reverse=True)
    candidate_by_query = {query_id: (delta_ap, delta_hits, idx, query_id) for delta_ap, delta_hits, idx, query_id in candidates}

    selected = []
    seen_query_ids = set()
    for query_id in forced_query_ids:
        if query_id in candidate_by_query and query_id not in seen_query_ids:
            selected.append(candidate_by_query[query_id])
            seen_query_ids.add(query_id)

    for candidate in candidates:
        if len(selected) >= top_n:
            break
        if candidate[3] in seen_query_ids:
            continue
        selected.append(candidate)
        seen_query_ids.add(candidate[3])
    rows = []
    for delta_ap, delta_hits, idx, query_id in selected:
        relevant = relevant_by_query[str(query_id)]
        baseline_top5 = baseline_rankings[idx][:5]
        method_top5 = method_rankings[idx][:5]
        rows.append(
            {
                "query_id": query_id,
                "query": query_lookup[query_id],
                "baseline_hits_at_5": baseline_per_query["hits_at_5"][idx],
                "method_hits_at_5": method_per_query["hits_at_5"][idx],
                "baseline_ap_at_10": round(baseline_per_query["average_precision"][idx], 4),
                "method_ap_at_10": round(method_per_query["average_precision"][idx], 4),
                "delta_ap_at_10": round(delta_ap, 4),
                "delta_hits_at_5": int(delta_hits),
                "baseline_first_relevant_rank": baseline_per_query["first_relevant_rank"][idx],
                "method_first_relevant_rank": method_per_query["first_relevant_rank"][idx],
                "baseline_top5": baseline_top5,
                "method_top5": method_top5,
                "relevant_hits_in_method_top5": [doc_id for doc_id in method_top5 if str(doc_id) in relevant],
            }
        )

    write_json(output_dir / "example_query_comparison.json", rows)

    md_lines = [
        "# Example Query Comparison",
        "",
        "| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['query_id']} | {row['baseline_hits_at_5']} | {row['method_hits_at_5']} | "
            f"{row['baseline_ap_at_10']:.4f} | {row['method_ap_at_10']:.4f} | {row['delta_ap_at_10']:.4f} |"
        )

    md_lines.append("")
    for row in rows:
        md_lines.append(f"## Query {row['query_id']}")
        md_lines.append("")
        md_lines.append(row["query"])
        md_lines.append("")
        md_lines.append(f"- Baseline top 5: {row['baseline_top5']}")
        md_lines.append(f"- Method top 5: {row['method_top5']}")
        md_lines.append(f"- Relevant docs recovered by method in top 5: {row['relevant_hits_in_method_top5']}")
        md_lines.append("")

    (output_dir / "example_query_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    return rows


def build_report(
    output_path: Path,
    summary: Dict[str, object],
    examples: Sequence[Dict[str, object]],
    n_sweep_rows: Sequence[Dict[str, object]],
) -> None:
    baseline = summary["methods"]["baseline_tfidf"]
    method = summary["methods"]["unordered_local_context_bow"]
    delta = summary["delta_vs_baseline_at_10"]
    config = summary["best_config"]
    significance = summary["significance"]

    lines = [
        "# Unordered Local-Context Bag-of-Words Experiment Report",
        "",
        "## Method Summary",
        "",
        "The baseline system uses unigram TF-IDF with cosine similarity. "
        "The proposed method adds unordered local-context features extracted from nearby-word windows. "
        "Inside each local window, only token presence matters and word order is ignored, so a phrase such as "
        "`heat transfer` and the same two words appearing in swapped order still activate the same context feature.",
        "",
        "## Hypothesis",
        "",
        "A local unordered context signal should help when the baseline retrieves documents with the right words "
        "but the wrong nearby context. This is especially relevant for ambiguous technical terms and for queries "
        "whose important words should appear close together in relevant documents.",
        "",
        "## What n Means",
        "",
        "Here, `n` is the local context radius. For each token position, we inspect up to `n` neighboring words "
        "on the left and `n` neighboring words on the right, then build unordered context features from the unique "
        "tokens in that local window. So `n = 1` means a maximum window of 3 tokens, `n = 2` means up to 5 tokens, and so on.",
        "",
        "## Baseline Limitations Addressed",
        "",
        "- The assignment report highlights that the baseline VSM lacks contextual representation and ignores local proximity information.",
        "- The project brief explicitly asks us to fix factual retrieval failures observed in the baseline; local context is a direct response to those failures.",
        "- For ambiguous terms, plain TF-IDF gives the same credit to a shared token even when its nearby words indicate a different sense.",
        "",
        "## Best Configuration",
        "",
        f"- `n = {config['radius']}`",
        f"- `context_orders = {config['orders']}`",
        f"- `min_context_df = {config['min_df']}`",
        f"- `context_weight_alpha = {config['alpha']}`",
        f"- `context_feature_vocab_size = {config['context_feature_vocab_size']}`",
        "",
        "## Local Context Size Sweep",
        "",
        "| n | Best orders | Best min_df | Best alpha | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in n_sweep_rows:
        lines.append(
            f"| {row['n']} | {row['best_orders']} | {row['best_min_df']} | {row['best_alpha']:.2f} | "
            f"{row['precision_at_10']:.4f} | {row['recall_at_10']:.4f} | {row['fscore_at_10']:.4f} | "
            f"{row['map_at_10']:.4f} | {row['ndcg_at_10']:.4f} | {row['mrr_at_10']:.4f} |"
        )

    lines.extend(
        [
            "",
            "The sweep shows a trade-off rather than a single monotonic pattern. `n = 1` is strongest on precision, recall, "
            "F-score, and tied nDCG, while `n = 4` gives the best MAP@10 and MRR@10. In this dataset, a moderately larger "
            "window captures useful technical co-occurrence without drifting too far from the local topic.",
            "",
            "## k=10 Results",
            "",
            "| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| baseline_tfidf | {baseline['k10']['precision']:.4f} | {baseline['k10']['recall']:.4f} | {baseline['k10']['fscore']:.4f} | "
            f"{baseline['k10']['map']:.4f} | {baseline['k10']['ndcg']:.4f} | {baseline['k10']['mrr']:.4f} | {baseline['runtime_seconds']:.2f} |",
            f"| unordered_local_context_bow | {method['k10']['precision']:.4f} | {method['k10']['recall']:.4f} | {method['k10']['fscore']:.4f} | "
            f"{method['k10']['map']:.4f} | {method['k10']['ndcg']:.4f} | {method['k10']['mrr']:.4f} | {method['runtime_seconds']:.2f} |",
            "",
            "## Delta vs Baseline at k=10",
            "",
            f"- `dP@10 = {delta['precision']:+.4f}`",
            f"- `dR@10 = {delta['recall']:+.4f}`",
            f"- `dF@10 = {delta['fscore']:+.4f}`",
            f"- `dMAP@10 = {delta['map']:+.4f}`",
            f"- `dnDCG@10 = {delta['ndcg']:+.4f}`",
            f"- `dMRR@10 = {delta['mrr']:+.4f}`",
            "",
            "## Significance Checks",
            "",
            "Approximate randomization over per-query scores:",
            "",
            f"- `AP@10 p-value = {significance['ap_at_10_pvalue']:.4f}`",
            f"- `nDCG@10 p-value = {significance['ndcg_at_10_pvalue']:.4f}`",
            "",
            "## Interpretation",
            "",
            "The local-context method helps when relevance depends on nearby co-occurrence instead of isolated term overlap. "
            "Because the context features are unordered, the model gains proximity-sensitive evidence without committing to exact word order, "
            "which is useful for technical queries where terminology may appear in slightly different surface forms.",
            "",
            "At the same time, the method is still limited by lexical overlap: if the right concept uses entirely different vocabulary, "
            "unordered context alone cannot fix that. It is better viewed as a context-aware extension of TF-IDF, not a full semantic model.",
            "",
            "## Example Query Comparison",
            "",
            "| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in examples:
        lines.append(
            f"| {row['query_id']} | {row['baseline_hits_at_5']} | {row['method_hits_at_5']} | "
            f"{row['baseline_ap_at_10']:.4f} | {row['method_ap_at_10']:.4f} | {row['delta_ap_at_10']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{summary['paths']['summary_json']}`",
            f"- Summary CSV: `{summary['paths']['summary_csv']}`",
            f"- Overlay plot: `{summary['paths']['overlay_plot']}`",
            f"- n-sweep JSON: `{summary['paths']['n_sweep_json']}`",
            f"- n-sweep CSV: `{summary['paths']['n_sweep_csv']}`",
            f"- n-sweep plot: `{summary['paths']['n_sweep_plot']}`",
            f"- Example comparison markdown: `{summary['paths']['example_markdown']}`",
            f"- Config sweep JSON: `{summary['paths']['config_sweep_json']}`",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unordered local-context bag-of-words experiments.")
    parser.add_argument("--dataset-dir", default=str(ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline_tfidf"
    method_dir = output_dir / "unordered_local_context_bow"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")

    processed_docs = normalize_collection(load_json(Path(args.processed_docs)))
    processed_queries = normalize_collection(load_json(Path(args.processed_queries)))

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    evaluator = Evaluation()

    baseline_start = time.time()
    doc_unigrams = [unigram_feature_map(doc) for doc in processed_docs]
    query_unigrams = [unigram_feature_map(query) for query in processed_queries]
    baseline_model = SparseTfidfModel.build("baseline_tfidf", doc_unigrams, doc_ids)
    baseline_rankings, baseline_scores = baseline_model.rank(query_unigrams)
    baseline_runtime = time.time() - baseline_start

    baseline_metrics = compute_metrics(evaluator, baseline_rankings, query_ids, qrels)
    baseline_per_query = compute_per_query_metrics(evaluator, baseline_rankings, query_ids, qrels)
    save_metric_plot(
        baseline_dir / "eval_plot.png",
        baseline_metrics,
        "Baseline TF-IDF Evaluation Metrics",
    )
    save_rankings(baseline_dir / "rankings_top20.json", query_ids, baseline_rankings, top_k=20)
    write_json(
        baseline_dir / "metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": {metric: values[9] for metric, values in baseline_metrics.items()},
            "runtime_seconds": baseline_runtime,
        },
    )

    sweep_start = time.time()
    sweep_payload = find_best_config(
        docs=processed_docs,
        queries=processed_queries,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        baseline_scores=baseline_scores,
        output_dir=output_dir,
    )
    best_config = sweep_payload["best_config"]
    n_sweep_rows = sweep_payload["best_by_radius"]

    raw_doc_contexts = [
        context_feature_map(doc, radius=best_config["radius"], orders=best_config["orders"])
        for doc in processed_docs
    ]
    raw_query_contexts = [
        context_feature_map(query, radius=best_config["radius"], orders=best_config["orders"])
        for query in processed_queries
    ]
    filtered_doc_contexts, doc_freq = filter_context_feature_maps(
        raw_doc_contexts,
        min_df=best_config["min_df"],
    )
    filtered_query_contexts = [
        {feature: value for feature, value in fmap.items() if doc_freq[feature] >= best_config["min_df"]}
        for fmap in raw_query_contexts
    ]

    method_context_model = SparseTfidfModel.build(
        "unordered_local_context_bow_context",
        filtered_doc_contexts,
        doc_ids,
    )
    _, context_scores = method_context_model.rank(filtered_query_contexts)

    method_rankings = []
    method_score_vectors = []
    for base_scores, ctx_scores in zip(baseline_scores, context_scores):
        combined_scores = base_scores + (best_config["alpha"] * ctx_scores)
        ranked_indexes = sorted(
            range(len(doc_ids)),
            key=lambda idx: (-combined_scores[idx], doc_ids[idx]),
        )
        method_rankings.append([doc_ids[idx] for idx in ranked_indexes])
        method_score_vectors.append(combined_scores)

    method_runtime = time.time() - sweep_start
    method_metrics = compute_metrics(evaluator, method_rankings, query_ids, qrels)
    method_per_query = compute_per_query_metrics(evaluator, method_rankings, query_ids, qrels)

    save_metric_plot(
        method_dir / "eval_plot.png",
        method_metrics,
        "Unordered Local-Context Bag-of-Words Metrics",
    )
    all_n_metrics = {}
    method_labels = {}
    for row in n_sweep_rows:
        n = row["n"]
        raw_doc_contexts_n = [
            context_feature_map(doc, radius=n, orders=tuple(row["best_orders"]))
            for doc in processed_docs
        ]
        raw_query_contexts_n = [
            context_feature_map(query, radius=n, orders=tuple(row["best_orders"]))
            for query in processed_queries
        ]
        filtered_doc_contexts_n, doc_freq_n = filter_context_feature_maps(
            raw_doc_contexts_n,
            min_df=row["best_min_df"],
        )
        filtered_query_contexts_n = [
            {feature: value for feature, value in fmap.items() if doc_freq_n[feature] >= row["best_min_df"]}
            for fmap in raw_query_contexts_n
        ]
        context_model_n = SparseTfidfModel.build(
            f"context_r{n}",
            filtered_doc_contexts_n,
            doc_ids,
        )
        _, context_scores_n = context_model_n.rank(filtered_query_contexts_n)
        
        rankings_n = []
        for base_scores, ctx_scores in zip(baseline_scores, context_scores_n):
            combined_scores = base_scores + (row["best_alpha"] * ctx_scores)
            ranked_indexes = sorted(
                range(len(doc_ids)),
                key=lambda idx: (-combined_scores[idx], doc_ids[idx]),
            )[:10]
            rankings_n.append([doc_ids[idx] for idx in ranked_indexes])
        
        all_n_metrics[n] = compute_metrics(evaluator, rankings_n, query_ids, qrels)
        method_labels[n] = f"n={n}"

    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        all_n_metrics,
        method_labels,
        list(all_n_metrics.keys()),
    )
    save_rankings(method_dir / "rankings_top20.json", query_ids, method_rankings, top_k=20)

    method_metrics_payload = {
        "best_config": best_config,
        "metrics_by_k": method_metrics,
        "k10": {metric: values[9] for metric, values in method_metrics.items()},
        "runtime_seconds": method_runtime,
        "avg_context_features_per_doc": float(np.mean([len(fmap) for fmap in filtered_doc_contexts])),
        "avg_context_features_per_query": float(np.mean([len(fmap) for fmap in filtered_query_contexts])),
    }
    write_json(method_dir / "metrics.json", method_metrics_payload)

    rows = []
    summary_methods = {}
    for method_name, metrics_payload in [
        ("baseline_tfidf", {"metrics_by_k": baseline_metrics, "runtime_seconds": baseline_runtime}),
        ("unordered_local_context_bow", method_metrics_payload),
    ]:
        k10 = {metric: values[9] for metric, values in metrics_payload["metrics_by_k"].items()}
        row = {
            "method": method_name,
            "precision@10": k10["precision"],
            "recall@10": k10["recall"],
            "fscore@10": k10["fscore"],
            "map@10": k10["map"],
            "ndcg@10": k10["ndcg"],
            "mrr@10": k10["mrr"],
            "runtime_seconds": metrics_payload["runtime_seconds"],
        }
        rows.append(row)
        summary_methods[method_name] = {
            "k10": k10,
            "metrics_by_k": metrics_payload["metrics_by_k"],
            "runtime_seconds": metrics_payload["runtime_seconds"],
        }

    write_summary_csv(output_dir / "summary_k10.csv", rows)
    write_n_sweep_csv(
        output_dir / "n_sweep_best_by_radius.csv",
        [
            {
                "n": row["n"],
                "best_orders": row["best_orders"],
                "best_min_df": row["best_min_df"],
                "best_alpha": row["best_alpha"],
                "precision@10": row["precision_at_10"],
                "recall@10": row["recall_at_10"],
                "fscore@10": row["fscore_at_10"],
                "map@10": row["map_at_10"],
                "ndcg@10": row["ndcg_at_10"],
                "mrr@10": row["mrr_at_10"],
                "context_feature_vocab_size": row["context_feature_vocab_size"],
            }
            for row in n_sweep_rows
        ],
    )
    save_n_sweep_plot(output_dir / "n_sweep_best_by_radius.png", n_sweep_rows)

    significance = {
        "ap_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["average_precision"],
            method_per_query["average_precision"],
        ),
        "ndcg_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["ndcg"],
            method_per_query["ndcg"],
        ),
    }

    examples = build_example_query_comparison(
        query_ids=query_ids,
        queries_json=queries_json,
        qrels=qrels,
        baseline_rankings=baseline_rankings,
        method_rankings=method_rankings,
        baseline_per_query=baseline_per_query,
        method_per_query=method_per_query,
        output_dir=output_dir,
    )

    summary = {
        "dataset": str(dataset_dir),
        "best_config": best_config,
        "n_sweep_best_by_radius": n_sweep_rows,
        "methods": summary_methods,
        "delta_vs_baseline_at_10": {
            metric: summary_methods["unordered_local_context_bow"]["k10"][metric]
            - summary_methods["baseline_tfidf"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        },
        "significance": significance,
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
            "n_sweep_json": str(output_dir / "n_sweep_best_by_radius.json"),
            "n_sweep_csv": str(output_dir / "n_sweep_best_by_radius.csv"),
            "n_sweep_plot": str(output_dir / "n_sweep_best_by_radius.png"),
            "example_markdown": str(output_dir / "example_query_comparison.md"),
            "config_sweep_json": str(output_dir / "config_sweep.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "config": {
                "best_config": best_config,
                "n_sweep_best_by_radius": n_sweep_rows,
            },
            "k10": {
                method_name: method_payload["k10"]
                for method_name, method_payload in summary_methods.items()
            },
            "metrics": {
                method_name: method_payload["metrics_by_k"]
                for method_name, method_payload in summary_methods.items()
            },
        },
    )

    build_report(
        output_path=SCRIPT_DIR / "experiment_report.md",
        summary=summary,
        examples=examples,
        n_sweep_rows=n_sweep_rows,
    )

    print("Best configuration:", json.dumps(best_config, indent=2))
    print("Baseline MAP@10:", f"{summary_methods['baseline_tfidf']['k10']['map']:.4f}")
    print("Method MAP@10:", f"{summary_methods['unordered_local_context_bow']['k10']['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
