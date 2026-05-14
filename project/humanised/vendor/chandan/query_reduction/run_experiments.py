#!/usr/bin/env python3
"""
Run query reduction experiments for the Cranfield collection.

Methods:
1. idf_topk: keep the top-k highest-IDF query terms.
2. prf_term_pruning: use first-pass top documents to keep the best-supported original query terms.

This script writes all artifacts inside project/chandan/query_reduction only and
also creates a unified overlay against the previously completed
unordered_local_context_bow method by reading its summary file.
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
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CHANDAN_DIR = SCRIPT_DIR.parent
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
    doc_freq: Dict[str, int]
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

        return cls(
            name=name,
            doc_ids=doc_ids,
            idf=idf,
            doc_freq=dict(doc_freq),
            postings=dict(postings),
        )

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


def flatten_query(query: Sequence[Sequence[str]]) -> List[str]:
    return [token for sentence in query for token in sentence]


def rebuild_query_from_tokens(tokens: Sequence[str]) -> List[List[str]]:
    return [list(tokens)] if tokens else [[]]


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
    seed: int = 17,
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


def save_rankings(path: Path, query_ids: Sequence[int], rankings: Sequence[Sequence[int]], top_k: int = 20) -> None:
    payload = {str(query_id): ranked_docs[:top_k] for query_id, ranked_docs in zip(query_ids, rankings)}
    write_json(path, payload)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def keep_topk_tokens(
    tokens: Sequence[str],
    model: SparseTfidfModel,
    keep_k: int,
) -> List[str]:
    scored = []
    for index, token in enumerate(tokens):
        scored.append((model.idf.get(token, 0.0), index, token))
    scored.sort(key=lambda item: (-item[0], item[1]))
    keep_count = min(max(1, keep_k), len(tokens))
    selected = sorted(scored[:keep_count], key=lambda item: item[1])
    return [token for _, _, token in selected]


def keep_prf_tokens(
    tokens: Sequence[str],
    model: SparseTfidfModel,
    top_doc_ids: Sequence[int],
    doc_id_to_index: Dict[int, int],
    doc_unigrams: Sequence[Dict[str, float]],
    keep_k: int,
    alpha: float,
) -> List[str]:
    scored = []
    for index, token in enumerate(tokens):
        idf_score = model.idf.get(token, 0.0)
        support = 0.0
        for doc_id in top_doc_ids:
            doc_index = doc_id_to_index[doc_id]
            support += doc_unigrams[doc_index].get(token, 0.0) * idf_score
        if top_doc_ids:
            support /= len(top_doc_ids)
        final_score = alpha * idf_score + (1.0 - alpha) * support
        scored.append((final_score, idf_score, index, token))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    keep_count = min(max(1, keep_k), len(tokens))
    selected = sorted(scored[:keep_count], key=lambda item: item[2])
    return [token for _, _, _, token in selected]


def reduce_queries(
    queries: Sequence[Sequence[Sequence[str]]],
    reducer_name: str,
    reducer_value,
    model: SparseTfidfModel,
    baseline_rankings: Sequence[Sequence[int]] | None = None,
    doc_id_to_index: Dict[int, int] | None = None,
    doc_unigrams: Sequence[Dict[str, float]] | None = None,
) -> Tuple[List[List[List[str]]], List[Dict[str, object]]]:
    reduced_queries = []
    preview = []
    for query_index, query in enumerate(queries):
        tokens = flatten_query(query)
        if reducer_name == "idf_topk":
            reduced_tokens = keep_topk_tokens(tokens, model, keep_k=int(reducer_value))
        elif reducer_name == "prf_term_pruning":
            reduced_tokens = keep_prf_tokens(
                tokens=tokens,
                model=model,
                top_doc_ids=baseline_rankings[query_index][: int(reducer_value["top_docs"])],
                doc_id_to_index=doc_id_to_index,
                doc_unigrams=doc_unigrams,
                keep_k=int(reducer_value["keep_k"]),
                alpha=float(reducer_value["alpha"]),
            )
        else:
            raise ValueError(f"Unsupported reducer: {reducer_name}")
        reduced_queries.append(rebuild_query_from_tokens(reduced_tokens))
        preview.append(
            {
                "query_index": query_index + 1,
                "original_len": len(tokens),
                "reduced_len": len(reduced_tokens),
                "original_tokens": tokens,
                "reduced_tokens": reduced_tokens,
            }
        )
    return reduced_queries, preview


def run_method_sweep(
    method_name: str,
    candidate_values: Sequence,
    queries: Sequence[Sequence[Sequence[str]]],
    query_ids: Sequence[int],
    qrels,
    model: SparseTfidfModel,
    output_dir: Path,
    baseline_rankings: Sequence[Sequence[int]] | None = None,
    doc_id_to_index: Dict[int, int] | None = None,
    doc_unigrams: Sequence[Dict[str, float]] | None = None,
) -> Dict[str, object]:
    evaluator = Evaluation()
    sweep_rows = []
    for candidate_value in candidate_values:
        reduced_queries, preview = reduce_queries(
            queries,
            method_name,
            candidate_value,
            model,
            baseline_rankings=baseline_rankings,
            doc_id_to_index=doc_id_to_index,
            doc_unigrams=doc_unigrams,
        )
        query_maps = [unigram_feature_map(query) for query in reduced_queries]
        rankings, _ = model.rank(query_maps)
        metrics = compute_metrics(evaluator, rankings, query_ids, qrels)
        avg_length = float(np.mean([sum(len(sentence) for sentence in query) for query in reduced_queries]))
        sweep_rows.append(
            {
                "value": candidate_value,
                "metrics_by_k": metrics,
                "k10": {metric: values[9] for metric, values in metrics.items()},
                "avg_reduced_query_len": avg_length,
                "preview": preview[:8],
            }
        )

    sweep_rows.sort(
        key=lambda row: (
            row["k10"]["map"],
            row["k10"]["ndcg"],
            row["k10"]["mrr"],
            row["k10"]["precision"],
        ),
        reverse=True,
    )
    write_json(output_dir / "sweep_results.json", sweep_rows)
    return {
        "best": sweep_rows[0],
        "all": sweep_rows,
    }


def save_tuning_combo_overlay(
    path: Path,
    baseline_metrics: Dict[str, List[float]],
    sweep_rows: Sequence[Dict[str, object]],
    best_row: Dict[str, object],
    title: str,
) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    overlap_with_baseline = all(
        np.allclose(
            np.array(best_row["metrics_by_k"][metric_name], dtype=np.float64),
            np.array(baseline_metrics[metric_name], dtype=np.float64),
            atol=1e-12,
        )
        for metric_name in METRIC_LABELS
    )
    best_label = "best_tuned_config"
    if overlap_with_baseline:
        best_label = "best_tuned_config (overlaps baseline)"

    for axis, metric_name in zip(axes, METRIC_LABELS):
        for row in sweep_rows:
            axis.plot(
                ks,
                row["metrics_by_k"][metric_name],
                color="tab:blue",
                alpha=0.16,
                linewidth=1.0,
                zorder=1,
            )

        axis.plot(
            ks,
            best_row["metrics_by_k"][metric_name],
            marker="s",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.6,
            linewidth=2.2,
            color="tab:red",
            label=best_label,
            zorder=2,
        )

        axis.plot(
            ks,
            baseline_metrics[metric_name],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=2.8,
            color="black",
            linestyle="--",
            label="baseline_tfidf",
            zorder=3,
        )

        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.suptitle(title, y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_unified_overlay(
    path: Path,
    metrics_by_method: Dict[str, Dict[str, List[float]]],
    method_order: Sequence[str],
) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for axis, metric_name in zip(axes, METRIC_LABELS):
        for method_name in method_order:
            axis.plot(
                ks,
                metrics_by_method[method_name][metric_name],
                marker="o",
                linewidth=2,
                label=method_name,
            )
        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
    fig.suptitle("Unified Comparison: Baseline, Local Context, and Query Reduction", y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_example_comparison(
    query_ids: Sequence[int],
    queries_json,
    qrels,
    baseline_rankings: Sequence[Sequence[int]],
    method_rankings: Sequence[Sequence[int]],
    baseline_per_query: Dict[str, List[float]],
    method_per_query: Dict[str, List[float]],
    output_dir: Path,
    forced_query_ids: Sequence[int] = (40, 64, 81, 90),
    top_n: int = 8,
) -> List[Dict[str, object]]:
    query_lookup = {item["query number"]: item["query"].strip() for item in queries_json}
    candidates = []
    for idx, query_id in enumerate(query_ids):
        delta_ap = method_per_query["average_precision"][idx] - baseline_per_query["average_precision"][idx]
        delta_hits = method_per_query["hits_at_5"][idx] - baseline_per_query["hits_at_5"][idx]
        candidates.append((delta_ap, delta_hits, idx, query_id))
    candidates.sort(reverse=True)
    candidate_by_query = {query_id: item for item in candidates for query_id in [item[3]]}

    selected = []
    seen = set()
    for query_id in forced_query_ids:
        if query_id in candidate_by_query and query_id not in seen:
            selected.append(candidate_by_query[query_id])
            seen.add(query_id)
    for candidate in candidates:
        if len(selected) >= top_n:
            break
        if candidate[3] in seen:
            continue
        selected.append(candidate)
        seen.add(candidate[3])

    rows = []
    for delta_ap, delta_hits, idx, query_id in selected:
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
                "baseline_top5": baseline_rankings[idx][:5],
                "method_top5": method_rankings[idx][:5],
            }
        )
    write_json(output_dir / "example_query_comparison.json", rows)
    lines = [
        "# Query Reduction Example Comparison",
        "",
        "| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['baseline_hits_at_5']} | {row['method_hits_at_5']} | "
            f"{row['baseline_ap_at_10']:.4f} | {row['method_ap_at_10']:.4f} | {row['delta_ap_at_10']:.4f} |"
        )
    (output_dir / "example_query_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def build_report(
    output_path: Path,
    summary: Dict[str, object],
    examples: Sequence[Dict[str, object]],
    unified_methods: Sequence[str],
) -> None:
    baseline = summary["methods"]["baseline_tfidf"]
    best_method_name = summary["best_query_reduction_method"]
    best_method = summary["methods"][best_method_name]
    delta_map = summary["delta_best_method_vs_baseline_at_10"]["map"]
    if delta_map > 0.0:
        result_line = "- `result = best query-reduction variant is above baseline on MAP@10`"
    elif delta_map < 0.0:
        result_line = "- `result = best query-reduction variant is below baseline on MAP@10`"
    else:
        result_line = "- `result = best query-reduction variant matches baseline on MAP@10`"

    lines = [
        "# Query Reduction Experiment Report",
        "",
        "## Method Summary",
        "",
        "Query reduction removes noisy or overly broad terms before ranking. This is a lightweight, non-deep-learning "
        "way to lower query-time cost and sometimes reduce ambiguity by keeping only the most informative parts of the query.",
        "",
        "## Implemented Methods",
        "",
        "- `idf_topk`: keep the top-k highest-IDF query terms.",
        "- `prf_term_pruning`: run baseline retrieval once, score original query terms by how strongly they are supported in the top retrieved documents, then keep only the strongest original terms.",
        "",
        "## Best Query Reduction Method",
        "",
        f"- `method = {best_method_name}`",
        f"- `best_parameter = {best_method['best_parameter']}`",
        result_line,
        "",
        "## k=10 Comparison",
        "",
        "| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name in unified_methods:
        method = summary["methods"][method_name]
        lines.append(
            f"| {method_name} | {method['k10']['precision']:.4f} | {method['k10']['recall']:.4f} | "
            f"{method['k10']['fscore']:.4f} | {method['k10']['map']:.4f} | {method['k10']['ndcg']:.4f} | {method['k10']['mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best-Method Delta vs Baseline at k=10",
            "",
            f"- `dMAP@10 = {summary['delta_best_method_vs_baseline_at_10']['map']:+.4f}`",
            f"- `dnDCG@10 = {summary['delta_best_method_vs_baseline_at_10']['ndcg']:+.4f}`",
            f"- `dMRR@10 = {summary['delta_best_method_vs_baseline_at_10']['mrr']:+.4f}`",
            "",
            "The query-reduction variants reduce some noisy cases, but in Cranfield many long query terms are actually useful technical clues rather than filler. "
            "So aggressive pruning tends to remove signal along with noise, which hurts the final ranking. In the current `chandan` results, "
            "`unordered_local_context_bow` remains the strongest method overall.",
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
            "## Unified Overlay",
            "",
            "The unified overlay compares the baseline, the earlier unordered local-context method, and all query-reduction methods in one figure.",
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{summary['paths']['summary_json']}`",
            f"- Summary CSV: `{summary['paths']['summary_csv']}`",
            f"- Unified overlay: `{summary['paths']['unified_overlay']}`",
            f"- IDF top-k all-combinations overlay: `{summary['paths']['idf_topk_tuning_overlay']}`",
            f"- IDF top-k extended all-combinations overlay: `{summary['paths']['idf_topk_extended_tuning_overlay']}`",
            f"- PRF pruning all-combinations overlay: `{summary['paths']['prf_tuning_overlay']}`",
            f"- PRF pruning extended all-combinations overlay: `{summary['paths']['prf_extended_tuning_overlay']}`",
            f"- Comparison summary: `{summary['paths']['comparison_summary']}`",
            f"- Example comparison markdown: `{summary['paths']['example_markdown']}`",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query reduction experiments.")
    parser.add_argument("--dataset-dir", default=str(ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "baseline_tfidf"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    topk_legacy_dir = SCRIPT_DIR / "idf_topk"
    topk_extended_dir = SCRIPT_DIR / "idf_topk_extended"
    prf_legacy_dir = SCRIPT_DIR / "prf_term_pruning"
    prf_extended_dir = SCRIPT_DIR / "prf_term_pruning_extended"

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
    baseline_rankings, _ = baseline_model.rank(query_unigrams)
    baseline_runtime = time.time() - baseline_start
    doc_id_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    baseline_metrics = compute_metrics(evaluator, baseline_rankings, query_ids, qrels)
    baseline_per_query = compute_per_query_metrics(evaluator, baseline_rankings, query_ids, qrels)

    save_metric_plot(baseline_dir / "eval_plot.png", baseline_metrics, "Baseline TF-IDF Evaluation Metrics")
    save_rankings(baseline_dir / "rankings_top20.json", query_ids, baseline_rankings)
    write_json(
        baseline_dir / "metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": {metric: values[9] for metric, values in baseline_metrics.items()},
            "runtime_seconds": baseline_runtime,
        },
    )

    legacy_idf_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    extended_idf_candidates = sorted(set(legacy_idf_candidates + [22, 24, 26, 28, 30, 35, 40]))

    prf_legacy_candidates = [
        {"top_docs": top_docs, "keep_k": keep_k, "alpha": alpha}
        for top_docs in [3, 5, 8, 10]
        for keep_k in [3, 4, 5, 6]
        for alpha in [0.3, 0.5, 0.7]
    ]
    prf_extended_candidates = [
        {"top_docs": top_docs, "keep_k": keep_k, "alpha": alpha}
        for top_docs in [3, 5, 8, 10, 12, 15]
        for keep_k in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
        for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    ]

    method_specs = [
        {
            "summary_name": "idf_topk_legacy",
            "reducer_name": "idf_topk",
            "candidates": legacy_idf_candidates,
            "output_dir": topk_legacy_dir / "output",
            "readme_path": topk_legacy_dir / "README.md",
            "method_title": "IDF Top-k Query Reduction (legacy grid)",
        },
        {
            "summary_name": "idf_topk_extended",
            "reducer_name": "idf_topk",
            "candidates": extended_idf_candidates,
            "output_dir": topk_extended_dir / "output",
            "readme_path": topk_extended_dir / "README.md",
            "method_title": "IDF Top-k Query Reduction (extended grid)",
        },
        {
            "summary_name": "prf_term_pruning_legacy",
            "reducer_name": "prf_term_pruning",
            "candidates": prf_legacy_candidates,
            "output_dir": prf_legacy_dir / "output",
            "readme_path": prf_legacy_dir / "README.md",
            "method_title": "Pseudo-Relevance-Feedback Query Pruning (legacy grid)",
        },
        {
            "summary_name": "prf_term_pruning_extended",
            "reducer_name": "prf_term_pruning",
            "candidates": prf_extended_candidates,
            "output_dir": prf_extended_dir / "output",
            "readme_path": prf_extended_dir / "README.md",
            "method_title": "Pseudo-Relevance-Feedback Query Pruning (extended grid)",
        },
    ]
    for path in [baseline_dir] + [spec["output_dir"] for spec in method_specs]:
        path.mkdir(parents=True, exist_ok=True)

    summary_methods = {
        "baseline_tfidf": {
            "k10": {metric: values[9] for metric, values in baseline_metrics.items()},
            "metrics_by_k": baseline_metrics,
            "runtime_seconds": baseline_runtime,
        }
    }
    method_rankings_by_name = {"baseline_tfidf": baseline_rankings}
    method_per_query_by_name = {"baseline_tfidf": baseline_per_query}

    for spec in method_specs:
        method_name = spec["reducer_name"]
        summary_name = spec["summary_name"]
        start_time = time.time()
        sweep_payload = run_method_sweep(
            method_name=method_name,
            candidate_values=spec["candidates"],
            queries=processed_queries,
            query_ids=query_ids,
            qrels=qrels,
            model=baseline_model,
            output_dir=spec["output_dir"],
            baseline_rankings=baseline_rankings,
            doc_id_to_index=doc_id_to_index,
            doc_unigrams=doc_unigrams,
        )
        best = sweep_payload["best"]
        all_rows = sweep_payload["all"]
        reduced_queries, reduction_preview = reduce_queries(
            processed_queries,
            reducer_name=method_name,
            reducer_value=best["value"],
            model=baseline_model,
            baseline_rankings=baseline_rankings,
            doc_id_to_index=doc_id_to_index,
            doc_unigrams=doc_unigrams,
        )
        reduced_query_maps = [unigram_feature_map(query) for query in reduced_queries]
        rankings, _ = baseline_model.rank(reduced_query_maps)
        runtime = time.time() - start_time
        metrics = compute_metrics(evaluator, rankings, query_ids, qrels)
        per_query = compute_per_query_metrics(evaluator, rankings, query_ids, qrels)

        save_metric_plot(spec["output_dir"] / "eval_plot.png", metrics, spec["method_title"])
        save_tuning_combo_overlay(
            spec["output_dir"] / "all_tuned_combinations_overlay.png",
            baseline_metrics,
            all_rows,
            best,
            f"Baseline vs All Tuned Combinations ({summary_name})",
        )
        save_rankings(spec["output_dir"] / "rankings_top20.json", query_ids, rankings)
        write_json(
            spec["output_dir"] / "reduction_preview.json",
            reduction_preview[:20],
        )
        write_json(
            spec["output_dir"] / "metrics.json",
            {
                "best_parameter": best["value"],
                "metrics_by_k": metrics,
                "k10": {metric: values[9] for metric, values in metrics.items()},
                "runtime_seconds": runtime,
                "avg_reduced_query_len": best["avg_reduced_query_len"],
            },
        )
        summary_methods[summary_name] = {
            "best_parameter": best["value"],
            "k10": {metric: values[9] for metric, values in metrics.items()},
            "metrics_by_k": metrics,
            "runtime_seconds": runtime,
            "avg_reduced_query_len": best["avg_reduced_query_len"],
        }
        method_rankings_by_name[summary_name] = rankings
        method_per_query_by_name[summary_name] = per_query

        spec["readme_path"].write_text(
            "\n".join(
                [
                    f"# {spec['method_title']}",
                    "",
                    f"- Method key: `{summary_name}`",
                    f"- Candidate count: `{len(spec['candidates'])}`",
                    f"- Best parameter: `{best['value']}`",
                    f"- Avg reduced query length: `{best['avg_reduced_query_len']:.2f}`",
                    f"- MAP@10: `{summary_methods[summary_name]['k10']['map']:.4f}`",
                    f"- nDCG@10: `{summary_methods[summary_name]['k10']['ndcg']:.4f}`",
                    f"- MRR@10: `{summary_methods[summary_name]['k10']['mrr']:.4f}`",
                ]
            ),
            encoding="utf-8",
        )

    query_reduction_variant_names = [spec["summary_name"] for spec in method_specs]
    legacy_variant_names = [name for name in query_reduction_variant_names if name.endswith("_legacy")]
    extended_variant_names = [name for name in query_reduction_variant_names if name.endswith("_extended")]

    def method_rank_key(name: str) -> Tuple[float, float, float]:
        return (
            summary_methods[name]["k10"]["map"],
            summary_methods[name]["k10"]["ndcg"],
            summary_methods[name]["k10"]["mrr"],
        )

    best_legacy_query_reduction_method = max(legacy_variant_names, key=method_rank_key)
    best_extended_query_reduction_method = max(extended_variant_names, key=method_rank_key)
    best_query_reduction_method = max(query_reduction_variant_names, key=method_rank_key)

    baseline_map = summary_methods["baseline_tfidf"]["k10"]["map"]
    improved_candidates_vs_baseline = [
        name for name in query_reduction_variant_names if summary_methods[name]["k10"]["map"] > baseline_map
    ]
    best_improved_query_reduction_method = None
    if improved_candidates_vs_baseline:
        best_improved_query_reduction_method = max(improved_candidates_vs_baseline, key=method_rank_key)

    previous_method_path = CHANDAN_DIR / "unordered_local_context_bow" / "output" / "summary.json"
    previous_method_summary = load_json(previous_method_path)
    unified_methods = ["baseline_tfidf", "unordered_local_context_bow", *query_reduction_variant_names]
    unified_metrics_by_method = {
        "baseline_tfidf": summary_methods["baseline_tfidf"]["metrics_by_k"],
        "unordered_local_context_bow": previous_method_summary["methods"]["unordered_local_context_bow"]["metrics_by_k"],
        **{
            method_name: summary_methods[method_name]["metrics_by_k"]
            for method_name in query_reduction_variant_names
        },
    }
    save_unified_overlay(output_dir / "unified_eval_overlay.png", unified_metrics_by_method, unified_methods)

    summary_rows = []
    for method_name in unified_methods:
        if method_name == "unordered_local_context_bow":
            method_payload = previous_method_summary["methods"][method_name]
        else:
            method_payload = summary_methods[method_name]
        summary_rows.append(
            {
                "method": method_name,
                "precision@10": method_payload["k10"]["precision"],
                "recall@10": method_payload["k10"]["recall"],
                "fscore@10": method_payload["k10"]["fscore"],
                "map@10": method_payload["k10"]["map"],
                "ndcg@10": method_payload["k10"]["ndcg"],
                "mrr@10": method_payload["k10"]["mrr"],
                "runtime_seconds": method_payload.get("runtime_seconds", 0.0),
            }
        )
    write_summary_csv(output_dir / "summary_k10.csv", summary_rows)

    best_method_name = best_query_reduction_method
    examples = build_example_comparison(
        query_ids=query_ids,
        queries_json=queries_json,
        qrels=qrels,
        baseline_rankings=baseline_rankings,
        method_rankings=method_rankings_by_name[best_method_name],
        baseline_per_query=baseline_per_query,
        method_per_query=method_per_query_by_name[best_method_name],
        output_dir=output_dir,
    )

    significance = {
        "ap_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["average_precision"],
            method_per_query_by_name[best_method_name]["average_precision"],
        ),
        "ndcg_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["ndcg"],
            method_per_query_by_name[best_method_name]["ndcg"],
        ),
    }

    summary = {
        "dataset": str(dataset_dir),
        "best_query_reduction_method": best_query_reduction_method,
        "best_legacy_query_reduction_method": best_legacy_query_reduction_method,
        "best_extended_query_reduction_method": best_extended_query_reduction_method,
        "best_improved_query_reduction_method_vs_baseline": best_improved_query_reduction_method,
        "query_reduction_variants": query_reduction_variant_names,
        "methods": {
            **summary_methods,
            "unordered_local_context_bow": previous_method_summary["methods"]["unordered_local_context_bow"],
        },
        "significance_best_method_vs_baseline": significance,
        "delta_best_method_vs_baseline_at_10": {
            metric: summary_methods[best_query_reduction_method]["k10"][metric]
            - summary_methods["baseline_tfidf"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        },
        "delta_extended_vs_legacy_at_10": {
            "idf_topk": {
                metric: summary_methods["idf_topk_extended"]["k10"][metric]
                - summary_methods["idf_topk_legacy"]["k10"][metric]
                for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
            },
            "prf_term_pruning": {
                metric: summary_methods["prf_term_pruning_extended"]["k10"][metric]
                - summary_methods["prf_term_pruning_legacy"]["k10"][metric]
                for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
            },
        },
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "unified_overlay": str(output_dir / "unified_eval_overlay.png"),
            "idf_topk_tuning_overlay": str((topk_legacy_dir / "output") / "all_tuned_combinations_overlay.png"),
            "idf_topk_extended_tuning_overlay": str((topk_extended_dir / "output") / "all_tuned_combinations_overlay.png"),
            "prf_tuning_overlay": str((prf_legacy_dir / "output") / "all_tuned_combinations_overlay.png"),
            "prf_extended_tuning_overlay": str((prf_extended_dir / "output") / "all_tuned_combinations_overlay.png"),
            "comparison_summary": str(output_dir / "comparison_summary.json"),
            "example_markdown": str(output_dir / "example_query_comparison.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "best_query_reduction_method": best_query_reduction_method,
            "k10": {
                method_name: (
                    previous_method_summary["methods"][method_name]["k10"]
                    if method_name == "unordered_local_context_bow"
                    else summary_methods[method_name]["k10"]
                )
                for method_name in unified_methods
            },
            "metrics": unified_metrics_by_method,
        },
    )

    build_report(
        output_path=SCRIPT_DIR / "experiment_report.md",
        summary=summary,
        examples=examples,
        unified_methods=unified_methods,
    )

    (SCRIPT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Query Reduction",
                "",
                "This folder contains query-side reduction methods only.",
                "",
                "Implemented method folders:",
                "- `idf_topk`",
                "- `idf_topk_extended`",
                "- `prf_term_pruning`",
                "- `prf_term_pruning_extended`",
                "",
                "Main runner:",
                "- `run_experiments.py`",
                "",
                "Unified comparison output:",
                "- `output/unified_eval_overlay.png`",
            ]
        ),
        encoding="utf-8",
    )

    print("Best legacy query reduction method:", best_legacy_query_reduction_method)
    print("Best extended query reduction method:", best_extended_query_reduction_method)
    print("Best overall query reduction method:", best_query_reduction_method)
    if best_improved_query_reduction_method:
        print("Best method improving baseline:", best_improved_query_reduction_method)
    else:
        print("No query-reduction variant beat baseline MAP@10 in this run.")
    print("Best MAP@10:", f"{summary_methods[best_query_reduction_method]['k10']['map']:.4f}")
    print("Unified overlay written to:", output_dir / "unified_eval_overlay.png")


if __name__ == "__main__":
    main()
