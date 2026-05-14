#!/usr/bin/env python3
"""
Tune n-gram retrieval on Cranfield and compare every tuned setting against a
baseline unigram TF-IDF run.

Usage (from repo root):
    python project/ritisha/ngram/run_ngram.py \
        --dataset cranfield/ \
        --out-folder project/ritisha/ngram/output_ngram/ \
        --n-values 2,3,4,5,6,7,8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


ASSIGNMENT_ROOT = Path(__file__).resolve().parents[3]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from evaluation import Evaluation  # noqa: E402
from inflectionReduction import InflectionReduction  # noqa: E402
from sentenceSegmentation import SentenceSegmentation  # noqa: E402
from stopwordRemoval import StopwordRemoval  # noqa: E402
from tokenization import Tokenization  # noqa: E402

from ngram_retrieval import NgramRetrieval


METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F-score",
    "map": "MAP",
    "ndcg": "nDCG",
    "mrr": "MRR",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def preprocess(texts: Sequence[str], segmenter_name: str, tokenizer_name: str) -> List[List[List[str]]]:
    segmenter = SentenceSegmentation()
    tokenizer = Tokenization()
    reducer = InflectionReduction()
    sw_remover = StopwordRemoval()

    processed = []
    for text in texts:
        if segmenter_name == "punkt":
            segmented = segmenter.punkt(text)
        else:
            segmented = segmenter.naive(text)

        if tokenizer_name == "ptb":
            tokenized = tokenizer.pennTreeBank(segmented)
        else:
            tokenized = tokenizer.naive(segmented)

        reduced = reducer.reduce(tokenized)
        cleaned = sw_remover.fromList(reduced)
        processed.append(cleaned)

    return processed


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


def save_consistent_overlay(
    path: Path,
    baseline_metrics: Dict[str, List[float]],
    tuned_metrics: Sequence[Dict[str, List[float]]],
    tuned_labels: Sequence[str],
    title: str,
) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(tuned_metrics))))
    for axis, metric_name in zip(axes, METRIC_LABELS):
        for idx, metrics in enumerate(tuned_metrics):
            axis.plot(
                ks,
                metrics[metric_name],
                marker="s",
                linewidth=1.8,
                color=colors[idx],
                label=tuned_labels[idx],
                zorder=1,
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
            zorder=2,
        )
        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.suptitle(title, y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_all_tuned_combinations_overlay(
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
                alpha=0.13,
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


def save_param_sweep_plot(path: Path, rows: Sequence[Dict[str, object]], param_name: str) -> None:
    param_values = [row[param_name] for row in rows]
    metric_keys = ["precision_at_10", "recall_at_10", "fscore_at_10", "map_at_10", "ndcg_at_10", "mrr_at_10"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for axis, metric_key in zip(axes, metric_keys):
        axis.plot(param_values, [row[metric_key] for row in rows], marker="o", linewidth=2)
        axis.set_title(metric_key.replace("_at_10", "").upper() + "@10")
        axis.set_xlabel(param_name)
        axis.set_ylabel("Score")
        axis.set_xticks(param_values)
        axis.grid(alpha=0.25)
    fig.suptitle(f"{param_name} Tuning Sweep", y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def write_n_sweep_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "n",
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
            writer.writerow(
                {
                    "n": row["n"],
                    "precision@10": row["precision_at_10"],
                    "recall@10": row["recall_at_10"],
                    "fscore@10": row["fscore_at_10"],
                    "map@10": row["map_at_10"],
                    "ndcg@10": row["ndcg_at_10"],
                    "mrr@10": row["mrr_at_10"],
                    "runtime_seconds": row["runtime_seconds"],
                }
            )


def run_retrieval(
    n_value: int,
    processed_docs: Sequence[Sequence[Sequence[str]]],
    processed_queries: Sequence[Sequence[Sequence[str]]],
    doc_ids: Sequence[int],
) -> Dict[str, object]:
    retriever = NgramRetrieval(n=n_value)
    start_time = time.time()
    retriever.buildIndex(processed_docs, doc_ids)
    rankings = retriever.rank(processed_queries)
    runtime = time.time() - start_time
    return {
        "rankings": rankings,
        "runtime_seconds": runtime,
    }


def build_report(output_path: Path, summary: Dict[str, object], n_sweep_rows: Sequence[Dict[str, object]]) -> None:
    baseline = summary["methods"]["baseline_tfidf"]
    best_method = summary["methods"]["ngram_best"]
    lines = [
        "# N-gram Tuning Report",
        "",
        "## Method Summary",
        "",
        "The n-gram model augments unigram TF-IDF with contiguous multi-word tokens. "
        "This sweep tunes the maximum n and compares each tuned configuration against the baseline unigram TF-IDF run.",
        "",
        "## Tuning Grid",
        "",
        f"- `tested_n_values = {summary['tested_n_values']}`",
        "- ranking uses cosine similarity over TF-IDF vectors",
        "",
        "## Best Configuration",
        "",
        f"- `best_n = {summary['best_n']}`",
        f"- `MAP@10 = {best_method['k10']['map']:.4f}`",
        f"- `nDCG@10 = {best_method['k10']['ndcg']:.4f}`",
        f"- `MRR@10 = {best_method['k10']['mrr']:.4f}`",
        "",
        "## n Sweep (k=10)",
        "",
        "| n | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in n_sweep_rows:
        lines.append(
            f"| {row['n']} | {row['precision_at_10']:.4f} | {row['recall_at_10']:.4f} | {row['fscore_at_10']:.4f} | "
            f"{row['map_at_10']:.4f} | {row['ndcg_at_10']:.4f} | {row['mrr_at_10']:.4f} | {row['runtime_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline vs Best at k=10",
            "",
            "| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| baseline_tfidf | {baseline['k10']['precision']:.4f} | {baseline['k10']['recall']:.4f} | {baseline['k10']['fscore']:.4f} | "
            f"{baseline['k10']['map']:.4f} | {baseline['k10']['ndcg']:.4f} | {baseline['k10']['mrr']:.4f} |",
            f"| ngram_best (n={summary['best_n']}) | {best_method['k10']['precision']:.4f} | {best_method['k10']['recall']:.4f} | "
            f"{best_method['k10']['fscore']:.4f} | {best_method['k10']['map']:.4f} | {best_method['k10']['ndcg']:.4f} | {best_method['k10']['mrr']:.4f} |",
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{summary['paths']['summary_json']}`",
            f"- Summary CSV: `{summary['paths']['summary_csv']}`",
            f"- n sweep JSON: `{summary['paths']['n_sweep_json']}`",
            f"- n sweep CSV: `{summary['paths']['n_sweep_csv']}`",
            f"- Overlay plot: `{summary['paths']['overlay_plot']}`",
            f"- All tuned combinations overlay: `{summary['paths']['all_configs_overlay']}`",
            f"- n sweep plot: `{summary['paths']['n_sweep_plot']}`",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_int_list(raw: str) -> List[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune n-gram retrieval on Cranfield")
    parser.add_argument("--dataset", default=str(ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--out-folder", default=str(Path(__file__).resolve().parent / "output_ngram"))
    parser.add_argument("--segmenter", default="punkt", choices=["punkt", "naive"])
    parser.add_argument("--tokenizer", default="ptb", choices=["ptb", "naive"])
    parser.add_argument("--n-values", default="2,3,4,5,6,7,8")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    docs = [item["body"] for item in docs_json]
    queries = [item["query"] for item in queries_json]

    print("Preprocessing documents...")
    processed_docs = preprocess(docs, args.segmenter, args.tokenizer)
    print("Preprocessing queries...")
    processed_queries = preprocess(queries, args.segmenter, args.tokenizer)

    evaluator = Evaluation()

    baseline_dir = output_dir / "baseline_tfidf"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_run = run_retrieval(1, processed_docs, processed_queries, doc_ids)
    baseline_metrics = compute_metrics(evaluator, baseline_run["rankings"], query_ids, qrels)
    save_metric_plot(baseline_dir / "eval_plot.png", baseline_metrics, "Baseline TF-IDF Metrics")
    save_rankings(baseline_dir / "rankings_top20.json", query_ids, baseline_run["rankings"], top_k=20)
    write_json(
        baseline_dir / "metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": {metric: values[9] for metric, values in baseline_metrics.items()},
            "runtime_seconds": baseline_run["runtime_seconds"],
            "n": 1,
        },
    )

    n_values = parse_int_list(args.n_values)
    if any(n < 2 for n in n_values):
        raise ValueError("--n-values must be >= 2 because n=1 is reserved for baseline TF-IDF.")

    sweep_rows = []
    tuned_metrics = []
    tuned_labels = []
    for n_value in n_values:
        print(f"Running n-gram sweep for n={n_value}...")
        run_payload = run_retrieval(n_value, processed_docs, processed_queries, doc_ids)
        metrics = compute_metrics(evaluator, run_payload["rankings"], query_ids, qrels)

        method_dir = output_dir / f"ngram_n{n_value}"
        method_dir.mkdir(parents=True, exist_ok=True)
        save_metric_plot(method_dir / "eval_plot.png", metrics, f"N-gram Metrics (n={n_value})")
        save_rankings(method_dir / "rankings_top20.json", query_ids, run_payload["rankings"], top_k=20)
        write_json(
            method_dir / "metrics.json",
            {
                "metrics_by_k": metrics,
                "k10": {metric: values[9] for metric, values in metrics.items()},
                "runtime_seconds": run_payload["runtime_seconds"],
                "n": n_value,
            },
        )

        sweep_row = {
            "n": n_value,
            "precision_at_10": metrics["precision"][9],
            "recall_at_10": metrics["recall"][9],
            "fscore_at_10": metrics["fscore"][9],
            "map_at_10": metrics["map"][9],
            "ndcg_at_10": metrics["ndcg"][9],
            "mrr_at_10": metrics["mrr"][9],
            "runtime_seconds": run_payload["runtime_seconds"],
            "metrics_by_k": metrics,
            "rankings": run_payload["rankings"],
        }
        sweep_rows.append(sweep_row)
        tuned_metrics.append(metrics)
        tuned_labels.append(f"ngram_n{n_value}")

    sweep_rows.sort(
        key=lambda row: (
            row["map_at_10"],
            row["ndcg_at_10"],
            row["mrr_at_10"],
            row["precision_at_10"],
        ),
        reverse=True,
    )
    best_row = sweep_rows[0]

    save_consistent_overlay(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        tuned_metrics,
        tuned_labels,
        "Baseline vs Tuned N-gram Combinations",
    )
    save_all_tuned_combinations_overlay(
        output_dir / "all_tuned_combinations_overlay.png",
        baseline_metrics,
        sweep_rows,
        best_row,
        "Baseline vs All Tuned N-gram Combinations",
    )
    save_param_sweep_plot(output_dir / "n_sweep_k10.png", sweep_rows, "n")

    write_json(
        output_dir / "n_sweep_results.json",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"rankings"}
            }
            for row in sweep_rows
        ],
    )
    write_n_sweep_csv(output_dir / "n_sweep_results.csv", sweep_rows)

    summary_methods = {
        "baseline_tfidf": {
            "k10": {metric: values[9] for metric, values in baseline_metrics.items()},
            "metrics_by_k": baseline_metrics,
            "runtime_seconds": baseline_run["runtime_seconds"],
        },
        "ngram_best": {
            "best_n": best_row["n"],
            "k10": {
                "precision": best_row["precision_at_10"],
                "recall": best_row["recall_at_10"],
                "fscore": best_row["fscore_at_10"],
                "map": best_row["map_at_10"],
                "ndcg": best_row["ndcg_at_10"],
                "mrr": best_row["mrr_at_10"],
            },
            "metrics_by_k": best_row["metrics_by_k"],
            "runtime_seconds": best_row["runtime_seconds"],
        },
    }

    summary_rows = [
        {
            "method": "baseline_tfidf",
            "precision@10": summary_methods["baseline_tfidf"]["k10"]["precision"],
            "recall@10": summary_methods["baseline_tfidf"]["k10"]["recall"],
            "fscore@10": summary_methods["baseline_tfidf"]["k10"]["fscore"],
            "map@10": summary_methods["baseline_tfidf"]["k10"]["map"],
            "ndcg@10": summary_methods["baseline_tfidf"]["k10"]["ndcg"],
            "mrr@10": summary_methods["baseline_tfidf"]["k10"]["mrr"],
            "runtime_seconds": summary_methods["baseline_tfidf"]["runtime_seconds"],
        }
    ]
    for row in sorted(sweep_rows, key=lambda item: item["n"]):
        summary_rows.append(
            {
                "method": f"ngram_n{row['n']}",
                "precision@10": row["precision_at_10"],
                "recall@10": row["recall_at_10"],
                "fscore@10": row["fscore_at_10"],
                "map@10": row["map_at_10"],
                "ndcg@10": row["ndcg_at_10"],
                "mrr@10": row["mrr_at_10"],
                "runtime_seconds": row["runtime_seconds"],
            }
        )
    write_summary_csv(output_dir / "summary_k10.csv", summary_rows)

    summary = {
        "dataset": str(dataset_dir),
        "best_n": best_row["n"],
        "tested_n_values": sorted(n_values),
        "methods": summary_methods,
        "delta_best_vs_baseline_at_10": {
            metric: summary_methods["ngram_best"]["k10"][metric] - summary_methods["baseline_tfidf"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        },
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "n_sweep_json": str(output_dir / "n_sweep_results.json"),
            "n_sweep_csv": str(output_dir / "n_sweep_results.csv"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
            "all_configs_overlay": str(output_dir / "all_tuned_combinations_overlay.png"),
            "n_sweep_plot": str(output_dir / "n_sweep_k10.png"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    build_report(SCRIPT_DIR / "experiment_report.md", summary, sorted(sweep_rows, key=lambda item: item["n"]))

    print("Best n:", best_row["n"])
    print("Baseline MAP@10:", f"{summary_methods['baseline_tfidf']['k10']['map']:.4f}")
    print("Best n-gram MAP@10:", f"{summary_methods['ngram_best']['k10']['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    main()