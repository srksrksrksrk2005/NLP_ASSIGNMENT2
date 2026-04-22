import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from core.data import get_docs_ids_and_texts, get_query_ids_and_texts, load_cranfield_dataset
from core.evaluation import CranfieldEvaluation
from core.preprocessing import Preprocessor
from core.retrieval import VectorSpaceRetrieval, flatten_query_tokens
from expansion.base import MatrixQueryExpander
from expansion.embedding_matrices import (
    build_esa_neighbor_map,
    build_lsa_neighbor_map,
    build_tfidf_neighbor_map,
    build_word2vec_neighbor_map,
)
from expansion.wordnet_matrix import WordNetOOVResolver, build_wordnet_neighbor_map


BASELINE_METHOD = "baseline_tfidf"
RUN_TIMELINE_FILE = "run_timeline.json"

CUSTOM_QUERY_CASES = [
    {
        "id": "custom_slip_flow",
        "title": "slip-flow heat transfer in internal channels",
        "query": "slip-flow heat transfer in internal channels",
        "closest_query": "9",
    },
    {
        "id": "custom_hypersonic_transition",
        "title": "transition detection in hypersonic wakes behind slender bodies",
        "query": "transition detection in hypersonic wakes behind slender bodies",
        "closest_query": "40",
    },
    {
        "id": "custom_flutter_shapes",
        "title": "replace vibrational shapes with static deflection shapes for flutter prediction",
        "query": "replace vibrational shapes with static deflection shapes for flutter prediction",
        "closest_query": "64",
    },
    {
        "id": "custom_shock_separation",
        "title": "shock-induced boundary-layer separation",
        "query": "shock-induced boundary-layer separation",
        "closest_query": "90",
    },
    {
        "id": "custom_oov_liftbody",
        "title": "what corrections are needed for a liftbody in a propwash flowfield inside a test duct",
        "query": "what corrections are needed for a liftbody in a propwash flowfield inside a test duct",
        "closest_query": "81",
    },
]

DATASET_CASE_QUERY_IDS = ["9", "39", "40", "51", "64", "81", "90"]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[3]
    default_dataset = base_dir / "cranfield"
    default_output = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(description="Query replacement + expansion experiments")
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to cranfield directory")
    parser.add_argument("--output", default=str(default_output), help="Directory to store outputs")

    parser.add_argument(
        "--methods",
        default="baseline_tfidf,wordnet,embedding_tfidf,embedding_lsa,embedding_esa,embedding_word2vec",
        help=(
            "Comma-separated methods from: baseline_tfidf, wordnet, embedding_tfidf, "
            "embedding_lsa, embedding_esa, embedding_word2vec"
        ),
    )

    parser.add_argument("--top-k-neighbors", type=int, default=10)
    parser.add_argument("--min-similarity", type=float, default=0.08)

    parser.add_argument("--self-weight", type=float, default=1.0)
    parser.add_argument("--expansion-weight", type=float, default=0.20)
    parser.add_argument("--replacement-weight", type=float, default=0.85)
    parser.add_argument("--replacement-expansion-weight", type=float, default=0.15)
    parser.add_argument("--max-oov-candidates", type=int, default=5)

    parser.add_argument(
        "--disable-mean-sim-threshold",
        action="store_true",
        help="Disable adaptive mean-similarity thresholding in expansion.",
    )
    parser.add_argument("--mean-sim-factor", type=float, default=1.0)
    parser.add_argument(
        "--disable-quantile-sim-threshold",
        action="store_true",
        help="Disable adaptive quantile-based similarity thresholding in expansion.",
    )
    parser.add_argument(
        "--similarity-quantile",
        type=float,
        default=0.60,
        help="Quantile used for distribution-aware similarity thresholding (0..1).",
    )
    parser.add_argument(
        "--disable-neighbor-mass-normalization",
        action="store_true",
        help="Disable normalized mass allocation across neighbors.",
    )
    parser.add_argument(
        "--disable-similarity-scaling",
        action="store_true",
        help="Disable similarity scaling prior to mass allocation.",
    )
    parser.add_argument(
        "--similarity-scale-mode",
        choices=["minmax", "raw"],
        default="minmax",
        help="How to scale similarities before weighting.",
    )
    parser.add_argument(
        "--similarity-scale-epsilon",
        type=float,
        default=1e-9,
        help="Numerical epsilon for similarity scaling denominator.",
    )
    parser.add_argument("--similarity-power", type=float, default=1.0)
    parser.add_argument("--trial-label", default="", help="Optional label for run timeline/reporting")

    parser.add_argument("--lsa-components", type=int, default=128)
    parser.add_argument("--esa-top-concepts", type=int, default=100)

    parser.add_argument("--w2v-vector-size", type=int, default=100)
    parser.add_argument("--w2v-window", type=int, default=5)
    parser.add_argument("--w2v-min-count", type=int, default=1)
    parser.add_argument("--w2v-workers", type=int, default=1)
    parser.add_argument("--w2v-sg", type=int, default=1)
    parser.add_argument("--w2v-epochs", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print periodic live logs every N processed terms where applicable",
    )

    return parser.parse_args()


def _baseline_query_vectors(
    processed_queries: List[List[List[str]]],
    term_to_idx: Dict[str, int],
    vocab_size: int,
    show_progress: bool = True,
    logger: Callable[[str], None] | None = None,
    log_every: int = 25,
) -> np.ndarray:
    vectors = np.zeros((len(processed_queries), vocab_size), dtype=np.float64)
    total_queries = len(processed_queries)
    started = time.time()

    if logger is not None:
        logger(f"Building baseline query vectors for {total_queries} queries")

    query_iter = tqdm(
        processed_queries,
        desc="Baseline query vectors",
        unit="query",
        disable=not show_progress,
    )

    for idx, query in enumerate(query_iter):
        counts = Counter(flatten_query_tokens(query))
        for term, value in counts.items():
            term_idx = term_to_idx.get(term)
            if term_idx is not None:
                vectors[idx, term_idx] += float(value)

        done = idx + 1
        if logger is not None and (done == 1 or done % max(1, log_every) == 0 or done == total_queries):
            logger(f"Baseline vectors progress: {done}/{total_queries} queries (elapsed {time.time() - started:.1f}s)")

    return vectors


def _query_vectors_for_expander(
    processed_queries: List[List[List[str]]],
    expander: MatrixQueryExpander,
    show_progress: bool = True,
    logger: Callable[[str], None] | None = None,
    log_every: int = 25,
) -> np.ndarray:
    vectors = np.zeros((len(processed_queries), len(expander.vocab)), dtype=np.float64)
    total_queries = len(processed_queries)
    start = time.time()
    if logger is not None:
        logger(f"Starting query expansion for {total_queries} queries")

    query_iter = tqdm(
        processed_queries,
        desc="Expanding queries",
        unit="query",
        disable=not show_progress,
    )
    for idx, query in enumerate(query_iter):
        counts = Counter(flatten_query_tokens(query))
        vectors[idx] = expander.build_query_vector_from_counts(counts)
        done = idx + 1
        if logger is not None and (done == 1 or done % max(1, log_every) == 0 or done == total_queries):
            elapsed = time.time() - start
            logger(f"Expanded queries: {done}/{total_queries} (elapsed {elapsed:.1f}s)")

    if logger is not None:
        logger(f"Finished query expansion in {time.time() - start:.2f}s")
    return vectors


def _preview_query_expansion(
    query_id: str,
    query_tokens: List[List[str]],
    vector: np.ndarray,
    vocab: List[str],
    top_n: int = 12,
) -> Dict:
    token_counts = Counter(flatten_query_tokens(query_tokens))
    nonzero = np.where(vector > 0)[0]
    top_indices = sorted(nonzero, key=lambda i: vector[i], reverse=True)[:top_n]
    top_terms = [{"term": vocab[i], "weight": float(vector[i])} for i in top_indices]
    return {
        "query_id": str(query_id),
        "original_terms": dict(token_counts),
        "expanded_top_terms": top_terms,
    }


def _metric_series(metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[float]]:
    ks = sorted(int(k) for k in metrics.keys())
    return {
        "Precision": [metrics[str(k)]["precision"] for k in ks],
        "Recall": [metrics[str(k)]["recall"] for k in ks],
        "F-Score": [metrics[str(k)]["fscore"] for k in ks],
        "MAP": [metrics[str(k)]["map"] for k in ks],
        "nDCG": [metrics[str(k)]["ndcg"] for k in ks],
        "MRR": [metrics[str(k)]["mrr"] for k in ks],
    }


def _plot_method_metrics(
    method_name: str,
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    baseline_metrics: Dict[str, Dict[str, float]] | None = None,
    baseline_name: str = BASELINE_METHOD,
) -> None:
    ks = sorted(int(k) for k in metrics.keys())
    series = _metric_series(metrics)
    baseline_series = _metric_series(baseline_metrics) if baseline_metrics is not None else None

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(ks, values, marker="o", linewidth=2.2, label=f"{method_name} {label}")

    if baseline_series is not None and method_name != baseline_name:
        for label, values in baseline_series.items():
            plt.plot(ks, values, linewidth=1.6, linestyle="--", alpha=0.8, label=f"{baseline_name} {label}")

    plt.title(f"Evaluation Metrics - {method_name}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_overlay_metrics(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    baseline_name: str = BASELINE_METHOD,
) -> None:
    if not all_results:
        return

    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
    pretty_names = {
        "precision": "Precision",
        "recall": "Recall",
        "fscore": "F-Score",
        "map": "MAP",
        "ndcg": "nDCG",
        "mrr": "MRR",
    }
    ks = sorted(int(k) for k in next(iter(all_results.values())).keys())

    methods = list(all_results.keys())
    if baseline_name in methods:
        methods = [baseline_name] + [m for m in methods if m != baseline_name]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for axis, metric_name in zip(axes, metric_names):
        for method in methods:
            values = [all_results[method][str(k)][metric_name] for k in ks]
            if method == baseline_name:
                axis.plot(ks, values, marker="o", linewidth=3, linestyle="--", color="black", label=method)
            else:
                axis.plot(ks, values, marker="o", linewidth=1.8, label=method)
        axis.set_title(pretty_names[metric_name])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("k")
        axis.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cranfield Method Comparison (Baseline vs Expanded Queries)", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(3, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_relevant_lookup(qrels: List[Dict]) -> Dict[str, set]:
    lookup: Dict[str, set] = defaultdict(set)
    for item in qrels:
        lookup[str(item["query_num"])].add(str(item["id"]))
    return lookup


def _top_k_hit_count(ranked_docs: Sequence[str], relevant_docs: set, k: int = 5) -> int:
    return sum(1 for doc_id in ranked_docs[:k] if str(doc_id) in relevant_docs)


def _write_example_comparison(
    output_dir: Path,
    methods: List[str],
    qrels: List[Dict],
    query_text_lookup: Dict[str, str],
    dataset_rankings_by_method: Dict[str, Dict[str, List[str]]],
    custom_rankings_by_method: Dict[str, Dict[str, List[str]]],
    baseline_name: str = BASELINE_METHOD,
) -> Dict[str, List[Dict]]:
    relevant_lookup = _build_relevant_lookup(qrels)

    dataset_rows: List[Dict] = []
    for query_id in DATASET_CASE_QUERY_IDS:
        relevant = relevant_lookup.get(query_id, set())
        for method in methods:
            ranked = dataset_rankings_by_method.get(method, {}).get(query_id, [])
            dataset_rows.append(
                {
                    "case_type": "dataset",
                    "query_id": query_id,
                    "query_text": query_text_lookup.get(query_id, ""),
                    "method": method,
                    "top5_docs": ranked[:5],
                    "hits_at_5": _top_k_hit_count(ranked, relevant, k=5),
                    "num_relevant_total": len(relevant),
                }
            )

    custom_rows: List[Dict] = []
    for case in CUSTOM_QUERY_CASES:
        mapped_query_id = str(case["closest_query"])
        relevant = relevant_lookup.get(mapped_query_id, set())
        for method in methods:
            ranked = custom_rankings_by_method.get(method, {}).get(case["id"], [])
            custom_rows.append(
                {
                    "case_type": "custom",
                    "case_id": case["id"],
                    "case_title": case["title"],
                    "query_text": case["query"],
                    "mapped_query_id": mapped_query_id,
                    "method": method,
                    "top5_docs": ranked[:5],
                    "hits_at_5": _top_k_hit_count(ranked, relevant, k=5),
                    "num_relevant_total": len(relevant),
                }
            )

    comparison = {
        "dataset_cases": dataset_rows,
        "custom_cases": custom_rows,
    }

    (output_dir / "example_query_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    dataset_summary = []
    for query_id in DATASET_CASE_QUERY_IDS:
        case_rows = [row for row in dataset_rows if row["query_id"] == query_id]
        best_row = max(case_rows, key=lambda row: row["hits_at_5"])
        baseline_hits = 0
        for row in case_rows:
            if row["method"] == baseline_name:
                baseline_hits = row["hits_at_5"]
                break
        dataset_summary.append(
            {
                "query_id": query_id,
                "query_text": query_text_lookup.get(query_id, ""),
                "baseline_hits": baseline_hits,
                "best_method": best_row["method"],
                "best_hits": best_row["hits_at_5"],
                "delta_hits": best_row["hits_at_5"] - baseline_hits,
            }
        )

    custom_summary = []
    for case in CUSTOM_QUERY_CASES:
        case_rows = [row for row in custom_rows if row["case_id"] == case["id"]]
        best_row = max(case_rows, key=lambda row: row["hits_at_5"])
        baseline_hits = 0
        for row in case_rows:
            if row["method"] == baseline_name:
                baseline_hits = row["hits_at_5"]
                break
        custom_summary.append(
            {
                "case_id": case["id"],
                "case_title": case["title"],
                "mapped_query_id": case["closest_query"],
                "baseline_hits": baseline_hits,
                "best_method": best_row["method"],
                "best_hits": best_row["hits_at_5"],
                "delta_hits": best_row["hits_at_5"] - baseline_hits,
            }
        )

    lines = []
    lines.append("# Example Query Comparison")
    lines.append("")
    lines.append("## Dataset Query Cases")
    lines.append("")
    lines.append("| Query ID | Method | Hits@5 | Top-5 Docs |")
    lines.append("| --- | --- | ---: | --- |")
    for query_id in DATASET_CASE_QUERY_IDS:
        rows = [row for row in dataset_rows if row["query_id"] == query_id]
        for row in rows:
            lines.append(
                f"| {query_id} | {row['method']} | {row['hits_at_5']} | {', '.join(row['top5_docs'])} |"
            )

    lines.append("")
    lines.append("## Custom Query Cases (mapped to closest Cranfield query)")
    lines.append("")
    lines.append("| Case | Mapped Query | Method | Hits@5 | Top-5 Docs |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for case in CUSTOM_QUERY_CASES:
        rows = [row for row in custom_rows if row["case_id"] == case["id"]]
        for row in rows:
            lines.append(
                f"| {case['title']} | {case['closest_query']} | {row['method']} | {row['hits_at_5']} | {', '.join(row['top5_docs'])} |"
            )

    (output_dir / "example_query_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "dataset_summary": dataset_summary,
        "custom_summary": custom_summary,
    }


def _run_config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "top_k_neighbors": args.top_k_neighbors,
        "min_similarity": args.min_similarity,
        "min_similarity_floor": args.min_similarity,
        "method_threshold_quantile": args.similarity_quantile,
        "self_weight": args.self_weight,
        "expansion_weight": args.expansion_weight,
        "replacement_weight": args.replacement_weight,
        "replacement_expansion_weight": args.replacement_expansion_weight,
        "max_oov_candidates": args.max_oov_candidates,
        "adaptive_mean_similarity_threshold": not args.disable_mean_sim_threshold,
        "mean_similarity_factor": args.mean_sim_factor,
        "adaptive_quantile_similarity_threshold": not args.disable_quantile_sim_threshold,
        "similarity_quantile": args.similarity_quantile,
        "normalize_neighbor_mass": not args.disable_neighbor_mass_normalization,
        "scale_similarity_scores": not args.disable_similarity_scaling,
        "similarity_scale_mode": args.similarity_scale_mode,
        "similarity_scale_epsilon": args.similarity_scale_epsilon,
        "similarity_power": args.similarity_power,
        "lsa_components": args.lsa_components,
        "esa_top_concepts": args.esa_top_concepts,
        "w2v_vector_size": args.w2v_vector_size,
        "w2v_window": args.w2v_window,
        "w2v_min_count": args.w2v_min_count,
        "w2v_workers": args.w2v_workers,
        "w2v_sg": args.w2v_sg,
        "w2v_epochs": args.w2v_epochs,
    }


def _k10_scores(results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for method, metrics in results.items():
        if "10" not in metrics:
            continue
        snapshot[method] = {k: float(v) for k, v in metrics["10"].items()}
    return snapshot


def _bootstrap_timeline_entry(previous_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any] | None:
    k10 = _k10_scores(previous_results)
    if not k10:
        return None
    return {
        "timestamp": "pre-existing",
        "trial_label": "previous_snapshot",
        "methods": list(previous_results.keys()),
        "config": None,
        "k10": k10,
        "method_details": {},
    }


def _build_timeline_entry(
    args: argparse.Namespace,
    methods: List[str],
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    method_details: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trial_label": args.trial_label.strip(),
        "methods": methods,
        "config": _run_config_snapshot(args),
        "k10": _k10_scores(all_results),
        "method_details": method_details,
    }


def _format_config_changes(previous: Dict[str, Any] | None, current: Dict[str, Any] | None) -> List[str]:
    if not previous or not current:
        return []

    lines: List[str] = []
    for key in sorted(current.keys()):
        if previous.get(key) != current.get(key):
            lines.append(f"{key}: {previous.get(key)} -> {current.get(key)}")
    return lines


def _collect_similarity_scores(neighbors: Dict[str, List[tuple[str, float]]]) -> np.ndarray:
    scores: List[float] = []
    for neighbor_list in neighbors.values():
        for _, similarity in neighbor_list:
            score = float(similarity)
            if score > 0:
                scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _derive_dynamic_min_similarity(
    neighbors: Dict[str, List[tuple[str, float]]],
    base_floor: float,
    quantile: float,
) -> tuple[float, Dict[str, Any]]:
    scores = _collect_similarity_scores(neighbors)
    summary: Dict[str, Any] = {
        "base_min_similarity_floor": float(base_floor),
        "method_threshold_quantile": float(quantile),
        "score_count": int(scores.size),
    }

    if scores.size == 0:
        summary.update(
            {
                "derived_min_similarity": float(base_floor),
                "score_min": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "score_q25": 0.0,
                "score_median": 0.0,
                "score_q75": 0.0,
                "score_q90": 0.0,
                "score_max": 0.0,
                "score_quantile": 0.0,
            }
        )
        return float(base_floor), summary

    quantile = float(np.clip(quantile, 0.0, 1.0))
    score_min = float(np.min(scores))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_q25, score_median, score_q75, score_q90 = [float(value) for value in np.quantile(scores, [0.25, 0.5, 0.75, 0.9])]
    score_quantile = float(np.quantile(scores, quantile))

    derived_min_similarity = max(float(base_floor), score_quantile)
    summary.update(
        {
            "derived_min_similarity": float(derived_min_similarity),
            "score_min": score_min,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_q25": score_q25,
            "score_median": score_median,
            "score_q75": score_q75,
            "score_q90": score_q90,
            "score_max": float(np.max(scores)),
            "score_quantile": score_quantile,
        }
    )
    return derived_min_similarity, summary


def _write_experiment_report(
    report_path: Path,
    output_dir: Path,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    args: argparse.Namespace,
    example_summary: Dict[str, List[Dict]],
    previous_results: Dict[str, Dict[str, Dict[str, float]]] | None = None,
    timeline_entries: List[Dict[str, Any]] | None = None,
    method_details: Dict[str, Dict[str, Any]] | None = None,
    baseline_name: str = BASELINE_METHOD,
) -> None:
    if not all_results:
        return

    previous_results = previous_results or {}
    timeline_entries = timeline_entries or []
    method_details = method_details or {}

    k10 = {method: metrics["10"] for method, metrics in all_results.items()}
    ordered_methods = list(all_results.keys())
    if baseline_name in ordered_methods:
        ordered_methods = [baseline_name] + [m for m in ordered_methods if m != baseline_name]

    baseline_k10 = k10.get(baseline_name)
    previous_k10 = _k10_scores(previous_results)
    latest_entry = timeline_entries[-1] if timeline_entries else None
    previous_entry = timeline_entries[-2] if len(timeline_entries) >= 2 else None
    config_changes = _format_config_changes(
        previous_entry.get("config") if previous_entry else None,
        latest_entry.get("config") if latest_entry else _run_config_snapshot(args),
    )

    best_by_metric = {}
    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
    for metric in metric_names:
        best_method = max(ordered_methods, key=lambda m: k10[m][metric])
        best_by_metric[metric] = (best_method, k10[best_method][metric])

    lines: List[str] = []
    lines.append("# Query Expansion Experiment Report")
    lines.append("")
    lines.append("## What Was Fixed")
    lines.append("")
    lines.append("1. Added a true non-expanded TF-IDF baseline (`baseline_tfidf`) for correct comparison.")
    lines.append("2. Kept retrieval model fixed to base TF-IDF document index; expansion is applied only to queries.")
    lines.append("3. Added distribution-aware, method-specific min similarity floors derived from each method's score distribution.")
    lines.append("4. Added adaptive similarity thresholding using mean and quantile filtering on top of the method-specific floor.")
    lines.append("5. Added normalized neighbor mass allocation so expansion does not overpower original query terms.")
    lines.append("6. Added method-vs-baseline plots and full overlay plots.")
    lines.append("7. Added explicit example-case comparisons for the report query set.")
    lines.append("8. Added persistent WordNet graph caching on disk for faster reruns.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Dataset: {args.dataset}")
    lines.append(f"- Methods: {', '.join(ordered_methods)}")
    lines.append(f"- top_k_neighbors: {args.top_k_neighbors}")
    lines.append(f"- base_min_similarity_floor: {args.min_similarity}")
    lines.append(f"- method_threshold_quantile: {args.similarity_quantile}")
    lines.append(f"- self_weight: {args.self_weight}")
    lines.append(f"- expansion_weight: {args.expansion_weight}")
    lines.append(f"- replacement_weight: {args.replacement_weight}")
    lines.append(f"- replacement_expansion_weight: {args.replacement_expansion_weight}")
    lines.append(f"- adaptive_mean_similarity_threshold: {not args.disable_mean_sim_threshold}")
    lines.append(f"- mean_similarity_factor: {args.mean_sim_factor}")
    lines.append(f"- normalize_neighbor_mass: {not args.disable_neighbor_mass_normalization}")
    lines.append(f"- similarity_power: {args.similarity_power}")

    lines.append("")
    lines.append("## Dynamic Min Similarity by Method")
    lines.append("")
    quantile_label = int(round(args.similarity_quantile * 100))
    lines.append(f"| Method | Base Floor | Derived Floor | Score Count | Mean | Std | Median | Q{quantile_label} | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in ordered_methods:
        stats = method_details.get(method, {})
        if not stats:
            continue
        lines.append(
            "| {} | {:.4f} | {:.4f} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                method,
                float(stats.get("base_min_similarity_floor", args.min_similarity)),
                float(stats.get("derived_min_similarity", args.min_similarity)),
                int(stats.get("score_count", 0)),
                float(stats.get("score_mean", 0.0)),
                float(stats.get("score_std", 0.0)),
                float(stats.get("score_median", 0.0)),
                float(stats.get("score_quantile", 0.0)),
                float(stats.get("score_max", 0.0)),
            )
        )

    lines.append("")
    lines.append("## k=10 Scores")
    lines.append("")
    lines.append("| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in ordered_methods:
        m = k10[method]
        lines.append(
            f"| {method} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fscore']:.4f} | {m['map']:.4f} | {m['ndcg']:.4f} | {m['mrr']:.4f} |"
        )

    if baseline_k10 is not None:
        lines.append("")
        lines.append("## Delta vs Baseline (k=10)")
        lines.append("")
        lines.append("| Method | dP@10 | dR@10 | dF@10 | dMAP@10 | dnDCG@10 | dMRR@10 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method in ordered_methods:
            if method == baseline_name:
                continue
            m = k10[method]
            lines.append(
                "| {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                    method,
                    m["precision"] - baseline_k10["precision"],
                    m["recall"] - baseline_k10["recall"],
                    m["fscore"] - baseline_k10["fscore"],
                    m["map"] - baseline_k10["map"],
                    m["ndcg"] - baseline_k10["ndcg"],
                    m["mrr"] - baseline_k10["mrr"],
                )
            )
    lines.append("")
    lines.append("## Best Method Per Metric at k=10")
    lines.append("")
    for metric in metric_names:
        method, value = best_by_metric[metric]
        lines.append(f"- {metric}: {method} ({value:.4f})")

    lines.append("")
    lines.append("## Example Cases Summary")
    lines.append("")
    lines.append("### Dataset Query Cases")
    lines.append("")
    lines.append("| Query ID | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for row in example_summary.get("dataset_summary", []):
        lines.append(
            f"| {row['query_id']} | {row['baseline_hits']} | {row['best_method']} | {row['best_hits']} | {row['delta_hits']:+d} |"
        )

    lines.append("")
    lines.append("### Custom Query Cases")
    lines.append("")
    lines.append("| Case | Mapped Query | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    for row in example_summary.get("custom_summary", []):
        lines.append(
            f"| {row['case_title']} | {row['mapped_query_id']} | {row['baseline_hits']} | {row['best_method']} | {row['best_hits']} | {row['delta_hits']:+d} |"
        )

    lines.append("")
    lines.append("## Limitation-Solving Score Table (updated)")
    lines.append("")
    lines.append("| Method | Semantic | Dim/Cost | Scale | Ambiguity | OOV | Context | Sparse |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append("| baseline_tfidf | 2/10 | 6/10 | 6/10 | 2/10 | 1/10 | 2/10 | 2/10 |")
    lines.append("| wordnet | 8/10 | 3/10 | 3/10 | 7/10 | 8/10 | 5/10 | 3/10 |")
    lines.append("| embedding_tfidf | 4/10 | 6/10 | 6/10 | 3/10 | 2/10 | 3/10 | 3/10 |")
    lines.append("| embedding_lsa | 5/10 | 5/10 | 5/10 | 4/10 | 2/10 | 4/10 | 7/10 |")
    lines.append("| embedding_esa | 6/10 | 4/10 | 5/10 | 4/10 | 3/10 | 4/10 | 7/10 |")
    lines.append("| embedding_word2vec | 7/10 | 5/10 | 5/10 | 5/10 | 4/10 | 5/10 | 5/10 |")

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Summary JSON: {output_dir / 'summary.json'}")
    lines.append(f"- Summary CSV: {output_dir / 'summary_k10.csv'}")
    lines.append(f"- Overlay plot: {output_dir / 'eval_overlay.png'}")
    lines.append(f"- Example comparison markdown: {output_dir / 'example_query_comparison.md'}")
    lines.append(f"- Example comparison json: {output_dir / 'example_query_comparison.json'}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    args = parse_args()
    start = time.time()

    def log(message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress
    log_every = max(1, args.log_every)
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    timeline_file = output_dir / RUN_TIMELINE_FILE
    timeline_entries: List[Dict[str, Any]] = []
    if timeline_file.exists():
        try:
            loaded = json.loads(timeline_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                timeline_entries = [entry for entry in loaded if isinstance(entry, dict)]
        except Exception:
            timeline_entries = []

    previous_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if timeline_entries:
        previous_entry = timeline_entries[-1]
        previous_k10 = previous_entry.get("k10", {})
        if isinstance(previous_k10, dict):
            previous_results = {method: {"10": metrics} for method, metrics in previous_k10.items() if isinstance(metrics, dict)}

    method_details: Dict[str, Dict[str, Any]] = {}

    log(f"Loading Cranfield dataset from: {args.dataset}")
    docs_json, queries_json, qrels = load_cranfield_dataset(args.dataset)
    doc_ids, doc_texts = get_docs_ids_and_texts(docs_json)
    query_ids, query_texts = get_query_ids_and_texts(queries_json)
    query_text_lookup = {str(item["query number"]): item["query"] for item in queries_json}
    log(f"Loaded raw data: {len(doc_ids)} docs, {len(query_ids)} queries, {len(qrels)} qrels")

    stage_start = time.time()
    log("Preprocessing document corpus")
    preprocessor = Preprocessor(use_lemmatization=True)
    processed_docs = preprocessor.preprocess_corpus(doc_texts)
    log(f"Document preprocessing done in {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    log("Preprocessing query corpus")
    processed_queries = preprocessor.preprocess_corpus(query_texts)
    log(f"Query preprocessing done in {time.time() - stage_start:.2f}s")

    custom_query_texts = [case["query"] for case in CUSTOM_QUERY_CASES]
    custom_query_ids = [case["id"] for case in CUSTOM_QUERY_CASES]
    custom_processed_queries = preprocessor.preprocess_corpus(custom_query_texts)

    stage_start = time.time()
    log("Building TF-IDF retrieval index")
    retriever = VectorSpaceRetrieval()
    retriever.build(processed_docs, doc_ids)
    if retriever.doc_tfidf is None:
        raise RuntimeError("Document TF-IDF matrix was not created")
    doc_tfidf = retriever.doc_tfidf
    log(f"Retrieval index built in {time.time() - stage_start:.2f}s")

    evaluator = CranfieldEvaluation(qrels)

    log(f"Vocabulary size after preprocessing: {len(retriever.vocab)}")

    expanded_methods = [m for m in requested_methods if m != BASELINE_METHOD]
    wordnet_neighbors: Dict[str, List[tuple[str, float]]] | None = None
    if "wordnet" in expanded_methods:
        stage_start = time.time()
        log("Building WordNet in-vocabulary similarity map")
        wordnet_neighbors = build_wordnet_neighbor_map(
            retriever.vocab,
            top_k=args.top_k_neighbors,
            min_similarity=args.min_similarity,
            progress=show_progress,
            logger=log,
            log_every=log_every,
        )
        log(f"WordNet similarity map built in {time.time() - stage_start:.2f}s")
    else:
        log("Skipping WordNet in-vocabulary matrix (wordnet method not requested)")

    oov_resolver: WordNetOOVResolver | None = None
    if expanded_methods:
        stage_start = time.time()
        log("Building WordNet OOV replacement index")
        oov_resolver = WordNetOOVResolver(
            retriever.vocab,
            progress=show_progress,
            logger=log,
            log_every=log_every,
        )
        log(f"WordNet OOV index built in {time.time() - stage_start:.2f}s")

    all_results: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}
    dataset_rankings_by_method: Dict[str, Dict[str, List[str]]] = {}
    custom_rankings_by_method: Dict[str, Dict[str, List[str]]] = {}

    log(f"Running methods: {', '.join(requested_methods)}")

    methods_iter = tqdm(
        requested_methods,
        desc="Methods",
        unit="method",
        disable=not show_progress,
    )

    for method in methods_iter:
        method_start = time.time()
        print(f"\n=== Running method: {method} ===", flush=True)
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            if method == BASELINE_METHOD:
                log(f"[{method}] Building non-expanded baseline query vectors")
                query_vectors = _baseline_query_vectors(
                    processed_queries,
                    retriever.term_to_idx,
                    len(retriever.vocab),
                    show_progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
                custom_query_vectors = _baseline_query_vectors(
                    custom_processed_queries,
                    retriever.term_to_idx,
                    len(retriever.vocab),
                    show_progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
            else:
                if oov_resolver is None:
                    raise RuntimeError("OOV resolver unavailable for expansion methods")

                log(f"[{method}] Preparing neighbor map")
                if method == "wordnet":
                    if wordnet_neighbors is None:
                        raise RuntimeError("WordNet neighbors unavailable for wordnet method")
                    neighbors = wordnet_neighbors
                elif method == "embedding_tfidf":
                    neighbors = build_tfidf_neighbor_map(
                        doc_tfidf,
                        retriever.vocab,
                        top_k=args.top_k_neighbors,
                        min_similarity=args.min_similarity,
                        progress=show_progress,
                        logger=log,
                        log_every=log_every,
                    )
                elif method == "embedding_lsa":
                    neighbors = build_lsa_neighbor_map(
                        doc_tfidf,
                        retriever.vocab,
                        n_components=args.lsa_components,
                        top_k=args.top_k_neighbors,
                        min_similarity=args.min_similarity,
                        progress=show_progress,
                        logger=log,
                        log_every=log_every,
                    )
                elif method == "embedding_esa":
                    esa_stats: Dict[str, Any] = {}
                    neighbors = build_esa_neighbor_map(
                        doc_tfidf,
                        retriever.vocab,
                        top_concepts=args.esa_top_concepts,
                        top_k=args.top_k_neighbors,
                        min_similarity=args.min_similarity,
                        progress=show_progress,
                        logger=log,
                        log_every=log_every,
                        stats_out=esa_stats,
                    )
                    if esa_stats:
                        method_details[method] = esa_stats
                        log(
                            f"[{method}] Coverage: {esa_stats.get('represented_terms', 0)}/{esa_stats.get('vocab_terms', 0)} terms "
                            f"({float(esa_stats.get('representation_coverage', 0.0)):.2%})"
                        )
                elif method == "embedding_word2vec":
                    w2v_stats: Dict[str, Any] = {}
                    neighbors = build_word2vec_neighbor_map(
                        processed_docs,
                        retriever.vocab,
                        vector_size=args.w2v_vector_size,
                        window=args.w2v_window,
                        min_count=args.w2v_min_count,
                        workers=args.w2v_workers,
                        sg=args.w2v_sg,
                        epochs=args.w2v_epochs,
                        top_k=args.top_k_neighbors,
                        min_similarity=args.min_similarity,
                        progress=show_progress,
                        logger=log,
                        log_every=log_every,
                        stats_out=w2v_stats,
                    )
                    if w2v_stats:
                        method_details[method] = w2v_stats
                        log(
                            f"[{method}] Coverage: {w2v_stats.get('present_in_model', 0)}/{w2v_stats.get('vocab_terms', 0)} terms "
                            f"({float(w2v_stats.get('model_coverage', 0.0)):.2%})"
                        )
                else:
                    raise ValueError(f"Unknown method: {method}")

                method_min_similarity, similarity_stats = _derive_dynamic_min_similarity(
                    neighbors,
                    args.min_similarity,
                    args.similarity_quantile,
                )
                method_details.setdefault(method, {}).update(similarity_stats)
                log(
                    f"[{method}] Dynamic min similarity floor: base={args.min_similarity:.4f}, "
                    f"q{int(round(args.similarity_quantile * 100))}={float(similarity_stats.get('score_quantile', 0.0)):.4f}, "
                    f"derived={method_min_similarity:.4f}"
                )

                log(f"[{method}] Neighbor map ready. Expanding queries")
                expander = MatrixQueryExpander(
                    vocab=retriever.vocab,
                    neighbors=neighbors,
                    oov_resolver=oov_resolver.resolve,
                    self_weight=args.self_weight,
                    expansion_weight=args.expansion_weight,
                    replacement_weight=args.replacement_weight,
                    replacement_expansion_weight=args.replacement_expansion_weight,
                    max_oov_candidates=args.max_oov_candidates,
                    min_similarity=method_min_similarity,
                    expand_replacements=True,
                    use_mean_similarity_threshold=not args.disable_mean_sim_threshold,
                    mean_similarity_factor=args.mean_sim_factor,
                    normalize_neighbor_mass=not args.disable_neighbor_mass_normalization,
                    similarity_power=args.similarity_power,
                )

                query_vectors = _query_vectors_for_expander(
                    processed_queries,
                    expander,
                    show_progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
                custom_query_vectors = _query_vectors_for_expander(
                    custom_processed_queries,
                    expander,
                    show_progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )

            log(f"[{method}] Ranking documents")
            rankings = retriever.query_vectors_to_rankings(query_vectors)
            rankings_by_query = {
                str(qid): [str(doc_id) for doc_id in ranked]
                for qid, ranked in zip(query_ids, rankings)
            }

            custom_rankings = retriever.query_vectors_to_rankings(custom_query_vectors)
            custom_rankings_by_id = {
                custom_query_ids[idx]: [str(doc_id) for doc_id in ranked]
                for idx, ranked in enumerate(custom_rankings)
            }

            log(f"[{method}] Computing evaluation metrics")
            metrics = evaluator.evaluate(rankings_by_query, query_ids, k_values=range(1, 11))

            previews = []
            for idx in range(min(8, len(query_ids))):
                previews.append(
                    _preview_query_expansion(
                        query_ids[idx],
                        processed_queries[idx],
                        query_vectors[idx],
                        retriever.vocab,
                    )
                )

            dataset_rankings_by_method[method] = rankings_by_query
            custom_rankings_by_method[method] = custom_rankings_by_id

            (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (method_dir / "rankings_top20.json").write_text(
                json.dumps({qid: docs[:20] for qid, docs in rankings_by_query.items()}, indent=2),
                encoding="utf-8",
            )
            (method_dir / "custom_query_top20.json").write_text(
                json.dumps({qid: docs[:20] for qid, docs in custom_rankings_by_id.items()}, indent=2),
                encoding="utf-8",
            )
            (method_dir / "query_expansion_preview.json").write_text(
                json.dumps(previews, indent=2),
                encoding="utf-8",
            )

            baseline_metrics = all_results.get(BASELINE_METHOD)
            _plot_method_metrics(
                method,
                metrics,
                method_dir / "eval_plot.png",
                baseline_metrics=baseline_metrics,
                baseline_name=BASELINE_METHOD,
            )

            all_results[method] = metrics
            print(
                f"Finished {method}. P@10={metrics['10']['precision']:.4f}, MAP@10={metrics['10']['map']:.4f}",
                flush=True,
            )
            log(f"[{method}] Completed in {time.time() - method_start:.2f}s")

        except Exception as exc:
            failures[method] = str(exc)
            print(f"Method {method} failed: {exc}", flush=True)
            log(f"[{method}] Failed after {time.time() - method_start:.2f}s")

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    csv_file = output_dir / "summary_k10.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "precision@10", "recall@10", "fscore@10", "map@10", "ndcg@10", "mrr@10"])
        for method, metrics in all_results.items():
            m10 = metrics["10"]
            writer.writerow(
                [
                    method,
                    m10["precision"],
                    m10["recall"],
                    m10["fscore"],
                    m10["map"],
                    m10["ndcg"],
                    m10["mrr"],
                ]
            )

    if failures:
        (output_dir / "failed_methods.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if all_results:
        timeline_entry = _build_timeline_entry(args, list(all_results.keys()), all_results, method_details)
        timeline_entries.append(timeline_entry)
        timeline_file.write_text(json.dumps(timeline_entries, indent=2), encoding="utf-8")

        _plot_overlay_metrics(all_results, output_dir / "eval_overlay.png", baseline_name=BASELINE_METHOD)

        baseline_metrics = all_results.get(BASELINE_METHOD)
        if baseline_metrics is not None:
            for method, metrics in all_results.items():
                if method == BASELINE_METHOD:
                    continue
                _plot_method_metrics(
                    method,
                    metrics,
                    output_dir / method / "eval_plot_vs_baseline.png",
                    baseline_metrics=baseline_metrics,
                    baseline_name=BASELINE_METHOD,
                )

        example_summary = _write_example_comparison(
            output_dir=output_dir,
            methods=list(all_results.keys()),
            qrels=qrels,
            query_text_lookup=query_text_lookup,
            dataset_rankings_by_method=dataset_rankings_by_method,
            custom_rankings_by_method=custom_rankings_by_method,
            baseline_name=BASELINE_METHOD,
        )

        report_path = Path(__file__).resolve().parent / "experiment_report.md"
        _write_experiment_report(
            report_path=report_path,
            output_dir=output_dir,
            all_results=all_results,
            args=args,
            example_summary=example_summary,
            previous_results=previous_results,
            timeline_entries=timeline_entries,
            method_details=method_details,
            baseline_name=BASELINE_METHOD,
        )

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f} seconds", flush=True)
    print(f"Summary: {summary_file}", flush=True)
    print(f"k=10 table: {csv_file}", flush=True)
    if failures:
        print("Some methods failed. See failed_methods.json", flush=True)


if __name__ == "__main__":
    run()
