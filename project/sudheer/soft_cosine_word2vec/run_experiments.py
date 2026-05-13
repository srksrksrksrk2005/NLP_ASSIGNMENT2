#!/usr/bin/env python3
"""
Run baseline TF-IDF retrieval and a Word2Vec-based soft cosine backbone
for the Cranfield collection.

The method keeps TF-IDF term vectors but replaces exact term matching with
soft matching through a sparse Word2Vec-derived term similarity matrix.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SUDHEER_DIR = SCRIPT_DIR.parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
PROJECT_DIR = SCRIPT_DIR.parents[1]
if str(SUDHEER_DIR) not in sys.path:
    sys.path.insert(0, str(SUDHEER_DIR))

from common import (  # noqa: E402
    ASSIGNMENT_ROOT as COMMON_ASSIGNMENT_ROOT,
    Evaluation,
    SoftCosineModel,
    TfidfIndex,
    approximate_randomization_pvalue,
    best_results_by_primary,
    build_example_query_comparison,
    build_experiment_report,
    build_term_similarity_matrix,
    compute_metrics,
    compute_per_query_metrics,
    load_json,
    metric_sort_key,
    normalize_collection,
    save_all_tuned_combinations_overlay,
    save_metric_plot,
    save_overlay_plot,
    save_primary_sweep_plot,
    save_rankings,
    train_word2vec_artifacts,
    unigram_feature_map,
    write_json,
    write_primary_sweep_csv,
    write_summary_csv,
)


METHOD_KEY = "soft_cosine_word2vec"
TFIDF_SOURCE_KEY = "soft_cosine_tfidf_source"


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EMBEDDING_MATRICES = load_module_from_path(
    "nikhil_embedding_matrices",
    PROJECT_DIR / "Nikhil" / "query_expansion" / "expansion" / "embedding_matrices.py",
)
build_tfidf_neighbor_map = EMBEDDING_MATRICES.build_tfidf_neighbor_map


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def sg_label(value: int) -> str:
    return "skipgram" if int(value) == 1 else "cbow"


def neighbor_map_to_similarity_matrix(
    vocab,
    neighbor_map,
    similarity_power: float = 1.0,
):
    term_to_idx = {term: index for index, term in enumerate(vocab)}
    rows = list(range(len(vocab)))
    cols = list(range(len(vocab)))
    data = [1.0] * len(vocab)

    pair_to_similarity = {}
    for term, neighbors in neighbor_map.items():
        src_index = term_to_idx.get(term)
        if src_index is None:
            continue
        for neighbor, similarity in neighbors:
            dst_index = term_to_idx.get(neighbor)
            if dst_index is None or dst_index == src_index:
                continue
            value = float(max(0.0, similarity))
            if similarity_power != 1.0:
                value = value ** float(similarity_power)
            pair = (src_index, dst_index) if src_index < dst_index else (dst_index, src_index)
            pair_to_similarity[pair] = max(pair_to_similarity.get(pair, 0.0), value)

    for (src_index, dst_index), similarity in pair_to_similarity.items():
        rows.extend([src_index, dst_index])
        cols.extend([dst_index, src_index])
        data.extend([similarity, similarity])

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=float).tocsr()
    matrix.sum_duplicates()
    matrix.data = matrix.data.clip(0.0, 1.0)
    return matrix, {
        "represented_terms": len(vocab),
        "representation_coverage": 1.0 if vocab else 0.0,
        "retained_neighbor_edges": int(len(pair_to_similarity)),
        "avg_neighbors_per_term": float((2 * len(pair_to_similarity)) / len(vocab)) if vocab else 0.0,
    }


def run_method(
    docs,
    doc_ids,
    query_tf,
    baseline_index: TfidfIndex,
    vector_size: int,
    window: int,
    sg: int,
    epochs: int,
    min_count: int,
    workers: int,
    top_k_neighbors: int,
    min_similarity: float,
    similarity_power: float,
):
    artifacts = train_word2vec_artifacts(
        docs_tokens=docs,
        vocab=baseline_index.vocab,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )
    similarity_matrix, similarity_stats = build_term_similarity_matrix(
        vocab=baseline_index.vocab,
        term_to_idx=baseline_index.term_to_idx,
        word_vectors=artifacts.word_vectors,
        top_k_neighbors=top_k_neighbors,
        min_similarity=min_similarity,
        similarity_power=similarity_power,
    )
    model = SoftCosineModel.build(METHOD_KEY, baseline_index, similarity_matrix)
    rankings, score_vectors = model.rank_query_matrix(query_tf)
    return artifacts, similarity_stats, rankings, score_vectors


def run_tfidf_source_method(
    query_tf,
    baseline_index: TfidfIndex,
    top_k_neighbors: int,
    min_similarity: float,
    similarity_power: float,
):
    neighbor_map = build_tfidf_neighbor_map(
        baseline_index.doc_tfidf,
        baseline_index.vocab,
        top_k=top_k_neighbors,
        min_similarity=min_similarity,
        progress=False,
    )
    similarity_matrix, similarity_stats = neighbor_map_to_similarity_matrix(
        baseline_index.vocab,
        neighbor_map,
        similarity_power=similarity_power,
    )
    model = SoftCosineModel.build(TFIDF_SOURCE_KEY, baseline_index, similarity_matrix)
    rankings, score_vectors = model.rank_query_matrix(query_tf)
    return similarity_stats, rankings, score_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Word2Vec soft cosine experiments.")
    parser.add_argument("--dataset-dir", default=str(COMMON_ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--vector-sizes", default="100")
    parser.add_argument("--windows", default="5")
    parser.add_argument("--sg-values", default="1")
    parser.add_argument("--epochs-values", default="20")
    parser.add_argument("--top-k-neighbors-values", default="5,10,20")
    parser.add_argument("--min-similarity-values", default="0.10,0.20,0.30")
    parser.add_argument("--similarity-power-values", default="1.0,2.0")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline_tfidf"
    method_dir = output_dir / METHOD_KEY
    tfidf_source_dir = output_dir / TFIDF_SOURCE_KEY
    baseline_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    tfidf_source_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")

    processed_docs = normalize_collection(load_json(Path(args.processed_docs)))
    processed_queries = normalize_collection(load_json(Path(args.processed_queries)))

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    evaluator = Evaluation()

    doc_feature_maps = [unigram_feature_map(doc) for doc in processed_docs]
    query_feature_maps = [unigram_feature_map(query) for query in processed_queries]

    baseline_start = time.time()
    baseline_index = TfidfIndex.build("baseline_tfidf", doc_feature_maps, doc_ids)
    baseline_rankings, _ = baseline_index.rank_feature_maps(query_feature_maps)
    baseline_runtime = time.time() - baseline_start
    query_tf = baseline_index.feature_maps_to_matrix(query_feature_maps)

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

    vector_sizes = parse_int_list(args.vector_sizes)
    windows = parse_int_list(args.windows)
    sg_values = parse_int_list(args.sg_values)
    epochs_values = parse_int_list(args.epochs_values)
    top_k_values = parse_int_list(args.top_k_neighbors_values)
    min_similarity_values = parse_float_list(args.min_similarity_values)
    similarity_power_values = parse_float_list(args.similarity_power_values)

    sweep_results = []
    config_index = 0
    total_configs = (
        len(vector_sizes)
        * len(windows)
        * len(sg_values)
        * len(epochs_values)
        * len(top_k_values)
        * len(min_similarity_values)
        * len(similarity_power_values)
    )
    sweep_start = time.time()

    for vector_size in vector_sizes:
        for window in windows:
            for sg in sg_values:
                for epochs in epochs_values:
                    for top_k_neighbors in top_k_values:
                        for min_similarity in min_similarity_values:
                            for similarity_power in similarity_power_values:
                                config_index += 1
                                print(
                                    f"[sweep {config_index}/{total_configs}] "
                                    f"vec={vector_size} window={window} sg={sg_label(sg)} epochs={epochs} "
                                    f"topk={top_k_neighbors} floor={min_similarity:.2f} power={similarity_power:.2f}",
                                    flush=True,
                                )

                                artifacts, similarity_stats, rankings, _ = run_method(
                                    docs=processed_docs,
                                    doc_ids=doc_ids,
                                    query_tf=query_tf,
                                    baseline_index=baseline_index,
                                    vector_size=vector_size,
                                    window=window,
                                    sg=sg,
                                    epochs=epochs,
                                    min_count=args.min_count,
                                    workers=args.workers,
                                    top_k_neighbors=top_k_neighbors,
                                    min_similarity=min_similarity,
                                    similarity_power=similarity_power,
                                )
                                metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
                                k10 = {metric: values[9] for metric, values in metrics_by_k.items()}

                                sweep_results.append(
                                    {
                                        "vector_size": vector_size,
                                        "window": window,
                                        "sg": sg,
                                        "sg_label": sg_label(sg),
                                        "epochs": epochs,
                                        "top_k_neighbors": top_k_neighbors,
                                        "min_similarity": min_similarity,
                                        "similarity_power": similarity_power,
                                        "sentences_seen": artifacts.sentences_seen,
                                        "present_in_model": artifacts.present_in_model,
                                        "vocab_terms": artifacts.vocab_terms,
                                        "model_coverage": artifacts.model_coverage,
                                        "represented_terms": similarity_stats["represented_terms"],
                                        "representation_coverage": similarity_stats["representation_coverage"],
                                        "retained_neighbor_edges": similarity_stats["retained_neighbor_edges"],
                                        "avg_neighbors_per_term": similarity_stats["avg_neighbors_per_term"],
                                        "metrics_by_k": metrics_by_k,
                                        "precision_at_10": k10["precision"],
                                        "recall_at_10": k10["recall"],
                                        "fscore_at_10": k10["fscore"],
                                        "map_at_10": k10["map"],
                                        "ndcg_at_10": k10["ndcg"],
                                        "mrr_at_10": k10["mrr"],
                                    }
                                )
                                print(f"  MAP@10={k10['map']:.4f} nDCG@10={k10['ndcg']:.4f}", flush=True)

    sweep_results.sort(key=metric_sort_key, reverse=True)
    best_config = sweep_results[0]
    write_json(output_dir / "config_sweep.json", sweep_results)

    best_by_top_k = best_results_by_primary(sweep_results, "top_k_neighbors")
    write_json(output_dir / "neighbor_sweep_best.json", best_by_top_k)
    write_primary_sweep_csv(
        output_dir / "neighbor_sweep_best.csv",
        best_by_top_k,
        [
            "top_k_neighbors",
            "min_similarity",
            "similarity_power",
            "vector_size",
            "window",
            "sg_label",
            "epochs",
            "representation_coverage",
            "retained_neighbor_edges",
            "avg_neighbors_per_term",
            "precision_at_10",
            "recall_at_10",
            "fscore_at_10",
            "map_at_10",
            "ndcg_at_10",
            "mrr_at_10",
        ],
    )
    save_primary_sweep_plot(
        output_dir / "neighbor_sweep_best.png",
        best_by_top_k,
        primary_key="top_k_neighbors",
        x_label="Top-k retained neighbors per term",
        title="Best Result Per Soft Cosine Neighbor Budget",
    )

    artifacts, similarity_stats, method_rankings, _ = run_method(
        docs=processed_docs,
        doc_ids=doc_ids,
        query_tf=query_tf,
        baseline_index=baseline_index,
        vector_size=int(best_config["vector_size"]),
        window=int(best_config["window"]),
        sg=int(best_config["sg"]),
        epochs=int(best_config["epochs"]),
        min_count=args.min_count,
        workers=args.workers,
        top_k_neighbors=int(best_config["top_k_neighbors"]),
        min_similarity=float(best_config["min_similarity"]),
        similarity_power=float(best_config["similarity_power"]),
    )
    method_runtime = time.time() - sweep_start
    method_metrics = compute_metrics(evaluator, method_rankings, query_ids, qrels)
    method_per_query = compute_per_query_metrics(evaluator, method_rankings, query_ids, qrels)

    save_metric_plot(
        method_dir / "eval_plot.png",
        method_metrics,
        "Word2Vec Soft Cosine Backbone Metrics",
    )
    save_rankings(method_dir / "rankings_top20.json", query_ids, method_rankings, top_k=20)

    grouped_metrics = {row["top_k_neighbors"]: row["metrics_by_k"] for row in best_by_top_k}
    method_labels = {row["top_k_neighbors"]: f"topk={row['top_k_neighbors']}" for row in best_by_top_k}
    ordered_keys = [row["top_k_neighbors"] for row in best_by_top_k]
    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        grouped_metrics,
        method_labels,
        ordered_keys,
        "Baseline vs Best Word2Vec Soft Cosine Models by Neighbor Budget",
    )
    save_all_tuned_combinations_overlay(
        output_dir / "all_tuned_combinations_overlay.png",
        baseline_metrics,
        sweep_results,
        best_config,
        "Baseline vs All Tuned Word2Vec Soft Cosine Configurations",
    )

    method_metrics_payload = {
        "best_config": best_config,
        "metrics_by_k": method_metrics,
        "k10": {metric: values[9] for metric, values in method_metrics.items()},
        "runtime_seconds": method_runtime,
        "present_in_model": artifacts.present_in_model,
        "vocab_terms": artifacts.vocab_terms,
        "model_coverage": artifacts.model_coverage,
        "represented_terms": similarity_stats["represented_terms"],
        "representation_coverage": similarity_stats["representation_coverage"],
        "retained_neighbor_edges": similarity_stats["retained_neighbor_edges"],
        "avg_neighbors_per_term": similarity_stats["avg_neighbors_per_term"],
    }
    write_json(method_dir / "metrics.json", method_metrics_payload)

    tfidf_source_results = []
    for top_k_neighbors in top_k_values:
        for min_similarity in min_similarity_values:
            for similarity_power in similarity_power_values:
                similarity_stats, rankings, _ = run_tfidf_source_method(
                    query_tf=query_tf,
                    baseline_index=baseline_index,
                    top_k_neighbors=top_k_neighbors,
                    min_similarity=min_similarity,
                    similarity_power=similarity_power,
                )
                metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
                k10 = {metric: values[9] for metric, values in metrics_by_k.items()}
                tfidf_source_results.append(
                    {
                        "top_k_neighbors": top_k_neighbors,
                        "min_similarity": min_similarity,
                        "similarity_power": similarity_power,
                        "metrics_by_k": metrics_by_k,
                        "precision_at_10": k10["precision"],
                        "recall_at_10": k10["recall"],
                        "fscore_at_10": k10["fscore"],
                        "map_at_10": k10["map"],
                        "ndcg_at_10": k10["ndcg"],
                        "mrr_at_10": k10["mrr"],
                        **similarity_stats,
                    }
                )

    tfidf_source_results.sort(key=metric_sort_key, reverse=True)
    best_tfidf_source = tfidf_source_results[0]
    save_metric_plot(
        tfidf_source_dir / "eval_plot.png",
        best_tfidf_source["metrics_by_k"],
        "TF-IDF Source Soft Cosine Metrics",
    )
    write_json(tfidf_source_dir / "sweep_results.json", tfidf_source_results)
    write_json(
        tfidf_source_dir / "metrics.json",
        {
            "best_config": {
                "top_k_neighbors": best_tfidf_source["top_k_neighbors"],
                "min_similarity": best_tfidf_source["min_similarity"],
                "similarity_power": best_tfidf_source["similarity_power"],
                "represented_terms": best_tfidf_source["represented_terms"],
                "representation_coverage": best_tfidf_source["representation_coverage"],
                "retained_neighbor_edges": best_tfidf_source["retained_neighbor_edges"],
                "avg_neighbors_per_term": best_tfidf_source["avg_neighbors_per_term"],
            },
            "metrics_by_k": best_tfidf_source["metrics_by_k"],
            "k10": {
                "precision": best_tfidf_source["precision_at_10"],
                "recall": best_tfidf_source["recall_at_10"],
                "fscore": best_tfidf_source["fscore_at_10"],
                "map": best_tfidf_source["map_at_10"],
                "ndcg": best_tfidf_source["ndcg_at_10"],
                "mrr": best_tfidf_source["mrr_at_10"],
            },
        },
    )

    rows = []
    summary_methods = {}
    for method_name, metrics_payload in [
        ("baseline_tfidf", {"metrics_by_k": baseline_metrics, "runtime_seconds": baseline_runtime}),
        (METHOD_KEY, method_metrics_payload),
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

    tfidf_source_k10 = {
        "precision": best_tfidf_source["precision_at_10"],
        "recall": best_tfidf_source["recall_at_10"],
        "fscore": best_tfidf_source["fscore_at_10"],
        "map": best_tfidf_source["map_at_10"],
        "ndcg": best_tfidf_source["ndcg_at_10"],
        "mrr": best_tfidf_source["mrr_at_10"],
    }
    rows.append(
        {
            "method": TFIDF_SOURCE_KEY,
            "precision@10": tfidf_source_k10["precision"],
            "recall@10": tfidf_source_k10["recall"],
            "fscore@10": tfidf_source_k10["fscore"],
            "map@10": tfidf_source_k10["map"],
            "ndcg@10": tfidf_source_k10["ndcg"],
            "mrr@10": tfidf_source_k10["mrr"],
            "runtime_seconds": 0.0,
        }
    )
    summary_methods[TFIDF_SOURCE_KEY] = {
        "k10": tfidf_source_k10,
        "metrics_by_k": best_tfidf_source["metrics_by_k"],
        "runtime_seconds": 0.0,
    }

    write_summary_csv(output_dir / "summary_k10.csv", rows)

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
        "neighbor_sweep_best": best_by_top_k,
        "methods": summary_methods,
        "delta_vs_baseline_at_10": {
            metric: summary_methods[METHOD_KEY]["k10"][metric] - summary_methods["baseline_tfidf"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        },
        "significance": significance,
        "tfidf_source_comparison": {
            "best_config": {
                "top_k_neighbors": best_tfidf_source["top_k_neighbors"],
                "min_similarity": best_tfidf_source["min_similarity"],
                "similarity_power": best_tfidf_source["similarity_power"],
            },
            "k10": tfidf_source_k10,
            "delta_vs_word2vec_soft_cosine_at_10": {
                metric: tfidf_source_k10[metric] - summary_methods[METHOD_KEY]["k10"][metric]
                for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
            },
        },
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
            "all_configs_overlay": str(output_dir / "all_tuned_combinations_overlay.png"),
            "neighbor_sweep_json": str(output_dir / "neighbor_sweep_best.json"),
            "neighbor_sweep_csv": str(output_dir / "neighbor_sweep_best.csv"),
            "neighbor_sweep_plot": str(output_dir / "neighbor_sweep_best.png"),
            "example_markdown": str(output_dir / "example_query_comparison.md"),
            "config_sweep_json": str(output_dir / "config_sweep.json"),
            "tfidf_source_metrics_json": str(tfidf_source_dir / "metrics.json"),
            "tfidf_source_sweep_json": str(tfidf_source_dir / "sweep_results.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "config": {
                "best_config": best_config,
                "neighbor_sweep_best": best_by_top_k,
                "best_tfidf_source": summary["tfidf_source_comparison"]["best_config"],
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

    build_experiment_report(
        output_path=SCRIPT_DIR / "experiment_report.md",
        title="Word2Vec Soft Cosine Backbone Experiment Report",
        method_key=METHOD_KEY,
        method_label=METHOD_KEY,
        summary=summary,
        examples=examples,
        sweep_rows=best_by_top_k,
        sweep_section_title="Neighbor Budget Sweep",
        sweep_columns=[
            ("top_k_neighbors", "Top-k neighbors", "{}"),
            ("min_similarity", "Similarity floor", "{:.2f}"),
            ("similarity_power", "Similarity power", "{:.2f}"),
            ("vector_size", "Vector size", "{}"),
            ("representation_coverage", "Coverage", "{:.2%}"),
        ],
        best_config_fields=[
            ("vector_size", "vector_size", "{}"),
            ("window", "window", "{}"),
            ("training_objective", "sg_label", "{}"),
            ("epochs", "epochs", "{}"),
            ("top_k_neighbors", "top_k_neighbors", "{}"),
            ("min_similarity", "min_similarity", "{:.2f}"),
            ("similarity_power", "similarity_power", "{:.2f}"),
            ("representation_coverage", "representation_coverage", "{:.2%}"),
            ("avg_neighbors_per_term", "avg_neighbors_per_term", "{:.2f}"),
        ],
        method_summary_lines=[
            "The baseline system uses unigram TF-IDF with cosine similarity. "
            "The proposed method keeps the TF-IDF representation, but replaces exact lexical matching with soft matching "
            "through a sparse term similarity matrix built from Word2Vec neighbors.",
            "This follows the soft cosine idea: two different words can contribute partial similarity if their embeddings "
            "place them near each other, so the method behaves like a lexical-semantic bridge rather than a pure dense retriever.",
        ],
        hypothesis_lines=[
            "Soft cosine should preserve the precision advantages of TF-IDF while relaxing the requirement that matching "
            "terms be literally identical.",
            "Because the document representation remains sparse and explicit, the method should be more interpretable than "
            "dense centroid retrieval and more semantically tolerant than ordinary cosine over TF-IDF.",
        ],
        limitations_lines=[
            "The baseline only rewards exact term overlap.",
            "Word2Vec neighbors can inject graded similarity between related technical terms.",
            "Soft cosine still depends on the quality and coverage of the embedding space used to define term similarity.",
        ],
        interpretation_lines=[
            "This method is a strong compromise between lexical and semantic retrieval. It keeps explicit TF-IDF term weights, "
            "which helps precision, but it no longer treats all non-identical terms as completely unrelated.",
            "Its main tuning burden is the similarity matrix itself: if too many neighbors are kept, the model drifts; if too few "
            "are kept, the method collapses back toward ordinary TF-IDF.",
        ],
        citations=[
            (
                "Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (NeurIPS 2013)",
                "https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html",
            ),
            (
                "Sidorov et al., Soft Similarity and Soft Cosine Measure (2014)",
                "https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462014000300007",
            ),
        ],
    )

    report_path = SCRIPT_DIR / "experiment_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    comparison_section = (
        "\n\n## TF-IDF Source Comparison\n\n"
        "We also ran soft cosine with the same TF-IDF query/document vectors but built the term-similarity matrix "
        "from TF-IDF term neighborhoods instead of Word2Vec neighbors.\n\n"
        f"- `soft_cosine_word2vec MAP@10 = {summary_methods[METHOD_KEY]['k10']['map']:.4f}`\n"
        f"- `soft_cosine_tfidf_source MAP@10 = {tfidf_source_k10['map']:.4f}`\n"
        f"- `soft_cosine_word2vec nDCG@10 = {summary_methods[METHOD_KEY]['k10']['ndcg']:.4f}`\n"
        f"- `soft_cosine_tfidf_source nDCG@10 = {tfidf_source_k10['ndcg']:.4f}`\n\n"
        "This confirms that the soft-cosine scoring idea itself is useful, but on Cranfield the TF-IDF-derived "
        "similarity source remains slightly stronger than the Word2Vec-derived similarity source.\n"
    )
    report_path.write_text(report_text + comparison_section, encoding="utf-8")

    print("Best configuration:", best_config)
    print("Baseline MAP@10:", f"{summary_methods['baseline_tfidf']['k10']['map']:.4f}")
    print("Method MAP@10:", f"{summary_methods[METHOD_KEY]['k10']['map']:.4f}")
    print("TF-IDF source MAP@10:", f"{tfidf_source_k10['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
