#!/usr/bin/env python3
"""
Run baseline TF-IDF retrieval and a Word2Vec centroid-retrieval backbone
for the Cranfield collection.

The method trains Word2Vec on the stopword-removed Cranfield document tokens,
represents each document/query by the mean of its in-vocabulary word vectors,
and ranks documents with cosine similarity in the dense embedding space.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUDHEER_DIR = SCRIPT_DIR.parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
if str(SUDHEER_DIR) not in sys.path:
    sys.path.insert(0, str(SUDHEER_DIR))

from common import (  # noqa: E402
    ASSIGNMENT_ROOT as COMMON_ASSIGNMENT_ROOT,
    DenseCosineModel,
    Evaluation,
    TfidfIndex,
    approximate_randomization_pvalue,
    best_results_by_primary,
    build_average_embeddings,
    build_example_query_comparison,
    build_experiment_report,
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


METHOD_KEY = "word2vec_average"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sg_label(value: int) -> str:
    return "skipgram" if int(value) == 1 else "cbow"


def run_method(
    docs,
    queries,
    doc_ids,
    vector_size: int,
    window: int,
    sg: int,
    epochs: int,
    min_count: int,
    workers: int,
    vocab,
):
    artifacts = train_word2vec_artifacts(
        docs_tokens=docs,
        vocab=vocab,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    doc_vectors, doc_stats = build_average_embeddings(
        collection=docs,
        word_vectors=artifacts.word_vectors,
        vector_size=vector_size,
        term_weights=None,
    )
    query_vectors, query_stats = build_average_embeddings(
        collection=queries,
        word_vectors=artifacts.word_vectors,
        vector_size=vector_size,
        term_weights=None,
    )

    model = DenseCosineModel.build(METHOD_KEY, doc_vectors, doc_ids)
    rankings, score_vectors = model.rank(query_vectors)
    return artifacts, doc_stats, query_stats, rankings, score_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Word2Vec average-backbone experiments.")
    parser.add_argument("--dataset-dir", default=str(COMMON_ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--vector-sizes", default="50,100,150")
    parser.add_argument("--windows", default="2,5")
    parser.add_argument("--sg-values", default="0,1")
    parser.add_argument("--epochs-values", default="20")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline_tfidf"
    method_dir = output_dir / METHOD_KEY
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

    doc_feature_maps = [unigram_feature_map(doc) for doc in processed_docs]
    query_feature_maps = [unigram_feature_map(query) for query in processed_queries]

    baseline_start = time.time()
    baseline_index = TfidfIndex.build("baseline_tfidf", doc_feature_maps, doc_ids)
    baseline_rankings, _ = baseline_index.rank_feature_maps(query_feature_maps)
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

    vector_sizes = parse_int_list(args.vector_sizes)
    windows = parse_int_list(args.windows)
    sg_values = parse_int_list(args.sg_values)
    epochs_values = parse_int_list(args.epochs_values)

    sweep_results = []
    config_index = 0
    total_configs = len(vector_sizes) * len(windows) * len(sg_values) * len(epochs_values)
    sweep_start = time.time()

    for vector_size in vector_sizes:
        for window in windows:
            for sg in sg_values:
                for epochs in epochs_values:
                    config_index += 1
                    print(
                        f"[sweep {config_index}/{total_configs}] "
                        f"vec={vector_size} window={window} sg={sg_label(sg)} epochs={epochs}",
                        flush=True,
                    )

                    artifacts, doc_stats, query_stats, rankings, _ = run_method(
                        docs=processed_docs,
                        queries=processed_queries,
                        doc_ids=doc_ids,
                        vector_size=vector_size,
                        window=window,
                        sg=sg,
                        epochs=epochs,
                        min_count=args.min_count,
                        workers=args.workers,
                        vocab=baseline_index.vocab,
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
                            "sentences_seen": artifacts.sentences_seen,
                            "present_in_model": artifacts.present_in_model,
                            "vocab_terms": artifacts.vocab_terms,
                            "model_coverage": artifacts.model_coverage,
                            "avg_in_vocab_tokens_per_doc": doc_stats["avg_in_vocab_tokens_per_item"],
                            "avg_in_vocab_tokens_per_query": query_stats["avg_in_vocab_tokens_per_item"],
                            "zero_vector_docs": doc_stats["zero_vector_items"],
                            "zero_vector_queries": query_stats["zero_vector_items"],
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

    best_by_vector_size = best_results_by_primary(sweep_results, "vector_size")
    write_json(output_dir / "vector_size_sweep_best.json", best_by_vector_size)
    write_primary_sweep_csv(
        output_dir / "vector_size_sweep_best.csv",
        best_by_vector_size,
        [
            "vector_size",
            "window",
            "sg_label",
            "epochs",
            "model_coverage",
            "avg_in_vocab_tokens_per_doc",
            "avg_in_vocab_tokens_per_query",
            "zero_vector_docs",
            "zero_vector_queries",
            "precision_at_10",
            "recall_at_10",
            "fscore_at_10",
            "map_at_10",
            "ndcg_at_10",
            "mrr_at_10",
        ],
    )
    save_primary_sweep_plot(
        output_dir / "vector_size_sweep_best.png",
        best_by_vector_size,
        primary_key="vector_size",
        x_label="Word2Vec embedding size",
        title="Best Result Per Word2Vec Embedding Size",
    )

    artifacts, doc_stats, query_stats, method_rankings, _ = run_method(
        docs=processed_docs,
        queries=processed_queries,
        doc_ids=doc_ids,
        vector_size=int(best_config["vector_size"]),
        window=int(best_config["window"]),
        sg=int(best_config["sg"]),
        epochs=int(best_config["epochs"]),
        min_count=args.min_count,
        workers=args.workers,
        vocab=baseline_index.vocab,
    )
    method_runtime = time.time() - sweep_start
    method_metrics = compute_metrics(evaluator, method_rankings, query_ids, qrels)
    method_per_query = compute_per_query_metrics(evaluator, method_rankings, query_ids, qrels)

    save_metric_plot(
        method_dir / "eval_plot.png",
        method_metrics,
        "Average Word2Vec Backbone Metrics",
    )
    save_rankings(method_dir / "rankings_top20.json", query_ids, method_rankings, top_k=20)

    grouped_metrics = {row["vector_size"]: row["metrics_by_k"] for row in best_by_vector_size}
    method_labels = {row["vector_size"]: f"vec={row['vector_size']}" for row in best_by_vector_size}
    ordered_keys = [row["vector_size"] for row in best_by_vector_size]
    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        grouped_metrics,
        method_labels,
        ordered_keys,
        "Baseline vs Best Word2Vec Average Models by Embedding Size",
    )
    save_all_tuned_combinations_overlay(
        output_dir / "all_tuned_combinations_overlay.png",
        baseline_metrics,
        sweep_results,
        best_config,
        "Baseline vs All Tuned Word2Vec Average Configurations",
    )

    method_metrics_payload = {
        "best_config": best_config,
        "metrics_by_k": method_metrics,
        "k10": {metric: values[9] for metric, values in method_metrics.items()},
        "runtime_seconds": method_runtime,
        "present_in_model": artifacts.present_in_model,
        "vocab_terms": artifacts.vocab_terms,
        "model_coverage": artifacts.model_coverage,
        "avg_in_vocab_tokens_per_doc": doc_stats["avg_in_vocab_tokens_per_item"],
        "avg_in_vocab_tokens_per_query": query_stats["avg_in_vocab_tokens_per_item"],
        "zero_vector_docs": doc_stats["zero_vector_items"],
        "zero_vector_queries": query_stats["zero_vector_items"],
    }
    write_json(method_dir / "metrics.json", method_metrics_payload)

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
        "vector_size_sweep_best": best_by_vector_size,
        "methods": summary_methods,
        "delta_vs_baseline_at_10": {
            metric: summary_methods[METHOD_KEY]["k10"][metric] - summary_methods["baseline_tfidf"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        },
        "significance": significance,
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
            "all_configs_overlay": str(output_dir / "all_tuned_combinations_overlay.png"),
            "vector_size_sweep_json": str(output_dir / "vector_size_sweep_best.json"),
            "vector_size_sweep_csv": str(output_dir / "vector_size_sweep_best.csv"),
            "vector_size_sweep_plot": str(output_dir / "vector_size_sweep_best.png"),
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
                "vector_size_sweep_best": best_by_vector_size,
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
        title="Average Word2Vec Backbone Experiment Report",
        method_key=METHOD_KEY,
        method_label=METHOD_KEY,
        summary=summary,
        examples=examples,
        sweep_rows=best_by_vector_size,
        sweep_section_title="Embedding Size Sweep",
        sweep_columns=[
            ("vector_size", "Vector size", "{}"),
            ("window", "Window", "{}"),
            ("sg_label", "Objective", "{}"),
            ("epochs", "Epochs", "{}"),
            ("model_coverage", "Coverage", "{:.2%}"),
        ],
        best_config_fields=[
            ("vector_size", "vector_size", "{}"),
            ("window", "window", "{}"),
            ("training_objective", "sg_label", "{}"),
            ("epochs", "epochs", "{}"),
            ("model_coverage", "model_coverage", "{:.2%}"),
            ("avg_in_vocab_tokens_per_doc", "avg_in_vocab_tokens_per_doc", "{:.2f}"),
            ("avg_in_vocab_tokens_per_query", "avg_in_vocab_tokens_per_query", "{:.2f}"),
        ],
        method_summary_lines=[
            "The baseline system uses unigram TF-IDF with cosine similarity. "
            "The proposed backbone trains Word2Vec on the Cranfield document corpus and represents each document "
            "and query by the average of its in-vocabulary word vectors.",
            "This follows the simplest compositional Word2Vec strategy: convert token embeddings into a single "
            "document/query centroid, then rank with cosine similarity in the dense vector space.",
        ],
        hypothesis_lines=[
            "A centroid embedding should recover some semantic similarity that TF-IDF misses, especially when a "
            "relevant document uses related technical wording instead of the exact same surface form.",
            "Because the method compresses the full token sequence into one dense vector, it should be fast at "
            "query time, but it may blur fine-grained distinctions and phrase structure.",
        ],
        limitations_lines=[
            "The baseline VSM depends strongly on exact lexical overlap.",
            "Dense centroids can partially address semantic mismatch between queries and documents.",
            "Averaging is still context-light, so ambiguity and phrase ordering are only weakly handled.",
        ],
        interpretation_lines=[
            "Average Word2Vec is the cleanest way to turn word embeddings into a retrieval backbone. "
            "Its main strength is simplicity: once the embedding space is trained, both documents and queries can "
            "be embedded and compared very cheaply.",
            "Its main weakness is that every item is compressed into a single centroid. Important rare words can be "
            "washed out by more frequent words, and the method has no direct representation of local order or phrase boundaries.",
        ],
        citations=[
            (
                "Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (NeurIPS 2013)",
                "https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html",
            ),
            (
                "Arora, Liang, and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (ICLR 2017)",
                "https://openreview.net/forum?id=SyK00v5xx",
            ),
        ],
    )

    print("Best configuration:", best_config)
    print("Baseline MAP@10:", f"{summary_methods['baseline_tfidf']['k10']['map']:.4f}")
    print("Method MAP@10:", f"{summary_methods[METHOD_KEY]['k10']['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
