#!/usr/bin/env python3
"""
Run baseline TF-IDF retrieval and Smooth Inverse Frequency (SIF) embeddings
for the Cranfield collection.

The method follows the report setup:
1. Load pre-trained 50d GloVe vectors.
2. Estimate corpus token probabilities from the Cranfield document collection.
3. Build SIF-weighted dense vectors for documents and queries.
4. Optionally remove the top principal components from document/query vectors.
5. Rank documents with cosine similarity in the dense space.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD


SCRIPT_DIR = Path(__file__).resolve().parent
CHANDAN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = CHANDAN_DIR.parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
SUDHEER_DIR = PROJECT_DIR / "sudheer"
if str(SUDHEER_DIR) not in sys.path:
    sys.path.insert(0, str(SUDHEER_DIR))

from common import (  # noqa: E402
    ASSIGNMENT_ROOT as COMMON_ASSIGNMENT_ROOT,
    DenseCosineModel,
    Evaluation,
    TfidfIndex,
    approximate_randomization_pvalue,
    best_results_by_primary,
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
    unigram_feature_map,
    write_json,
    write_primary_sweep_csv,
    write_summary_csv,
)


METHOD_KEY = "sif_embeddings"
DEFAULT_LOCAL_GLOVE = ASSIGNMENT_ROOT / "project" / "pretrained_models" / "glove.6B.50d.w2v.txt"
DEFAULT_GENSIM_MODEL = "glove-wiki-gigaword-50"


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def format_smoothing(value: float) -> str:
    return f"{value:.0e}"


def flatten_collection(collection: Sequence[Sequence[Sequence[str]]]) -> list[list[str]]:
    return [[token for sentence in item for token in sentence] for item in collection]


def load_pretrained_glove_vectors(
    vocab: Sequence[str],
    glove_path: Path,
    gensim_model_name: str,
) -> dict:
    try:
        from gensim.models import KeyedVectors
    except Exception as exc:
        raise ImportError(
            "gensim is required to load the pre-trained GloVe vectors for SIF."
        ) from exc

    keyed_vectors = None
    source = None
    source_label = None

    if glove_path.exists():
        print(f"Loading local GloVe vectors from {glove_path}...", flush=True)
        keyed_vectors = KeyedVectors.load_word2vec_format(str(glove_path), binary=False)
        source = "local_glove_w2v"
        source_label = glove_path.name
    else:
        print(
            f"Local GloVe file not found at {glove_path}. Falling back to gensim downloader "
            f"('{gensim_model_name}').",
            flush=True,
        )
        try:
            import gensim.downloader as api

            keyed_vectors = api.load(gensim_model_name)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the pre-trained GloVe vectors. "
                f"Tried local path '{glove_path}' and gensim model '{gensim_model_name}'."
            ) from exc
        source = "gensim_downloader"
        source_label = gensim_model_name

    word_vectors = {}
    for term in vocab:
        if term in keyed_vectors:
            word_vectors[term] = np.asarray(keyed_vectors[term], dtype=np.float64)

    present_in_model = len(word_vectors)
    vocab_terms = len(vocab)
    model_coverage = float(present_in_model / vocab_terms) if vocab_terms else 0.0
    return {
        "word_vectors": word_vectors,
        "vector_size": int(keyed_vectors.vector_size),
        "present_in_model": present_in_model,
        "vocab_terms": vocab_terms,
        "model_coverage": model_coverage,
        "source": source,
        "source_label": source_label,
    }


def compute_corpus_probabilities(flat_docs: Sequence[Sequence[str]]) -> tuple[Dict[str, float], dict]:
    token_counts = Counter()
    total_tokens = 0
    for doc in flat_docs:
        token_counts.update(doc)
        total_tokens += len(doc)

    probabilities = {
        token: float(count / total_tokens)
        for token, count in token_counts.items()
        if total_tokens > 0
    }
    stats = {
        "total_tokens": int(total_tokens),
        "unique_terms": int(len(token_counts)),
    }
    return probabilities, stats


def build_sif_item_embeddings(
    collection_flat: Sequence[Sequence[str]],
    word_vectors: Dict[str, np.ndarray],
    vector_size: int,
    word_probabilities: Dict[str, float],
    smoothing: float,
) -> tuple[np.ndarray, dict]:
    embeddings = np.zeros((len(collection_flat), vector_size), dtype=np.float64)
    in_vocab_token_counts = []
    nonzero_items = 0

    for item_index, tokens in enumerate(collection_flat):
        weighted_sum = np.zeros(vector_size, dtype=np.float64)
        in_vocab_tokens = 0

        for token in tokens:
            vector = word_vectors.get(token)
            if vector is None:
                continue

            probability = float(word_probabilities.get(token, 0.0))
            weight = float(smoothing / (smoothing + probability))
            weighted_sum += weight * vector
            in_vocab_tokens += 1

        if in_vocab_tokens > 0:
            embeddings[item_index] = weighted_sum / float(in_vocab_tokens)
            nonzero_items += 1

        in_vocab_token_counts.append(in_vocab_tokens)

    zero_vector_items = len(collection_flat) - nonzero_items
    return embeddings, {
        "avg_in_vocab_tokens_per_item": float(np.mean(in_vocab_token_counts)) if in_vocab_token_counts else 0.0,
        "zero_vector_items": int(zero_vector_items),
        "nonzero_vector_ratio": float(nonzero_items / len(collection_flat)) if collection_flat else 0.0,
    }


def remove_principal_components(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if n_components <= 0:
        return doc_embeddings, query_embeddings, {
            "pc_removal_applied": False,
            "explained_variance_ratio": [],
            "docs_used_for_svd": int(len(doc_embeddings)),
        }

    nonzero_doc_mask = np.linalg.norm(doc_embeddings, axis=1) > 0.0
    svd_input = doc_embeddings[nonzero_doc_mask]
    if len(svd_input) < max(2, n_components):
        return doc_embeddings, query_embeddings, {
            "pc_removal_applied": False,
            "explained_variance_ratio": [],
            "docs_used_for_svd": int(len(svd_input)),
            "pc_removal_skipped": True,
        }

    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd.fit(svd_input)
    principal_components = np.asarray(svd.components_, dtype=np.float64)

    doc_centered = doc_embeddings - doc_embeddings.dot(principal_components.T).dot(principal_components)
    query_centered = query_embeddings - query_embeddings.dot(principal_components.T).dot(principal_components)
    return doc_centered, query_centered, {
        "pc_removal_applied": True,
        "explained_variance_ratio": [float(value) for value in svd.explained_variance_ratio_],
        "docs_used_for_svd": int(len(svd_input)),
    }


def run_sif_configuration(
    docs_flat: Sequence[Sequence[str]],
    queries_flat: Sequence[Sequence[str]],
    doc_ids: Sequence[int],
    query_ids: Sequence[int],
    qrels,
    evaluator: Evaluation,
    word_vectors: Dict[str, np.ndarray],
    vector_size: int,
    word_probabilities: Dict[str, float],
    smoothing: float,
    n_components: int,
) -> dict:
    started = time.time()
    doc_embeddings, doc_stats = build_sif_item_embeddings(
        docs_flat,
        word_vectors,
        vector_size,
        word_probabilities,
        smoothing,
    )
    query_embeddings, query_stats = build_sif_item_embeddings(
        queries_flat,
        word_vectors,
        vector_size,
        word_probabilities,
        smoothing,
    )
    doc_embeddings, query_embeddings, pc_stats = remove_principal_components(
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        n_components=n_components,
    )

    model = DenseCosineModel.build(METHOD_KEY, doc_embeddings, doc_ids)
    rankings, _ = model.rank(query_embeddings)
    metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
    k10 = {metric: values[9] for metric, values in metrics_by_k.items()}
    runtime_seconds = time.time() - started

    return {
        "a": float(smoothing),
        "a_label": format_smoothing(smoothing),
        "n_components": int(n_components),
        "runtime_seconds": runtime_seconds,
        "doc_stats": doc_stats,
        "query_stats": query_stats,
        "pc_stats": pc_stats,
        "metrics_by_k": metrics_by_k,
        "precision_at_10": k10["precision"],
        "recall_at_10": k10["recall"],
        "fscore_at_10": k10["fscore"],
        "map_at_10": k10["map"],
        "ndcg_at_10": k10["ndcg"],
        "mrr_at_10": k10["mrr"],
        "rankings": rankings,
        "doc_embeddings": doc_embeddings,
        "query_embeddings": query_embeddings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIF embeddings experiments.")
    parser.add_argument("--dataset-dir", default=str(COMMON_ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--glove-path", default=str(DEFAULT_LOCAL_GLOVE))
    parser.add_argument("--gensim-model", default=DEFAULT_GENSIM_MODEL)
    parser.add_argument("--a-values", default="0.01,0.001,0.0001")
    parser.add_argument("--pc-values", default="0,1,2")
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
    docs_flat = flatten_collection(processed_docs)
    queries_flat = flatten_collection(processed_queries)

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

    retrieval_vocab = sorted({token for doc in docs_flat for token in doc} | {token for query in queries_flat for token in query})
    glove_artifacts = load_pretrained_glove_vectors(
        vocab=retrieval_vocab,
        glove_path=Path(args.glove_path),
        gensim_model_name=args.gensim_model,
    )
    word_probabilities, probability_stats = compute_corpus_probabilities(docs_flat)

    a_values = parse_float_list(args.a_values)
    pc_values = parse_int_list(args.pc_values)

    sweep_results = []
    best_result = None
    total_configs = len(a_values) * len(pc_values)
    config_counter = 0
    sweep_started = time.time()

    for smoothing in a_values:
        for n_components in pc_values:
            config_counter += 1
            print(
                f"[sweep {config_counter}/{total_configs}] "
                f"a={format_smoothing(smoothing)} remove_pcs={n_components}",
                flush=True,
            )
            result = run_sif_configuration(
                docs_flat=docs_flat,
                queries_flat=queries_flat,
                doc_ids=doc_ids,
                query_ids=query_ids,
                qrels=qrels,
                evaluator=evaluator,
                word_vectors=glove_artifacts["word_vectors"],
                vector_size=glove_artifacts["vector_size"],
                word_probabilities=word_probabilities,
                smoothing=smoothing,
                n_components=n_components,
            )
            print(
                f"  MAP@10={result['map_at_10']:.4f} "
                f"nDCG@10={result['ndcg_at_10']:.4f} "
                f"MRR@10={result['mrr_at_10']:.4f}",
                flush=True,
            )
            sweep_results.append(
                {
                    "a": result["a"],
                    "a_label": result["a_label"],
                    "n_components": result["n_components"],
                    "runtime_seconds": result["runtime_seconds"],
                    "vector_size": glove_artifacts["vector_size"],
                    "embedding_source": glove_artifacts["source"],
                    "embedding_source_label": glove_artifacts["source_label"],
                    "present_in_model": glove_artifacts["present_in_model"],
                    "vocab_terms": glove_artifacts["vocab_terms"],
                    "model_coverage": glove_artifacts["model_coverage"],
                    "corpus_probability_total_tokens": probability_stats["total_tokens"],
                    "corpus_probability_unique_terms": probability_stats["unique_terms"],
                    "avg_in_vocab_tokens_per_doc": result["doc_stats"]["avg_in_vocab_tokens_per_item"],
                    "avg_in_vocab_tokens_per_query": result["query_stats"]["avg_in_vocab_tokens_per_item"],
                    "zero_vector_docs": result["doc_stats"]["zero_vector_items"],
                    "zero_vector_queries": result["query_stats"]["zero_vector_items"],
                    "pc_removal_applied": result["pc_stats"]["pc_removal_applied"],
                    "pc_explained_variance_ratio": result["pc_stats"]["explained_variance_ratio"],
                    "docs_used_for_svd": result["pc_stats"]["docs_used_for_svd"],
                    "metrics_by_k": result["metrics_by_k"],
                    "precision_at_10": result["precision_at_10"],
                    "recall_at_10": result["recall_at_10"],
                    "fscore_at_10": result["fscore_at_10"],
                    "map_at_10": result["map_at_10"],
                    "ndcg_at_10": result["ndcg_at_10"],
                    "mrr_at_10": result["mrr_at_10"],
                }
            )

            if best_result is None or metric_sort_key(result) > metric_sort_key(best_result):
                best_result = result

    if best_result is None:
        raise RuntimeError("No SIF sweep results were produced.")

    method_runtime = time.time() - sweep_started
    sweep_results_sorted = sorted(sweep_results, key=metric_sort_key, reverse=True)
    write_json(output_dir / "config_sweep.json", sweep_results)
    write_primary_sweep_csv(
        output_dir / "config_sweep_k10.csv",
        sweep_results,
        [
            "a",
            "a_label",
            "n_components",
            "runtime_seconds",
            "vector_size",
            "embedding_source_label",
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

    best_by_a = best_results_by_primary(sweep_results_sorted, "a")
    write_json(output_dir / "a_sweep_best.json", best_by_a)
    write_primary_sweep_csv(
        output_dir / "a_sweep_best.csv",
        best_by_a,
        [
            "a",
            "a_label",
            "n_components",
            "vector_size",
            "embedding_source_label",
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
        output_dir / "a_sweep_best.png",
        best_by_a,
        primary_key="a",
        x_label="SIF smoothing parameter a",
        title="Best Result Per SIF Smoothing Parameter",
    )

    method_metrics = best_result["metrics_by_k"]
    method_per_query = compute_per_query_metrics(evaluator, best_result["rankings"], query_ids, qrels)
    save_metric_plot(
        method_dir / "eval_plot.png",
        method_metrics,
        "Pure SIF Embeddings Metrics",
    )
    save_rankings(method_dir / "rankings_top20.json", query_ids, best_result["rankings"], top_k=20)

    method_metrics_payload = {
        "best_config": next(
            row
            for row in sweep_results_sorted
            if row["a"] == best_result["a"] and row["n_components"] == best_result["n_components"]
        ),
        "metrics_by_k": method_metrics,
        "k10": {metric: values[9] for metric, values in method_metrics.items()},
        "runtime_seconds": method_runtime,
        "best_config_runtime_seconds": best_result["runtime_seconds"],
        "vector_size": glove_artifacts["vector_size"],
        "embedding_source": glove_artifacts["source"],
        "embedding_source_label": glove_artifacts["source_label"],
        "present_in_model": glove_artifacts["present_in_model"],
        "vocab_terms": glove_artifacts["vocab_terms"],
        "model_coverage": glove_artifacts["model_coverage"],
        "corpus_probability_total_tokens": probability_stats["total_tokens"],
        "corpus_probability_unique_terms": probability_stats["unique_terms"],
        "avg_in_vocab_tokens_per_doc": best_result["doc_stats"]["avg_in_vocab_tokens_per_item"],
        "avg_in_vocab_tokens_per_query": best_result["query_stats"]["avg_in_vocab_tokens_per_item"],
        "zero_vector_docs": best_result["doc_stats"]["zero_vector_items"],
        "zero_vector_queries": best_result["query_stats"]["zero_vector_items"],
        "pc_removal_applied": best_result["pc_stats"]["pc_removal_applied"],
        "pc_explained_variance_ratio": best_result["pc_stats"]["explained_variance_ratio"],
        "docs_used_for_svd": best_result["pc_stats"]["docs_used_for_svd"],
    }
    write_json(method_dir / "metrics.json", method_metrics_payload)

    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        {"pure_sif": method_metrics},
        {"pure_sif": "pure_sif"},
        ["pure_sif"],
        "Baseline vs Pure SIF Embeddings",
    )
    save_all_tuned_combinations_overlay(
        output_dir / "all_tuned_combinations_overlay.png",
        baseline_metrics,
        sweep_results,
        method_metrics_payload["best_config"],
        "Baseline vs All Tuned SIF Configurations",
    )

    rows = []
    summary_methods = {}
    for method_name, metrics_payload in [
        ("baseline_tfidf", {"metrics_by_k": baseline_metrics, "runtime_seconds": baseline_runtime}),
        (METHOD_KEY, method_metrics_payload),
    ]:
        k10 = {metric: values[9] for metric, values in metrics_payload["metrics_by_k"].items()}
        rows.append(
            {
                "method": method_name,
                "precision@10": k10["precision"],
                "recall@10": k10["recall"],
                "fscore@10": k10["fscore"],
                "map@10": k10["map"],
                "ndcg@10": k10["ndcg"],
                "mrr@10": k10["mrr"],
                "runtime_seconds": metrics_payload["runtime_seconds"],
            }
        )
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
        method_rankings=best_result["rankings"],
        baseline_per_query=baseline_per_query,
        method_per_query=method_per_query,
        output_dir=output_dir,
    )

    summary = {
        "dataset": str(dataset_dir),
        "best_config": method_metrics_payload["best_config"],
        "a_sweep_best": best_by_a,
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
            "a_sweep_json": str(output_dir / "a_sweep_best.json"),
            "a_sweep_csv": str(output_dir / "a_sweep_best.csv"),
            "a_sweep_plot": str(output_dir / "a_sweep_best.png"),
            "config_sweep_json": str(output_dir / "config_sweep.json"),
            "config_sweep_csv": str(output_dir / "config_sweep_k10.csv"),
            "example_markdown": str(output_dir / "example_query_comparison.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "comparison_summary.json",
        {
            "config": {
                "best_config": method_metrics_payload["best_config"],
                "a_sweep_best": best_by_a,
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

    report_rows = sorted(sweep_results, key=lambda row: (-row["a"], row["n_components"]))
    build_experiment_report(
        output_path=SCRIPT_DIR / "experiment_report.md",
        title="Smooth Inverse Frequency Embeddings Experiment Report",
        method_key=METHOD_KEY,
        method_label=METHOD_KEY,
        summary=summary,
        examples=examples,
        sweep_rows=report_rows,
        sweep_section_title="SIF Configuration Sweep",
        sweep_columns=[
            ("a_label", "Smoothing a", "{}"),
            ("n_components", "Removed PCs", "{}"),
        ],
        best_config_fields=[
            ("smoothing_a", "a", "{}"),
            ("removed_principal_components", "n_components", "{}"),
            ("vector_size", "vector_size", "{}"),
            ("embedding_source", "embedding_source_label", "{}"),
            ("model_coverage", "model_coverage", "{:.2%}"),
            ("avg_in_vocab_tokens_per_doc", "avg_in_vocab_tokens_per_doc", "{:.2f}"),
            ("avg_in_vocab_tokens_per_query", "avg_in_vocab_tokens_per_query", "{:.2f}"),
        ],
        method_summary_lines=[
            "The baseline system uses unigram TF-IDF with cosine similarity. "
            "The proposed SIF backbone replaces sparse lexical vectors with one dense vector per query/document.",
            "Each token vector is downweighted by the smooth inverse frequency term `a / (a + p(w))`, "
            "where `p(w)` is estimated from the Cranfield document collection. The resulting dense vectors are "
            "optionally purified by subtracting the top principal components learned from document embeddings.",
        ],
        hypothesis_lines=[
            "Plain embedding averages often let frequent words dominate the final centroid. "
            "SIF should help by suppressing common terms while keeping rare technical vocabulary more visible.",
            "Principal-component removal should further improve retrieval by subtracting broad discourse directions "
            "that are shared across many documents but are not specific to the current query intent.",
        ],
        limitations_lines=[
            "The TF-IDF baseline depends heavily on exact lexical overlap.",
            "SIF tries to address vocabulary mismatch without falling back to direct token matching.",
            "Because the final representation is still a single dense vector, rare multi-term technical intent can still be blurred.",
        ],
        interpretation_lines=[
            "SIF is a stronger dense baseline than plain average embeddings because it addresses two known failure modes: "
            "dominance by frequent tokens and over-shared corpus directions.",
            "Even with those corrections, the method still compresses each item into one centroid. "
            "That compression is especially costly on Cranfield, where exact rare terminology often carries most of the relevance signal.",
        ],
        citations=[
            (
                "Arora, Liang, and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (ICLR 2017)",
                "https://openreview.net/forum?id=SyK00v5xx",
            ),
            (
                "Pennington, Socher, and Manning, GloVe: Global Vectors for Word Representation (EMNLP 2014)",
                "https://aclanthology.org/D14-1162/",
            ),
        ],
    )

    print("Best configuration:", method_metrics_payload["best_config"])
    print("Baseline MAP@10:", f"{summary_methods['baseline_tfidf']['k10']['map']:.4f}")
    print("Pure SIF MAP@10:", f"{summary_methods[METHOD_KEY]['k10']['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
