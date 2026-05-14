#!/usr/bin/env python3
"""
Run SIF transfer experiments on adapted token spaces derived from:
1. TF-IDF term-document profiles
2. LSA term embeddings
3. ESA term concept vectors

Each backbone supplies a vector per token, after which we run the same SIF
pipeline used in the pure embedding experiment: weighted averaging followed by
optional principal-component removal and cosine ranking.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
SUDHEER_DIR = ASSIGNMENT_ROOT
if str(SUDHEER_DIR) not in sys.path:
    sys.path.insert(0, str(SUDHEER_DIR))

from common import (  # noqa: E402
    ASSIGNMENT_ROOT as COMMON_ASSIGNMENT_ROOT,
    DenseCosineModel,
    Evaluation,
    TfidfIndex,
    approximate_randomization_pvalue,
    build_example_query_comparison,
    compute_metrics,
    compute_per_query_metrics,
    load_json,
    metric_sort_key,
    normalize_collection,
    save_metric_plot,
    save_overlay_plot,
    save_rankings,
    unigram_feature_map,
    write_json,
    write_summary_csv,
)


DEFAULT_BACKBONES = "tfidf_term,lsa_term,esa_term"


@dataclass
class BackboneSpec:
    name: str
    label: str
    notes: str
    model: KeyedVectors
    doc_token_coverage: float
    query_token_coverage: float


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def format_smoothing(value: float) -> str:
    return f"{value:.0e}"


def flatten_collection(collection: Sequence[Sequence[Sequence[str]]]) -> list[list[str]]:
    return [[token for sentence in item for token in sentence] for item in collection]


def flatten_to_texts(collection: Sequence[Sequence[Sequence[str]]]) -> list[str]:
    return [" ".join(token for sentence in item for token in sentence) for item in collection]


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


def count_covered_tokens(collection_flat: Sequence[Sequence[str]], model: KeyedVectors) -> tuple[int, int]:
    covered = 0
    total = 0
    for item in collection_flat:
        total += len(item)
        for token in item:
            if token in model:
                covered += 1
    return covered, total


def compute_coverage_ratio(collection_flat: Sequence[Sequence[str]], model: KeyedVectors) -> float:
    covered, total = count_covered_tokens(collection_flat, model)
    return float(covered / total) if total else 0.0


def keep_top_k_per_row(matrix: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    if top_k <= 0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)

    rows = []
    cols = []
    data = []
    for row_index in range(matrix.shape[0]):
        start = matrix.indptr[row_index]
        end = matrix.indptr[row_index + 1]
        row_cols = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size == 0:
            continue

        if row_data.size > top_k:
            keep_idx = np.argpartition(row_data, -top_k)[-top_k:]
            row_cols = row_cols[keep_idx]
            row_data = row_data[keep_idx]

        positive = row_data > 0
        row_cols = row_cols[positive]
        row_data = row_data[positive]
        if row_data.size == 0:
            continue

        rows.extend([row_index] * len(row_cols))
        cols.extend(row_cols.tolist())
        data.extend(row_data.astype(np.float32).tolist())

    return sparse.csr_matrix((data, (rows, cols)), shape=matrix.shape, dtype=np.float32)


def make_keyed_vectors(vocab: Sequence[str], vectors: np.ndarray) -> KeyedVectors:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    model = KeyedVectors(vector_size=int(vectors.shape[1]))
    model.add_vectors(list(vocab), vectors)
    return model


def build_term_spaces(
    processed_docs: Sequence[Sequence[Sequence[str]]],
    lsa_components: int,
    esa_top_concepts: int,
) -> Dict[str, tuple[list[str], np.ndarray]]:
    doc_texts = flatten_to_texts(processed_docs)
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
        sublinear_tf=True,
        max_df=1.0,
        min_df=1,
        norm="l2",
        ngram_range=(1, 1),
    )
    doc_term_matrix = vectorizer.fit_transform(doc_texts)
    vocab = list(vectorizer.get_feature_names_out())
    term_document = doc_term_matrix.T.tocsr()

    tfidf_term_vectors = term_document.toarray().astype(np.float32)

    max_components = min(term_document.shape[0] - 1, term_document.shape[1] - 1)
    components = max(2, min(lsa_components, max_components))
    lsa_vectors = TruncatedSVD(
        n_components=components,
        random_state=42,
    ).fit_transform(term_document).astype(np.float32)

    esa_sparse = keep_top_k_per_row(term_document, esa_top_concepts)
    esa_vectors = esa_sparse.toarray().astype(np.float32)

    return {
        "tfidf_term": (vocab, tfidf_term_vectors),
        "lsa_term": (vocab, lsa_vectors),
        "esa_term": (vocab, esa_vectors),
    }


def build_backbones(
    docs_flat: Sequence[Sequence[str]],
    queries_flat: Sequence[Sequence[str]],
    processed_docs: Sequence[Sequence[Sequence[str]]],
    lsa_components: int,
    esa_top_concepts: int,
) -> Dict[str, BackboneSpec]:
    term_spaces = build_term_spaces(processed_docs, lsa_components, esa_top_concepts)

    specs = {
        "tfidf_term": ("TF-IDF term space", "Adapted application on term-document profile vectors."),
        "lsa_term": ("LSA term space", "Adapted application on latent term embeddings from SVD."),
        "esa_term": ("ESA term space", "Adapted application on explicit concept vectors."),
    }

    backbones: Dict[str, BackboneSpec] = {}
    for name, (label, notes) in specs.items():
        vocab, vectors = term_spaces[name]
        model = make_keyed_vectors(vocab, vectors)
        backbones[name] = BackboneSpec(
            name=name,
            label=label,
            notes=notes,
            model=model,
            doc_token_coverage=compute_coverage_ratio(docs_flat, model),
            query_token_coverage=compute_coverage_ratio(queries_flat, model),
        )
    return backbones


def build_sif_item_embeddings(
    collection_flat: Sequence[Sequence[str]],
    model: KeyedVectors,
    word_probabilities: Dict[str, float],
    smoothing: float,
) -> tuple[np.ndarray, dict]:
    vector_size = int(model.vector_size)
    embeddings = np.zeros((len(collection_flat), vector_size), dtype=np.float64)
    in_vocab_token_counts = []
    nonzero_items = 0

    for item_index, tokens in enumerate(collection_flat):
        weighted_sum = np.zeros(vector_size, dtype=np.float64)
        in_vocab_tokens = 0

        for token in tokens:
            if token not in model:
                continue
            probability = float(word_probabilities.get(token, 0.0))
            weight = float(smoothing / (smoothing + probability))
            weighted_sum += weight * model.get_vector(token, norm=False)
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
    max_components = min(svd_input.shape[0] - 1, svd_input.shape[1] - 1) if svd_input.size else 0
    effective_components = min(n_components, max_components)
    if effective_components <= 0:
        return doc_embeddings, query_embeddings, {
            "pc_removal_applied": False,
            "explained_variance_ratio": [],
            "docs_used_for_svd": int(len(svd_input)),
            "pc_removal_skipped": True,
        }

    svd = TruncatedSVD(n_components=effective_components, n_iter=7, random_state=42)
    svd.fit(svd_input)
    principal_components = np.asarray(svd.components_, dtype=np.float64)

    doc_centered = doc_embeddings - doc_embeddings.dot(principal_components.T).dot(principal_components)
    query_centered = query_embeddings - query_embeddings.dot(principal_components.T).dot(principal_components)
    return doc_centered, query_centered, {
        "pc_removal_applied": True,
        "explained_variance_ratio": [float(value) for value in svd.explained_variance_ratio_],
        "docs_used_for_svd": int(len(svd_input)),
        "effective_components": int(effective_components),
    }


def evaluate_backbone(
    backbone: BackboneSpec,
    evaluator: Evaluation,
    docs_flat: Sequence[Sequence[str]],
    queries_flat: Sequence[Sequence[str]],
    doc_ids: Sequence[int],
    query_ids: Sequence[int],
    qrels,
    word_probabilities: Dict[str, float],
    a_values: Sequence[float],
    n_components_values: Sequence[int],
) -> dict:
    sweep_results = []
    best_run = None

    for smoothing in a_values:
        for n_components in n_components_values:
            started = time.time()
            doc_embeddings, doc_stats = build_sif_item_embeddings(
                docs_flat,
                backbone.model,
                word_probabilities,
                smoothing,
            )
            query_embeddings, query_stats = build_sif_item_embeddings(
                queries_flat,
                backbone.model,
                word_probabilities,
                smoothing,
            )
            doc_embeddings, query_embeddings, pc_stats = remove_principal_components(
                doc_embeddings,
                query_embeddings,
                n_components,
            )
            model = DenseCosineModel.build(f"sif_{backbone.name}", doc_embeddings, doc_ids)
            rankings, _ = model.rank(query_embeddings)
            metrics = compute_metrics(evaluator, rankings, query_ids, qrels)
            k10 = {metric: values[9] for metric, values in metrics.items()}

            result = {
                "a": float(smoothing),
                "a_label": format_smoothing(smoothing),
                "n_components": int(n_components),
                "runtime_seconds": time.time() - started,
                "metrics_by_k": metrics,
                "k10": k10,
                "precision_at_10": k10["precision"],
                "recall_at_10": k10["recall"],
                "fscore_at_10": k10["fscore"],
                "map_at_10": k10["map"],
                "ndcg_at_10": k10["ndcg"],
                "mrr_at_10": k10["mrr"],
                "avg_in_vocab_tokens_per_doc": doc_stats["avg_in_vocab_tokens_per_item"],
                "avg_in_vocab_tokens_per_query": query_stats["avg_in_vocab_tokens_per_item"],
                "zero_vector_docs": doc_stats["zero_vector_items"],
                "zero_vector_queries": query_stats["zero_vector_items"],
                "pc_removal_applied": pc_stats["pc_removal_applied"],
                "pc_explained_variance_ratio": pc_stats["explained_variance_ratio"],
                "docs_used_for_svd": pc_stats["docs_used_for_svd"],
                "effective_components": pc_stats.get("effective_components", 0),
                "rankings": rankings,
            }
            sweep_results.append(result)

            if best_run is None or metric_sort_key(result) > metric_sort_key(best_run):
                best_run = result

    if best_run is None:
        raise RuntimeError(f"No SIF results were produced for backbone '{backbone.name}'.")

    return {
        "name": backbone.name,
        "label": backbone.label,
        "notes": backbone.notes,
        "doc_token_coverage": backbone.doc_token_coverage,
        "query_token_coverage": backbone.query_token_coverage,
        "best": best_run,
        "sweep": sweep_results,
    }


def save_k10_comparison_plot(
    path: Path,
    baseline_k10: dict,
    backbone_results: Sequence[dict],
) -> None:
    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
    metric_labels = {
        "precision": "P@10",
        "recall": "R@10",
        "fscore": "F@10",
        "map": "MAP@10",
        "ndcg": "nDCG@10",
        "mrr": "MRR@10",
    }

    labels = ["baseline_tfidf"] + [row["label"] for row in backbone_results]
    values_by_metric = {
        metric: [baseline_k10[metric]] + [row["best"]["k10"][metric] for row in backbone_results]
        for metric in metric_names
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    x = np.arange(len(labels))
    for axis, metric in zip(axes, metric_names):
        bars = axis.bar(x, values_by_metric[metric], color=["black"] + ["tab:blue"] * len(backbone_results))
        axis.set_title(metric_labels[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=22, ha="right")
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
        for bar in bars:
            height = bar.get_height()
            axis.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("SIF Transfer Backbones at k=10", y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.30, bottom=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_transfer_report(
    path: Path,
    summary: dict,
    best_overall: dict,
    example_rows: Sequence[dict],
) -> None:
    baseline = summary["baseline"]["k10"]
    backbones = summary["backbones"]
    lines = [
        "# SIF Transferability Report",
        "",
        "## What this experiment does",
        "",
        "Instead of using only pre-trained word vectors, this experiment adapts `TF-IDF`, `LSA`, and `ESA` into token-level vector spaces and then runs the same SIF pipeline on top of those token vectors.",
        "",
        "## Best backbone per requested family",
        "",
        "| Backbone | Coverage (docs) | Coverage (queries) | Best a | Removed PCs | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in backbones:
        best = row["best"]
        config = best["config"]
        lines.append(
            f"| {row['label']} | {row['doc_token_coverage']:.2%} | {row['query_token_coverage']:.2%} | "
            f"{config['a_label']} | {config['n_components']} | {best['k10']['precision']:.4f} | "
            f"{best['k10']['recall']:.4f} | {best['k10']['fscore']:.4f} | {best['k10']['map']:.4f} | "
            f"{best['k10']['ndcg']:.4f} | {best['k10']['mrr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline reference",
            "",
            f"- `baseline_tfidf MAP@10 = {baseline['map']:.4f}`",
            f"- `best transferred SIF backbone = {best_overall['label']}`",
            f"- `best transferred SIF MAP@10 = {best_overall['best']['k10']['map']:.4f}`",
            "",
            "## Notes",
            "",
            "- TF-IDF term space uses each term's document-distribution profile as its token vector.",
            "- LSA term space uses each term's latent SVD embedding as its token vector.",
            "- ESA term space uses each term's pruned explicit concept activation vector as its token vector.",
            "- SIF itself is unchanged: weighted averaging plus optional principal-component removal.",
            "",
        ]
    )

    if example_rows:
        lines.extend(
            [
                "## Example Query Comparison For Best Backbone",
                "",
                "| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in example_rows:
            lines.append(
                f"| {row['query_id']} | {row['baseline_hits_at_5']} | {row['method_hits_at_5']} | "
                f"{row['baseline_ap_at_10']:.4f} | {row['method_ap_at_10']:.4f} | {row['delta_ap_at_10']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Output Files",
            "",
            f"- `summary.json`: `{summary['paths']['summary_json']}`",
            f"- `summary_k10.csv`: `{summary['paths']['summary_csv']}`",
            f"- `eval_overlay.png`: `{summary['paths']['overlay_plot']}`",
            f"- `k10_backbone_bars.png`: `{summary['paths']['k10_plot']}`",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIF transfer experiments on TF-IDF, LSA, and ESA token spaces.")
    parser.add_argument("--dataset-dir", default=str(COMMON_ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output_transferability"))
    parser.add_argument("--backbones", default=DEFAULT_BACKBONES)
    parser.add_argument("--a-values", default="0.01,0.001,0.0001")
    parser.add_argument("--n-components", default="0,1,2")
    parser.add_argument("--lsa-components", type=int, default=128)
    parser.add_argument("--esa-top-concepts", type=int, default=100)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline_tfidf"
    baseline_dir.mkdir(parents=True, exist_ok=True)

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
    baseline_k10 = {metric: values[9] for metric, values in baseline_metrics.items()}
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
            "k10": baseline_k10,
            "runtime_seconds": baseline_runtime,
        },
    )

    backbone_names = parse_csv_list(args.backbones)
    backbones = build_backbones(
        docs_flat=docs_flat,
        queries_flat=queries_flat,
        processed_docs=processed_docs,
        lsa_components=args.lsa_components,
        esa_top_concepts=args.esa_top_concepts,
    )
    missing = [name for name in backbone_names if name not in backbones]
    if missing:
        raise ValueError(f"Unknown backbone(s): {', '.join(missing)}")

    word_probabilities, probability_stats = compute_corpus_probabilities(docs_flat)
    a_values = parse_float_list(args.a_values)
    n_component_values = parse_int_list(args.n_components)

    backbone_results = []
    summary_rows = [
        {
            "method": "baseline_tfidf",
            "precision@10": baseline_k10["precision"],
            "recall@10": baseline_k10["recall"],
            "fscore@10": baseline_k10["fscore"],
            "map@10": baseline_k10["map"],
            "ndcg@10": baseline_k10["ndcg"],
            "mrr@10": baseline_k10["mrr"],
            "runtime_seconds": baseline_runtime,
        }
    ]

    for backbone_name in backbone_names:
        backbone = backbones[backbone_name]
        print(f"Running SIF transfer on {backbone.label}...", flush=True)
        result = evaluate_backbone(
            backbone=backbone,
            evaluator=evaluator,
            docs_flat=docs_flat,
            queries_flat=queries_flat,
            doc_ids=doc_ids,
            query_ids=query_ids,
            qrels=qrels,
            word_probabilities=word_probabilities,
            a_values=a_values,
            n_components_values=n_component_values,
        )

        method_key = f"sif_{backbone.name}"
        method_dir = output_dir / method_key
        method_dir.mkdir(parents=True, exist_ok=True)

        best = result["best"]
        best_metrics = best["metrics_by_k"]
        method_per_query = compute_per_query_metrics(evaluator, best["rankings"], query_ids, qrels)

        save_metric_plot(
            method_dir / "eval_plot.png",
            best_metrics,
            f"SIF on {backbone.label}",
        )
        save_rankings(method_dir / "rankings_top20.json", query_ids, best["rankings"], top_k=20)
        write_json(method_dir / "config_sweep.json", result["sweep"])
        write_json(
            method_dir / "metrics.json",
            {
                "backbone": result["name"],
                "label": result["label"],
                "notes": result["notes"],
                "doc_token_coverage": result["doc_token_coverage"],
                "query_token_coverage": result["query_token_coverage"],
                "best_config": {
                    "a": best["a"],
                    "a_label": best["a_label"],
                    "n_components": best["n_components"],
                },
                "metrics_by_k": best_metrics,
                "k10": best["k10"],
                "runtime_seconds": sum(row["runtime_seconds"] for row in result["sweep"]),
                "significance": {
                    "ap_at_10_pvalue": approximate_randomization_pvalue(
                        baseline_per_query["average_precision"],
                        method_per_query["average_precision"],
                    ),
                    "ndcg_at_10_pvalue": approximate_randomization_pvalue(
                        baseline_per_query["ndcg"],
                        method_per_query["ndcg"],
                    ),
                },
            },
        )

        backbone_results.append(result)
        summary_rows.append(
            {
                "method": method_key,
                "precision@10": best["k10"]["precision"],
                "recall@10": best["k10"]["recall"],
                "fscore@10": best["k10"]["fscore"],
                "map@10": best["k10"]["map"],
                "ndcg@10": best["k10"]["ndcg"],
                "mrr@10": best["k10"]["mrr"],
                "runtime_seconds": sum(row["runtime_seconds"] for row in result["sweep"]),
            }
        )

    backbone_results.sort(key=lambda row: row["best"]["k10"]["map"], reverse=True)
    best_overall = backbone_results[0]

    grouped_metrics = {
        row["name"]: row["best"]["metrics_by_k"]
        for row in backbone_results
    }
    method_labels = {
        row["name"]: row["label"]
        for row in backbone_results
    }
    ordered_keys = [row["name"] for row in backbone_results]
    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        grouped_metrics,
        method_labels,
        ordered_keys,
        "Baseline vs SIF Transfer Backbones",
    )
    save_k10_comparison_plot(output_dir / "k10_backbone_bars.png", baseline_k10, backbone_results)

    best_per_query = compute_per_query_metrics(
        evaluator,
        best_overall["best"]["rankings"],
        query_ids,
        qrels,
    )
    example_rows = build_example_query_comparison(
        query_ids=query_ids,
        queries_json=queries_json,
        qrels=qrels,
        baseline_rankings=baseline_rankings,
        method_rankings=best_overall["best"]["rankings"],
        baseline_per_query=baseline_per_query,
        method_per_query=best_per_query,
        output_dir=output_dir,
    )

    write_summary_csv(output_dir / "summary_k10.csv", summary_rows)

    summary = {
        "dataset": str(dataset_dir),
        "settings": {
            "backbones": backbone_names,
            "a_values": a_values,
            "n_components": n_component_values,
            "lsa_components": args.lsa_components,
            "esa_top_concepts": args.esa_top_concepts,
            "corpus_probability_total_tokens": probability_stats["total_tokens"],
            "corpus_probability_unique_terms": probability_stats["unique_terms"],
        },
        "baseline": {
            "metrics_by_k": baseline_metrics,
            "k10": baseline_k10,
            "runtime_seconds": baseline_runtime,
        },
        "backbones": [
            {
                "name": row["name"],
                "label": row["label"],
                "notes": row["notes"],
                "doc_token_coverage": row["doc_token_coverage"],
                "query_token_coverage": row["query_token_coverage"],
                "best": {
                    "config": {
                        "a": row["best"]["a"],
                        "a_label": row["best"]["a_label"],
                        "n_components": row["best"]["n_components"],
                    },
                    "runtime_seconds": row["best"]["runtime_seconds"],
                    "k10": row["best"]["k10"],
                    "metrics_by_k": row["best"]["metrics_by_k"],
                },
                "sweep": [
                    {
                        "a": item["a"],
                        "a_label": item["a_label"],
                        "n_components": item["n_components"],
                        "runtime_seconds": item["runtime_seconds"],
                        "k10": item["k10"],
                        "metrics_by_k": item["metrics_by_k"],
                    }
                    for item in row["sweep"]
                ],
            }
            for row in backbone_results
        ],
        "best_backbone": {
            "name": best_overall["name"],
            "label": best_overall["label"],
            "best_config": {
                "a": best_overall["best"]["a"],
                "a_label": best_overall["best"]["a_label"],
                "n_components": best_overall["best"]["n_components"],
            },
            "k10": best_overall["best"]["k10"],
        },
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
            "k10_plot": str(output_dir / "k10_backbone_bars.png"),
            "example_markdown": str(output_dir / "example_query_comparison.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    write_transfer_report(
        SCRIPT_DIR / "transferability_report.md",
        summary=summary,
        best_overall=best_overall,
        example_rows=example_rows,
    )

    print("Best transferred SIF backbone:", best_overall["label"])
    print("Best transferred MAP@10:", f"{best_overall['best']['k10']['map']:.4f}")
    print("Summary written to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
