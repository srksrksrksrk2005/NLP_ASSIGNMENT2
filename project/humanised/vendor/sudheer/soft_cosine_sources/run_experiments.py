#!/usr/bin/env python3
"""
Run soft cosine retrieval with four different term-similarity sources:

1. TF-IDF term similarities
2. LSA term similarities
3. ESA term similarities
4. WordNet term similarities

Each source is tuned with a small source-specific sweep, then evaluated against
the same source-specific TF-IDF baseline built from the same vectorizer setup.
The script writes one plot/report set per source plus a combined comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_DIR = Path(__file__).resolve().parent
SUDHEER_DIR = SCRIPT_DIR.parent
def find_ancestor_containing(relative_path: Path) -> Path:
    for candidate in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
        if (candidate / relative_path).exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {relative_path} above {SCRIPT_DIR}")


PROJECT_ROOT = find_ancestor_containing(Path("Nikhil") / "query_expansion" / "expansion" / "embedding_matrices.py")
ASSIGNMENT_ROOT = find_ancestor_containing(Path("cranfield") / "cran_docs.json")
NIKHIL_EXPANSION_DIR = PROJECT_ROOT / "Nikhil" / "query_expansion" / "expansion"
for path in [str(ASSIGNMENT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from common import (  # noqa: E402
    Evaluation,
    SoftCosineModel,
    approximate_randomization_pvalue,
    best_results_by_primary,
    build_example_query_comparison,
    compute_metrics,
    compute_per_query_metrics,
    load_json,
    metric_sort_key,
    save_metric_plot,
    save_overlay_plot,
    save_primary_sweep_plot,
    save_rankings,
    write_json,
    write_summary_csv,
)

EMBEDDING_MATRICES = load_module_from_path(
    "nikhil_embedding_matrices",
    NIKHIL_EXPANSION_DIR / "embedding_matrices.py",
)
WORDNET_MATRIX = load_module_from_path(
    "nikhil_wordnet_matrix",
    NIKHIL_EXPANSION_DIR / "wordnet_matrix.py",
)

build_tfidf_neighbor_map = EMBEDDING_MATRICES.build_tfidf_neighbor_map
build_lsa_neighbor_map = EMBEDDING_MATRICES.build_lsa_neighbor_map
build_esa_neighbor_map = EMBEDDING_MATRICES.build_esa_neighbor_map
build_wordnet_neighbor_map = WORDNET_MATRIX.build_wordnet_neighbor_map


METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F-score",
    "map": "MAP",
    "ndcg": "nDCG",
    "mrr": "MRR",
}


@dataclass
class TextTfidfIndex:
    doc_ids: List[int]
    vocab: List[str]
    term_to_idx: Dict[str, int]
    idf: np.ndarray
    doc_tf: sparse.csr_matrix
    doc_tfidf: sparse.csr_matrix
    vectorizer: TfidfVectorizer

    @classmethod
    def build(
        cls,
        doc_texts: Sequence[str],
        doc_ids: Sequence[int],
        sublinear_tf: bool,
        max_df: float,
        min_df: int,
        ngram_max: int,
    ) -> "TextTfidfIndex":
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            dtype=np.float64,
            use_idf=False,
            norm=None,
            sublinear_tf=sublinear_tf,
            max_df=max_df,
            min_df=min_df,
            ngram_range=(1, ngram_max),
        )
        doc_tf = vectorizer.fit_transform(doc_texts).tocsr()
        binary_tf = doc_tf.copy()
        binary_tf.data = np.ones_like(binary_tf.data)
        df = np.maximum(1.0, np.asarray(binary_tf.sum(axis=0)).ravel())
        idf = np.log(float(max(1, len(doc_texts))) / df)
        doc_tfidf = doc_tf.multiply(idf).tocsr()
        vocab = list(vectorizer.get_feature_names_out())
        term_to_idx = {term: index for index, term in enumerate(vocab)}
        return cls(
            doc_ids=list(doc_ids),
            vocab=vocab,
            term_to_idx=term_to_idx,
            idf=idf,
            doc_tf=doc_tf,
            doc_tfidf=doc_tfidf,
            vectorizer=vectorizer,
        )

    def transform_queries(self, query_texts: Sequence[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(query_texts).tocsr()


def flatten_document(document: Sequence[Sequence[str]]) -> str:
    return " ".join(token for sentence in document for token in sentence)


def flatten_collection(collection: Sequence[Sequence[Sequence[str]]]) -> List[str]:
    return [flatten_document(item) for item in collection]


def documents_to_feature_maps(collection: Sequence[Sequence[Sequence[str]]]) -> List[Dict[str, float]]:
    feature_maps = []
    for item in collection:
        counts: Dict[str, float] = {}
        for sentence in item:
            for token in sentence:
                counts[token] = counts.get(token, 0.0) + 1.0
        feature_maps.append(counts)
    return feature_maps


def normalize_neighbor_map_stats(vocab: Sequence[str], neighbor_map: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
    represented_terms = sum(1 for term in vocab if neighbor_map.get(term))
    neighbor_counts = [len(neighbor_map.get(term, ())) for term in vocab]
    edge_pairs = set()
    for term, neighbors in neighbor_map.items():
        for neighbor, _ in neighbors:
            pair = (term, neighbor) if term <= neighbor else (neighbor, term)
            edge_pairs.add(pair)
    return {
        "represented_terms": represented_terms,
        "representation_coverage": float(represented_terms / len(vocab)) if vocab else 0.0,
        "retained_neighbor_edges": int(len(edge_pairs)),
        "avg_neighbors_per_term": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
    }


def neighbor_map_to_similarity_matrix(
    vocab: Sequence[str],
    neighbor_map: Dict[str, List[Tuple[str, float]]],
    similarity_power: float = 1.0,
) -> sparse.csr_matrix:
    term_to_idx = {term: index for index, term in enumerate(vocab)}
    rows = list(range(len(vocab)))
    cols = list(range(len(vocab)))
    data = [1.0] * len(vocab)

    pair_to_similarity: Dict[Tuple[int, int], float] = {}
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

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float64).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.clip(matrix.data, 0.0, 1.0)
    return matrix


def filter_neighbor_map(
    neighbor_map: Dict[str, List[Tuple[str, float]]],
    top_k: int,
    min_similarity: float,
) -> Dict[str, List[Tuple[str, float]]]:
    filtered_map: Dict[str, List[Tuple[str, float]]] = {}
    for term, neighbors in neighbor_map.items():
        filtered_neighbors = [(neighbor, similarity) for neighbor, similarity in neighbors if similarity >= min_similarity]
        filtered_map[term] = filtered_neighbors[:top_k]
    return filtered_map


def compute_sweep_metrics(
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


def k10(metrics_by_k: Dict[str, List[float]]) -> Dict[str, float]:
    return {metric_name: metric_values[9] for metric_name, metric_values in metrics_by_k.items()}


def source_label(source_key: str) -> str:
    return {
        "tfidf": "TF-IDF soft cosine",
        "lsa": "LSA soft cosine",
        "esa": "ESA soft cosine",
        "wordnet": "WordNet soft cosine",
    }[source_key]


SOURCE_SPECS = [
    {
        "key": "tfidf",
        "primary_param": "top_k_neighbors",
        "cache_params": (),
        "vectorizer": {"sublinear_tf": True, "max_df": 0.9, "min_df": 2, "ngram_max": 1},
        "search_space": {"top_k_neighbors": [5, 10], "min_similarity": [0.05]},
        "build": lambda index, cfg, stats: build_tfidf_neighbor_map(
            index.doc_tfidf,
            index.vocab,
            top_k=cfg["top_k_neighbors"],
            min_similarity=cfg["min_similarity"],
            progress=False,
        ),
    },
    {
        "key": "lsa",
        "primary_param": "n_components",
        "cache_params": ("n_components",),
        "vectorizer": {"sublinear_tf": True, "max_df": 0.95, "min_df": 2, "ngram_max": 1},
        "search_space": {"n_components": [150, 250], "top_k_neighbors": [10], "min_similarity": [0.05, 0.10]},
        "build": lambda index, cfg, stats: build_lsa_neighbor_map(
            index.doc_tfidf,
            index.vocab,
            n_components=cfg["n_components"],
            top_k=cfg["top_k_neighbors"],
            min_similarity=cfg["min_similarity"],
            progress=False,
        ),
    },
    {
        "key": "esa",
        "primary_param": "top_concepts",
        "cache_params": ("top_concepts",),
        "vectorizer": {"sublinear_tf": True, "max_df": 0.9, "min_df": 2, "ngram_max": 1},
        "search_space": {"top_concepts": [25, 50], "top_k_neighbors": [10], "min_similarity": [0.0, 0.02]},
        "build": lambda index, cfg, stats: build_esa_neighbor_map(
            index.doc_tfidf,
            index.vocab,
            top_concepts=cfg["top_concepts"],
            top_k=cfg["top_k_neighbors"],
            min_similarity=cfg["min_similarity"],
            progress=False,
            stats_out=stats,
        ),
    },
    {
        "key": "wordnet",
        "primary_param": "top_k_neighbors",
        "cache_params": (),
        "vectorizer": {"sublinear_tf": True, "max_df": 0.95, "min_df": 2, "ngram_max": 1},
        "search_space": {"top_k_neighbors": [5, 10], "min_similarity": [0.05]},
        "build": lambda index, cfg, stats: build_wordnet_neighbor_map(
            index.vocab,
            top_k=cfg["top_k_neighbors"],
            min_similarity=cfg["min_similarity"],
            progress=False,
            cache_dir=SCRIPT_DIR / "cache" / "wordnet",
        ),
    },
]


def build_configs(search_space: Dict[str, Sequence[object]]) -> List[Dict[str, object]]:
    keys = list(search_space)
    configs = []
    for values in itertools.product(*[search_space[key] for key in keys]):
        configs.append({key: value for key, value in zip(keys, values)})
    return configs


def run_source_sweep(
    source_spec: Dict[str, object],
    docs_texts: Sequence[str],
    queries_texts: Sequence[str],
    queries_json,
    query_ids: Sequence[int],
    qrels,
    doc_ids: Sequence[int],
    output_dir: Path,
) -> Dict[str, object]:
    source_key = str(source_spec["key"])
    source_dir = output_dir / source_key
    source_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_cfg = source_spec["vectorizer"]
    index = TextTfidfIndex.build(
        doc_texts=docs_texts,
        doc_ids=doc_ids,
        sublinear_tf=bool(vectorizer_cfg["sublinear_tf"]),
        max_df=float(vectorizer_cfg["max_df"]),
        min_df=int(vectorizer_cfg["min_df"]),
        ngram_max=int(vectorizer_cfg["ngram_max"]),
    )
    query_tf = index.transform_queries(queries_texts)

    evaluator = Evaluation()

    baseline_start = time.time()
    baseline_model = SoftCosineModel.build(f"baseline_{source_key}", index, sparse.identity(len(index.vocab), format="csr", dtype=np.float64))
    baseline_rankings, _ = baseline_model.rank_query_matrix(query_tf)
    baseline_runtime = time.time() - baseline_start
    baseline_metrics = compute_sweep_metrics(evaluator, baseline_rankings, query_ids, qrels)
    baseline_per_query = compute_per_query_metrics(evaluator, baseline_rankings, query_ids, qrels)

    save_metric_plot(source_dir / "baseline_eval_plot.png", baseline_metrics, f"Baseline TF-IDF Metrics ({source_label(source_key)})")
    save_rankings(source_dir / "baseline_rankings_top20.json", query_ids, baseline_rankings, top_k=20)
    write_json(
        source_dir / "baseline_metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": k10(baseline_metrics),
            "runtime_seconds": baseline_runtime,
            "vectorizer": vectorizer_cfg,
        },
    )

    sweep_results: List[Dict[str, object]] = []
    configs = build_configs(source_spec["search_space"])
    total = len(configs)
    sweep_start = time.time()
    cache_params = tuple(source_spec.get("cache_params", ()))
    base_neighbor_map_cache: Dict[Tuple[object, ...], Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]] = {}

    for config_index, config in enumerate(configs, start=1):
        config_start = time.time()
        print(
            f"[{source_key}] sweep {config_index}/{total} "
            + " ".join(f"{name}={value}" for name, value in config.items()),
            flush=True,
        )

        cache_key = tuple(config[param] for param in cache_params)
        if cache_key not in base_neighbor_map_cache:
            base_config: Dict[str, object] = {}
            for param_name, values in source_spec["search_space"].items():
                if param_name in cache_params:
                    base_config[param_name] = config[param_name]
                elif param_name.startswith("min_"):
                    base_config[param_name] = min(values)
                else:
                    base_config[param_name] = max(values)

            base_stats: Dict[str, float] = {}
            base_neighbor_map = source_spec["build"](index, base_config, base_stats)
            base_neighbor_map_cache[cache_key] = (base_neighbor_map, dict(base_stats))
        base_neighbor_map, base_stats = base_neighbor_map_cache[cache_key]

        top_k_value = config.get("top_k_neighbors")
        if top_k_value is None:
            top_k_value = config.get("top_concepts")
        if top_k_value is None:
            top_k_value = config[source_spec["primary_param"]]

        neighbor_map = filter_neighbor_map(base_neighbor_map, int(top_k_value), float(config.get("min_similarity", 0.0)))
        stats = normalize_neighbor_map_stats(index.vocab, neighbor_map)
        stats.update(base_stats)
        similarity_matrix = neighbor_map_to_similarity_matrix(index.vocab, neighbor_map)
        model = SoftCosineModel.build(source_key, index, similarity_matrix)
        rankings, _ = model.rank_query_matrix(query_tf)
        metrics_by_k = compute_sweep_metrics(evaluator, rankings, query_ids, qrels)
        per_query = compute_per_query_metrics(evaluator, rankings, query_ids, qrels)

        row = {
            **config,
            "metrics_by_k": metrics_by_k,
            "k10": k10(metrics_by_k),
            "precision_at_10": k10(metrics_by_k)["precision"],
            "recall_at_10": k10(metrics_by_k)["recall"],
            "fscore_at_10": k10(metrics_by_k)["fscore"],
            "map_at_10": k10(metrics_by_k)["map"],
            "ndcg_at_10": k10(metrics_by_k)["ndcg"],
            "mrr_at_10": k10(metrics_by_k)["mrr"],
            "runtime_seconds": time.time() - config_start,
            "vectorizer": vectorizer_cfg,
            "rankings": rankings,
            "per_query": per_query,
            **stats,
        }
        sweep_results.append(row)
        print(
            f"  MAP@10={row['k10']['map']:.4f} nDCG@10={row['k10']['ndcg']:.4f}",
            flush=True,
        )

    sweep_results.sort(key=metric_sort_key, reverse=True)
    best_config = sweep_results[0]
    write_json(source_dir / "sweep_results.json", sweep_results)

    best_by_primary = best_results_by_primary(sweep_results, source_spec["primary_param"])
    write_json(source_dir / "sweep_best.json", best_by_primary)
    sweep_fieldnames = [
        source_spec["primary_param"],
        *[name for name in source_spec["search_space"] if name != source_spec["primary_param"]],
        "precision@10",
        "recall@10",
        "fscore@10",
        "map@10",
        "ndcg@10",
        "mrr@10",
        "runtime_seconds",
        "represented_terms",
        "representation_coverage",
        "retained_neighbor_edges",
        "avg_neighbors_per_term",
    ]
    write_summary_csv(
        source_dir / "sweep_best.csv",
        [
            {
                **row,
                "precision@10": row["k10"]["precision"],
                "recall@10": row["k10"]["recall"],
                "fscore@10": row["k10"]["fscore"],
                "map@10": row["k10"]["map"],
                "ndcg@10": row["k10"]["ndcg"],
                "mrr@10": row["k10"]["mrr"],
            }
            for row in best_by_primary
        ],
        fieldnames=sweep_fieldnames,
    )
    save_primary_sweep_plot(
        source_dir / "sweep_best.png",
        [
            {
                **row,
                "precision_at_10": row["k10"]["precision"],
                "recall_at_10": row["k10"]["recall"],
                "fscore_at_10": row["k10"]["fscore"],
                "map_at_10": row["k10"]["map"],
                "ndcg_at_10": row["k10"]["ndcg"],
                "mrr_at_10": row["k10"]["mrr"],
            }
            for row in best_by_primary
        ],
        primary_key=source_spec["primary_param"],
        x_label=source_spec["primary_param"],
        title=f"Best Result Per {source_label(source_key)} {source_spec['primary_param']}",
    )

    # Final best-source rerun for outputs and example comparisons.
    best_neighbor_map = source_spec["build"](index, best_config, {})
    best_similarity_matrix = neighbor_map_to_similarity_matrix(index.vocab, best_neighbor_map)
    best_model = SoftCosineModel.build(source_key, index, best_similarity_matrix)
    best_rankings, _ = best_model.rank_query_matrix(query_tf)
    best_metrics = compute_sweep_metrics(evaluator, best_rankings, query_ids, qrels)
    best_per_query = compute_per_query_metrics(evaluator, best_rankings, query_ids, qrels)
    best_runtime = time.time() - sweep_start

    save_metric_plot(
        source_dir / "eval_plot.png",
        best_metrics,
        f"{source_label(source_key)} Metrics",
    )
    save_rankings(source_dir / "rankings_top20.json", query_ids, best_rankings, top_k=20)
    write_json(
        source_dir / "metrics.json",
        {
            "source": source_key,
            "source_label": source_label(source_key),
            "best_config": {k: v for k, v in best_config.items() if k not in {"metrics_by_k", "rankings", "per_query"}},
            "metrics_by_k": best_metrics,
            "k10": k10(best_metrics),
            "runtime_seconds": best_runtime,
            "vectorizer": vectorizer_cfg,
            "baseline_k10": k10(baseline_metrics),
            "baseline_runtime_seconds": baseline_runtime,
            "per_query": best_per_query,
        },
    )

    examples = build_example_query_comparison(
        query_ids=query_ids,
        queries_json=queries_json,
        qrels=qrels,
        baseline_rankings=baseline_rankings,
        method_rankings=best_rankings,
        baseline_per_query=baseline_per_query,
        method_per_query=best_per_query,
        output_dir=source_dir,
    )

    significance = {
        "ap_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["average_precision"],
            best_per_query["average_precision"],
        ),
        "ndcg_at_10_pvalue": approximate_randomization_pvalue(
            baseline_per_query["ndcg"],
            best_per_query["ndcg"],
        ),
    }

    return {
        "source": source_key,
        "label": source_label(source_key),
        "vectorizer": vectorizer_cfg,
        "baseline": {
            "metrics_by_k": baseline_metrics,
            "k10": k10(baseline_metrics),
            "runtime_seconds": baseline_runtime,
        },
        "best_config": {k: v for k, v in best_config.items() if k not in {"metrics_by_k", "rankings", "per_query"}},
        "best_metrics_by_k": best_metrics,
        "best_k10": k10(best_metrics),
        "best_runtime_seconds": best_runtime,
        "sweep_results": sweep_results,
        "best_by_primary": best_by_primary,
        "examples": examples,
        "significance": significance,
        "paths": {
            "source_dir": str(source_dir),
            "baseline_eval_plot": str(source_dir / "baseline_eval_plot.png"),
            "eval_plot": str(source_dir / "eval_plot.png"),
            "sweep_best_plot": str(source_dir / "sweep_best.png"),
            "metrics_json": str(source_dir / "metrics.json"),
            "sweep_best_json": str(source_dir / "sweep_best.json"),
            "sweep_best_csv": str(source_dir / "sweep_best.csv"),
            "rankings_top20": str(source_dir / "rankings_top20.json"),
            "example_markdown": str(source_dir / "example_query_comparison.md"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft cosine comparison across TF-IDF, LSA, ESA, and WordNet sources.")
    parser.add_argument("--dataset-dir", default=str(ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")
    processed_docs = load_json(Path(args.processed_docs))
    processed_queries = load_json(Path(args.processed_queries))

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    doc_texts = flatten_collection(processed_docs)
    query_texts = flatten_collection(processed_queries)

    source_results = []
    for source_spec in SOURCE_SPECS:
        print(f"[{source_spec['key']}] starting source-specific sweep", flush=True)
        source_results.append(
            run_source_sweep(
                source_spec=source_spec,
                docs_texts=doc_texts,
                queries_texts=query_texts,
                queries_json=queries_json,
                query_ids=query_ids,
                qrels=qrels,
                doc_ids=doc_ids,
                output_dir=output_dir,
            )
        )

    baseline_dir = output_dir / "baseline_tfidf"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    # Baseline summary uses the first source configuration's baseline as the common reference.
    baseline_metrics = source_results[0]["baseline"]["metrics_by_k"]
    baseline_k10 = source_results[0]["baseline"]["k10"]
    write_json(
        baseline_dir / "metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": baseline_k10,
            "runtime_seconds": source_results[0]["baseline"]["runtime_seconds"],
            "source": source_results[0]["source"],
        },
    )

    grouped_metrics = {row["source"]: row["best_metrics_by_k"] for row in source_results}
    labels = {row["source"]: row["label"] for row in source_results}
    ordered_keys = [row["source"] for row in source_results]
    save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        grouped_metrics,
        labels,
        ordered_keys,
        "Baseline vs Best Soft Cosine Sources",
    )

    best_source_result = max(
        source_results,
        key=lambda row: (row["best_k10"]["map"], row["best_k10"]["ndcg"], row["best_k10"]["mrr"]),
    )

    summary_rows = [
        {
            "method": "baseline_tfidf",
            "source": "baseline",
            "source_label": "baseline_tfidf",
            "primary_param": "-",
            "primary_value": "-",
            "precision@10": baseline_k10["precision"],
            "recall@10": baseline_k10["recall"],
            "fscore@10": baseline_k10["fscore"],
            "map@10": baseline_k10["map"],
            "ndcg@10": baseline_k10["ndcg"],
            "mrr@10": baseline_k10["mrr"],
            "runtime_seconds": source_results[0]["baseline"]["runtime_seconds"],
        }
    ]
    for row in source_results:
        best_config = row["best_config"]
        primary_param = next(iter([spec for spec in SOURCE_SPECS if spec["key"] == row["source"]][0]["search_space"].keys()))
        summary_rows.append(
            {
                "method": row["source"],
                "source": row["source"],
                "source_label": row["label"],
                "primary_param": primary_param,
                "primary_value": best_config[primary_param],
                "precision@10": row["best_k10"]["precision"],
                "recall@10": row["best_k10"]["recall"],
                "fscore@10": row["best_k10"]["fscore"],
                "map@10": row["best_k10"]["map"],
                "ndcg@10": row["best_k10"]["ndcg"],
                "mrr@10": row["best_k10"]["mrr"],
                "runtime_seconds": row["best_runtime_seconds"],
            }
        )

    write_summary_csv(
        output_dir / "summary_k10.csv",
        summary_rows,
        fieldnames=[
            "method",
            "source",
            "source_label",
            "primary_param",
            "primary_value",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
            "runtime_seconds",
        ],
    )

    comparison_summary = {
        "dataset": str(dataset_dir),
        "sources": {
            row["source"]: {
                "label": row["label"],
                "vectorizer": row["vectorizer"],
                "baseline": row["baseline"],
                "best_config": row["best_config"],
                "best_k10": row["best_k10"],
                "best_metrics_by_k": row["best_metrics_by_k"],
                "best_runtime_seconds": row["best_runtime_seconds"],
                "best_by_primary": row["best_by_primary"],
                "significance": row["significance"],
                "paths": row["paths"],
            }
            for row in source_results
        },
        "best_source": best_source_result["source"],
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "comparison_summary_json": str(output_dir / "comparison_summary.json"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
        },
    }
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    write_json(
        output_dir / "summary.json",
        {
            "dataset": str(dataset_dir),
            "baseline": {
                "metrics_by_k": baseline_metrics,
                "k10": baseline_k10,
                "runtime_seconds": source_results[0]["baseline"]["runtime_seconds"],
            },
            "sources": comparison_summary["sources"],
            "best_source": best_source_result["source"],
            "paths": comparison_summary["paths"],
        },
    )

    report_lines = ["# Soft Cosine Source Comparison Report", ""]
    report_lines.append(
        "This comparison reuses the same Cranfield preprocessing and tunes four soft-cosine sources: TF-IDF, LSA, ESA, and WordNet."
    )
    report_lines.append("")
    report_lines.append("## Best Results")
    report_lines.append("")
    report_lines.append("| Source | Primary Param | Best Value | MAP@10 | nDCG@10 | MRR@10 |")
    report_lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in source_results:
        primary_param = next(iter([spec for spec in SOURCE_SPECS if spec["key"] == row["source"]][0]["search_space"].keys()))
        report_lines.append(
            f"| {row['label']} | {primary_param} | {row['best_config'][primary_param]} | {row['best_k10']['map']:.4f} | {row['best_k10']['ndcg']:.4f} | {row['best_k10']['mrr']:.4f} |"
        )
    report_lines.append("")
    report_lines.append(f"Best source overall: {best_source_result['label']} with MAP@10={best_source_result['best_k10']['map']:.4f}")
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("- Each source has its own tuned sweep plot under its output folder.")
    report_lines.append("- WordNet uses unigram vocabulary because its synonym graph is lexical rather than phrasal.")
    report_lines.append("- TF-IDF, LSA, and ESA use the same pretokenized Cranfield texts with their own tuned vectorizer settings.")
    report_lines.append("")
    (output_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Comparison written to:", output_dir)
    print("Best source:", best_source_result["label"], f"MAP@10={best_source_result['best_k10']['map']:.4f}")


if __name__ == "__main__":
    main()
