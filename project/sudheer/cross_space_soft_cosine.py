#!/usr/bin/env python3
"""
Cross-compare multiple q,d vector spaces with compatible soft-cosine similarity
matrices S.

Why "compatible" matters:
- Sparse lexical spaces (TF-IDF, bigram TF-IDF, local-context BoW) can use
  term-level similarity sources.
- Dense latent spaces (LSA, ESA concept vectors) need an S matrix defined over
  their own feature dimensions.

This script therefore forms a principled grid of combinations instead of mixing
    incompatible spaces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSIGNMENT_ROOT = PROJECT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from common import (  # noqa: E402
    Evaluation,
    SoftCosineModel,
    TfidfIndex,
    build_term_similarity_matrix,
    compute_metrics,
    feature_maps_to_sparse_matrix,
    load_json,
    train_word2vec_artifacts,
    unigram_feature_map,
    write_json,
    write_summary_csv,
)


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


EMBEDDING_MATRICES = load_module_from_path(
    "nikhil_embedding_matrices",
    PROJECT_DIR / "Nikhil" / "query_expansion" / "expansion" / "embedding_matrices.py",
)
WORDNET_MATRIX = load_module_from_path(
    "nikhil_wordnet_matrix",
    PROJECT_DIR / "Nikhil" / "query_expansion" / "expansion" / "wordnet_matrix.py",
)
CHANDAN_LOCAL = load_module_from_path(
    "chandan_local_context",
    PROJECT_DIR / "chandan" / "unordered_local_context_bow" / "run_experiments.py",
)
ESA_RETRIEVAL_MODULE = load_module_from_path(
    "ram_esa_retrieval",
    PROJECT_DIR / "ramakrishna" / "esa" / "esa_retrieval.py",
)

build_tfidf_neighbor_map = EMBEDDING_MATRICES.build_tfidf_neighbor_map
build_lsa_neighbor_map = EMBEDDING_MATRICES.build_lsa_neighbor_map
build_esa_neighbor_map = EMBEDDING_MATRICES.build_esa_neighbor_map
build_wordnet_neighbor_map = WORDNET_MATRIX.build_wordnet_neighbor_map
context_feature_map = CHANDAN_LOCAL.context_feature_map
filter_context_feature_maps = CHANDAN_LOCAL.filter_context_feature_maps
ESARetrieval = ESA_RETRIEVAL_MODULE.ESARetrieval


def flatten_document(document):
    return " ".join(token for sentence in document for token in sentence)


def flatten_collection(collection):
    return [flatten_document(item) for item in collection]


class TextTfidfIndex:
    def __init__(self, doc_ids, vocab, term_to_idx, idf, doc_tfidf, vectorizer):
        self.doc_ids = list(doc_ids)
        self.vocab = list(vocab)
        self.term_to_idx = dict(term_to_idx)
        self.idf = idf
        self.doc_tfidf = doc_tfidf.tocsr()
        self.vectorizer = vectorizer

    @classmethod
    def build(
        cls,
        doc_texts: Sequence[str],
        doc_ids: Sequence[int],
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
            sublinear_tf=True,
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
        term_to_idx = {term: idx for idx, term in enumerate(vocab)}
        return cls(doc_ids, vocab, term_to_idx, idf, doc_tfidf, vectorizer)

    def transform_queries(self, query_texts: Sequence[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(query_texts).tocsr()


def neighbor_map_to_similarity_matrix(
    vocab: Sequence[str],
    neighbor_map: Dict[str, List[Tuple[str, float]]],
    similarity_power: float = 1.0,
) -> sparse.csr_matrix:
    term_to_idx = {term: idx for idx, term in enumerate(vocab)}
    rows = list(range(len(vocab)))
    cols = list(range(len(vocab)))
    data = [1.0] * len(vocab)

    pair_to_similarity = {}
    for term, neighbors in neighbor_map.items():
        src_idx = term_to_idx.get(term)
        if src_idx is None:
            continue
        for neighbor, similarity in neighbors:
            dst_idx = term_to_idx.get(neighbor)
            if dst_idx is None or dst_idx == src_idx:
                continue
            value = max(0.0, float(similarity))
            if similarity_power != 1.0:
                value = value ** similarity_power
            pair = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
            pair_to_similarity[pair] = max(pair_to_similarity.get(pair, 0.0), value)

    for (src_idx, dst_idx), similarity in pair_to_similarity.items():
        rows.extend([src_idx, dst_idx])
        cols.extend([dst_idx, src_idx])
        data.extend([similarity, similarity])

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float64).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.clip(matrix.data, 0.0, 1.0)
    return matrix


def build_local_context_feature_maps(
    docs_tokens: Sequence[Sequence[Sequence[str]]],
    queries_tokens: Sequence[Sequence[Sequence[str]]],
    radius: int = 4,
    orders: Sequence[int] = (2,),
    min_df: int = 3,
    alpha: float = 0.8,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    doc_unigrams = [unigram_feature_map(doc) for doc in docs_tokens]
    query_unigrams = [unigram_feature_map(query) for query in queries_tokens]

    raw_doc_contexts = [context_feature_map(doc, radius=radius, orders=orders) for doc in docs_tokens]
    raw_query_contexts = [context_feature_map(query, radius=radius, orders=orders) for query in queries_tokens]
    filtered_docs, doc_freq = filter_context_feature_maps(raw_doc_contexts, min_df=min_df)
    filtered_queries = [
        {feature: value for feature, value in fmap.items() if doc_freq[feature] >= min_df}
        for fmap in raw_query_contexts
    ]

    doc_maps = []
    for base, ctx in zip(doc_unigrams, filtered_docs):
        merged = dict(base)
        for feature, value in ctx.items():
            merged[feature] = alpha * float(value)
        doc_maps.append(merged)

    query_maps = []
    for base, ctx in zip(query_unigrams, filtered_queries):
        merged = dict(base)
        for feature, value in ctx.items():
            merged[feature] = alpha * float(value)
        query_maps.append(merged)

    return doc_maps, query_maps


def build_sparse_similarity(
    source_name: str,
    index,
    docs_tokens,
) -> sparse.csr_matrix:
    if source_name == "identity":
        return sparse.identity(len(index.vocab), format="csr", dtype=np.float64)
    if source_name == "tfidf":
        neighbor_map = build_tfidf_neighbor_map(index.doc_tfidf, index.vocab, top_k=5, min_similarity=0.05, progress=False)
        return neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    if source_name == "lsa":
        neighbor_map = build_lsa_neighbor_map(index.doc_tfidf, index.vocab, n_components=150, top_k=10, min_similarity=0.05, progress=False)
        return neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    if source_name == "esa":
        neighbor_map = build_esa_neighbor_map(index.doc_tfidf, index.vocab, top_concepts=50, top_k=10, min_similarity=0.02, progress=False, stats_out={})
        return neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    if source_name == "wordnet":
        neighbor_map = build_wordnet_neighbor_map(
            index.vocab,
            top_k=5,
            min_similarity=0.05,
            progress=False,
            cache_dir=SCRIPT_DIR / "soft_cosine_sources" / "cache" / "wordnet",
        )
        return neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    if source_name == "word2vec":
        artifacts = train_word2vec_artifacts(
            docs_tokens=docs_tokens,
            vocab=index.vocab,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            sg=1,
            epochs=20,
        )
        matrix, _ = build_term_similarity_matrix(
            vocab=index.vocab,
            term_to_idx=index.term_to_idx,
            word_vectors=artifacts.word_vectors,
            top_k_neighbors=10,
            min_similarity=0.10,
            similarity_power=2.0,
        )
        return matrix
    raise ValueError(f"Unknown sparse source: {source_name}")


def build_dense_feature_similarity(
    doc_vectors: np.ndarray,
    source_name: str,
    top_k: int = 25,
    min_similarity: float = 0.05,
    similarity_power: float = 1.0,
) -> np.ndarray:
    n_features = doc_vectors.shape[1]
    if source_name == "identity":
        return np.eye(n_features, dtype=np.float64)
    if source_name != "feature_cosine":
        raise ValueError(f"Unknown dense source: {source_name}")

    feature_matrix = doc_vectors.T.astype(np.float64)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    normalized = feature_matrix / np.maximum(norms, 1e-12)

    n_neighbors = min(n_features, top_k + 1)
    if n_neighbors <= 1:
        return np.eye(n_features, dtype=np.float64)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(normalized)
    distances, indices = nn.kneighbors(normalized, return_distance=True)

    S = np.eye(n_features, dtype=np.float64)
    for src in range(n_features):
        for distance, dst in zip(distances[src], indices[src]):
            if src == dst:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < min_similarity:
                continue
            if similarity_power != 1.0:
                similarity = similarity ** similarity_power
            if similarity > S[src, dst]:
                S[src, dst] = similarity
                S[dst, src] = similarity
    return S


def rank_dense_soft_cosine(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    doc_ids: Sequence[int],
    similarity_matrix: np.ndarray,
) -> List[List[int]]:
    doc_vectors = np.asarray(doc_vectors, dtype=np.float64)
    query_vectors = np.asarray(query_vectors, dtype=np.float64)
    S = np.asarray(similarity_matrix, dtype=np.float64)

    doc_soft = doc_vectors @ S
    query_soft = query_vectors @ S
    numerator = query_soft @ doc_vectors.T
    doc_norm_sq = np.sum(doc_soft * doc_vectors, axis=1)
    query_norm_sq = np.sum(query_soft * query_vectors, axis=1)
    denom = np.sqrt(
        np.maximum(query_norm_sq[:, np.newaxis], 0.0) * np.maximum(doc_norm_sq[np.newaxis, :], 0.0)
    ) + 1e-12
    scores = numerator / denom

    rankings = []
    for query_scores in scores:
        order = np.argsort(-query_scores, kind="mergesort")
        rankings.append([doc_ids[index] for index in order])
    return rankings


def build_lsa_space(doc_texts, query_texts, doc_ids):
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.float64,
        sublinear_tf=True,
        max_df=0.95,
        min_df=2,
        norm="l2",
        ngram_range=(1, 1),
    )
    doc_term = vectorizer.fit_transform(doc_texts)
    query_term = vectorizer.transform(query_texts)
    svd = TruncatedSVD(n_components=min(250, min(doc_term.shape) - 1), random_state=42)
    doc_vectors = svd.fit_transform(doc_term)
    query_vectors = svd.transform(query_term)
    return {
        "doc_ids": list(doc_ids),
        "doc_vectors": doc_vectors,
        "query_vectors": query_vectors,
    }


def build_esa_space(docs_tokens, queries_tokens, doc_ids):
    retriever = ESARetrieval(
        top_concepts=25,
        min_similarity=0.0,
        random_state=42,
        sublinear_tf=True,
        max_df=0.9,
        min_df=1,
        norm="l2",
        ngram_range=(1, 2),
    )
    retriever.build_index(docs_tokens, doc_ids, concept_docs=None, concept_ids=None, concept_source="cranfield")
    _, payload = retriever.rank(queries_tokens, return_scores=True)
    return {
        "doc_ids": list(doc_ids),
        "doc_vectors": retriever.doc_concept_matrix.astype(np.float64),
        "query_vectors": payload["query_concept_matrix"].astype(np.float64),
    }


def save_all_metrics_overlay(path: Path, all_metrics: Dict[str, Dict[str, List[float]]]) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    metric_order = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
    labels = list(all_metrics.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(labels))))

    for axis, metric_name in zip(axes, metric_order):
        for idx, label in enumerate(labels):
            axis.plot(
                ks,
                all_metrics[label][metric_name],
                linewidth=1.7,
                marker="o",
                markersize=3,
                color=colors[idx],
                label=label,
            )
        axis.set_title(metric_name.upper())
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.suptitle("Cross-Space Soft Cosine: Metrics vs k", y=0.999, fontsize=15)
    fig.subplots_adjust(top=0.83, hspace=0.34, wspace=0.24)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_single_metric_overlay(path: Path, all_metrics: Dict[str, Dict[str, List[float]]], metric_name: str, title: str) -> None:
    ks = list(range(1, 11))
    labels = list(all_metrics.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(labels))))

    plt.figure(figsize=(14, 7))
    for idx, label in enumerate(labels):
        plt.plot(
            ks,
            all_metrics[label][metric_name],
            linewidth=2,
            marker="o",
            markersize=4,
            color=colors[idx],
            label=label,
        )
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(metric_name.upper())
    plt.xticks(ks)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_metric_table(path: Path, all_metrics: Dict[str, Dict[str, List[float]]], metric_name: str) -> None:
    rows = []
    for combo, metrics in all_metrics.items():
        row = {"combination": combo}
        for k, value in enumerate(metrics[metric_name], start=1):
            row[f"k{k}"] = value
        rows.append(row)
    fieldnames = ["combination"] + [f"k{k}" for k in range(1, 11)]
    write_summary_csv(path, rows, fieldnames=fieldnames)


def main() -> None:
    dataset_dir = ASSIGNMENT_ROOT / "cranfield"
    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")
    processed_docs = load_json(ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt")
    processed_queries = load_json(ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt")

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    doc_texts = flatten_collection(processed_docs)
    query_texts = flatten_collection(processed_queries)
    evaluator = Evaluation()

    sparse_spaces = {}

    unigram_index = TextTfidfIndex.build(doc_texts, doc_ids, max_df=0.95, min_df=2, ngram_max=1)
    sparse_spaces["unigram_tfidf"] = {
        "index": unigram_index,
        "query_tf": unigram_index.transform_queries(query_texts),
        "s_sources": ["identity", "tfidf", "lsa", "esa", "wordnet", "word2vec"],
    }

    bigram_index = TextTfidfIndex.build(doc_texts, doc_ids, max_df=0.95, min_df=2, ngram_max=2)
    sparse_spaces["bigram_tfidf"] = {
        "index": bigram_index,
        "query_tf": bigram_index.transform_queries(query_texts),
        "s_sources": ["identity", "tfidf", "lsa", "esa", "word2vec"],
    }

    local_doc_maps, local_query_maps = build_local_context_feature_maps(processed_docs, processed_queries)
    local_index = TfidfIndex.build("local_context_bow", local_doc_maps, doc_ids)
    local_query_tf = feature_maps_to_sparse_matrix(local_query_maps, local_index.term_to_idx, len(local_index.vocab))
    sparse_spaces["local_context_bow"] = {
        "index": local_index,
        "query_tf": local_query_tf,
        "s_sources": ["identity", "tfidf", "lsa"],
    }

    dense_spaces = {
        "lsa_latent": {
            **build_lsa_space(doc_texts, query_texts, doc_ids),
            "s_sources": ["identity", "feature_cosine"],
        },
        "esa_concept": {
            **build_esa_space(processed_docs, processed_queries, doc_ids),
            "s_sources": ["identity", "feature_cosine"],
        },
    }

    rows = []
    all_metrics = {}

    for qd_space, payload in sparse_spaces.items():
        index = payload["index"]
        query_tf = payload["query_tf"]
        for s_source in payload["s_sources"]:
            S = build_sparse_similarity(s_source, index, processed_docs)
            model = SoftCosineModel.build(f"{qd_space}__{s_source}", index, S)
            rankings, _ = model.rank_query_matrix(query_tf)
            metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
            combo_name = f"{qd_space}__S={s_source}"
            all_metrics[combo_name] = metrics_by_k
            rows.append(
                {
                    "qd_space": qd_space,
                    "s_source": s_source,
                    "space_type": "sparse",
                    "precision@10": metrics_by_k["precision"][9],
                    "recall@10": metrics_by_k["recall"][9],
                    "fscore@10": metrics_by_k["fscore"][9],
                    "map@10": metrics_by_k["map"][9],
                    "ndcg@10": metrics_by_k["ndcg"][9],
                    "mrr@10": metrics_by_k["mrr"][9],
                }
            )

    for qd_space, payload in dense_spaces.items():
        doc_vectors = payload["doc_vectors"]
        query_vectors = payload["query_vectors"]
        for s_source in payload["s_sources"]:
            S = build_dense_feature_similarity(doc_vectors, s_source)
            rankings = rank_dense_soft_cosine(doc_vectors, query_vectors, payload["doc_ids"], S)
            metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
            combo_name = f"{qd_space}__S={s_source}"
            all_metrics[combo_name] = metrics_by_k
            rows.append(
                {
                    "qd_space": qd_space,
                    "s_source": s_source,
                    "space_type": "dense",
                    "precision@10": metrics_by_k["precision"][9],
                    "recall@10": metrics_by_k["recall"][9],
                    "fscore@10": metrics_by_k["fscore"][9],
                    "map@10": metrics_by_k["map"][9],
                    "ndcg@10": metrics_by_k["ndcg"][9],
                    "mrr@10": metrics_by_k["mrr"][9],
                }
            )

    output_dir = SCRIPT_DIR / "cross_space_soft_cosine" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows.sort(key=lambda row: (row["map@10"], row["ndcg@10"], row["mrr@10"]), reverse=True)
    write_summary_csv(
        output_dir / "cross_space_soft_cosine_summary_k10.csv",
        rows,
        fieldnames=[
            "qd_space",
            "s_source",
            "space_type",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
        ],
    )

    for metric_name in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]:
        write_metric_table(output_dir / f"{metric_name}_by_k.csv", all_metrics, metric_name)

    save_all_metrics_overlay(output_dir / "all_metrics_overlay.png", all_metrics)
    save_single_metric_overlay(output_dir / "map_overlay.png", all_metrics, "map", "Cross-Space Soft Cosine MAP vs k")
    save_single_metric_overlay(output_dir / "ndcg_overlay.png", all_metrics, "ndcg", "Cross-Space Soft Cosine nDCG vs k")

    best_by_map = max(rows, key=lambda row: row["map@10"])
    best_by_ndcg = max(rows, key=lambda row: row["ndcg@10"])
    best_by_mrr = max(rows, key=lambda row: row["mrr@10"])

    write_json(
        output_dir / "cross_space_soft_cosine_summary.json",
        {
            "rows": rows,
            "best_by_map": best_by_map,
            "best_by_ndcg": best_by_ndcg,
            "best_by_mrr": best_by_mrr,
            "recommended_primary_metric": "map@10",
            "recommended_primary_metric_reason": (
                "MAP@10 is the best main metric here because this experiment compares full ranking quality "
                "across the top 10 results, not just the first hit. nDCG@10 is the best secondary metric "
                "because it rewards relevant documents appearing higher in the ranked list."
            ),
            "all_metrics": all_metrics,
        },
    )

    print("Wrote:", output_dir / "cross_space_soft_cosine_summary_k10.csv")
    print("Wrote:", output_dir / "all_metrics_overlay.png")
    print("Best by MAP@10:", json.dumps(best_by_map, indent=2))


if __name__ == "__main__":
    main()
