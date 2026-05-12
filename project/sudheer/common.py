from __future__ import annotations

import csv
import importlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


SUDHEER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SUDHEER_DIR.parent
ASSIGNMENT_ROOT = PROJECT_DIR.parent
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

K10_KEYS = [
    "precision_at_10",
    "recall_at_10",
    "fscore_at_10",
    "map_at_10",
    "ndcg_at_10",
    "mrr_at_10",
]


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


def flatten_tokens(item: Sequence[Sequence[str]]) -> Iterable[str]:
    for sentence in item:
        for token in sentence:
            yield token


def unigram_feature_map(item: Sequence[Sequence[str]]) -> Dict[str, float]:
    counts = Counter()
    for sentence in item:
        counts.update(sentence)
    return {term: float(value) for term, value in counts.items()}


def feature_maps_to_sparse_matrix(
    feature_maps: Sequence[Dict[str, float]],
    term_to_idx: Dict[str, int],
    n_terms: int,
) -> sparse.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for row_index, fmap in enumerate(feature_maps):
        for term, value in fmap.items():
            term_index = term_to_idx.get(term)
            if term_index is None or value <= 0.0:
                continue
            rows.append(row_index)
            cols.append(term_index)
            data.append(float(value))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(feature_maps), n_terms), dtype=np.float64)


def _l2_normalize_rows(matrix: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()) + eps
    inv = sparse.diags(1.0 / row_norms)
    return sparse.csr_matrix(inv.dot(matrix)).tocsr()


@dataclass
class TfidfIndex:
    name: str
    doc_ids: List[int]
    vocab: List[str]
    term_to_idx: Dict[str, int]
    idf: np.ndarray
    doc_freq: Dict[str, int]
    doc_tf: sparse.csr_matrix
    doc_tfidf: sparse.csr_matrix
    doc_tfidf_normalized: sparse.csr_matrix

    @classmethod
    def build(
        cls,
        name: str,
        feature_maps: Sequence[Dict[str, float]],
        doc_ids: Sequence[int],
    ) -> "TfidfIndex":
        doc_ids = list(doc_ids)
        vocab = sorted({term for fmap in feature_maps for term in fmap})
        term_to_idx = {term: index for index, term in enumerate(vocab)}
        doc_tf = feature_maps_to_sparse_matrix(feature_maps, term_to_idx, len(vocab))

        binary_tf = doc_tf.copy()
        binary_tf.data = np.ones_like(binary_tf.data)
        df = np.maximum(1.0, np.asarray(binary_tf.sum(axis=0)).ravel())
        idf = np.log(float(max(1, len(feature_maps))) / df)

        doc_tfidf = doc_tf.multiply(idf)
        doc_tfidf_normalized = _l2_normalize_rows(doc_tfidf.tocsr())
        doc_freq = {term: int(df[term_to_idx[term]]) for term in vocab}
        return cls(
            name=name,
            doc_ids=doc_ids,
            vocab=vocab,
            term_to_idx=term_to_idx,
            idf=idf,
            doc_freq=doc_freq,
            doc_tf=doc_tf.tocsr(),
            doc_tfidf=doc_tfidf.tocsr(),
            doc_tfidf_normalized=doc_tfidf_normalized,
        )

    def feature_maps_to_matrix(self, feature_maps: Sequence[Dict[str, float]]) -> sparse.csr_matrix:
        return feature_maps_to_sparse_matrix(feature_maps, self.term_to_idx, len(self.vocab))

    def idf_by_term(self) -> Dict[str, float]:
        return {term: float(self.idf[index]) for term, index in self.term_to_idx.items()}

    def rank_feature_maps(
        self,
        query_feature_maps: Sequence[Dict[str, float]],
    ) -> Tuple[List[List[int]], List[np.ndarray]]:
        return self.rank_query_matrix(self.feature_maps_to_matrix(query_feature_maps))

    def rank_query_matrix(
        self,
        query_tf: sparse.csr_matrix,
    ) -> Tuple[List[List[int]], List[np.ndarray]]:
        query_tfidf = query_tf.multiply(self.idf)
        query_norm = _l2_normalize_rows(query_tfidf.tocsr())
        scores = query_norm.dot(self.doc_tfidf_normalized.T).toarray()

        rankings = []
        score_vectors = []
        for query_scores in scores:
            order = np.argsort(-query_scores, kind="mergesort")
            rankings.append([self.doc_ids[index] for index in order])
            score_vectors.append(query_scores.copy())
        return rankings, score_vectors


@dataclass
class DenseCosineModel:
    name: str
    doc_ids: List[int]
    doc_vectors: np.ndarray
    doc_vectors_normalized: np.ndarray

    @classmethod
    def build(
        cls,
        name: str,
        doc_vectors: np.ndarray,
        doc_ids: Sequence[int],
    ) -> "DenseCosineModel":
        doc_vectors = np.asarray(doc_vectors, dtype=np.float64)
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        normalized = np.divide(
            doc_vectors,
            safe_norms,
            out=np.zeros_like(doc_vectors, dtype=np.float64),
            where=safe_norms > 0.0,
        )
        return cls(
            name=name,
            doc_ids=list(doc_ids),
            doc_vectors=doc_vectors,
            doc_vectors_normalized=normalized,
        )

    def rank(self, query_vectors: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
        query_vectors = np.asarray(query_vectors, dtype=np.float64)
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        normalized = np.divide(
            query_vectors,
            safe_norms,
            out=np.zeros_like(query_vectors, dtype=np.float64),
            where=safe_norms > 0.0,
        )
        scores = normalized @ self.doc_vectors_normalized.T

        rankings = []
        score_vectors = []
        for query_scores in scores:
            order = np.argsort(-query_scores, kind="mergesort")
            rankings.append([self.doc_ids[index] for index in order])
            score_vectors.append(query_scores.copy())
        return rankings, score_vectors


@dataclass
class Word2VecArtifacts:
    vector_size: int
    word_vectors: Dict[str, np.ndarray]
    vocab_terms: int
    present_in_model: int
    model_coverage: float
    sentences_seen: int
    embedding_source: str = "scratch"
    pretrained_model_name: str | None = None
    fine_tuned: bool = False


@dataclass
class SoftCosineModel:
    name: str
    doc_ids: List[int]
    idf: np.ndarray
    doc_tfidf: sparse.csr_matrix
    term_similarity: sparse.csr_matrix
    doc_norm_sq: np.ndarray

    @classmethod
    def build(
        cls,
        name: str,
        tfidf_index: TfidfIndex,
        term_similarity: sparse.csr_matrix,
    ) -> "SoftCosineModel":
        doc_tfidf = tfidf_index.doc_tfidf.tocsr()
        doc_soft = doc_tfidf.dot(term_similarity)
        doc_norm_sq = np.asarray(doc_soft.multiply(doc_tfidf).sum(axis=1)).ravel()
        return cls(
            name=name,
            doc_ids=list(tfidf_index.doc_ids),
            idf=tfidf_index.idf.copy(),
            doc_tfidf=doc_tfidf,
            term_similarity=term_similarity.tocsr(),
            doc_norm_sq=doc_norm_sq,
        )

    def rank_query_matrix(self, query_tf: sparse.csr_matrix) -> Tuple[List[List[int]], List[np.ndarray]]:
        query_tfidf = query_tf.multiply(self.idf).tocsr()
        query_soft = query_tfidf.dot(self.term_similarity)
        numerator = query_soft.dot(self.doc_tfidf.T).toarray()
        query_norm_sq = np.asarray(query_soft.multiply(query_tfidf).sum(axis=1)).ravel()
        denom = np.sqrt(
            np.maximum(query_norm_sq[:, np.newaxis], 0.0) * np.maximum(self.doc_norm_sq[np.newaxis, :], 0.0)
        ) + 1e-12
        scores = numerator / denom

        rankings = []
        score_vectors = []
        for query_scores in scores:
            order = np.argsort(-query_scores, kind="mergesort")
            rankings.append([self.doc_ids[index] for index in order])
            score_vectors.append(query_scores.copy())
        return rankings, score_vectors


def train_word2vec_artifacts(
    docs_tokens: Sequence[Sequence[Sequence[str]]],
    vocab: Sequence[str],
    vector_size: int,
    window: int,
    min_count: int = 1,
    workers: int = 1,
    sg: int = 1,
    epochs: int = 20,
    initial_word_vectors: Dict[str, np.ndarray] | None = None,
    embedding_source: str = "scratch",
    pretrained_model_name: str | None = None,
) -> Word2VecArtifacts:
    try:
        word2vec_cls = importlib.import_module("gensim.models").Word2Vec
    except Exception as exc:
        raise ImportError(
            "gensim is required to run the Sudheer Word2Vec experiments."
        ) from exc

    sentences = [sentence for doc in docs_tokens for sentence in doc if sentence]
    if not sentences:
        return Word2VecArtifacts(
            vector_size=vector_size,
            word_vectors={},
            vocab_terms=len(vocab),
            present_in_model=0,
            model_coverage=0.0,
            sentences_seen=0,
            embedding_source=embedding_source,
            pretrained_model_name=pretrained_model_name,
            fine_tuned=initial_word_vectors is not None,
        )

    if initial_word_vectors is None:
        model = word2vec_cls(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs,
            seed=42,
        )
    else:
        model = word2vec_cls(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=42,
        )
        model.build_vocab(sentences)
        if hasattr(model.wv, "vectors_lockf"):
            model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
        for term, vector in initial_word_vectors.items():
            term_index = model.wv.key_to_index.get(term)
            if term_index is None:
                continue
            vector = np.asarray(vector, dtype=np.float64)
            if vector.shape[0] != vector_size:
                continue
            model.wv.vectors[term_index] = vector
        model.train(sentences, total_examples=len(sentences), epochs=epochs)

    word_vectors = {}
    for term in vocab:
        if term in model.wv:
            word_vectors[term] = np.asarray(model.wv[term], dtype=np.float64)

    present_in_model = len(word_vectors)
    vocab_terms = len(vocab)
    coverage = float(present_in_model / vocab_terms) if vocab_terms else 0.0
    return Word2VecArtifacts(
        vector_size=vector_size,
        word_vectors=word_vectors,
        vocab_terms=vocab_terms,
        present_in_model=present_in_model,
        model_coverage=coverage,
        sentences_seen=len(sentences),
        embedding_source=embedding_source if initial_word_vectors is None else "finetuned",
        pretrained_model_name=pretrained_model_name,
        fine_tuned=initial_word_vectors is not None,
    )


def load_pretrained_word2vec_artifacts(
    model_name: str,
    vocab: Sequence[str],
) -> Word2VecArtifacts:
    try:
        downloader = importlib.import_module("gensim.downloader")
    except Exception as exc:
        raise ImportError(
            "gensim is required to load pretrained word embeddings."
        ) from exc

    try:
        keyed_vectors = downloader.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load pretrained embeddings '{model_name}'. "
            "Check network access or the gensim downloader cache."
        ) from exc

    word_vectors = {}
    for term in vocab:
        if term in keyed_vectors:
            word_vectors[term] = np.asarray(keyed_vectors[term], dtype=np.float64)

    present_in_model = len(word_vectors)
    vocab_terms = len(vocab)
    coverage = float(present_in_model / vocab_terms) if vocab_terms else 0.0
    return Word2VecArtifacts(
        vector_size=int(keyed_vectors.vector_size),
        word_vectors=word_vectors,
        vocab_terms=vocab_terms,
        present_in_model=present_in_model,
        model_coverage=coverage,
        sentences_seen=0,
        embedding_source="pretrained",
        pretrained_model_name=model_name,
        fine_tuned=False,
    )


def build_average_embeddings(
    collection: Sequence[Sequence[Sequence[str]]],
    word_vectors: Dict[str, np.ndarray],
    vector_size: int,
    term_weights: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    embeddings = np.zeros((len(collection), vector_size), dtype=np.float64)
    in_vocab_token_counts = []
    nonzero_items = 0

    for item_index, item in enumerate(collection):
        counts = Counter(flatten_tokens(item))
        weighted_sum = np.zeros(vector_size, dtype=np.float64)
        total_weight = 0.0
        in_vocab_tokens = 0

        for term, tf in counts.items():
            vector = word_vectors.get(term)
            if vector is None:
                continue
            term_weight = 1.0 if term_weights is None else float(term_weights.get(term, 0.0))
            weight = float(tf) * term_weight
            if weight <= 0.0:
                continue
            weighted_sum += weight * vector
            total_weight += weight
            in_vocab_tokens += int(tf)

        if total_weight > 0.0:
            embeddings[item_index] = weighted_sum / total_weight
            nonzero_items += 1

        in_vocab_token_counts.append(in_vocab_tokens)

    zero_vector_items = len(collection) - nonzero_items
    return embeddings, {
        "avg_in_vocab_tokens_per_item": float(np.mean(in_vocab_token_counts)) if in_vocab_token_counts else 0.0,
        "zero_vector_items": int(zero_vector_items),
        "nonzero_vector_ratio": float(nonzero_items / len(collection)) if collection else 0.0,
    }


def build_term_similarity_matrix(
    vocab: Sequence[str],
    term_to_idx: Dict[str, int],
    word_vectors: Dict[str, np.ndarray],
    top_k_neighbors: int,
    min_similarity: float,
    similarity_power: float = 1.0,
) -> Tuple[sparse.csr_matrix, Dict[str, float]]:
    rows = list(range(len(vocab)))
    cols = list(range(len(vocab)))
    data = [1.0] * len(vocab)

    available_terms = [term for term in vocab if term in word_vectors]
    if not available_terms:
        identity = sparse.identity(len(vocab), format="csr", dtype=np.float64)
        return identity, {
            "represented_terms": 0,
            "representation_coverage": 0.0,
            "retained_neighbor_edges": 0,
            "avg_neighbors_per_term": 0.0,
        }

    matrix = np.vstack([word_vectors[term] for term in available_terms]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    n_neighbors = min(len(available_terms), top_k_neighbors + 1)
    if n_neighbors <= 1:
        identity = sparse.identity(len(vocab), format="csr", dtype=np.float64)
        return identity, {
            "represented_terms": len(available_terms),
            "representation_coverage": float(len(available_terms) / len(vocab)) if vocab else 0.0,
            "retained_neighbor_edges": 0,
            "avg_neighbors_per_term": 0.0,
        }

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix, return_distance=True)

    retained_edges = 0
    neighbor_counts = []
    pair_to_similarity: Dict[Tuple[int, int], float] = {}
    for src_position, term in enumerate(available_terms):
        src_index = term_to_idx[term]
        local_count = 0
        for distance, neighbor_position in zip(distances[src_position], indices[src_position]):
            if neighbor_position == src_position:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < min_similarity:
                continue
            similarity = similarity ** similarity_power
            neighbor_term = available_terms[neighbor_position]
            dst_index = term_to_idx[neighbor_term]
            if src_index == dst_index:
                continue
            pair = (src_index, dst_index) if src_index < dst_index else (dst_index, src_index)
            pair_to_similarity[pair] = max(pair_to_similarity.get(pair, 0.0), similarity)
            local_count += 1
            retained_edges += 1
        neighbor_counts.append(local_count)

    for (src_index, dst_index), similarity in pair_to_similarity.items():
        rows.extend([src_index, dst_index])
        cols.extend([dst_index, src_index])
        data.extend([similarity, similarity])

    similarity_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(vocab), len(vocab)),
        dtype=np.float64,
    ).tocsr()
    similarity_matrix.sum_duplicates()
    similarity_matrix.data = np.clip(similarity_matrix.data, 0.0, 1.0)

    represented_terms = len(available_terms)
    return similarity_matrix, {
        "represented_terms": represented_terms,
        "representation_coverage": float(represented_terms / len(vocab)) if vocab else 0.0,
            "retained_neighbor_edges": int(len(pair_to_similarity)),
            "avg_neighbors_per_term": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        }


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
    grouped_metrics: Dict[object, Dict[str, List[float]]],
    method_labels: Dict[object, str],
    ordered_keys: Sequence[object],
    title: str,
) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(ordered_keys))))
    for axis, metric_name in zip(axes, METRIC_LABELS):
        for color_index, key in enumerate(ordered_keys):
            axis.plot(
                ks,
                grouped_metrics[key][metric_name],
                marker="s",
                linewidth=1.8,
                color=colors[color_index],
                label=method_labels[key],
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
    sweep_results: Sequence[Dict[str, object]],
    best_config: Dict[str, object],
    title: str,
) -> None:
    ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    overlap_with_baseline = all(
        np.allclose(
            np.array(best_config["metrics_by_k"][metric_name], dtype=np.float64),
            np.array(baseline_metrics[metric_name], dtype=np.float64),
            atol=1e-12,
        )
        for metric_name in METRIC_LABELS
    )
    best_label = "best_tuned_config"
    if overlap_with_baseline:
        best_label = "best_tuned_config (overlaps baseline)"

    for axis, metric_name in zip(axes, METRIC_LABELS):
        for row in sweep_results:
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
            best_config["metrics_by_k"][metric_name],
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


def save_primary_sweep_plot(
    path: Path,
    rows: Sequence[Dict[str, object]],
    primary_key: str,
    x_label: str,
    title: str,
) -> None:
    x_values = [row[primary_key] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for axis, metric_key in zip(axes, K10_KEYS):
        axis.plot(x_values, [row[metric_key] for row in rows], marker="o", linewidth=2)
        axis.set_title(metric_key.replace("_at_10", "").upper() + "@10")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Score")
        axis.set_xticks(x_values)
        axis.grid(alpha=0.25)

    fig.suptitle(title, y=0.992, fontsize=15)
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rankings(
    path: Path,
    query_ids: Sequence[int],
    rankings: Sequence[Sequence[int]],
    top_k: int = 20,
) -> None:
    payload = {
        str(query_id): ranked_docs[:top_k]
        for query_id, ranked_docs in zip(query_ids, rankings)
    }
    write_json(path, payload)


def write_summary_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_primary_sweep_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_sort_key(row: Dict[str, object]) -> Tuple[float, float, float, float, float, float]:
    return (
        float(row["map_at_10"]),
        float(row["ndcg_at_10"]),
        float(row["mrr_at_10"]),
        float(row["precision_at_10"]),
        float(row["recall_at_10"]),
        float(row["fscore_at_10"]),
    )


def best_results_by_primary(
    sweep_results: Sequence[Dict[str, object]],
    primary_key: str,
) -> List[Dict[str, object]]:
    best = {}
    for row in sweep_results:
        primary_value = row[primary_key]
        current = best.get(primary_value)
        if current is None or metric_sort_key(row) > metric_sort_key(current):
            best[primary_value] = row
    return [best[key] for key in sorted(best)]


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
    for index, query_id in enumerate(query_ids):
        delta_ap = method_per_query["average_precision"][index] - baseline_per_query["average_precision"][index]
        delta_hits = method_per_query["hits_at_5"][index] - baseline_per_query["hits_at_5"][index]
        candidates.append((delta_ap, delta_hits, index, query_id))

    candidates.sort(reverse=True)
    candidate_by_query = {query_id: candidate for candidate in candidates for query_id in [candidate[3]]}

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
    for delta_ap, delta_hits, index, query_id in selected:
        relevant = relevant_by_query[str(query_id)]
        baseline_top5 = baseline_rankings[index][:5]
        method_top5 = method_rankings[index][:5]
        rows.append(
            {
                "query_id": query_id,
                "query": query_lookup[query_id],
                "baseline_hits_at_5": baseline_per_query["hits_at_5"][index],
                "method_hits_at_5": method_per_query["hits_at_5"][index],
                "baseline_ap_at_10": round(baseline_per_query["average_precision"][index], 4),
                "method_ap_at_10": round(method_per_query["average_precision"][index], 4),
                "delta_ap_at_10": round(delta_ap, 4),
                "delta_hits_at_5": int(delta_hits),
                "baseline_first_relevant_rank": baseline_per_query["first_relevant_rank"][index],
                "method_first_relevant_rank": method_per_query["first_relevant_rank"][index],
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


def _format_report_value(value, fmt: str) -> str:
    if fmt == "{}":
        return str(value)
    return fmt.format(value)


def build_experiment_report(
    output_path: Path,
    title: str,
    method_key: str,
    method_label: str,
    summary: Dict[str, object],
    examples: Sequence[Dict[str, object]],
    sweep_rows: Sequence[Dict[str, object]],
    sweep_section_title: str,
    sweep_columns: Sequence[Tuple[str, str, str]],
    best_config_fields: Sequence[Tuple[str, str, str]],
    method_summary_lines: Sequence[str],
    hypothesis_lines: Sequence[str],
    limitations_lines: Sequence[str],
    interpretation_lines: Sequence[str],
    citations: Sequence[Tuple[str, str]],
) -> None:
    baseline = summary["methods"]["baseline_tfidf"]
    method = summary["methods"][method_key]
    delta = summary["delta_vs_baseline_at_10"]
    config = summary["best_config"]
    significance = summary["significance"]

    lines = [f"# {title}", ""]

    if method_summary_lines:
        lines.extend(["## Method Summary", ""])
        for paragraph in method_summary_lines:
            lines.extend([paragraph, ""])

    if hypothesis_lines:
        lines.extend(["## Hypothesis", ""])
        for paragraph in hypothesis_lines:
            lines.extend([paragraph, ""])

    if limitations_lines:
        lines.extend(["## Baseline Limitations Addressed", ""])
        for bullet in limitations_lines:
            lines.append(f"- {bullet}")
        lines.append("")

    lines.extend(["## Best Configuration", ""])
    for label, key, fmt in best_config_fields:
        lines.append(f"- `{label} = {_format_report_value(config.get(key), fmt)}`")
    lines.append("")

    if sweep_rows:
        lines.extend([f"## {sweep_section_title}", ""])
        table_headers = [header for _, header, _ in sweep_columns] + ["P@10", "R@10", "F@10", "MAP@10", "nDCG@10", "MRR@10"]
        lines.append("| " + " | ".join(table_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(table_headers)) + " |")
        for row in sweep_rows:
            values = [
                _format_report_value(row[key], fmt)
                for key, _, fmt in sweep_columns
            ]
            values.extend(
                [
                    f"{row['precision_at_10']:.4f}",
                    f"{row['recall_at_10']:.4f}",
                    f"{row['fscore_at_10']:.4f}",
                    f"{row['map_at_10']:.4f}",
                    f"{row['ndcg_at_10']:.4f}",
                    f"{row['mrr_at_10']:.4f}",
                ]
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            "## k=10 Results",
            "",
            "| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| baseline_tfidf | {baseline['k10']['precision']:.4f} | {baseline['k10']['recall']:.4f} | {baseline['k10']['fscore']:.4f} | "
            f"{baseline['k10']['map']:.4f} | {baseline['k10']['ndcg']:.4f} | {baseline['k10']['mrr']:.4f} | {baseline['runtime_seconds']:.2f} |",
            f"| {method_label} | {method['k10']['precision']:.4f} | {method['k10']['recall']:.4f} | {method['k10']['fscore']:.4f} | "
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
        ]
    )

    if interpretation_lines:
        lines.extend(["## Interpretation", ""])
        for paragraph in interpretation_lines:
            lines.extend([paragraph, ""])

    if examples:
        lines.extend(
            [
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
        lines.append("")

    if citations:
        lines.extend(["## Research References", ""])
        for label, url in citations:
            lines.append(f"- {label}: {url}")
        lines.append("")

    lines.extend(["## Output Files", ""])
    for label, value in summary["paths"].items():
        lines.append(f"- {label}: `{value}`")

    output_path.write_text("\n".join(lines), encoding="utf-8")
