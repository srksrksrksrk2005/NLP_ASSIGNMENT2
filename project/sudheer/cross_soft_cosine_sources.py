#!/usr/bin/env python3
"""
Cross-compare soft-cosine similarity sources against multiple compatible q,d spaces.

Important constraint:
Soft cosine needs q, d, and S to live in the same feature space. So this script
compares term-based q,d spaces only:
1. unigram TF-IDF q,d
2. bigram TF-IDF q,d

Against multiple similarity sources S:
1. TF-IDF-derived S
2. LSA-derived S
3. ESA-derived S
4. WordNet-derived S
5. Word2Vec-derived S
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


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
    build_term_similarity_matrix,
    compute_metrics,
    load_json,
    train_word2vec_artifacts,
    write_json,
    write_summary_csv,
)


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
WORDNET_MATRIX = load_module_from_path(
    "nikhil_wordnet_matrix",
    PROJECT_DIR / "Nikhil" / "query_expansion" / "expansion" / "wordnet_matrix.py",
)

build_tfidf_neighbor_map = EMBEDDING_MATRICES.build_tfidf_neighbor_map
build_lsa_neighbor_map = EMBEDDING_MATRICES.build_lsa_neighbor_map
build_esa_neighbor_map = EMBEDDING_MATRICES.build_esa_neighbor_map
build_wordnet_neighbor_map = WORDNET_MATRIX.build_wordnet_neighbor_map


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


def flatten_document(document):
    return " ".join(token for sentence in document for token in sentence)


def flatten_collection(collection):
    return [flatten_document(item) for item in collection]


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


def run_source(index: TextTfidfIndex, query_tf: sparse.csr_matrix, source_name: str, docs_tokens):
    if source_name == "tfidf":
        neighbor_map = build_tfidf_neighbor_map(index.doc_tfidf, index.vocab, top_k=5, min_similarity=0.05, progress=False)
        S = neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    elif source_name == "lsa":
        neighbor_map = build_lsa_neighbor_map(index.doc_tfidf, index.vocab, n_components=150, top_k=10, min_similarity=0.05, progress=False)
        S = neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    elif source_name == "esa":
        neighbor_map = build_esa_neighbor_map(index.doc_tfidf, index.vocab, top_concepts=50, top_k=10, min_similarity=0.02, progress=False, stats_out={})
        S = neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    elif source_name == "wordnet":
        neighbor_map = build_wordnet_neighbor_map(index.vocab, top_k=5, min_similarity=0.05, progress=False, cache_dir=SCRIPT_DIR / "soft_cosine_sources" / "cache" / "wordnet")
        S = neighbor_map_to_similarity_matrix(index.vocab, neighbor_map, similarity_power=1.0)
    elif source_name == "word2vec":
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
        S, _ = build_term_similarity_matrix(
            vocab=index.vocab,
            term_to_idx=index.term_to_idx,
            word_vectors=artifacts.word_vectors,
            top_k_neighbors=10,
            min_similarity=0.10,
            similarity_power=2.0,
        )
    else:
        raise ValueError(f"Unknown source: {source_name}")

    model = SoftCosineModel.build(f"{source_name}_soft_cosine", index, S)
    rankings, _ = model.rank_query_matrix(query_tf)
    return rankings


def save_map_bar_plot(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    labels = [f"{row['qd_space']}\nS={row['s_source']}" for row in rows]
    values = [row["map@10"] for row in rows]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(rows)), values)
    plt.xticks(range(len(rows)), labels, rotation=45, ha="right")
    plt.ylabel("MAP@10")
    plt.title("Cross Soft Cosine Comparison")
    plt.grid(axis="y", alpha=0.25)

    best_index = int(np.argmax(values))
    bars[best_index].set_color("tab:red")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


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

    qd_spaces = {
        "unigram_tfidf": {"max_df": 0.95, "min_df": 2, "ngram_max": 1},
        "bigram_tfidf": {"max_df": 0.95, "min_df": 2, "ngram_max": 2},
    }
    s_sources = ["tfidf", "lsa", "esa", "wordnet", "word2vec"]

    evaluator = Evaluation()
    rows = []
    all_metrics = {}

    for qd_space, cfg in qd_spaces.items():
        index = TextTfidfIndex.build(
            doc_texts=doc_texts,
            doc_ids=doc_ids,
            max_df=cfg["max_df"],
            min_df=cfg["min_df"],
            ngram_max=cfg["ngram_max"],
        )
        query_tf = index.transform_queries(query_texts)

        for s_source in s_sources:
            rankings = run_source(index, query_tf, s_source, processed_docs)
            metrics_by_k = compute_metrics(evaluator, rankings, query_ids, qrels)
            k10 = {metric: values[9] for metric, values in metrics_by_k.items()}

            row = {
                "qd_space": qd_space,
                "s_source": s_source,
                "precision@10": k10["precision"],
                "recall@10": k10["recall"],
                "fscore@10": k10["fscore"],
                "map@10": k10["map"],
                "ndcg@10": k10["ndcg"],
                "mrr@10": k10["mrr"],
            }
            rows.append(row)
            all_metrics[f"{qd_space}__{s_source}"] = metrics_by_k

    output_dir = SCRIPT_DIR / "cross_soft_cosine" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(
        output_dir / "cross_soft_cosine_summary_k10.csv",
        rows,
        fieldnames=[
            "qd_space",
            "s_source",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
        ],
    )
    write_json(
        output_dir / "cross_soft_cosine_summary.json",
        {
            "rows": rows,
            "best_by_map": max(rows, key=lambda row: row["map@10"]),
            "all_metrics": all_metrics,
        },
    )
    save_map_bar_plot(output_dir / "cross_soft_cosine_map_bar.png", rows)

    print("Wrote:", output_dir / "cross_soft_cosine_summary_k10.csv")
    print("Wrote:", output_dir / "cross_soft_cosine_map_bar.png")


if __name__ == "__main__":
    main()
