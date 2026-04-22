from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import sparse


class VectorSpaceRetrieval:
    """TF-IDF document index and cosine-ranking in vocabulary space."""

    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.vocab: List[str] = []
        self.term_to_idx: Dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.doc_tf: sparse.csr_matrix | None = None
        self.doc_tfidf: sparse.csr_matrix | None = None
        self.doc_tfidf_normalized: sparse.csr_matrix | None = None

    @staticmethod
    def _flatten_tokens(doc_sentences: Sequence[Sequence[str]]) -> List[str]:
        return [token for sentence in doc_sentences for token in sentence]

    def build(self, docs_tokens: Sequence[Sequence[Sequence[str]]], doc_ids: Sequence[str]) -> None:
        self.doc_ids = list(doc_ids)

        vocab_set = set()
        doc_counters: List[Counter] = []
        for doc in docs_tokens:
            tokens = self._flatten_tokens(doc)
            counter = Counter(tokens)
            doc_counters.append(counter)
            vocab_set.update(counter.keys())

        self.vocab = sorted(vocab_set)
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocab)}

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for doc_idx, counter in enumerate(doc_counters):
            for term, tf in counter.items():
                rows.append(doc_idx)
                cols.append(self.term_to_idx[term])
                data.append(float(tf))

        n_docs = len(doc_counters)
        n_terms = len(self.vocab)
        self.doc_tf = sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float64)

        # IDF with assignment-style definition log(N / df).
        binary_tf = self.doc_tf.copy()
        binary_tf.data = np.ones_like(binary_tf.data)
        df = np.maximum(1.0, np.asarray(binary_tf.sum(axis=0)).ravel())
        self.idf = np.log(float(n_docs) / df)

        self.doc_tfidf = self.doc_tf.multiply(self.idf)
        self.doc_tfidf_normalized = _l2_normalize_rows(self.doc_tfidf)

    def query_vectors_to_rankings(self, query_vectors: np.ndarray) -> List[List[str]]:
        if self.idf is None or self.doc_tfidf_normalized is None:
            raise RuntimeError("Index not built. Call build() first.")

        if query_vectors.ndim != 2:
            raise ValueError("query_vectors must be a 2D array")
        if query_vectors.shape[1] != len(self.vocab):
            raise ValueError("Query vectors must align with vocabulary size")

        query_tfidf = query_vectors * self.idf[np.newaxis, :]
        query_tfidf_sparse = sparse.csr_matrix(query_tfidf)
        query_norm = _l2_normalize_rows(query_tfidf_sparse)
        cosine = query_norm.dot(self.doc_tfidf_normalized.T).toarray()

        rankings: List[List[str]] = []
        for q_idx in range(cosine.shape[0]):
            order = np.argsort(cosine[q_idx])[::-1]
            rankings.append([self.doc_ids[i] for i in order])
        return rankings



def _l2_normalize_rows(matrix: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()) + eps
    inv = sparse.diags(1.0 / row_norms)
    return sparse.csr_matrix(inv.dot(matrix)).tocsr()



def flatten_query_tokens(query_sentences: Sequence[Sequence[str]]) -> Iterable[str]:
    for sentence in query_sentences:
        for token in sentence:
            yield token
