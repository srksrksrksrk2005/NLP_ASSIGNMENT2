import importlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse


NeighborMap = Dict[str, List[Tuple[str, float]]]



def build_tfidf_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    top_k: int = 10,
    min_similarity: float = 0.05,
) -> NeighborMap:
    term_document = doc_tfidf.T.tocsr()
    return _neighbor_map_from_matrix(list(vocab), term_document, top_k, min_similarity)



def build_lsa_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    n_components: int = 128,
    top_k: int = 10,
    min_similarity: float = 0.05,
) -> NeighborMap:
    term_document = doc_tfidf.T.tocsr()

    max_components = min(term_document.shape[0] - 1, term_document.shape[1] - 1)
    if max_components < 2:
        return {term: [] for term in vocab}

    components = max(2, min(n_components, max_components))

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=components, random_state=42)
    term_embeddings = svd.fit_transform(term_document)
    return _neighbor_map_from_matrix(list(vocab), term_embeddings, top_k, min_similarity)



def build_esa_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    top_concepts: int = 100,
    top_k: int = 10,
    min_similarity: float = 0.05,
) -> NeighborMap:
    """
    ESA-style concept vectors where each term keeps only top concept activations.
    Here, Cranfield documents are treated as explicit concepts.
    """
    term_document = doc_tfidf.T.tocsr()
    concept_vectors = _keep_top_k_per_row(term_document, top_concepts)
    return _neighbor_map_from_matrix(list(vocab), concept_vectors, top_k, min_similarity)



def build_word2vec_neighbor_map(
    docs_tokens: Sequence[Sequence[Sequence[str]]],
    vocab: Sequence[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 1,
    sg: int = 1,
    epochs: int = 20,
    top_k: int = 10,
    min_similarity: float = 0.05,
) -> NeighborMap:
    try:
        word2vec_cls = importlib.import_module("gensim.models").Word2Vec
    except Exception as exc:
        raise ImportError("gensim is required for word2vec neighbor map") from exc

    sentences = [sentence for doc in docs_tokens for sentence in doc if sentence]
    if not sentences:
        return {term: [] for term in vocab}

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

    available_terms: List[str] = []
    vectors: List[np.ndarray] = []
    for term in vocab:
        if term in model.wv:
            available_terms.append(term)
            vectors.append(model.wv[term])

    if not available_terms:
        return {term: [] for term in vocab}

    matrix = np.vstack(vectors)
    partial = _neighbor_map_from_matrix(available_terms, matrix, top_k, min_similarity)

    full_map = {term: [] for term in vocab}
    full_map.update(partial)
    return full_map



def _neighbor_map_from_matrix(
    vocab: List[str],
    matrix,
    top_k: int,
    min_similarity: float,
) -> NeighborMap:
    if not vocab:
        return {}

    n_neighbors = min(len(vocab), top_k + 1)
    if n_neighbors <= 1:
        return {term: [] for term in vocab}

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix, return_distance=True)

    neighbor_map: NeighborMap = {}
    for i, term in enumerate(vocab):
        row: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            sim = 1.0 - float(dist)
            if sim >= min_similarity:
                row.append((vocab[idx], sim))
        neighbor_map[term] = row[:top_k]

    return neighbor_map



def _keep_top_k_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    if k <= 0:
        return sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)

    matrix = matrix.tocsr()
    new_data = []
    new_indices = []
    new_indptr = [0]

    n_rows = int(matrix.shape[0]) if matrix.shape is not None else 0
    for row in range(n_rows):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]

        if row_data.size <= k:
            chosen = np.arange(row_data.size)
        else:
            chosen = np.argpartition(row_data, -k)[-k:]
            chosen = chosen[np.argsort(row_data[chosen])[::-1]]

        new_data.extend(row_data[chosen].tolist())
        new_indices.extend(row_indices[chosen].tolist())
        new_indptr.append(len(new_data))

    return sparse.csr_matrix(
        (np.array(new_data), np.array(new_indices), np.array(new_indptr)),
        shape=matrix.shape,
        dtype=matrix.dtype,
    )
