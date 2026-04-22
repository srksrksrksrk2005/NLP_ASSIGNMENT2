import importlib
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm


NeighborMap = Dict[str, List[Tuple[str, float]]]



def build_tfidf_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    top_k: int = 10,
    min_similarity: float = 0.05,
    progress: bool = False,
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
) -> NeighborMap:
    if logger is not None:
        logger("Building TF-IDF term-term neighbor map")
    term_document = doc_tfidf.T.tocsr()
    return _neighbor_map_from_matrix(
        list(vocab),
        term_document,
        top_k,
        min_similarity,
        progress=progress,
        progress_desc="TF-IDF neighbors",
        logger=logger,
        log_every=log_every,
    )



def build_lsa_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    n_components: int = 128,
    top_k: int = 10,
    min_similarity: float = 0.05,
    progress: bool = False,
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
) -> NeighborMap:
    term_document = doc_tfidf.T.tocsr()

    max_components = min(term_document.shape[0] - 1, term_document.shape[1] - 1)
    if max_components < 2:
        return {term: [] for term in vocab}

    components = max(2, min(n_components, max_components))

    from sklearn.decomposition import TruncatedSVD

    started = time.time()
    if logger is not None:
        logger(f"LSA SVD fitting started with n_components={components}")

    svd = TruncatedSVD(n_components=components, random_state=42)
    term_embeddings = svd.fit_transform(term_document)
    if logger is not None:
        logger(f"LSA SVD fitting finished in {time.time() - started:.2f}s")
    return _neighbor_map_from_matrix(
        list(vocab),
        term_embeddings,
        top_k,
        min_similarity,
        progress=progress,
        progress_desc="LSA neighbors",
        logger=logger,
        log_every=log_every,
    )



def build_esa_neighbor_map(
    doc_tfidf: sparse.csr_matrix,
    vocab: Sequence[str],
    top_concepts: int = 100,
    top_k: int = 10,
    min_similarity: float = 0.05,
    progress: bool = False,
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
    stats_out: Dict | None = None,
) -> NeighborMap:
    """
    ESA-style concept vectors where each term keeps only top concept activations.
    Here, Cranfield documents are treated as explicit concepts.
    """
    term_document = doc_tfidf.T.tocsr()
    if stats_out is not None:
        row_nnz = np.diff(term_document.indptr)
        represented = int(np.count_nonzero(row_nnz))
        vocab_count = len(vocab)
        stats_out.update(
            {
                "vocab_terms": vocab_count,
                "represented_terms": represented,
                "representation_coverage": float(represented / vocab_count) if vocab_count else 0.0,
            }
        )
    concept_vectors = _keep_top_k_per_row(
        term_document,
        top_concepts,
        progress=progress,
        progress_desc="ESA concept pruning",
        logger=logger,
        log_every=log_every,
    )
    return _neighbor_map_from_matrix(
        list(vocab),
        concept_vectors,
        top_k,
        min_similarity,
        progress=progress,
        progress_desc="ESA neighbors",
        logger=logger,
        log_every=log_every,
    )



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
    progress: bool = False,
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
    stats_out: Dict | None = None,
) -> NeighborMap:
    try:
        word2vec_cls = importlib.import_module("gensim.models").Word2Vec
    except Exception as exc:
        raise ImportError("gensim is required for word2vec neighbor map") from exc

    sentences = [sentence for doc in docs_tokens for sentence in doc if sentence]
    if not sentences:
        return {term: [] for term in vocab}

    if logger is not None:
        logger(
            f"Word2Vec training started: {len(sentences)} sentences, vector_size={vector_size}, "
            f"window={window}, epochs={epochs}"
        )
    started = time.time()

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
    if logger is not None:
        logger(f"Word2Vec training finished in {time.time() - started:.2f}s")

    available_terms: List[str] = []
    vectors: List[np.ndarray] = []
    for term in vocab:
        if term in model.wv:
            available_terms.append(term)
            vectors.append(model.wv[term])

    if stats_out is not None:
        vocab_count = len(vocab)
        in_model = len(available_terms)
        stats_out.update(
            {
                "vocab_terms": vocab_count,
                "present_in_model": in_model,
                "model_coverage": float(in_model / vocab_count) if vocab_count else 0.0,
            }
        )

    if not available_terms:
        return {term: [] for term in vocab}

    matrix = np.vstack(vectors)
    partial = _neighbor_map_from_matrix(
        available_terms,
        matrix,
        top_k,
        min_similarity,
        progress=progress,
        progress_desc="Word2Vec neighbors",
        logger=logger,
        log_every=log_every,
    )

    full_map = {term: [] for term in vocab}
    full_map.update(partial)
    return full_map



def _neighbor_map_from_matrix(
    vocab: List[str],
    matrix,
    top_k: int,
    min_similarity: float,
    progress: bool = False,
    progress_desc: str = "Matrix neighbors",
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
) -> NeighborMap:
    if not vocab:
        return {}

    n_neighbors = min(len(vocab), top_k + 1)
    if n_neighbors <= 1:
        return {term: [] for term in vocab}

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    if logger is not None:
        logger(f"{progress_desc}: fitting nearest-neighbor index")
    started = time.time()
    nn.fit(matrix)
    if logger is not None:
        logger(f"{progress_desc}: index fitted in {time.time() - started:.2f}s")

    started = time.time()
    if logger is not None:
        logger(f"{progress_desc}: computing nearest neighbors")
    distances, indices = nn.kneighbors(matrix, return_distance=True)
    if logger is not None:
        logger(f"{progress_desc}: nearest-neighbor search done in {time.time() - started:.2f}s")

    neighbor_map: NeighborMap = {}
    total_terms = len(vocab)
    started = time.time()
    index_iter = tqdm(range(len(vocab)), desc=progress_desc, unit="term", disable=not progress)
    for i in index_iter:
        term = vocab[i]
        row: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            sim = 1.0 - float(dist)
            if sim >= min_similarity:
                row.append((vocab[idx], sim))
        neighbor_map[term] = row[:top_k]

        done = i + 1
        if logger is not None and (done == 1 or done % max(1, log_every) == 0 or done == total_terms):
            elapsed = time.time() - started
            logger(f"{progress_desc}: {done}/{total_terms} terms processed (elapsed {elapsed:.1f}s)")

    if logger is not None:
        logger(f"{progress_desc}: completed in {time.time() - started:.2f}s")

    return neighbor_map



def _keep_top_k_per_row(
    matrix: sparse.csr_matrix,
    k: int,
    progress: bool = False,
    progress_desc: str = "ESA row pruning",
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
) -> sparse.csr_matrix:
    if k <= 0:
        return sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)

    matrix = matrix.tocsr()
    new_data = []
    new_indices = []
    new_indptr = [0]

    n_rows = int(matrix.shape[0]) if matrix.shape is not None else 0
    started = time.time()
    if logger is not None:
        logger(f"{progress_desc}: pruning started for {n_rows} rows with top_k={k}")

    row_iter = tqdm(range(n_rows), desc=progress_desc, unit="term", disable=not progress)
    for row in row_iter:
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

        done = row + 1
        if logger is not None and (done == 1 or done % max(1, log_every) == 0 or done == n_rows):
            elapsed = time.time() - started
            logger(f"{progress_desc}: {done}/{n_rows} rows processed (elapsed {elapsed:.1f}s)")

    if logger is not None:
        logger(f"{progress_desc}: completed in {time.time() - started:.2f}s")

    return sparse.csr_matrix(
        (np.array(new_data), np.array(new_indices), np.array(new_indptr)),
        shape=matrix.shape,
        dtype=matrix.dtype,
    )
