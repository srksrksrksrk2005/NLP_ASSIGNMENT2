from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import time
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import nltk
from nltk.corpus import wordnet as wn
from tqdm.auto import tqdm


CACHE_VERSION = "wordnet_neighbors_v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "wordnet"


def ensure_wordnet_resources() -> None:
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


@lru_cache(maxsize=50000)
def _synsets(term: str) -> List[Any]:
    return list(wn.synsets(term))


@lru_cache(maxsize=500000)
def _best_similarity(term_a: str, term_b: str) -> float:
    synsets_a = _synsets(term_a)
    synsets_b = _synsets(term_b)
    if not synsets_a or not synsets_b:
        return 0.0

    best = 0.0
    for sa in synsets_a:
        if sa is None:
            continue
        for sb in synsets_b:
            if sb is None:
                continue
            score = sa.wup_similarity(sb)
            if score is None:
                score = sa.path_similarity(sb)
            if score is None:
                continue
            if score > best:
                best = float(score)
    return best



def _normalize_lemma(lemma: str) -> str:
    return lemma.lower().replace("_", " ").strip()



def _lemma_candidates(term: str) -> Set[str]:
    candidates: Set[str] = set()
    for syn in _synsets(term):
        if syn is None:
            continue
        for lemma in syn.lemma_names():
            normalized = _normalize_lemma(lemma)
            if " " not in normalized:
                candidates.add(normalized)

        # Add nearby lexical alternatives beyond direct synonyms.
        for related in syn.hypernyms() + syn.hyponyms() + syn.similar_tos():
            for lemma in related.lemma_names():
                normalized = _normalize_lemma(lemma)
                if " " not in normalized:
                    candidates.add(normalized)
    return candidates



def build_wordnet_neighbor_map(
    vocab: Sequence[str],
    top_k: int = 10,
    min_similarity: float = 0.05,
    progress: bool = False,
    logger: Callable[[str], None] | None = None,
    log_every: int = 1000,
    cache_dir: str | Path | None = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a sparse word-word similarity map over in-vocabulary terms using WordNet.
    """
    ensure_wordnet_resources()

    cache_path = _cache_path(vocab, top_k, min_similarity, cache_dir)
    if cache_path.exists():
        try:
            if logger is not None:
                logger(f"WordNet matrix cache hit: {cache_path}")
            with gzip.open(cache_path, "rb") as handle:
                cached_neighbors = pickle.load(handle)
            if logger is not None:
                logger("WordNet matrix loaded from disk cache")
            return cached_neighbors
        except Exception as exc:
            if logger is not None:
                logger(f"WordNet matrix cache load failed, rebuilding: {exc}")

    vocab_set = set(vocab)
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    total_terms = len(vocab)
    started = time.time()

    if logger is not None:
        logger(
            f"WordNet matrix build started: {total_terms} terms, top_k={top_k}, min_similarity={min_similarity}"
        )

    vocab_iter = tqdm(vocab, desc="WordNet vocab graph", unit="term", disable=not progress)
    for idx, term in enumerate(vocab_iter, start=1):
        candidates = _lemma_candidates(term)
        candidates = {cand for cand in candidates if cand in vocab_set and cand != term}

        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            sim = _best_similarity(term, cand)
            if sim >= min_similarity:
                scored.append((cand, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        neighbors[term] = scored[:top_k]

        if logger is not None and (idx == 1 or idx % max(1, log_every) == 0 or idx == total_terms):
            elapsed = time.time() - started
            logger(f"WordNet matrix progress: {idx}/{total_terms} terms (elapsed {elapsed:.1f}s)")

    if logger is not None:
        logger(f"WordNet matrix build finished in {time.time() - started:.2f}s")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb") as handle:
            pickle.dump(neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if logger is not None:
            logger(f"WordNet matrix cached at {cache_path}")
    except Exception as exc:
        if logger is not None:
            logger(f"WordNet matrix cache save failed: {exc}")

    return neighbors


class WordNetOOVResolver:
    """Resolve OOV terms into in-vocabulary replacements using WordNet similarity."""

    def __init__(
        self,
        vocab: Sequence[str],
        progress: bool = False,
        logger: Callable[[str], None] | None = None,
        log_every: int = 1000,
    ) -> None:
        ensure_wordnet_resources()
        self.vocab = list(vocab)
        self.vocab_set = set(self.vocab)

        self.lemma_to_vocab: Dict[str, Set[str]] = defaultdict(set)
        total_terms = len(self.vocab)
        started = time.time()
        if logger is not None:
            logger(f"WordNet OOV index build started for {total_terms} terms")

        vocab_iter = tqdm(self.vocab, desc="WordNet OOV index", unit="term", disable=not progress)
        for idx, term in enumerate(vocab_iter, start=1):
            self.lemma_to_vocab[term].add(term)
            for cand in _lemma_candidates(term):
                self.lemma_to_vocab[cand].add(term)

            if logger is not None and (idx == 1 or idx % max(1, log_every) == 0 or idx == total_terms):
                elapsed = time.time() - started
                logger(f"WordNet OOV index progress: {idx}/{total_terms} terms (elapsed {elapsed:.1f}s)")

        self._cache: Dict[Tuple[str, int, float], List[Tuple[str, float]]] = {}
        if logger is not None:
            logger(f"WordNet OOV index build finished in {time.time() - started:.2f}s")

    def resolve(self, term: str, max_candidates: int, min_similarity: float) -> List[Tuple[str, float]]:
        key = (term, max_candidates, min_similarity)
        if key in self._cache:
            return self._cache[key]

        if term in self.vocab_set:
            self._cache[key] = [(term, 1.0)]
            return self._cache[key]

        candidates = set()
        for lemma in _lemma_candidates(term):
            candidates.update(self.lemma_to_vocab.get(lemma, set()))

        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            sim = _best_similarity(term, cand)
            if sim >= min_similarity:
                scored.append((cand, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        self._cache[key] = scored[:max_candidates]
        return self._cache[key]



def _cache_path(
    vocab: Sequence[str],
    top_k: int,
    min_similarity: float,
    cache_dir: str | Path | None,
) -> Path:
    hasher = hashlib.sha256()
    hasher.update(CACHE_VERSION.encode("utf-8"))
    hasher.update(f"|top_k={top_k}|min_similarity={min_similarity:.6f}".encode("utf-8"))
    for term in vocab:
        hasher.update(term.encode("utf-8"))
        hasher.update(b"\0")
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return cache_root / f"{hasher.hexdigest()}.pkl.gz"
