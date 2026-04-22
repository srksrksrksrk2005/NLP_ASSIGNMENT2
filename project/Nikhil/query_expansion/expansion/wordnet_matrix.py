from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Set, Tuple

import nltk
from nltk.corpus import wordnet as wn


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
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a sparse word-word similarity map over in-vocabulary terms using WordNet.
    """
    ensure_wordnet_resources()

    vocab_set = set(vocab)
    neighbors: Dict[str, List[Tuple[str, float]]] = {}

    for term in vocab:
        candidates = _lemma_candidates(term)
        candidates = {cand for cand in candidates if cand in vocab_set and cand != term}

        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            sim = _best_similarity(term, cand)
            if sim >= min_similarity:
                scored.append((cand, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        neighbors[term] = scored[:top_k]

    return neighbors


class WordNetOOVResolver:
    """Resolve OOV terms into in-vocabulary replacements using WordNet similarity."""

    def __init__(self, vocab: Sequence[str]) -> None:
        ensure_wordnet_resources()
        self.vocab = list(vocab)
        self.vocab_set = set(self.vocab)

        self.lemma_to_vocab: Dict[str, Set[str]] = defaultdict(set)
        for term in self.vocab:
            self.lemma_to_vocab[term].add(term)
            for cand in _lemma_candidates(term):
                self.lemma_to_vocab[cand].add(term)

        self._cache: Dict[Tuple[str, int, float], List[Tuple[str, float]]] = {}

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
