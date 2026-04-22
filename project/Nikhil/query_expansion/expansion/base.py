from collections import Counter
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

NeighborMap = Dict[str, List[Tuple[str, float]]]
OOVResolver = Callable[[str, int, float], List[Tuple[str, float]]]


class MatrixQueryExpander:
    """
    Shared query replacement + expansion engine based on a word-word similarity map.

    Expansion for in-vocabulary terms:
        q[t] += self_weight * tf(t)
        q[n] += expansion_weight * tf(t) * sim(t, n)

    OOV replacement:
        q[r] += replacement_weight * tf(oov) * sim(oov, r)
        (optional) spread from each replacement term using neighbor map.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        neighbors: NeighborMap,
        oov_resolver: Optional[OOVResolver] = None,
        self_weight: float = 1.0,
        expansion_weight: float = 0.35,
        replacement_weight: float = 1.0,
        replacement_expansion_weight: Optional[float] = None,
        max_oov_candidates: int = 5,
        min_similarity: float = 0.05,
        expand_replacements: bool = True,
    ) -> None:
        self.vocab = list(vocab)
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocab)}
        self.neighbors = neighbors
        self.oov_resolver = oov_resolver

        self.self_weight = float(self_weight)
        self.expansion_weight = float(expansion_weight)
        self.replacement_weight = float(replacement_weight)
        self.replacement_expansion_weight = (
            float(replacement_expansion_weight)
            if replacement_expansion_weight is not None
            else float(expansion_weight)
        )
        self.max_oov_candidates = int(max_oov_candidates)
        self.min_similarity = float(min_similarity)
        self.expand_replacements = bool(expand_replacements)

    def build_query_vector_from_tokens(self, query_tokens: Iterable[str]) -> np.ndarray:
        return self.build_query_vector_from_counts(Counter(query_tokens))

    def build_query_vector_from_counts(self, term_counts: Mapping[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float64)

        for term, count in term_counts.items():
            if count <= 0:
                continue
            count_value = float(count)

            if term in self.term_to_idx:
                idx = self.term_to_idx[term]
                vec[idx] += self.self_weight * count_value
                self._spread_from_term(vec, term, count_value * self.expansion_weight)
                continue

            if self.oov_resolver is None:
                continue

            replacements = self.oov_resolver(term, self.max_oov_candidates, self.min_similarity)
            for repl_term, sim in replacements:
                if repl_term not in self.term_to_idx:
                    continue
                if sim < self.min_similarity:
                    continue

                repl_idx = self.term_to_idx[repl_term]
                base_mass = self.replacement_weight * count_value * float(sim)
                vec[repl_idx] += base_mass

                if self.expand_replacements:
                    self._spread_from_term(
                        vec,
                        repl_term,
                        base_mass * self.replacement_expansion_weight,
                    )

        return vec

    def _spread_from_term(self, vec: np.ndarray, term: str, base_mass: float) -> None:
        if base_mass <= 0:
            return
        for neighbor, sim in self.neighbors.get(term, []):
            idx = self.term_to_idx.get(neighbor)
            if idx is None:
                continue
            if sim < self.min_similarity:
                continue
            vec[idx] += base_mass * float(sim)
