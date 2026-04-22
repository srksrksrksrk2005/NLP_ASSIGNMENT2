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
        q[n] += expansion_weight * tf(t) distributed over filtered similar terms

    OOV replacement:
        q[r] += replacement_weight * tf(oov) distributed over filtered replacements
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
        use_mean_similarity_threshold: bool = True,
        mean_similarity_factor: float = 1.0,
        normalize_neighbor_mass: bool = True,
        similarity_power: float = 1.0,
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
        self.use_mean_similarity_threshold = bool(use_mean_similarity_threshold)
        self.mean_similarity_factor = float(mean_similarity_factor)
        self.normalize_neighbor_mass = bool(normalize_neighbor_mass)
        self.similarity_power = float(similarity_power)

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
            replacements = [(term_name, sim) for term_name, sim in replacements if term_name in self.term_to_idx]
            replacements = self._filter_by_similarity(replacements)
            if not replacements:
                continue

            if self.normalize_neighbor_mass:
                replacement_mass = self.replacement_weight * count_value
                replacement_weights = self._normalized_similarity_weights(replacements)
                for (repl_term, _), repl_weight in zip(replacements, replacement_weights):
                    repl_idx = self.term_to_idx[repl_term]
                    base_mass = replacement_mass * float(repl_weight)
                    vec[repl_idx] += base_mass

                    if self.expand_replacements:
                        self._spread_from_term(
                            vec,
                            repl_term,
                            base_mass * self.replacement_expansion_weight,
                        )
            else:
                for repl_term, sim in replacements:
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

        candidates = self._filter_by_similarity(self.neighbors.get(term, []))
        if not candidates:
            return

        if self.normalize_neighbor_mass:
            weights = self._normalized_similarity_weights(candidates)
            for (neighbor, _), weight in zip(candidates, weights):
                idx = self.term_to_idx.get(neighbor)
                if idx is None:
                    continue
                vec[idx] += base_mass * float(weight)
        else:
            for neighbor, sim in candidates:
                idx = self.term_to_idx.get(neighbor)
                if idx is None:
                    continue
                vec[idx] += base_mass * float(sim)

    def _filter_by_similarity(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        filtered = [(term, float(sim)) for term, sim in candidates if float(sim) >= self.min_similarity]
        if not filtered:
            return []

        if not self.use_mean_similarity_threshold:
            return filtered

        sims = np.array([sim for _, sim in filtered], dtype=np.float64)
        mean_sim = float(np.mean(sims))
        adaptive_threshold = max(self.min_similarity, self.mean_similarity_factor * mean_sim)
        tightened = [(term, sim) for term, sim in filtered if sim >= adaptive_threshold]
        if tightened:
            return tightened

        # Fallback so expansion is not accidentally disabled when thresholding is strict.
        return [max(filtered, key=lambda item: item[1])]

    def _normalized_similarity_weights(self, candidates: List[Tuple[str, float]]) -> np.ndarray:
        sims = np.array([max(0.0, sim) for _, sim in candidates], dtype=np.float64)
        sims = np.power(sims, self.similarity_power)
        denom = float(np.sum(sims))
        if denom <= 0:
            return np.full(len(candidates), 1.0 / float(len(candidates)), dtype=np.float64)
        return sims / denom
