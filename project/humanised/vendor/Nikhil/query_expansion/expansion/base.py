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
        use_quantile_similarity_threshold: bool = True,
        similarity_quantile: float = 0.60,
        normalize_neighbor_mass: bool = True,
        scale_similarity_scores: bool = True,
        similarity_scale_mode: str = "minmax",
        similarity_scale_epsilon: float = 1e-9,
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
        self.use_quantile_similarity_threshold = bool(use_quantile_similarity_threshold)
        self.similarity_quantile = float(np.clip(similarity_quantile, 0.0, 1.0))
        self.normalize_neighbor_mass = bool(normalize_neighbor_mass)
        self.scale_similarity_scores = bool(scale_similarity_scores)
        self.similarity_scale_mode = str(similarity_scale_mode).strip().lower()
        if self.similarity_scale_mode not in {"raw", "minmax"}:
            self.similarity_scale_mode = "minmax"
        self.similarity_scale_epsilon = max(1e-12, float(similarity_scale_epsilon))
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
            replacements, adaptive_threshold = self._filter_by_similarity(replacements)
            if not replacements:
                continue

            if self.normalize_neighbor_mass:
                replacement_mass = self.replacement_weight * count_value
                replacement_weights = self._normalized_similarity_weights(replacements, adaptive_threshold)
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
                    scaled_sim = self._scale_similarity_values(np.array([float(sim)]), adaptive_threshold)[0]
                    base_mass = self.replacement_weight * count_value * float(scaled_sim)
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

        candidates, adaptive_threshold = self._filter_by_similarity(self.neighbors.get(term, []))
        if not candidates:
            return

        if self.normalize_neighbor_mass:
            weights = self._normalized_similarity_weights(candidates, adaptive_threshold)
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
                scaled_sim = self._scale_similarity_values(np.array([float(sim)]), adaptive_threshold)[0]
                vec[idx] += base_mass * float(scaled_sim)

    def _filter_by_similarity(self, candidates: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], float]:
        filtered = [(term, float(sim)) for term, sim in candidates if float(sim) >= self.min_similarity]
        if not filtered:
            return [], self.min_similarity

        thresholds = [self.min_similarity]
        sims = np.array([sim for _, sim in filtered], dtype=np.float64)

        if self.use_mean_similarity_threshold:
            mean_sim = float(np.mean(sims))
            thresholds.append(self.mean_similarity_factor * mean_sim)

        if self.use_quantile_similarity_threshold:
            quantile_value = float(np.quantile(sims, self.similarity_quantile))
            thresholds.append(quantile_value)

        adaptive_threshold = float(max(thresholds))
        tightened = [(term, sim) for term, sim in filtered if sim >= adaptive_threshold]
        if tightened:
            return tightened, adaptive_threshold

        # Fallback so expansion is not accidentally disabled when thresholding is strict.
        best_term, best_sim = max(filtered, key=lambda item: item[1])
        return [(best_term, best_sim)], max(self.min_similarity, float(best_sim) - self.similarity_scale_epsilon)

    def _normalized_similarity_weights(
        self,
        candidates: List[Tuple[str, float]],
        adaptive_threshold: float,
    ) -> np.ndarray:
        sims = np.array([max(0.0, sim) for _, sim in candidates], dtype=np.float64)
        sims = self._scale_similarity_values(sims, adaptive_threshold)
        sims = np.power(sims, self.similarity_power)
        denom = float(np.sum(sims))
        if denom <= 0:
            return np.full(len(candidates), 1.0 / float(len(candidates)), dtype=np.float64)
        return sims / denom

    def _scale_similarity_values(self, sims: np.ndarray, adaptive_threshold: float) -> np.ndarray:
        sims = np.maximum(0.0, sims.astype(np.float64, copy=False))
        if not self.scale_similarity_scores or self.similarity_scale_mode == "raw":
            return sims

        max_sim = float(np.max(sims)) if sims.size else 0.0
        denom = max_sim - adaptive_threshold
        if denom <= self.similarity_scale_epsilon:
            return np.ones_like(sims, dtype=np.float64)

        scaled = (sims - adaptive_threshold) / (denom + self.similarity_scale_epsilon)
        return np.clip(scaled, 0.0, 1.0)
