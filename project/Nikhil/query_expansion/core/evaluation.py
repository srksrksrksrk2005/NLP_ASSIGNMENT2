import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


class CranfieldEvaluation:
    """Evaluation metrics aligned with the assignment setup."""

    def __init__(self, qrels: Sequence[Dict]) -> None:
        self.relevant_by_query: Dict[str, set] = defaultdict(set)
        self.position_by_query: Dict[str, Dict[str, int]] = defaultdict(dict)

        for item in qrels:
            qid = str(item["query_num"])
            doc_id = str(item["id"])
            position = int(item["position"])
            self.relevant_by_query[qid].add(doc_id)
            self.position_by_query[qid][doc_id] = position

    def evaluate(
        self,
        rankings_by_query: Dict[str, List[str]],
        query_ids: Sequence[str],
        k_values: Iterable[int] = range(1, 11),
    ) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}

        for k in k_values:
            p_scores: List[float] = []
            r_scores: List[float] = []
            f_scores: List[float] = []
            ap_scores: List[float] = []
            ndcg_scores: List[float] = []
            rr_scores: List[float] = []

            for qid in query_ids:
                ranked = rankings_by_query[str(qid)]
                relevant = self.relevant_by_query[str(qid)]
                position_map = self.position_by_query[str(qid)]

                p_scores.append(self._precision_at_k(ranked, relevant, k))
                r_scores.append(self._recall_at_k(ranked, relevant, k))
                f_scores.append(self._fscore_at_k(ranked, relevant, k))
                ap_scores.append(self._average_precision_at_k(ranked, relevant, k))
                ndcg_scores.append(self._ndcg_at_k(ranked, relevant, position_map, k))
                rr_scores.append(self._reciprocal_rank_at_k(ranked, relevant, k))

            metrics[str(k)] = {
                "precision": _safe_mean(p_scores),
                "recall": _safe_mean(r_scores),
                "fscore": _safe_mean(f_scores),
                "map": _safe_mean(ap_scores),
                "ndcg": _safe_mean(ndcg_scores),
                "mrr": _safe_mean(rr_scores),
            }

        return metrics

    @staticmethod
    def _precision_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
        top_k = [str(doc_id) for doc_id in ranked[:k]]
        if not top_k:
            return 0.0
        hits = sum(1 for doc in top_k if doc in relevant)
        return hits / float(len(top_k))

    @staticmethod
    def _recall_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
        if not relevant:
            return 0.0
        top_k = [str(doc_id) for doc_id in ranked[:k]]
        hits = sum(1 for doc in top_k if doc in relevant)
        return hits / float(len(relevant))

    @staticmethod
    def _fscore_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
        p = CranfieldEvaluation._precision_at_k(ranked, relevant, k)
        r = CranfieldEvaluation._recall_at_k(ranked, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def _average_precision_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
        if not relevant:
            return 0.0
        top_k = [str(doc_id) for doc_id in ranked[:k]]
        hit_count = 0
        precision_sum = 0.0
        for i, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant:
                hit_count += 1
                precision_sum += hit_count / float(i)
        return precision_sum / float(len(relevant))

    @staticmethod
    def _ndcg_at_k(
        ranked: Sequence[str],
        relevant: set,
        position_map: Dict[str, int],
        k: int,
    ) -> float:
        def relevance(doc_id: str) -> float:
            # Cranfield positions are lower-is-better; convert to gain.
            pos = position_map.get(doc_id, 5)
            return max(0.0, 5.0 - float(pos))

        top_k = [str(doc_id) for doc_id in ranked[:k]]
        dcg = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant:
                dcg += relevance(doc_id) / math.log2(rank + 1)

        ideal_docs = sorted(relevant, key=lambda d: position_map.get(d, 5))[:k]
        idcg = 0.0
        for rank, doc_id in enumerate(ideal_docs, start=1):
            idcg += relevance(doc_id) / math.log2(rank + 1)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _reciprocal_rank_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
        top_k = [str(doc_id) for doc_id in ranked[:k]]
        for i, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant:
                return 1.0 / float(i)
        return 0.0



def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
