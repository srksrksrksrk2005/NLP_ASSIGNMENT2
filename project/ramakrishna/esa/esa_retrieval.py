import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ESARetrieval:
    def __init__(
        self,
        top_concepts=25,
        min_similarity=0.0,
        random_state=42,
        sublinear_tf=True,
        max_df=0.9,
        min_df=1,
        norm="l2",
        ngram_range=(1, 2),
    ):
        self.top_concepts = top_concepts
        self.min_similarity = min_similarity
        self.random_state = random_state
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range

        self.vectorizer = None
        self.doc_tfidf_matrix = None
        self.doc_concept_matrix = None
        self.doc_ids = None
        self.vocab_size = None
        self.actual_concepts = None
        self.avg_active_concepts = None
        self.concept_density = None
        self.concept_source = None
        self.concept_ids = None
        self.concept_tfidf_matrix = None

    def _load_prebuilt_index(self, prebuilt_index_path):
        bundle = joblib.load(prebuilt_index_path)
        required_keys = [
            "vectorizer",
            "doc_ids",
            "concept_ids",
            "doc_tfidf_matrix",
            "concept_tfidf_matrix",
            "doc_concept_matrix",
        ]
        missing = [key for key in required_keys if key not in bundle]
        if missing:
            raise ValueError(
                "Prebuilt ESA bundle is missing required keys: " + ", ".join(missing)
            )

        self.vectorizer = bundle["vectorizer"]
        self.doc_ids = list(bundle["doc_ids"])
        self.concept_ids = list(bundle["concept_ids"])
        self.doc_tfidf_matrix = bundle["doc_tfidf_matrix"]
        self.concept_tfidf_matrix = bundle["concept_tfidf_matrix"]
        self.doc_concept_matrix = bundle["doc_concept_matrix"]
        self.vocab_size = int(bundle.get("vocab_size", self.doc_tfidf_matrix.shape[1]))
        self.actual_concepts = int(bundle.get("actual_concepts", min(self.top_concepts, len(self.concept_ids))))
        self.avg_active_concepts = float(bundle.get("avg_active_concepts", 0.0))
        self.concept_density = float(bundle.get("concept_density", 0.0))
        self.concept_source = bundle.get("concept_source", "prebuilt")
        return self

    @staticmethod
    def _flatten_document(document):
        if isinstance(document, str):
            return document
        return " ".join(token for sentence in document for token in sentence)

    @staticmethod
    def _normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    def _prune_and_normalize(self, similarity_matrix):
        similarity_matrix = np.asarray(similarity_matrix, dtype=np.float32)
        if similarity_matrix.ndim != 2:
            raise ValueError("Expected a 2D similarity matrix.")

        n_rows, n_cols = similarity_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return np.zeros((n_rows, n_cols), dtype=np.float32)

        pruned = np.zeros_like(similarity_matrix, dtype=np.float32)
        active_counts = []

        for row_index in range(n_rows):
            row = np.clip(similarity_matrix[row_index], 0.0, None)
            candidates = np.flatnonzero(row >= self.min_similarity)
            if candidates.size == 0:
                continue

            if self.top_concepts > 0 and candidates.size > self.top_concepts:
                ranked_candidates = candidates[
                    np.argpartition(row[candidates], -self.top_concepts)[-self.top_concepts:]
                ]
                ranked_candidates = ranked_candidates[np.argsort(row[ranked_candidates])[::-1]]
            else:
                ranked_candidates = candidates[np.argsort(row[candidates])[::-1]]

            values = row[ranked_candidates]
            valid = values > 0
            ranked_candidates = ranked_candidates[valid]
            values = values[valid]
            pruned[row_index, ranked_candidates] = values
            active_counts.append(len(ranked_candidates))

        self.avg_active_concepts = float(np.mean(active_counts)) if active_counts else 0.0
        self.concept_density = float(np.count_nonzero(pruned)) / float(pruned.size)
        return self._normalize_rows(pruned)

    def build_index(
        self,
        docs,
        doc_ids,
        concept_docs=None,
        concept_ids=None,
        concept_source=None,
        prebuilt_index_path=None,
    ):
        if prebuilt_index_path:
            return self._load_prebuilt_index(prebuilt_index_path)

        doc_texts = [self._flatten_document(doc) for doc in docs]
        concept_docs = docs if concept_docs is None else concept_docs
        concept_ids = doc_ids if concept_ids is None else concept_ids
        concept_texts = [self._flatten_document(concept) for concept in concept_docs]

        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            dtype=np.float32,
            sublinear_tf=self.sublinear_tf,
            max_df=self.max_df,
            min_df=self.min_df,
            norm=self.norm,
            ngram_range=self.ngram_range,
        )

        fit_texts = doc_texts if concept_docs is docs else doc_texts + concept_texts
        self.vectorizer.fit(fit_texts)

        self.doc_tfidf_matrix = self.vectorizer.transform(doc_texts)
        self.concept_tfidf_matrix = self.vectorizer.transform(concept_texts)
        self.vocab_size = int(self.doc_tfidf_matrix.shape[1])

        doc_similarity = cosine_similarity(
            self.doc_tfidf_matrix,
            self.concept_tfidf_matrix,
            dense_output=True,
        ).astype(np.float32)
        if doc_similarity.shape[0] == doc_similarity.shape[1] and concept_docs is docs:
            np.fill_diagonal(doc_similarity, 1.0)

        self.doc_concept_matrix = self._prune_and_normalize(doc_similarity)
        self.doc_ids = list(doc_ids)
        self.concept_ids = list(concept_ids)
        self.actual_concepts = min(self.top_concepts, len(self.concept_ids))
        self.concept_source = concept_source

    def rank(self, queries, return_scores=False):
        if self.vectorizer is None or self.doc_tfidf_matrix is None:
            raise ValueError("Call build_index before rank.")

        query_texts = [self._flatten_document(query) for query in queries]
        query_tfidf_matrix = self.vectorizer.transform(query_texts)

        query_similarity = cosine_similarity(
            query_tfidf_matrix,
            self.concept_tfidf_matrix,
            dense_output=True,
        ).astype(np.float32)
        query_concept_matrix = self._prune_and_normalize(query_similarity)

        scores = np.dot(query_concept_matrix, self.doc_concept_matrix.T)
        ranked_doc_ids = []

        for row in scores:
            ranked_indices = np.argsort(row)[::-1]
            ranked_doc_ids.append([self.doc_ids[index] for index in ranked_indices])

        if return_scores:
            return ranked_doc_ids, {
                "query_concept_matrix": query_concept_matrix,
                "scores": scores,
            }

        return ranked_doc_ids
