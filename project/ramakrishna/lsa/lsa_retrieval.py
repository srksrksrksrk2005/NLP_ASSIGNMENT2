import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSARetrieval:
    def __init__(
        self,
        n_components=250,
        random_state=42,
        sublinear_tf=True,
        max_df=0.95,
        min_df=2,
        norm="l2",
        ngram_range=(1, 1),
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.svd = None
        self.doc_ids = None
        self.doc_vectors = None
        self.actual_components = None
        self.explained_variance = None
        self.vocab_size = None

    @staticmethod
    def _flatten_document(document):
        return " ".join(token for sentence in document for token in sentence)

    @staticmethod
    def _normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    def build_index(self, docs, doc_ids):
        doc_texts = [self._flatten_document(doc) for doc in docs]

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
        doc_term_matrix = self.vectorizer.fit_transform(doc_texts)
        self.vocab_size = int(doc_term_matrix.shape[1])

        max_components = min(doc_term_matrix.shape) - 1
        if max_components < 1:
            raise ValueError("Not enough data to compute an LSA projection.")

        self.actual_components = min(self.n_components, max_components)
        self.svd = TruncatedSVD(
            n_components=self.actual_components,
            random_state=self.random_state,
        )

        doc_latent = self.svd.fit_transform(doc_term_matrix)
        self.doc_vectors = self._normalize_rows(doc_latent)
        self.doc_ids = list(doc_ids)
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

    def rank(self, queries):
        if self.vectorizer is None or self.svd is None or self.doc_vectors is None:
            raise ValueError("Call build_index before rank.")

        query_texts = [self._flatten_document(query) for query in queries]
        query_term_matrix = self.vectorizer.transform(query_texts)
        query_latent = self.svd.transform(query_term_matrix)
        query_vectors = self._normalize_rows(query_latent)

        similarities = np.dot(query_vectors, self.doc_vectors.T)
        ranked_doc_ids = []

        for row in similarities:
            ranked_indices = np.argsort(row)[::-1]
            ranked_doc_ids.append([self.doc_ids[index] for index in ranked_indices])

        return ranked_doc_ids
