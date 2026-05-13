import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class HybridRetrieval:
    """
    Hybrid retrieval that combines TF-IDF cosine similarity with
    LSA cosine similarity using a weighted sum:
        score = alpha * tfidf_score + (1 - alpha) * lsa_score
    alpha is the tfidf_weight parameter (default 0.2, so 80% LSA).
    """

    def __init__(self, n_components=240, tfidf_weight=0.2,
                 random_state=42, sublinear_tf=True,
                 max_df=0.95, min_df=2, norm="l2",
                 ngram_range=(1, 1)):
        self.n_components = n_components
        self.tfidf_weight = tfidf_weight
        self.random_state = random_state
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range

        self.vectorizer = None
        self.svd = None
        self.doc_ids = None
        self.tfidf_doc_mat = None   # normalized tfidf for direct cosine
        self.lsa_doc_vecs = None    # normalized lsa vectors
        self.actual_components = None
        self.explained_variance = None
        self.vocab_size = None

    def _flatten(self, doc):
        return " ".join(tok for sent in doc for tok in sent)

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + 1e-12)

    def build_index(self, docs, doc_ids):
        texts = [self._flatten(d) for d in docs]

        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False,
            dtype=np.float32,
            sublinear_tf=self.sublinear_tf, max_df=self.max_df,
            min_df=self.min_df, norm=self.norm,
            ngram_range=self.ngram_range,
        )
        tfidf_mat = self.vectorizer.fit_transform(texts)
        self.vocab_size = tfidf_mat.shape[1]

        # keep a normalized copy for direct tfidf similarity
        self.tfidf_doc_mat = normalize(tfidf_mat, norm="l2", axis=1)

        # now do the SVD part
        max_k = min(tfidf_mat.shape) - 1
        self.actual_components = min(self.n_components, max_k)

        self.svd = TruncatedSVD(
            n_components=self.actual_components,
            random_state=self.random_state,
        )
        lsa_mat = self.svd.fit_transform(tfidf_mat)
        self.lsa_doc_vecs = self._normalize(lsa_mat)
        self.doc_ids = list(doc_ids)
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

    def rank(self, queries):
        """
        For each query, compute both tfidf and lsa similarities,
        then combine them with the weighted sum.
        """
        texts = [self._flatten(q) for q in queries]
        q_tfidf = self.vectorizer.transform(texts)

        # tfidf branch
        q_tfidf_norm = normalize(q_tfidf, norm="l2", axis=1)
        tfidf_scores = q_tfidf_norm.dot(self.tfidf_doc_mat.T).toarray()

        # lsa branch
        q_lsa = self.svd.transform(q_tfidf)
        q_lsa_norm = self._normalize(q_lsa)
        lsa_scores = q_lsa_norm @ self.lsa_doc_vecs.T

        # weighted combination
        alpha = self.tfidf_weight
        combined = alpha * tfidf_scores + (1.0 - alpha) * lsa_scores

        results = []
        for row in combined:
            order = np.argsort(row)[::-1]
            results.append([self.doc_ids[i] for i in order])
        return results
