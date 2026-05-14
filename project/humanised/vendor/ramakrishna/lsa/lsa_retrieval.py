import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSARetrieval:
    """
    Latent Semantic Analysis based retrieval.
    We build a TF-IDF matrix, reduce it with SVD, and then
    rank docs by cosine similarity in the latent space.
    """

    def __init__(self, n_components=250, random_state=42,
                 sublinear_tf=True, max_df=0.95, min_df=2,
                 norm="l2", ngram_range=(1, 1)):
        self.n_components = n_components
        self.random_state = random_state
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.min_df = min_df
        self.norm = norm
        self.ngram_range = ngram_range

        # these get set after build_index
        self.vectorizer = None
        self.svd = None
        self.doc_ids = None
        self.doc_vectors = None
        self.actual_components = None
        self.explained_variance = None
        self.vocab_size = None

    def _flatten(self, doc):
        # join all tokens from all sentences into one string
        return " ".join(tok for sent in doc for tok in sent)

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + 1e-12)

    def build_index(self, docs, doc_ids):
        """Build the TF-IDF matrix and apply SVD."""
        texts = [self._flatten(d) for d in docs]

        # build tfidf - we pass pre-tokenized text so just split on spaces
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

        # make sure we don't ask for more components than possible
        max_k = min(tfidf_mat.shape) - 1
        self.actual_components = min(self.n_components, max_k)

        self.svd = TruncatedSVD(
            n_components=self.actual_components,
            random_state=self.random_state,
        )
        doc_latent = self.svd.fit_transform(tfidf_mat)
        self.doc_vectors = self._normalize(doc_latent)
        self.doc_ids = list(doc_ids)
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

    def rank(self, queries):
        """Rank documents for each query using cosine similarity in LSA space."""
        texts = [self._flatten(q) for q in queries]
        q_tfidf = self.vectorizer.transform(texts)
        q_latent = self.svd.transform(q_tfidf)
        q_vecs = self._normalize(q_latent)

        sims = q_vecs @ self.doc_vectors.T
        results = []
        for row in sims:
            order = np.argsort(row)[::-1]
            results.append([self.doc_ids[i] for i in order])
        return results
