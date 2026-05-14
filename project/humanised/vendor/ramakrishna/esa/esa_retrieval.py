import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ESARetrieval:
    """
    Explicit Semantic Analysis retrieval.
    
    The idea: represent each document as a vector of similarities to a set
    of "concept" documents. Then compare query and document in this concept
    space. We use TF-IDF cosine similarity to measure how much each doc/query
    relates to each concept.
    
    We also use bigrams (ngram_range=(1,2)) by default to capture multi-word
    technical phrases like "boundary layer" or "heat transfer".
    """

    def __init__(self, top_concepts=25, min_similarity=0.0,
                 random_state=42, sublinear_tf=True,
                 max_df=0.9, min_df=1, norm="l2",
                 ngram_range=(1, 2)):
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

    def _flatten(self, doc):
        if isinstance(doc, str):
            return doc
        return " ".join(tok for sent in doc for tok in sent)

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + 1e-12)

    def _load_prebuilt_index(self, path):
        """Load a pre-saved ESA index from disk."""
        bundle = joblib.load(path)
        needed = ["vectorizer", "doc_ids", "concept_ids",
                  "doc_tfidf_matrix", "concept_tfidf_matrix", "doc_concept_matrix"]
        missing = [k for k in needed if k not in bundle]
        if missing:
            raise ValueError("Prebuilt ESA bundle missing: " + ", ".join(missing))

        self.vectorizer = bundle["vectorizer"]
        self.doc_ids = list(bundle["doc_ids"])
        self.concept_ids = list(bundle["concept_ids"])
        self.doc_tfidf_matrix = bundle["doc_tfidf_matrix"]
        self.concept_tfidf_matrix = bundle["concept_tfidf_matrix"]
        self.doc_concept_matrix = bundle["doc_concept_matrix"]
        self.vocab_size = int(bundle.get("vocab_size", self.doc_tfidf_matrix.shape[1]))
        self.actual_concepts = int(bundle.get("actual_concepts",
                                              min(self.top_concepts, len(self.concept_ids))))
        self.avg_active_concepts = float(bundle.get("avg_active_concepts", 0.0))
        self.concept_density = float(bundle.get("concept_density", 0.0))
        self.concept_source = bundle.get("concept_source", "prebuilt")

    def _prune_concepts(self, sim_matrix):
        """
        For each row (document/query), keep only the top_concepts highest
        similarity values and zero out the rest. Then normalize each row.
        This makes the concept vectors sparse and focused.
        """
        sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
        n_rows, n_cols = sim_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return np.zeros((n_rows, n_cols), dtype=np.float32)

        pruned = np.zeros_like(sim_matrix)
        active_counts = []

        for i in range(n_rows):
            row = np.clip(sim_matrix[i], 0.0, None)
            candidates = np.flatnonzero(row >= self.min_similarity)
            if candidates.size == 0:
                continue

            # pick top concepts
            if self.top_concepts > 0 and candidates.size > self.top_concepts:
                # argpartition is faster than full sort for top-k
                top_idx = candidates[
                    np.argpartition(row[candidates], -self.top_concepts)[-self.top_concepts:]
                ]
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            else:
                top_idx = candidates[np.argsort(row[candidates])[::-1]]

            vals = row[top_idx]
            valid = vals > 0
            top_idx = top_idx[valid]
            vals = vals[valid]
            pruned[i, top_idx] = vals
            active_counts.append(len(top_idx))

        self.avg_active_concepts = float(np.mean(active_counts)) if active_counts else 0.0
        self.concept_density = float(np.count_nonzero(pruned)) / float(pruned.size)
        return self._normalize(pruned)

    def build_index(self, docs, doc_ids, concept_docs=None, concept_ids=None,
                    concept_source=None, prebuilt_index_path=None):
        """
        Build the ESA index. If concept_docs is None, we use the documents
        themselves as concepts (self-referencing ESA).
        """
        if prebuilt_index_path:
            return self._load_prebuilt_index(prebuilt_index_path)

        doc_texts = [self._flatten(d) for d in docs]

        # if no separate concept corpus, use the docs themselves
        if concept_docs is None:
            concept_docs = docs
            concept_ids = doc_ids
        concept_texts = [self._flatten(c) for c in concept_docs]

        # build shared tfidf vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False,
            dtype=np.float32,
            sublinear_tf=self.sublinear_tf, max_df=self.max_df,
            min_df=self.min_df, norm=self.norm,
            ngram_range=self.ngram_range,
        )

        # fit on all texts so vocab covers both docs and concepts
        if concept_docs is docs:
            self.vectorizer.fit(doc_texts)
        else:
            self.vectorizer.fit(doc_texts + concept_texts)

        self.doc_tfidf_matrix = self.vectorizer.transform(doc_texts)
        self.concept_tfidf_matrix = self.vectorizer.transform(concept_texts)
        self.vocab_size = self.doc_tfidf_matrix.shape[1]

        # compute doc-concept similarity matrix
        sim = cosine_similarity(
            self.doc_tfidf_matrix, self.concept_tfidf_matrix,
            dense_output=True
        ).astype(np.float32)

        # if using self-referencing, set diagonal to 1
        if sim.shape[0] == sim.shape[1] and concept_docs is docs:
            np.fill_diagonal(sim, 1.0)

        # prune to top concepts and normalize
        self.doc_concept_matrix = self._prune_concepts(sim)
        self.doc_ids = list(doc_ids)
        self.concept_ids = list(concept_ids)
        self.actual_concepts = min(self.top_concepts, len(self.concept_ids))
        self.concept_source = concept_source

    def rank(self, queries):
        """Rank docs by cosine similarity in concept space."""
        q_texts = [self._flatten(q) for q in queries]
        q_tfidf = self.vectorizer.transform(q_texts)

        # project queries into concept space
        q_sim = cosine_similarity(
            q_tfidf, self.concept_tfidf_matrix,
            dense_output=True
        ).astype(np.float32)
        q_concepts = self._prune_concepts(q_sim)

        # rank by dot product in concept space
        scores = q_concepts @ self.doc_concept_matrix.T
        results = []
        for row in scores:
            order = np.argsort(row)[::-1]
            results.append([self.doc_ids[i] for i in order])
        return results
