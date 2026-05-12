"""
Blocks 2 & 3: Retrieval and Ranking
Integrates multiple retrieval modes (TF-IDF, N-gram, local BOW) with ranking modes (LSA, ESA, TF-IDF)
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RetrievalRankingPipeline:
    """Combined retrieval and ranking pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.retrieval_config = config.get("block2_retrieval_mode", {})
        self.ranking_config = config.get("block3_ranking_mode", {})
        self.retrieval_type = self.retrieval_config.get("retrieval_type", "tfidf")
        self.ranking_type = self.ranking_config.get("ranking_type", "tfidf")
        
        self.doc_vectors = None
        self.doc_ids = None
        self.vectorizer = None
        self.ranking_model = None
    
    def build_tfidf_index(self, docs, doc_ids):
        """Build TF-IDF retrieval index"""
        params = self.retrieval_config.get("retrieval_params", {}).get("tfidf", {})
        
        self.vectorizer = TfidfVectorizer(
            max_features=params.get("max_features", 5000),
            max_df=params.get("max_df", 0.95),
            min_df=params.get("min_df", 2),
            norm=params.get("norm", "l2"),
            sublinear_tf=params.get("sublinear_tf", True)
        )
        
        # Flatten documents if they're lists of tokens
        doc_texts = []
        for doc in docs:
            if isinstance(doc, list):
                doc_texts.append(" ".join(doc))
            else:
                doc_texts.append(doc)
        
        try:
            self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        except ValueError as e:
            # Handle small corpora where pruning removes all terms (e.g., in tests)
            msg = str(e)
            if "After pruning, no terms remain" in msg or "no terms remain" in msg:
                # Retry with relaxed parameters suitable for tiny corpora
                fallback_max_df = 1.0
                fallback_min_df = 1
                self.vectorizer = TfidfVectorizer(
                    max_features=params.get("max_features", 5000),
                    max_df=fallback_max_df,
                    min_df=fallback_min_df,
                    norm=params.get("norm", "l2"),
                    sublinear_tf=params.get("sublinear_tf", True)
                )
                self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
            else:
                raise

        self.doc_ids = list(doc_ids)
    
    def build_ngram_index(self, docs, doc_ids):
        """Build N-gram retrieval index"""
        params = self.retrieval_config.get("retrieval_params", {}).get("ngram", {})
        ngram_range = tuple(params.get("ngram_range", [1, 2]))
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            analyzer=params.get("analyzer", "char"),
            max_features=5000
        )
        
        doc_texts = []
        for doc in docs:
            if isinstance(doc, list):
                doc_texts.append(" ".join(doc))
            else:
                doc_texts.append(doc)
        
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        self.doc_ids = list(doc_ids)
    
    def build_local_bow_index(self, docs, doc_ids):
        """Build local bag-of-words (windowed) index"""
        params = self.retrieval_config.get("retrieval_params", {}).get("local_bow", {})
        window_size = params.get("window_size", 5)
        
        # Create windowed document representations
        windowed_docs = []
        for doc in docs:
            if isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = doc
            
            windows = []
            for i in range(len(tokens) - window_size + 1):
                window = " ".join(tokens[i:i + window_size])
                windows.append(window)
            
            if windows:
                windowed_docs.append(" ".join(windows))
            else:
                windowed_docs.append(" ".join(tokens))
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(windowed_docs)
        self.doc_ids = list(doc_ids)
    
    def build_retrieval_index(self, docs, doc_ids):
        """Build retrieval index based on configured retrieval type"""
        if self.retrieval_type == "tfidf":
            self.build_tfidf_index(docs, doc_ids)
        elif self.retrieval_type == "ngram":
            self.build_ngram_index(docs, doc_ids)
        elif self.retrieval_type == "local_bow":
            self.build_local_bow_index(docs, doc_ids)
        else:
            raise ValueError(f"Unknown retrieval type: {self.retrieval_type}")
    
    def rank_tfidf(self, query_vectors):
        """Rank using TF-IDF cosine similarity"""
        if isinstance(query_vectors, np.ndarray) and query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        query_vectors_sparse = self._convert_to_sparse_tfidf(query_vectors)
        similarities = cosine_similarity(query_vectors_sparse, self.doc_vectors)
        
        ranked_results = []
        for scores in similarities:
            ranked_indices = np.argsort(scores)[::-1]
            ranked_docs = [(self.doc_ids[idx], scores[idx]) for idx in ranked_indices]
            ranked_results.append(ranked_docs)
        
        return ranked_results
    
    def rank_lsa(self, query_vectors):
        """Rank using LSA"""
        if self.ranking_model is None:
            params = self.ranking_config.get("ranking_params", {}).get("lsa", {})
            n_components = params.get("n_components", 250)
            
            # Convert sparse to dense for SVD
            if hasattr(self.doc_vectors, 'toarray'):
                doc_dense = self.doc_vectors.toarray()
            else:
                doc_dense = self.doc_vectors
            
            n_components = min(n_components, min(doc_dense.shape) - 1)
            self.ranking_model = TruncatedSVD(
                n_components=n_components,
                random_state=params.get("random_state", 42)
            )
            self.doc_latent = self.ranking_model.fit_transform(doc_dense)
            self.doc_latent = self._normalize_rows(self.doc_latent)
        
        # Project queries
        if isinstance(query_vectors, np.ndarray) and query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        query_latent = self.ranking_model.transform(query_vectors)
        query_latent = self._normalize_rows(query_latent)
        
        # Rank by latent similarity
        similarities = np.dot(query_latent, self.doc_latent.T)
        
        ranked_results = []
        for scores in similarities:
            ranked_indices = np.argsort(scores)[::-1]
            ranked_docs = [(self.doc_ids[idx], scores[idx]) for idx in ranked_indices]
            ranked_results.append(ranked_docs)
        
        return ranked_results
    
    def rank_esa(self, query_vectors):
        """Rank using ESA (Explicit Semantic Analysis)"""
        params = self.ranking_config.get("ranking_params", {}).get("esa", {})
        top_concepts = params.get("top_concepts", 25)
        
        # Convert doc vectors to dense
        if hasattr(self.doc_vectors, 'toarray'):
            doc_dense = self.doc_vectors.toarray()
        else:
            doc_dense = self.doc_vectors
        
        # Compute document-document similarity (concept matrix)
        doc_similarity = cosine_similarity(doc_dense, doc_dense)
        np.fill_diagonal(doc_similarity, 1.0)
        
        # Prune to top_concepts
        doc_concept_matrix = self._prune_concepts(doc_similarity, top_concepts)
        
        # Project queries
        if isinstance(query_vectors, np.ndarray) and query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # Query to concept similarity
        query_concept_similarity = cosine_similarity(query_vectors, doc_dense)
        query_concept_matrix = self._prune_concepts(query_concept_similarity, top_concepts)
        
        # Rank
        similarities = np.dot(query_concept_matrix, doc_concept_matrix.T)
        
        ranked_results = []
        for scores in similarities:
            ranked_indices = np.argsort(scores)[::-1]
            ranked_docs = [(self.doc_ids[idx], scores[idx]) for idx in ranked_indices]
            ranked_results.append(ranked_docs)
        
        return ranked_results
    
    def rank(self, query_vectors):
        """Rank documents for queries using configured ranking type"""
        if self.ranking_type == "tfidf":
            return self.rank_tfidf(query_vectors)
        elif self.ranking_type == "lsa":
            return self.rank_lsa(query_vectors)
        elif self.ranking_type == "esa":
            return self.rank_esa(query_vectors)
        else:
            raise ValueError(f"Unknown ranking type: {self.ranking_type}")
    
    @staticmethod
    def _normalize_rows(matrix):
        """Normalize matrix rows"""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)
    
    @staticmethod
    def _prune_concepts(matrix, top_k):
        """Prune matrix to top-k values per row"""
        pruned = np.zeros_like(matrix)
        for i, row in enumerate(matrix):
            if top_k > 0 and np.count_nonzero(row) > top_k:
                top_indices = np.argsort(row)[-top_k:]
                pruned[i, top_indices] = row[top_indices]
            else:
                pruned[i] = row
        return RetrievalRankingPipeline._normalize_rows(pruned)
    
    @staticmethod
    def _convert_to_sparse_tfidf(dense_vectors):
        """Convert dense vectors to sparse format (if needed)"""
        from scipy.sparse import csr_matrix
        if hasattr(dense_vectors, 'toarray'):
            return dense_vectors
        return csr_matrix(dense_vectors)


def main():
    """Example usage of RetrievalRankingPipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Blocks 2 & 3: Retrieval and Ranking")
    parser.add_argument("--config", type=str, default="configs/default_config.json",
                        help="Config file path")
    parser.add_argument("--retrieval", type=str, choices=["tfidf", "ngram", "local_bow"],
                        help="Retrieval type")
    parser.add_argument("--ranking", type=str, choices=["tfidf", "lsa", "esa"],
                        help="Ranking type")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config if args provided
    if args.retrieval:
        config["block2_retrieval_mode"]["retrieval_type"] = args.retrieval
    if args.ranking:
        config["block3_ranking_mode"]["ranking_type"] = args.ranking
    
    # Initialize pipeline
    pipeline = RetrievalRankingPipeline(config)
    print(f"Retrieval type: {pipeline.retrieval_type}")
    print(f"Ranking type: {pipeline.ranking_type}")
    print("Pipeline ready for indexing and ranking")


if __name__ == "__main__":
    main()
