"""
Block 1: Query Processing
Handles query expansion (LSA, ESA, WordNet, Word2Vec, TF-IDF) and reduction.
Outputs IDF-weighted query vectors with optional batch processing.
"""

import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import wordnet


class QueryProcessor:
    """Main query processor combining expansion, reduction, and IDF weighting"""
    
    def __init__(self, config, preprocessing_module=None):
        """
        Initialize query processor
        
        Args:
            config: Configuration dictionary
            preprocessing_module: Module with preprocessing functions
        """
        self.config = config
        self.preprocessing = preprocessing_module
        self.block1_config = config.get("block1_query_processing", {})
        self.expansion_mode = self.block1_config.get("expansion_mode", "none")
        self.reduction_enabled = self.block1_config.get("reduction_enabled", False)
        self.tfidf_vectorizer = None
        self.expansion_model = None
        self.query_vectors = []
    
    def preprocess_query(self, query):
        """Apply preprocessing steps"""
        if self.preprocessing is None:
            return query.split()
        return self.preprocessing.preprocess_text(query)
    
    def expand_query_lsa(self, query_tokens, docs_tokens):
        """Expand query using LSA (Latent Semantic Analysis)"""
        params = self.block1_config.get("expansion_params", {}).get("lsa", {})
        n_components = params.get("n_components", 100)
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=params.get("max_features", 5000),
                max_df=params.get("max_df", 0.95),
                min_df=params.get("min_df", 2)
            )
            # Fit on all documents
            doc_texts = [" ".join(doc) for doc in docs_tokens]
            self.tfidf_vectorizer.fit(doc_texts)
        
        query_text = " ".join(query_tokens)
        query_vector = self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        
        # Optionally apply LSA
        if not hasattr(self, '_lsa_model'):
            doc_texts = [" ".join(doc) for doc in docs_tokens]
            doc_matrix = self.tfidf_vectorizer.transform(doc_texts).toarray()
            self._lsa_model = TruncatedSVD(n_components=min(n_components, doc_matrix.shape[1] - 1))
            self._lsa_model.fit(doc_matrix)
        
        query_latent = self._lsa_model.transform([query_vector])[0]
        self.last_query_latent = query_latent
        return query_vector
    
    def expand_query_wordnet(self, query_tokens):
        """Expand query using WordNet synonyms"""
        params = self.block1_config.get("expansion_params", {}).get("wordnet", {})
        synset_limit = params.get("synset_limit", 3)
        
        expanded_tokens = list(query_tokens)
        for token in query_tokens:
            synsets = wordnet.synsets(token)
            for synset in synsets[:synset_limit]:
                for lemma in synset.lemmas():
                    if lemma.name() not in expanded_tokens:
                        expanded_tokens.append(lemma.name())
        
        return expanded_tokens
    
    def expand_query_tfidf(self, query_tokens, docs_tokens):
        """Expand query using TF-IDF similarity"""
        query_text = " ".join(query_tokens)
        
        if self.tfidf_vectorizer is None:
            params = self.block1_config.get("expansion_params", {}).get("tfidf", {})
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=params.get("max_features", 5000),
                max_df=params.get("max_df", 0.95),
                min_df=params.get("min_df", 2)
            )
            doc_texts = [" ".join(doc) for doc in docs_tokens]
            self.tfidf_vectorizer.fit(doc_texts)
        
        return self.tfidf_vectorizer.transform([query_text]).toarray()[0]
    
    def reduce_query(self, query_tokens):
        """Reduce query to most important keywords"""
        if not self.reduction_enabled:
            return query_tokens
        
        params = self.block1_config.get("reduction_params", {})
        top_k = params.get("top_k", 10)
        method = params.get("method", "keyword_extraction")
        
        if method == "keyword_extraction":
            # Simple keyword extraction: keep tokens with highest TF-IDF scores
            from sklearn.feature_extraction.text import TfidfVectorizer as TV
            vectorizer = TV(max_features=top_k)
            query_text = " ".join(query_tokens)
            try:
                vectorizer.fit_transform([query_text])
                keywords = vectorizer.get_feature_names_out()
                return list(keywords)
            except:
                return query_tokens[:top_k]
        
        return query_tokens
    
    def process_query(self, query, docs_tokens=None, return_vector=True):
        """
        Process a single query through expansion, reduction, and vectorization
        
        Args:
            query: Query string
            docs_tokens: Tokenized documents (required for some expansion methods)
            return_vector: If True, return IDF-weighted vector
        
        Returns:
            If return_vector: numpy array (IDF-weighted vector)
            Else: list of query tokens
        """
        # Preprocessing
        query_tokens = self.preprocess_query(query)
        
        # Expansion
        if self.expansion_mode == "wordnet":
            query_tokens = self.expand_query_wordnet(query_tokens)
        elif self.expansion_mode == "lsa" and docs_tokens:
            query_vector = self.expand_query_lsa(query_tokens, docs_tokens)
            return query_vector
        elif self.expansion_mode == "tfidf" and docs_tokens:
            query_vector = self.expand_query_tfidf(query_tokens, docs_tokens)
            return query_vector
        # "none", "esa", "word2vec" modes: token-level only
        
        # Reduction
        query_tokens = self.reduce_query(query_tokens)
        
        # Vectorization
        if return_vector and self.tfidf_vectorizer:
            query_text = " ".join(query_tokens)
            return self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        
        return query_tokens if not return_vector else np.array([1.0] * len(query_tokens))
    
    def process_batch(self, queries, docs_tokens=None, return_vectors=True):
        """
        Process multiple queries
        
        Args:
            queries: List of query strings
            docs_tokens: Tokenized documents
            return_vectors: If True, return vectors; else return tokens
        
        Returns:
            List of processed queries (vectors or tokens)
        """
        processed_queries = []
        for query in queries:
            processed = self.process_query(query, docs_tokens, return_vectors)
            processed_queries.append(processed)
        
        if return_vectors:
            # Pad vectors to same length
            max_len = max(len(v) for v in processed_queries)
            padded = []
            for v in processed_queries:
                if len(v) < max_len:
                    padded_v = np.zeros(max_len)
                    padded_v[:len(v)] = v
                    padded.append(padded_v)
                else:
                    padded.append(v)
            return np.array(padded)
        
        return processed_queries
    
    def fit_tfidf(self, docs_tokens):
        """Fit TF-IDF vectorizer on documents"""
        if self.tfidf_vectorizer is None:
            params = self.block1_config.get("expansion_params", {}).get("tfidf", {})
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=params.get("max_features", 5000),
                max_df=params.get("max_df", 0.95),
                min_df=params.get("min_df", 2)
            )
            doc_texts = [" ".join(doc) for doc in docs_tokens]
            self.tfidf_vectorizer.fit(doc_texts)
    
    def get_query_vector(self, query):
        """Get IDF-weighted vector for a query"""
        query_tokens = self.preprocess_query(query)
        query_text = " ".join(query_tokens)
        
        if self.tfidf_vectorizer:
            return self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        
        # Fallback to simple representation
        return np.array([1.0] * len(query_tokens))


def main():
    """Example usage of QueryProcessor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Block 1: Query Processing")
    parser.add_argument("--config", type=str, default="configs/default_config.json",
                        help="Config file path")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--mode", type=str, choices=["lsa", "esa", "tfidf", "wordnet", "word2vec", "none"],
                        default="none", help="Query expansion mode")
    parser.add_argument("--reduce", action="store_true", help="Enable query reduction")
    parser.add_argument("--batch", action="store_true", help="Process batch of queries")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config based on CLI args
    config["block1_query_processing"]["expansion_mode"] = args.mode
    config["block1_query_processing"]["reduction_enabled"] = args.reduce
    
    # Initialize processor
    processor = QueryProcessor(config)
    
    if args.query:
        result = processor.process_query(args.query, return_vector=True)
        print(f"Query: {args.query}")
        print(f"Processed vector shape: {result.shape}")
        print(f"Vector (first 10 elements): {result[:10]}")
    else:
        print("Use --query to process a single query")


if __name__ == "__main__":
    main()
