import json
import os
import sys
import numpy as np
import math
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet

# Add project root to sys.path
BASE_DIR = Path("c:/Users/srksr/NLP/NLP_project/NLP_ASSIGNMENT2")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "project/Nikhil/query_expansion"))

from core.preprocessing import Preprocessor

def load_data():
    with open(BASE_DIR / "cranfield/cran_docs.json", "r") as f:
        docs_json = json.load(f)
    with open(BASE_DIR / "project/limitation_test_set_expanded.json", "r") as f:
        test_set = json.load(f)
    return docs_json, test_set

def get_map_at_k(ranked_ids, relevant_ids, k=10):
    if not relevant_ids:
        return 0.0
    rel_set = set(str(rid) for rid in relevant_ids)
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if str(doc_id) in rel_set:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / min(len(rel_set), k)

class SimpleSearchSystem:
    def __init__(self, doc_bodies, doc_ids):
        self.doc_ids = doc_ids
        self.preprocessor = Preprocessor()
        
        print("Preprocessing docs...")
        self.processed_docs = [self.preprocessor.preprocess_text(t) for t in doc_bodies]
        self.doc_strings = [" ".join(token for sent in doc for token in sent) for doc in self.processed_docs]
        
        # 1. Baseline TF-IDF
        print("Building Baseline...")
        self.vectorizer = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None)
        self.doc_tfidf = self.vectorizer.fit_transform(self.doc_strings)
        
        # 2. N-Gram (Bigram)
        print("Building N-Gram index...")
        self.ngram_vectorizer = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2))
        self.doc_ngram_tfidf = self.ngram_vectorizer.fit_transform(self.doc_strings)
        
        # 3. LSA (Latent Semantic Analysis)
        print("Building LSA index...")
        self.svd = TruncatedSVD(n_components=128, random_state=42)
        self.doc_lsa = self.svd.fit_transform(self.doc_tfidf)
        
        # 4. ESA (Simplified: Doc Similarity as Concepts)
        print("Building ESA index...")
        self.doc_concepts = cosine_similarity(self.doc_tfidf, self.doc_tfidf)
        # Prune ESA (keep top 25 per doc)
        for i in range(len(self.doc_concepts)):
            row = self.doc_concepts[i]
            top_idx = np.argsort(row)[-25:]
            mask = np.ones(len(row), dtype=bool)
            mask[top_idx] = False
            self.doc_concepts[i, mask] = 0
        
        # 5. Local Context BoW Logic
        print("Preparing Local Context index...")
        self.doc_context_features = [self._extract_local_context(doc) for doc in self.processed_docs]
        self.context_vectorizer = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None)
        self.doc_context_tfidf = self.context_vectorizer.fit_transform(self.doc_context_features)
        
        # IDF for Query Reduction
        self.idf_dict = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))

    def _extract_local_context(self, processed_doc, radius=2):
        features = []
        for sentence in processed_doc:
            for i in range(len(sentence)):
                start = max(0, i - radius)
                end = min(len(sentence), i + radius + 1)
                bag = sorted(set(sentence[start:end]))
                if len(bag) >= 2:
                    for combo in combinations(bag, 2):
                        features.append("ctx:" + "__".join(combo))
        return " ".join(features)

    def _wordnet_expand(self, tokens):
        expanded = list(tokens)
        for t in tokens:
            syns = wordnet.synsets(t)
            for syn in syns[:2]: # Top 2 synsets
                for lemma in syn.lemmas()[:2]: # Top 2 lemmas
                    name = lemma.name().replace("_", " ").lower()
                    if name != t:
                        expanded.append(name)
        return expanded

    def rank(self, query_text, method="baseline"):
        q_proc = self.preprocessor.preprocess_text(query_text)
        q_tokens = [t for s in q_proc for t in s]
        
        if method == "wordnet":
            q_tokens = self._wordnet_expand(q_tokens)
        
        if method == "query_reduction":
            scored = sorted([(self.idf_dict.get(t, 0.0), t) for t in q_tokens], reverse=True)
            q_tokens = [t for s, t in scored[:6]]
            
        q_str = " ".join(q_tokens)
        
        if method in ["baseline", "query_reduction", "wordnet"]:
            q_vec = self.vectorizer.transform([q_str])
            scores = cosine_similarity(q_vec, self.doc_tfidf).flatten()
        elif method == "ngram":
            q_vec = self.ngram_vectorizer.transform([q_str])
            scores = cosine_similarity(q_vec, self.doc_ngram_tfidf).flatten()
        elif method == "lsa":
            q_vec = self.vectorizer.transform([q_str])
            q_lsa = self.svd.transform(q_vec)
            scores = cosine_similarity(q_lsa, self.doc_lsa).flatten()
        elif method == "esa":
            q_vec = self.vectorizer.transform([q_str])
            q_sim = cosine_similarity(q_vec, self.doc_tfidf).flatten()
            # Map query to concepts
            top_idx = np.argsort(q_sim)[-25:]
            q_concept_vec = np.zeros(len(self.doc_ids))
            q_concept_vec[top_idx] = q_sim[top_idx]
            scores = np.dot(q_concept_vec, self.doc_concepts.T)
        elif method == "local_context":
            ctx_str = self._extract_local_context(q_proc)
            q_vec = self.vectorizer.transform([q_str])
            q_ctx_vec = self.context_vectorizer.transform([ctx_str])
            
            s1 = cosine_similarity(q_vec, self.doc_tfidf).flatten()
            s2 = cosine_similarity(q_ctx_vec, self.doc_context_tfidf).flatten()
            scores = s1 + 0.5 * s2
        else:
            return []
            
        indices = np.argsort(scores)[::-1]
        return [self.doc_ids[i] for i in indices]

# Main execution logic
docs_json, test_set = load_data()
doc_ids = [str(d["id"]) for d in docs_json]
doc_bodies = [d["body"] for d in docs_json]

search_system = SimpleSearchSystem(doc_bodies, doc_ids)

results_summary = {}
methods = ["baseline", "ngram", "lsa", "esa", "query_reduction", "wordnet", "local_context"]

for lim_name, lim_data in test_set["limitations"].items():
    print(f"Testing limitation: {lim_name}")
    all_queries = []
    for q in lim_data["cranfield_queries"]:
        all_queries.append((q["query"], q["relevant_docs"]))
    for q in lim_data["custom_queries"]:
        all_queries.append((q["query"], q["expected_relevant_docs"]))
        
    results_summary[lim_name] = {}
    for m in methods:
        scores = []
        for q_text, rel_ids in all_queries:
            if not rel_ids: continue
            rankings = search_system.rank(q_text, method=m)
            scores.append(get_map_at_k(rankings, rel_ids))
        results_summary[lim_name][m] = np.mean(scores) if scores else 0.0

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Print table
print("\nMAP@10 Performance per Limitation:")
header = f"{'Limitation':<40} | {'Base':<6} | {'NGram':<6} | {'LSA':<6} | {'ESA':<6} | {'Reduc':<6} | {'WNet':<6} | {'LCtx':<6}"
print(header)
print("-" * len(header))
for lim, res in results_summary.items():
    print(f"{lim:<40} | {res['baseline']:<6.3f} | {res['ngram']:<6.3f} | {res['lsa']:<6.3f} | {res['esa']:<6.3f} | {res['query_reduction']:<6.3f} | {res['wordnet']:<6.3f} | {res['local_context']:<6.3f}")
