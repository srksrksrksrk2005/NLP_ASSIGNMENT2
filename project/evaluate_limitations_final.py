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

# Add project paths
BASE_DIR = Path("c:/Users/srksr/NLP/NLP_project/NLP_ASSIGNMENT2")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "project/Nikhil/query_expansion"))
sys.path.append(str(BASE_DIR / "project/ramakrishna/lsa"))
sys.path.append(str(BASE_DIR / "project/ramakrishna/esa"))
sys.path.append(str(BASE_DIR / "project/ritisha/ngram"))
sys.path.append(str(BASE_DIR / "project/ritisha/wordnet"))

from core.preprocessing import Preprocessor
from core.retrieval import VectorSpaceRetrieval
from expansion.base import MatrixQueryExpander
from expansion.wordnet_matrix import build_wordnet_neighbor_map, WordNetOOVResolver
from expansion.embedding_matrices import (
    build_tfidf_neighbor_map,
    build_lsa_neighbor_map,
    build_esa_neighbor_map,
)
from lsa_retrieval import LSARetrieval
from esa_retrieval import ESARetrieval
from ngram_retrieval import NgramRetrieval
from wordnet_retrieval import WordNetRetrieval

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

# Main execution logic
docs_json, test_set = load_data()
preprocessor = Preprocessor()
doc_ids = [str(d["id"]) for d in docs_json]
processed_docs = [preprocessor.preprocess_text(d["body"]) for d in docs_json]

print("Initializing all retrieval systems...")
systems = {}

# 1. Base
systems["baseline"] = VectorSpaceRetrieval()
systems["baseline"].build(processed_docs, doc_ids)

# 2. Ngram (best n=3)
print("Building N-gram (n=3)...")
systems["ngram"] = NgramRetrieval(n=3)
systems["ngram"].buildIndex(processed_docs, doc_ids)

# 3. WordNet (Ritisha - Lesk) - best ms=1, cw=5
print("Building WordNet (ms=1, cw=5)...")
systems["wordnet_lesk"] = WordNetRetrieval(max_synonyms=1, context_window=5)
systems["wordnet_lesk"].buildIndex(processed_docs, doc_ids)

# 4. LSA/ESA (Ramakrishna)
print("Building LSA...")
systems["lsa"] = LSARetrieval(n_components=128)
systems["lsa"].build_index(processed_docs, doc_ids)
print("Building ESA...")
systems["esa"] = ESARetrieval(top_concepts=50)
systems["esa"].build_index(processed_docs, doc_ids)

# 5. Query Expansion variants (Nikhil)
print("Building Expansion Matrices...")
vocab = systems["baseline"].vocab
wn_map = build_wordnet_neighbor_map(vocab, top_k=10, min_similarity=0.1)
wn_oov = WordNetOOVResolver(vocab)
# FIX: pass wn_oov.resolve instead of wn_oov
systems["exp_wordnet"] = MatrixQueryExpander(vocab, wn_map, wn_oov.resolve)

tfidf_map = build_tfidf_neighbor_map(systems["baseline"].doc_tfidf, vocab, top_k=10)
systems["exp_tfidf"] = MatrixQueryExpander(vocab, tfidf_map)

lsa_map = build_lsa_neighbor_map(systems["baseline"].doc_tfidf, vocab, top_k=10)
systems["exp_lsa"] = MatrixQueryExpander(vocab, lsa_map)

results = {}
methods = ["baseline", "ngram", "wordnet_lesk", "lsa", "esa", "exp_wordnet", "exp_tfidf", "exp_lsa"]

for lim_name, lim_data in test_set["limitations"].items():
    print(f"Evaluating {lim_name}...")
    all_q = []
    for q in lim_data["cranfield_queries"]:
        all_q.append((preprocessor.preprocess_text(q["query"]), q["relevant_docs"]))
    for q in lim_data["custom_queries"]:
        all_q.append((preprocessor.preprocess_text(q["query"]), q["expected_relevant_docs"]))
        
    results[lim_name] = {}
    for m in methods:
        scores = []
        for q_proc, rel_ids in all_q:
            if not rel_ids: continue
            if m.startswith("exp_"):
                q_vec = systems[m].build_query_vector_from_counts(Counter([t for s in q_proc for t in s]))
                rankings = systems["baseline"].query_vectors_to_rankings(q_vec[np.newaxis, :])[0]
            elif m == "baseline":
                q_vec = np.zeros(len(systems["baseline"].vocab))
                counts = Counter([t for s in q_proc for t in s])
                for t, c in counts.items():
                    if t in systems["baseline"].term_to_idx:
                        q_vec[systems["baseline"].term_to_idx[t]] = c
                rankings = systems["baseline"].query_vectors_to_rankings(q_vec[np.newaxis, :])[0]
            else:
                rankings = systems[m].rank([q_proc])[0]
            scores.append(get_map_at_k(rankings, rel_ids))
        results[lim_name][m] = np.mean(scores) if scores else 0.0

with open("evaluation_detailed_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDetailed MAP@10 Results (Tuned):")
# Print a nice table
header = f"{'Limitation':<40} | {'Base':<6} | {'Ngram':<6} | {'WNetL':<6} | {'LSA':<6} | {'ESA':<6} | {'ExWN':<6} | {'ExTF':<6} | {'ExLSA':<6}"
print(header)
print("-" * len(header))
for lim, res in results.items():
    print(f"{lim:<40} | {res['baseline']:<6.3f} | {res['ngram']:<6.3f} | {res['wordnet_lesk']:<6.3f} | {res['lsa']:<6.3f} | {res['esa']:<6.3f} | {res['exp_wordnet']:<6.3f} | {res['exp_tfidf']:<6.3f} | {res['exp_lsa']:<6.3f}")
