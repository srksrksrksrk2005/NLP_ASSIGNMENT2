import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project paths
BASE_DIR = Path("c:/Users/srksr/NLP/NLP_project/NLP_ASSIGNMENT2")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "project/ritisha/ngram"))
sys.path.append(str(BASE_DIR / "project/ritisha/wordnet"))
sys.path.append(str(BASE_DIR / "project/Nikhil/query_expansion"))

from core.preprocessing import Preprocessor
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

# Main tuning logic
docs_json, test_set = load_data()
preprocessor = Preprocessor()
doc_ids = [str(d["id"]) for d in docs_json]
processed_docs = [preprocessor.preprocess_text(d["body"]) for d in docs_json]

# All queries for tuning
all_queries = []
for lim in test_set["limitations"].values():
    for q in lim["cranfield_queries"]:
        all_queries.append((preprocessor.preprocess_text(q["query"]), q["relevant_docs"]))
    for q in lim["custom_queries"]:
        all_queries.append((preprocessor.preprocess_text(q["query"]), q["expected_relevant_docs"]))

def run_ngram_tuning():
    print("\n--- Tuning N-gram ---", flush=True)
    results = []
    for n in [1, 2, 3]:
        print(f"Testing n={n}...", flush=True)
        engine = NgramRetrieval(n=n)
        engine.buildIndex(processed_docs, doc_ids)
        
        scores = []
        for q_proc, rel_ids in all_queries:
            rankings = engine.rank([q_proc])[0]
            scores.append(get_map_at_k(rankings, rel_ids))
        
        avg_map = np.mean(scores)
        print(f"n={n}: MAP@10 = {avg_map:.4f}", flush=True)
        results.append((n, avg_map))
    return results

def run_wordnet_tuning():
    print("\n--- Tuning WordNet (Lesk) ---", flush=True)
    results = []
    # VERY small sweep to save time
    for ms in [1, 2]:
        for cw in [5]:
            print(f"Testing max_synonyms={ms}, context_window={cw}...", flush=True)
            engine = WordNetRetrieval(max_synonyms=ms, context_window=cw)
            engine.buildIndex(processed_docs, doc_ids)
            
            scores = []
            for q_proc, rel_ids in all_queries:
                rankings = engine.rank([q_proc])[0]
                scores.append(get_map_at_k(rankings, rel_ids))
            
            avg_map = np.mean(scores)
            print(f"ms={ms}, cw={cw}: MAP@10 = {avg_map:.4f}", flush=True)
            results.append((ms, cw, avg_map))
    return results

if __name__ == "__main__":
    ngram_res = run_ngram_tuning()
    wordnet_res = run_wordnet_tuning()
    
    best_n = max(ngram_res, key=lambda x: x[1])[0]
    best_wn = max(wordnet_res, key=lambda x: x[2])
    
    print("\n" + "="*50)
    print(f"BEST N-gram: n={best_n}")
    print(f"BEST WordNet: max_synonyms={best_wn[0]}, context_window={best_wn[1]}")
    print("="*50)
