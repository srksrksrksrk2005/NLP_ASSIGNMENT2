import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ReducedQueryRetriever:
    """
    Retrieval model that reduces query terms using IDF and term pruning 
    before performing standard TF-IDF retrieval.
    """
    def __init__(self, keep_top_k=5):
        self.keep_top_k = keep_top_k
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
        self.tfidf_mtx = None
        self.docs_id = []
        self.feat_names = []
        self.idf_dict = {}

    def fit(self, docs):
        self.docs_id = [d['id'] for d in docs]
        texts = [d['text'] for d in docs]
        self.tfidf_mtx = self.vectorizer.fit_transform(texts)
        self.feat_names = self.vectorizer.get_feature_names_out()
        
        # cache IDF for quick lookup
        for w, idx in self.vectorizer.vocabulary_.items():
            self.idf_dict[w] = self.vectorizer.idf_[idx]
            
    def prune_query(self, query):
        tokens = query.split()
        scored = []
        for t in tokens:
            if t in self.idf_dict:
                scored.append((t, self.idf_dict[t]))
        # sort by IDF (rarity) descending
        scored.sort(key=lambda x: x[1], reverse=True)
        # keep top k
        reduced = [t for t, _ in scored[:self.keep_top_k]]
        return " ".join(reduced)
            
    def search(self, query):
        q_reduced = self.prune_query(query)
        q_v = self.vectorizer.transform([q_reduced])
        sims = cosine_similarity(q_v, self.tfidf_mtx)[0]
        results = [(self.docs_id[i], sims[i]) for i in range(len(self.docs_id))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

def main():
    import os
    base = "../../cranfield"
    doc_file = os.path.join(base, "cran_docs.json")
    query_file = os.path.join(base, "cran_queries.json")
    qrels_file = os.path.join(base, "cran_qrels.json")
    
    docs = json.load(open(doc_file))
    queries = json.load(open(query_file))
    qrels = json.load(open(qrels_file))
    
    rt = ReducedQueryRetriever(keep_top_k=7)
    rt.fit(docs)
    
    ap_scores = []
    for q in queries:
        ranked = rt.search(q['query'])
        truth = {x['id'] for x in qrels if x['query_num'] == q['query_number']}
        
        h = 0
        s = 0
        for i, (did, _) in enumerate(ranked[:10]):
            if str(did) in truth:
                h += 1
                s += h / (i + 1)
        if truth:
            ap_scores.append(s / min(len(truth), 10))
        else:
            ap_scores.append(0)
            
    m = np.mean(ap_scores)
    print(f"Reduction (k=7) MAP@10: {m:.4f}")
    
    plt.plot(ap_scores, label='Query Reduction')
    plt.legend()
    plt.savefig("query_reduction.png")

if __name__ == '__main__':
    main()
