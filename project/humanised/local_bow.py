import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalContextBoW:
    """
    Local Context Bag-of-Words Retriever.
    Extracts unordered pairs within a window to capture phrase contexts.
    """
    def __init__(self, window_size=3):
        self.window = window_size
        self.primary_vect = TfidfVectorizer(max_df=0.9, min_df=2)
        self.context_vect = TfidfVectorizer(max_df=0.9, min_df=2)
        
        self.primary_mtx = None
        self.context_mtx = None
        self.doc_ids = []

    def _extract_pairs(self, text):
        tokens = text.split()
        pairs = []
        for i, t in enumerate(tokens):
            start = max(0, i - self.window)
            end = min(len(tokens), i + self.window + 1)
            for j in range(start, end):
                if i != j:
                    t1, t2 = sorted([t, tokens[j]])
                    pairs.append(f"{t1}_{t2}")
        return " ".join(pairs)

    def fit(self, docs):
        self.doc_ids = [d['id'] for d in docs]
        texts = [d['text'] for d in docs]
        
        self.primary_mtx = self.primary_vect.fit_transform(texts)
        
        pair_texts = [self._extract_pairs(t) for t in texts]
        self.context_mtx = self.context_vect.fit_transform(pair_texts)

    def search(self, query, alpha=0.7):
        q_prim = self.primary_vect.transform([query])
        q_pairs = self._extract_pairs(query)
        q_ctx = self.context_vect.transform([q_pairs])
        
        sim_prim = cosine_similarity(q_prim, self.primary_mtx)[0]
        sim_ctx = cosine_similarity(q_ctx, self.context_mtx)[0]
        
        final_sims = alpha * sim_prim + (1 - alpha) * sim_ctx
        
        res = [(self.doc_ids[i], final_sims[i]) for i in range(len(self.doc_ids))]
        res.sort(key=lambda x: x[1], reverse=True)
        return res

def main():
    import os
    base_dir = "../../cranfield"
    docs = json.load(open(os.path.join(base_dir, "cran_docs.json")))
    queries = json.load(open(os.path.join(base_dir, "cran_queries.json")))
    qrels = json.load(open(os.path.join(base_dir, "cran_qrels.json")))

    mdl = LocalContextBoW(window_size=3)
    mdl.fit(docs)
    
    aps = []
    for q in queries:
        ranked = mdl.search(q['query'], alpha=0.8)
        rels = {x['id'] for x in qrels if x['query_num'] == q['query_number']}
        
        h = 0
        s = 0
        for i, (did, _) in enumerate(ranked[:10]):
            if str(did) in rels:
                h += 1
                s += h / (i+1)
        aps.append(s / min(len(rels), 10) if rels else 0)
        
    print(f"Local Context BoW MAP@10: {np.mean(aps):.4f}")
    plt.plot(aps, 'k.')
    plt.title("Local BoW AP@10")
    plt.savefig("local_bow_results.png")

if __name__ == '__main__':
    main()
