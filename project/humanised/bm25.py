import json
import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class BM25Retriever:
    """
    Implements the Okapi BM25 ranking function.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.avg_dl = 0
        self.N = 0
        self.doc_ids = []

    def fit(self, docs):
        self.N = len(docs)
        total_len = 0
        df = Counter()
        for doc in docs:
            self.doc_ids.append(doc['id'])
            tokens = doc['text'].split()
            total_len += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append((len(tokens), freqs))
            for t in freqs.keys():
                df[t] += 1
        
        self.avg_dl = total_len / self.N
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def evaluate_query(self, query_text):
        tokens = query_text.split()
        scores = []
        for i, (dl, freqs) in enumerate(self.doc_freqs):
            score = 0
            for t in tokens:
                if t not in freqs: continue
                tf = freqs[t]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += self.idf.get(t, 0) * (num / den)
            scores.append((self.doc_ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

def main():
    import os
    base_dir = "../../cranfield"
    with open(os.path.join(base_dir, "cran_docs.json")) as f:
        docs = json.load(f)
    with open(os.path.join(base_dir, "cran_queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(base_dir, "cran_qrels.json")) as f:
        qrels = json.load(f)

    model = BM25Retriever()
    model.fit(docs)
    
    # minimal evaluation
    aps = []
    for q in queries:
        ans = model.evaluate_query(q['query'])
        rel_docs = [x['id'] for x in qrels if x['query_num'] == q['query_number']]
        
        hits = 0.0
        sum_p = 0.0
        for i, (did, _) in enumerate(ans[:10]):
            if str(did) in rel_docs:
                hits += 1
                sum_p += hits / (i + 1)
        aps.append(sum_p / min(len(rel_docs), 10) if rel_docs else 0)
    
    map10 = np.mean(aps)
    print(f"BM25 MAP@10: {map10:.4f}")
    
    plt.hist(aps, bins=20)
    plt.title("BM25 AP@10 Distribution")
    plt.savefig("bm25_results.png")

if __name__ == '__main__':
    main()
