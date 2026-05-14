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
            tokens = doc['body'].split()
            total_len += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append((len(tokens), freqs))
            for t in freqs.keys():
                df[t] += 1
        
        self.avg_dl = total_len / self.N
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def search(self, query_text):
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
    import sys
    import json
    import matplotlib.pyplot as plt
    sys.path.append("../../")
    from evaluation import Evaluation
    
    base_dir = "../../cranfield"
    docs = json.load(open(os.path.join(base_dir, "cran_docs.json")))
    queries = json.load(open(os.path.join(base_dir, "cran_queries.json")))
    
    # Inject optimal dataset features
    try:
        prep_docs = json.load(open("../../output/stopword_removed_docs.txt"))
        prep_queries = json.load(open("../../output/stopword_removed_queries.txt"))
        for i, d in enumerate(docs): d['body'] = " ".join(t for s in prep_docs[i] for t in s)
        for i, q in enumerate(queries): q['query'] = " ".join(t for s in prep_queries[i] for t in s)
    except:
        pass
    qrels = json.load(open(os.path.join(base_dir, "cran_qrels.json")))

    mdl = BM25Retriever()
    mdl.fit(docs)
    
    doc_IDs_ordered = []
    query_ids = []
    
    for q in queries:
        ranked = mdl.search(q['query'])
        doc_IDs_ordered.append([str(did) for did, _ in ranked])
        query_ids.append(q['query number'])

    evaluator = Evaluation()
    
    ks = list(range(1, 11))
    
    # Metrics for Proposed Model
    p_prec, p_rec, p_f1, p_map, p_ndcg, p_mrr = [], [], [], [], [], []
    for k in ks:
        p_prec.append(evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k))
        p_rec.append(evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k))
        p_f1.append(evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k))
        p_map.append(evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k))
        p_ndcg.append(evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k))
        p_mrr.append(evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k))
        
    print(f"Model MAP@10: {p_map[-1]:.4f}")
    
    # Baseline comparison (Simple TF-IDF)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    base_vec = TfidfVectorizer()
    doc_mat = base_vec.fit_transform([d["body"] for d in docs])
    
    base_docs_ordered = []
    for q in queries:
        q_vec = base_vec.transform([q["query"]])
        sims = cosine_similarity(q_vec, doc_mat)[0]
        res = list(zip([d["id"] for d in docs], sims))
        res.sort(key=lambda x: x[1], reverse=True)
        base_docs_ordered.append([str(did) for did, _ in res])
        
    # Metrics for Baseline
    b_prec, b_rec, b_f1, b_map, b_ndcg, b_mrr = [], [], [], [], [], []
    for k in ks:
        b_prec.append(evaluator.meanPrecision(base_docs_ordered, query_ids, qrels, k))
        b_rec.append(evaluator.meanRecall(base_docs_ordered, query_ids, qrels, k))
        b_f1.append(evaluator.meanFscore(base_docs_ordered, query_ids, qrels, k))
        b_map.append(evaluator.meanAveragePrecision(base_docs_ordered, query_ids, qrels, k))
        b_ndcg.append(evaluator.meanNDCG(base_docs_ordered, query_ids, qrels, k))
        b_mrr.append(evaluator.meanReciprocalRank(base_docs_ordered, query_ids, qrels, k))
        
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    colors = ["b", "g", "r", "c", "m", "k"]
    labels = ["Precision", "Recall", "F-score", "MAP", "nDCG", "MRR"]
    proposed_metrics = [p_prec, p_rec, p_f1, p_map, p_ndcg, p_mrr]
    baseline_metrics = [b_prec, b_rec, b_f1, b_map, b_ndcg, b_mrr]
    
    for i in range(6):
        plt.plot(ks, proposed_metrics[i], label=f"Proposed {labels[i]}", color=colors[i], marker="o")
        plt.plot(ks, baseline_metrics[i], label=f"Baseline {labels[i]}", color=colors[i], linestyle="--", marker="x")
        
    plt.title("Evaluation Metrics @k - Proposed vs Baseline")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists("output"):
        os.makedirs("output")
        
    script_name = os.path.basename(__file__).replace(".py", "")
    plt.savefig(f"output/{script_name}_metrics_k.png")

if __name__ == '__main__':
    main()
