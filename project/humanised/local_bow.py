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
        texts = [d['body'] for d in docs]
        
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

    mdl = LocalContextBoW(window_size=3)
    mdl.fit(docs)
    
    doc_IDs_ordered = []
    query_ids = []
    
    for q in queries:
        ranked = mdl.search(q['query'], alpha=0.8)
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
