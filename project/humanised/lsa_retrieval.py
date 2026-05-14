import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class LSAModel:
    """
    Retrieval model using Latent Semantic Analysis (SVD on TF-IDF).
    """
    def __init__(self, components=250):
        self.components = components
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        self.svd = TruncatedSVD(n_components=components, random_state=42)
        self.doc_matrix = None
        self.doc_ids = []
        
    def fit(self, docs):
        self.doc_ids = [d['id'] for d in docs]
        texts = [d['text'] for d in docs]
        
        tfidf_m = self.vectorizer.fit_transform(texts)
        self.doc_matrix = self.svd.fit_transform(tfidf_m)
        
    def search(self, query):
        q_vec = self.vectorizer.transform([query])
        q_lsa = self.svd.transform(q_vec)
        
        sims = cosine_similarity(q_lsa, self.doc_matrix)[0]
        res = list(zip(self.doc_ids, sims))
        res.sort(key=lambda x: x[1], reverse=True)
        return res

def main():
    import os
    base_dir = "../../cranfield"
    docs = json.load(open(os.path.join(base_dir, "cran_docs.json")))
    queries = json.load(open(os.path.join(base_dir, "cran_queries.json")))
    qrels = json.load(open(os.path.join(base_dir, "cran_qrels.json")))

    mdl = LSAModel()
    mdl.fit(docs)
    
    aps = []
    for q in queries:
        ranked = mdl.search(q['query'])
        rels = [x['id'] for x in qrels if x['query_num'] == q['query_number']]
        
        h = 0.0
        s = 0.0
        for i, (did, _) in enumerate(ranked[:10]):
            if str(did) in rels:
                h += 1
                s += h / (i+1)
        aps.append(s / min(len(rels), 10) if rels else 0)
        
    print(f"LSA MAP@10: {np.mean(aps):.4f}")
    plt.plot(aps, 'o', alpha=0.5)
    plt.title("LSA AP@10 per query")
    plt.savefig("lsa_plot.png")

if __name__ == '__main__':
    main()
