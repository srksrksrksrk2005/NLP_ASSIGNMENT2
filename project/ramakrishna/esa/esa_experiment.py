import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR.parent / "lsa"))

from evaluation import Evaluation  # type: ignore
from hybrid_retrieval import HybridRetrieval  # type: ignore
from informationRetrieval import InformationRetrieval  # type: ignore
from inflectionReduction import InflectionReduction  # type: ignore
from lsa_retrieval import LSARetrieval  # type: ignore
from sentenceSegmentation import SentenceSegmentation  # type: ignore
from stopwordRemoval import StopwordRemoval  # type: ignore
from tokenization import Tokenization  # type: ignore
from esa_retrieval import ESARetrieval

# list of metrics we track
METRICS = [
    ("precision", "Precision"), ("recall", "Recall"), 
    ("fscore", "F-Score"), ("map", "MAP"), 
    ("ndcg", "nDCG"), ("mrr", "MRR")
]

class ESAExperiment:
    """Helper to run experiments comparing TF-IDF, LSA, Hybrid, and ESA."""
    
    def __init__(self, args):
        self.args = args
        self.out_folder = Path(args.out_folder)
        self.out_folder.mkdir(parents=True, exist_ok=True)

        self.seg = SentenceSegmentation()
        self.tok = Tokenization()
        self.inf = InflectionReduction()
        self.stp = StopwordRemoval()
        self.ev = Evaluation()

    def preprocess(self, texts):
        out = [self.seg.punkt(t) for t in texts]
        out = [self.tok.pennTreeBank(t) for t in out]
        out = [self.inf.reduce(t) for t in out]
        out = [self.stp.fromList(t) for t in out]
        return out

    def load_dataset(self):
        dataset = Path(self.args.dataset)
        queries = json.load(open(dataset / "cran_queries.json"))
        docs = json.load(open(dataset / "cran_docs.json"))
        qrels = json.load(open(dataset / "cran_qrels.json"))
        return queries, docs, qrels

    def load_concept_corpus(self):
        # currently only supporting cranfield self-concepts
        if self.args.concept_source == "cranfield":
            return None, None, "cranfield"
        raise ValueError("Only cranfield concept source is supported for now.")

    def evaluate(self, ranked, qids, qrels):
        res = {k: [] for k, _ in METRICS}
        for k in range(1, 11):
            res["precision"].append(self.ev.meanPrecision(ranked, qids, qrels, k))
            res["recall"].append(self.ev.meanRecall(ranked, qids, qrels, k))
            res["fscore"].append(self.ev.meanFscore(ranked, qids, qrels, k))
            res["map"].append(self.ev.meanAveragePrecision(ranked, qids, qrels, k))
            res["ndcg"].append(self.ev.meanNDCG(ranked, qids, qrels, k))
            res["mrr"].append(self.ev.meanReciprocalRank(ranked, qids, qrels, k))
        return res

    def rank_tfidf(self, p_docs, doc_ids, p_queries):
        ir = InformationRetrieval()
        ir.buildIndex(p_docs, doc_ids)
        return ir.rank(p_queries), ir

    def rank_lsa(self, p_docs, doc_ids, p_queries):
        lsa = LSARetrieval(
            n_components=self.args.lsa_components,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df, min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        lsa.build_index(p_docs, doc_ids)
        return lsa.rank(p_queries), lsa

    def rank_hybrid(self, p_docs, doc_ids, p_queries):
        hyb = HybridRetrieval(
            n_components=self.args.hybrid_lsa_components,
            tfidf_weight=self.args.tfidf_weight,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df, min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        hyb.build_index(p_docs, doc_ids)
        return hyb.rank(p_queries), hyb

    def rank_esa(self, p_docs, doc_ids, p_queries, concept_docs=None, 
                 concept_ids=None, concept_source=None, prebuilt_index_path=None):
        esa = ESARetrieval(
            top_concepts=self.args.esa_top_concepts,
            min_similarity=self.args.esa_min_similarity,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df, min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        esa.build_index(p_docs, doc_ids, concept_docs=concept_docs, 
                        concept_ids=concept_ids, concept_source=concept_source,
                        prebuilt_index_path=prebuilt_index_path)
        return esa.rank(p_queries), esa

    def plot_results(self, all_metrics):
        """Plot ESA vs TFIDF and ESA vs All"""
        methods = ["tfidf", "lsa", "hybrid", "esa"]
        colors = {"tfidf": "k", "lsa": "b", "hybrid": "g", "esa": "r"}
        
        # Plot all methods overlay
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for ax, (m_key, m_name) in zip(axes.flat, METRICS):
            for algo in methods:
                ax.plot(range(1, 11), all_metrics[algo][m_key], 
                        label=algo.upper(), c=colors[algo], lw=2)
            ax.set_title(m_name)
            ax.grid(alpha=0.3)
        
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(self.out_folder / "esa_vs_all_metrics.png")
        plt.close()

    def run_evaluation(self):
        q_json, d_json, qrels = self.load_dataset()
        
        q_ids = [q["query number"] for q in q_json]
        q_texts = [q["query"] for q in q_json]
        d_ids = [d["id"] for d in d_json]
        d_texts = [d["body"] for d in d_json]

        p_queries = self.preprocess(q_texts)
        p_docs = self.preprocess(d_texts)

        c_docs, c_ids, c_src = None, None, None
        if not self.args.prebuilt_index_path:
            c_docs, c_ids, c_src = self.load_concept_corpus()

        # run all models
        r_tfidf, m_tfidf = self.rank_tfidf(p_docs, d_ids, p_queries)
        r_lsa, m_lsa = self.rank_lsa(p_docs, d_ids, p_queries)
        r_hyb, m_hyb = self.rank_hybrid(p_docs, d_ids, p_queries)
        r_esa, m_esa = self.rank_esa(p_docs, d_ids, p_queries, c_docs, c_ids, c_src, 
                                     self.args.prebuilt_index_path)

        all_metrics = {
            "tfidf": self.evaluate(r_tfidf, q_ids, qrels),
            "lsa": self.evaluate(r_lsa, q_ids, qrels),
            "hybrid": self.evaluate(r_hyb, q_ids, qrels),
            "esa": self.evaluate(r_esa, q_ids, qrels),
        }

        self.plot_results(all_metrics)

        # save summary json
        with open(self.out_folder / "esa_metrics_summary.json", "w") as f:
            json.dump({m: all_metrics[m] for m in all_metrics}, f, indent=2)

        return {"metrics": all_metrics}
