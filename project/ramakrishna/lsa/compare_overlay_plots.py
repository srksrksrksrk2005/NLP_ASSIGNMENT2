import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = CURRENT_DIR.parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from evaluation import Evaluation
from inflectionReduction import InflectionReduction
from informationRetrieval import InformationRetrieval
from sentenceSegmentation import SentenceSegmentation
from stopwordRemoval import StopwordRemoval
from tokenization import Tokenization

from hybrid_retrieval import HybridRetrieval
from lsa_retrieval import LSARetrieval


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("fscore", "F-Score"),
    ("map", "MAP"),
    ("ndcg", "nDCG"),
    ("mrr", "MRR"),
]

METHOD_ORDER = ["tfidf", "lsa", "hybrid"]
METHOD_LABELS = {
    "tfidf": "Standard TF-IDF",
    "lsa": "Tuned LSA",
    "hybrid": "Tuned Hybrid (TF-IDF + LSA)",
}
METHOD_STYLES = {
    "tfidf": {"color": "black", "linestyle": "--", "marker": "o"},
    "lsa": {"color": "tab:blue", "linestyle": "-", "marker": "s"},
    "hybrid": {"color": "tab:green", "linestyle": "-", "marker": "^"},
}


class OverlayComparer:
    def __init__(self, args):
        self.args = args
        os.makedirs(self.args.out_folder, exist_ok=True)

        self.tokenizer = Tokenization()
        self.sentence_segmenter = SentenceSegmentation()
        self.inflection_reducer = InflectionReduction()
        self.stopword_remover = StopwordRemoval()
        self.evaluator = Evaluation()

    def segment_sentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentence_segmenter.naive(text)
        return self.sentence_segmenter.punkt(text)

    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        return self.tokenizer.pennTreeBank(text)

    def reduce_inflection(self, text):
        return self.inflection_reducer.reduce(text)

    def remove_stopwords(self, text):
        return self.stopword_remover.fromList(text)

    def preprocess_texts(self, texts):
        segmented = [self.segment_sentences(text) for text in texts]
        tokenized = [self.tokenize(text) for text in segmented]
        reduced = [self.reduce_inflection(text) for text in tokenized]
        return [self.remove_stopwords(text) for text in reduced]

    def load_dataset(self):
        dataset = Path(self.args.dataset)
        queries_json = json.load(open(dataset / "cran_queries.json", "r"))
        docs_json = json.load(open(dataset / "cran_docs.json", "r"))
        qrels = json.load(open(dataset / "cran_qrels.json", "r"))
        return queries_json, docs_json, qrels

    def evaluate_rankings(self, ranked_docs, query_ids, qrels):
        metric_values = {metric_key: [] for metric_key, _ in METRICS}

        for k in range(1, 11):
            metric_values["precision"].append(
                self.evaluator.meanPrecision(ranked_docs, query_ids, qrels, k)
            )
            metric_values["recall"].append(
                self.evaluator.meanRecall(ranked_docs, query_ids, qrels, k)
            )
            metric_values["fscore"].append(
                self.evaluator.meanFscore(ranked_docs, query_ids, qrels, k)
            )
            metric_values["map"].append(
                self.evaluator.meanAveragePrecision(ranked_docs, query_ids, qrels, k)
            )
            metric_values["ndcg"].append(
                self.evaluator.meanNDCG(ranked_docs, query_ids, qrels, k)
            )
            metric_values["mrr"].append(
                self.evaluator.meanReciprocalRank(ranked_docs, query_ids, qrels, k)
            )

        return metric_values

    def rank_tfidf(self, processed_docs, doc_ids, processed_queries):
        tfidf_retriever = InformationRetrieval()
        tfidf_retriever.buildIndex(processed_docs, doc_ids)
        return tfidf_retriever.rank(processed_queries)

    def rank_lsa(self, processed_docs, doc_ids, processed_queries):
        lsa_retriever = LSARetrieval(
            n_components=self.args.lsa_components,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df,
            min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        lsa_retriever.build_index(processed_docs, doc_ids)
        ranked_docs = lsa_retriever.rank(processed_queries)
        return ranked_docs, lsa_retriever

    def rank_hybrid(self, processed_docs, doc_ids, processed_queries):
        hybrid_retriever = HybridRetrieval(
            n_components=self.args.hybrid_lsa_components,
            tfidf_weight=self.args.tfidf_weight,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df,
            min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        hybrid_retriever.build_index(processed_docs, doc_ids)
        ranked_docs = hybrid_retriever.rank(processed_queries)
        return ranked_docs, hybrid_retriever

    def save_overlay_plots(self, all_metrics):
        k_values = list(range(1, 11))

        figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for axis, (metric_key, metric_name) in zip(axes.flat, METRICS):
            for method_key in METHOD_ORDER:
                style = METHOD_STYLES[method_key]
                axis.plot(
                    k_values,
                    all_metrics[method_key][metric_key],
                    label=METHOD_LABELS[method_key],
                    linewidth=2,
                    markersize=4,
                    **style,
                )
            axis.set_title(metric_name)
            axis.set_xlabel("k")
            axis.set_ylabel("Score")
            axis.grid(alpha=0.25)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        figure.suptitle("Cranfield Comparison: TF-IDF vs LSA vs Hybrid", y=0.98)
        figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        figure.savefig(Path(self.args.out_folder) / "overlay_all_metrics.png")
        plt.close(figure)

        for metric_key, metric_name in METRICS:
            plt.figure(figsize=(9, 5))
            for method_key in METHOD_ORDER:
                style = METHOD_STYLES[method_key]
                plt.plot(
                    k_values,
                    all_metrics[method_key][metric_key],
                    label=METHOD_LABELS[method_key],
                    linewidth=2,
                    markersize=5,
                    **style,
                )
            plt.title(f"{metric_name} Overlay")
            plt.xlabel("k")
            plt.ylabel("Score")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(self.args.out_folder) / f"overlay_{metric_key}.png")
            plt.close()

    def save_summary(self, all_metrics, lsa_retriever, hybrid_retriever):
        summary = {
            "config": {
                "segmenter": self.args.segmenter,
                "tokenizer": self.args.tokenizer,
                "vectorizer": {
                    "sublinear_tf": not self.args.disable_sublinear_tf,
                    "max_df": self.args.max_df,
                    "min_df": self.args.min_df,
                    "norm": self.args.tfidf_norm,
                    "ngram_range": [1, self.args.ngram_max],
                },
                "lsa": {
                    "requested_components": self.args.lsa_components,
                    "used_components": lsa_retriever.actual_components,
                    "explained_variance_ratio_sum": lsa_retriever.explained_variance,
                },
                "hybrid": {
                    "requested_components": self.args.hybrid_lsa_components,
                    "used_components": hybrid_retriever.actual_components,
                    "tfidf_weight": self.args.tfidf_weight,
                    "explained_variance_ratio_sum": hybrid_retriever.explained_variance,
                },
            },
            "metrics": all_metrics,
            "k10": {
                method_key: {
                    metric_key: all_metrics[method_key][metric_key][-1]
                    for metric_key, _ in METRICS
                }
                for method_key in METHOD_ORDER
            },
        }

        summary_path = Path(self.args.out_folder) / "comparison_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def run(self):
        queries_json, docs_json, qrels = self.load_dataset()
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processed_queries = self.preprocess_texts(queries)
        processed_docs = self.preprocess_texts(docs)

        ranked_tfidf = self.rank_tfidf(processed_docs, doc_ids, processed_queries)
        ranked_lsa, lsa_retriever = self.rank_lsa(processed_docs, doc_ids, processed_queries)
        ranked_hybrid, hybrid_retriever = self.rank_hybrid(
            processed_docs, doc_ids, processed_queries
        )

        all_metrics = {
            "tfidf": self.evaluate_rankings(ranked_tfidf, query_ids, qrels),
            "lsa": self.evaluate_rankings(ranked_lsa, query_ids, qrels),
            "hybrid": self.evaluate_rankings(ranked_hybrid, query_ids, qrels),
        }

        self.save_overlay_plots(all_metrics)
        self.save_summary(all_metrics, lsa_retriever, hybrid_retriever)

        print("Overlay plots saved in:", self.args.out_folder)
        print("K=10 summary")
        for method_key in METHOD_ORDER:
            print(f"\n{METHOD_LABELS[method_key]}")
            print(f"Precision@10: {all_metrics[method_key]['precision'][-1]:.4f}")
            print(f"Recall@10: {all_metrics[method_key]['recall'][-1]:.4f}")
            print(f"Fscore@10: {all_metrics[method_key]['fscore'][-1]:.4f}")
            print(f"MAP@10: {all_metrics[method_key]['map'][-1]:.4f}")
            print(f"nDCG@10: {all_metrics[method_key]['ndcg'][-1]:.4f}")
            print(f"MRR@10: {all_metrics[method_key]['mrr'][-1]:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate overlay plots for TF-IDF, LSA, and Hybrid retrieval"
    )
    parser.add_argument(
        "-dataset",
        default=str(ASSIGNMENT_ROOT / "cranfield"),
        help="Path to the Cranfield dataset folder",
    )
    parser.add_argument(
        "-out_folder",
        default=str(CURRENT_DIR / "output_compare"),
        help="Folder where comparison plots and summary will be written",
    )
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])

    parser.add_argument("-lsa_components", type=int, default=250)
    parser.add_argument("-hybrid_lsa_components", type=int, default=240)
    parser.add_argument("-tfidf_weight", type=float, default=0.2)
    parser.add_argument("-random_state", type=int, default=42)

    parser.add_argument("-max_df", type=float, default=0.95)
    parser.add_argument("-min_df", type=int, default=2)
    parser.add_argument("-tfidf_norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("-ngram_max", type=int, default=1, choices=[1, 2])
    parser.add_argument("-disable_sublinear_tf", action="store_true")
    return parser


if __name__ == "__main__":
    start_time = time.time()
    arguments = build_parser().parse_args()
    if arguments.tfidf_norm == "none":
        arguments.tfidf_norm = None

    comparer = OverlayComparer(arguments)
    comparer.run()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")