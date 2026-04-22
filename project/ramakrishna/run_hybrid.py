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
from sentenceSegmentation import SentenceSegmentation
from stopwordRemoval import StopwordRemoval
from tokenization import Tokenization

from hybrid_retrieval import HybridRetrieval


class HybridSearchEngine:
    def __init__(self, args):
        self.args = args
        os.makedirs(self.args.out_folder, exist_ok=True)

        self.tokenizer = Tokenization()
        self.sentence_segmenter = SentenceSegmentation()
        self.inflection_reducer = InflectionReduction()
        self.stopword_remover = StopwordRemoval()
        self.retriever = HybridRetrieval(
            n_components=self.args.lsa_components,
            tfidf_weight=self.args.tfidf_weight,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df,
            min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
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

    def _load_dataset(self):
        dataset = Path(self.args.dataset)
        queries_json = json.load(open(dataset / "cran_queries.json", "r"))
        docs_json = json.load(open(dataset / "cran_docs.json", "r"))
        qrels = json.load(open(dataset / "cran_qrels.json", "r"))
        return queries_json, docs_json, qrels

    def _save_summary(self, summary):
        summary_path = Path(self.args.out_folder) / "hybrid_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def evaluate_dataset(self):
        queries_json, docs_json, qrels = self._load_dataset()
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processed_queries = self.preprocess_texts(queries)
        processed_docs = self.preprocess_texts(docs)

        self.retriever.build_index(processed_docs, doc_ids)
        ranked_docs = self.retriever.rank(processed_queries)

        precisions = []
        recalls = []
        fscores = []
        maps = []
        ndcgs = []
        mrrs = []

        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(ranked_docs, query_ids, qrels, k)
            recall = self.evaluator.meanRecall(ranked_docs, query_ids, qrels, k)
            fscore = self.evaluator.meanFscore(ranked_docs, query_ids, qrels, k)
            map_score = self.evaluator.meanAveragePrecision(ranked_docs, query_ids, qrels, k)
            ndcg = self.evaluator.meanNDCG(ranked_docs, query_ids, qrels, k)
            mrr = self.evaluator.meanReciprocalRank(ranked_docs, query_ids, qrels, k)

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)
            maps.append(map_score)
            ndcgs.append(ndcg)
            mrrs.append(mrr)

            print(f"Precision, Recall, F-score @ {k}: {precision}, {recall}, {fscore}")
            print(f"MAP, nDCG, MRR @ {k}: {map_score}, {ndcg}, {mrr}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), maps, label="MAP")
        plt.plot(range(1, 11), ndcgs, label="nDCG")
        plt.plot(range(1, 11), mrrs, label="MRR")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset (Hybrid TF-IDF + LSA)")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(Path(self.args.out_folder) / "hybrid_eval_plot.png")

        summary = {
            "lsa_components_requested": self.args.lsa_components,
            "lsa_components_used": self.retriever.actual_components,
            "tfidf_weight": self.retriever.tfidf_weight,
            "explained_variance_ratio_sum": self.retriever.explained_variance,
            "vectorizer": {
                "sublinear_tf": self.retriever.sublinear_tf,
                "max_df": self.retriever.max_df,
                "min_df": self.retriever.min_df,
                "norm": self.retriever.norm,
                "ngram_range": list(self.retriever.ngram_range),
                "vocab_size": self.retriever.vocab_size,
            },
            "precision": precisions,
            "recall": recalls,
            "fscore": fscores,
            "map": maps,
            "ndcg": ndcgs,
            "mrr": mrrs,
        }
        self._save_summary(summary)

        print("\nHybrid summary")
        print(f"Components used: {self.retriever.actual_components}")
        print(f"TF-IDF weight: {self.retriever.tfidf_weight}")
        print(f"Explained variance ratio sum: {self.retriever.explained_variance:.4f}")
        print(
            "Vectorizer config: "
            f"sublinear_tf={self.retriever.sublinear_tf}, "
            f"max_df={self.retriever.max_df}, "
            f"min_df={self.retriever.min_df}, "
            f"norm={self.retriever.norm}, "
            f"ngram_range={self.retriever.ngram_range}, "
            f"vocab_size={self.retriever.vocab_size}"
        )
        print(f"Output folder: {self.args.out_folder}")

    def handle_custom_query(self):
        _, docs_json, _ = self._load_dataset()
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processed_docs = self.preprocess_texts(docs)
        self.retriever.build_index(processed_docs, doc_ids)

        print("Enter query below")
        query = input()
        processed_query = self.preprocess_texts([query])
        ranked_docs = self.retriever.rank(processed_query)[0]

        print("\nTop five document IDs:")
        for doc_id in ranked_docs[:5]:
            print(doc_id)


def build_parser():
    parser = argparse.ArgumentParser(description="Hybrid TF-IDF + LSA retrieval for Cranfield")
    parser.add_argument(
        "-dataset",
        default=str(ASSIGNMENT_ROOT / "cranfield"),
        help="Path to the Cranfield dataset folder",
    )
    parser.add_argument(
        "-out_folder",
        default=str(CURRENT_DIR / "output_hybrid"),
        help="Folder where plots and summaries will be written",
    )
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])
    parser.add_argument("-lsa_components", type=int, default=250)
    parser.add_argument("-tfidf_weight", type=float, default=0.5)
    parser.add_argument("-random_state", type=int, default=42)
    parser.add_argument("-max_df", type=float, default=0.95)
    parser.add_argument("-min_df", type=int, default=2)
    parser.add_argument("-tfidf_norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("-ngram_max", type=int, default=1, choices=[1, 2])
    parser.add_argument("-disable_sublinear_tf", action="store_true")
    parser.add_argument("-custom", action="store_true")
    return parser


if __name__ == "__main__":
    start_time = time.time()
    args = build_parser().parse_args()
    if args.tfidf_norm == "none":
        args.tfidf_norm = None
    engine = HybridSearchEngine(args)

    if args.custom:
        engine.handle_custom_query()
    else:
        engine.evaluate_dataset()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
