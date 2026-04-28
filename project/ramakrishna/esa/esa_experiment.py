from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

CURRENT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = CURRENT_DIR.parents[2]
LSA_ROOT = CURRENT_DIR.parent / "lsa"
for path in [str(ASSIGNMENT_ROOT), str(LSA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation import Evaluation
from hybrid_retrieval import HybridRetrieval
from informationRetrieval import InformationRetrieval
from inflectionReduction import InflectionReduction
from lsa_retrieval import LSARetrieval
from sentenceSegmentation import SentenceSegmentation
from stopwordRemoval import StopwordRemoval
from tokenization import Tokenization

from esa_retrieval import ESARetrieval


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("fscore", "F-Score"),
    ("map", "MAP"),
    ("ndcg", "nDCG"),
    ("mrr", "MRR"),
]

METHOD_ORDER = ["tfidf", "lsa", "hybrid", "esa"]
COMPARE_ORDER = ["tfidf", "esa"]

METHOD_LABELS = {
    "tfidf": "Standard TF-IDF",
    "lsa": "Tuned LSA",
    "hybrid": "Tuned Hybrid (TF-IDF + LSA)",
    "esa": "Tuned ESA",
}

METHOD_STYLES = {
    "tfidf": {"color": "black", "linestyle": "--", "marker": "o"},
    "lsa": {"color": "tab:blue", "linestyle": "-", "marker": "s"},
    "hybrid": {"color": "tab:green", "linestyle": "-", "marker": "^"},
    "esa": {"color": "tab:red", "linestyle": "-", "marker": "D"},
}


class ESAExperiment:
    def __init__(self, args):
        self.args = args
        self.out_folder = Path(self.args.out_folder)
        self.out_folder.mkdir(parents=True, exist_ok=True)

        self.tokenizer = Tokenization()
        self.sentence_segmenter = SentenceSegmentation()
        self.inflection_reducer = InflectionReduction()
        self.stopword_remover = StopwordRemoval()
        self.evaluator = Evaluation()
        self._concept_cache = None

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

    def load_concept_corpus(self):
        if self.args.concept_source == "cranfield":
            return None, None, "cranfield"

        if self._concept_cache is not None:
            return self._concept_cache

        if self.args.concept_source == "20news-technical":
            categories = [
                "sci.space",
                "sci.electronics",
                "sci.med",
                "comp.graphics",
                "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware",
                "comp.windows.x",
            ]
            dataset = fetch_20newsgroups(
                subset="train",
                categories=categories,
                remove=("headers", "footers", "quotes"),
                shuffle=True,
                random_state=self.args.random_state,
            )
            concept_texts = dataset.data
            if self.args.concept_limit and self.args.concept_limit > 0:
                concept_texts = concept_texts[: self.args.concept_limit]
            concept_ids = [f"20news-{index + 1}" for index in range(len(concept_texts))]
            self._concept_cache = (concept_texts, concept_ids, "20news-technical")
            return self._concept_cache

        raise ValueError(
            f"Unsupported concept source: {self.args.concept_source}. "
            "Use 'cranfield' or '20news-technical'."
        )

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
        return tfidf_retriever.rank(processed_queries), tfidf_retriever

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
        return lsa_retriever.rank(processed_queries), lsa_retriever

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
        return hybrid_retriever.rank(processed_queries), hybrid_retriever

    def rank_esa(
        self,
        processed_docs,
        doc_ids,
        processed_queries,
        concept_docs=None,
        concept_ids=None,
        concept_source=None,
        prebuilt_index_path=None,
    ):
        esa_retriever = ESARetrieval(
            top_concepts=self.args.esa_top_concepts,
            min_similarity=self.args.esa_min_similarity,
            random_state=self.args.random_state,
            sublinear_tf=not self.args.disable_sublinear_tf,
            max_df=self.args.max_df,
            min_df=self.args.min_df,
            norm=self.args.tfidf_norm,
            ngram_range=(1, self.args.ngram_max),
        )
        esa_retriever.build_index(
            processed_docs,
            doc_ids,
            concept_docs=concept_docs,
            concept_ids=concept_ids,
            concept_source=concept_source,
            prebuilt_index_path=prebuilt_index_path,
        )
        return esa_retriever.rank(processed_queries), esa_retriever

    @staticmethod
    def _first_relevant_rank(ranked_docs, relevant_set):
        for index, doc_id in enumerate(ranked_docs, start=1):
            if str(doc_id) in relevant_set:
                return index
        return None

    @staticmethod
    def _count_relevant(ranked_docs, relevant_set, k=5):
        return sum(1 for doc_id in ranked_docs[:k] if str(doc_id) in relevant_set)

    def build_query_rows(self, rankings, query_ids, queries, qrels, k=10):
        rows = []
        for index, query_id in enumerate(query_ids):
            relevant_docs = self.evaluator._get_true_doc_ids(qrels, query_id)
            relevant_set = {str(doc_id) for doc_id in relevant_docs}
            relevant_with_position = self.evaluator._get_true_doc_ids_with_position(
                qrels, query_id
            )

            row = {
                "query_id": str(query_id),
                "query": queries[index],
                "methods": {},
            }

            for method_key, ranked_docs in rankings.items():
                ranked = ranked_docs[index]
                row["methods"][method_key] = {
                    "ap10": self.evaluator.queryAveragePrecision(
                        ranked, query_id, relevant_with_position, k
                    ),
                    "p5": self.evaluator.queryPrecision(
                        ranked, query_id, relevant_docs, 5
                    ),
                    "first_relevant_rank": self._first_relevant_rank(
                        ranked, relevant_set
                    ),
                    "top5_relevant": self._count_relevant(ranked, relevant_set, 5),
                    "top5_docs": [str(doc_id) for doc_id in ranked[:5]],
                }
            rows.append(row)
        return rows

    @staticmethod
    def _format_rank(rank):
        return str(rank) if rank is not None else "-"

    def _format_method_cell(self, stats):
        return (
            f"{stats['ap10']:.3f} / {self._format_rank(stats['first_relevant_rank'])} / "
            f"{stats['top5_relevant']}"
        )

    def save_overlay_plots(self, all_metrics, methods, file_prefix, title):
        k_values = list(range(1, 11))

        figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for axis, (metric_key, metric_name) in zip(axes.flat, METRICS):
            for method_key in methods:
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
        figure.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
        figure.suptitle(title, y=0.98)
        figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        figure.savefig(self.out_folder / f"{file_prefix}_all_metrics.png")
        plt.close(figure)

    def save_per_metric_plots(self, all_metrics, methods, file_prefix, title_prefix):
        k_values = list(range(1, 11))
        for metric_key, metric_name in METRICS:
            plt.figure(figsize=(9, 5))
            for method_key in methods:
                style = METHOD_STYLES[method_key]
                plt.plot(
                    k_values,
                    all_metrics[method_key][metric_key],
                    label=METHOD_LABELS[method_key],
                    linewidth=2,
                    markersize=5,
                    **style,
                )
            plt.title(f"{title_prefix} - {metric_name}")
            plt.xlabel("k")
            plt.ylabel("Score")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_folder / f"{file_prefix}_{metric_key}.png")
            plt.close()

    def save_summary(self, all_metrics, retrievers):
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
                    "used_components": retrievers["lsa"].actual_components,
                    "explained_variance_ratio_sum": retrievers["lsa"].explained_variance,
                },
                "hybrid": {
                    "requested_components": self.args.hybrid_lsa_components,
                    "used_components": retrievers["hybrid"].actual_components,
                    "tfidf_weight": self.args.tfidf_weight,
                    "explained_variance_ratio_sum": retrievers["hybrid"].explained_variance,
                },
                "esa": {
                    "requested_top_concepts": self.args.esa_top_concepts,
                    "used_top_concepts": retrievers["esa"].actual_concepts,
                    "min_similarity": self.args.esa_min_similarity,
                    "concept_source": retrievers["esa"].concept_source,
                    "avg_active_concepts": retrievers["esa"].avg_active_concepts,
                    "concept_density": retrievers["esa"].concept_density,
                    "vocab_size": retrievers["esa"].vocab_size,
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

        summary_path = self.out_folder / "comparison_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)
        return summary

    def save_esa_summary(self, esa_metrics, esa_retriever):
        summary = {
            "esa_top_concepts_requested": self.args.esa_top_concepts,
            "esa_top_concepts_used": esa_retriever.actual_concepts,
            "esa_min_similarity": self.args.esa_min_similarity,
            "concept_source": esa_retriever.concept_source,
            "avg_active_concepts": esa_retriever.avg_active_concepts,
            "concept_density": esa_retriever.concept_density,
            "vectorizer": {
                "sublinear_tf": esa_retriever.sublinear_tf,
                "max_df": esa_retriever.max_df,
                "min_df": esa_retriever.min_df,
                "norm": esa_retriever.norm,
                "ngram_range": list(esa_retriever.ngram_range),
                "vocab_size": esa_retriever.vocab_size,
            },
            "precision": esa_metrics["precision"],
            "recall": esa_metrics["recall"],
            "fscore": esa_metrics["fscore"],
            "map": esa_metrics["map"],
            "ndcg": esa_metrics["ndcg"],
            "mrr": esa_metrics["mrr"],
        }

        summary_path = self.out_folder / "esa_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)
        return summary

    @staticmethod
    def _rank_sort_gain(row, primary, challengers):
        baseline = max(row["methods"][method]["ap10"] for method in challengers)
        return row["methods"][primary]["ap10"] - baseline

    def select_case_rows(self, rows, primary="esa", challengers=("tfidf", "lsa"), limit=5):
        candidates = [
            row
            for row in rows
            if all(
                row["methods"][primary]["ap10"] >= row["methods"][challenger]["ap10"]
                for challenger in challengers
            )
        ]
        candidates.sort(
            key=lambda row: self._rank_sort_gain(row, primary, challengers),
            reverse=True,
        )
        return candidates[:limit]

    def select_hybrid_favorites(self, rows, primary="hybrid", secondary="esa", limit=5):
        candidates = [
            row
            for row in rows
            if row["methods"][primary]["ap10"] >= row["methods"][secondary]["ap10"]
        ]
        candidates.sort(
            key=lambda row: row["methods"][primary]["ap10"] - row["methods"][secondary]["ap10"],
            reverse=True,
        )
        return candidates[:limit]

    def write_analysis_markdown(self, summary, rows, tfidf_metrics, lsa_metrics, hybrid_metrics, esa_metrics):
        esa_wins = self.select_case_rows(rows, primary="esa", challengers=("tfidf", "lsa"), limit=5)
        hybrid_wins = self.select_hybrid_favorites(rows, primary="hybrid", secondary="esa", limit=4)

        lines = []
        lines.append("# ESA Analysis Notes")
        lines.append("")
        lines.append("## ESA Setup")
        lines.append(
            "We used a Cranfield document-concept ESA variant: each document is treated as an explicit concept, "
            "queries are projected into that concept space through TF-IDF similarity, and the top concepts are "
            "retained before ranking."
        )
        lines.append("")
        lines.append("## Best ESA Configuration")
        lines.append(f"- `esa_top_concepts = {self.args.esa_top_concepts}`")
        lines.append(f"- `esa_min_similarity = {self.args.esa_min_similarity}`")
        lines.append(f"- `sublinear_tf = {not self.args.disable_sublinear_tf}`")
        lines.append(f"- `max_df = {self.args.max_df}`")
        lines.append(f"- `min_df = {self.args.min_df}`")
        lines.append(f"- `ngram_range = (1, {self.args.ngram_max})`")
        lines.append("")
        lines.append("## Comparison at k=10")
        lines.append("")
        lines.append("| Method | Precision | Recall | F-score | MAP | nDCG | MRR |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for method_key, metrics in [
            ("tfidf", tfidf_metrics),
            ("lsa", lsa_metrics),
            ("hybrid", hybrid_metrics),
            ("esa", esa_metrics),
        ]:
            lines.append(
                "| {label} | {p:.4f} | {r:.4f} | {f:.4f} | {m:.4f} | {n:.4f} | {rr:.4f} |".format(
                    label=METHOD_LABELS[method_key],
                    p=metrics["precision"][-1],
                    r=metrics["recall"][-1],
                    f=metrics["fscore"][-1],
                    m=metrics["map"][-1],
                    n=metrics["ndcg"][-1],
                    rr=metrics["mrr"][-1],
                )
            )
        lines.append("")
        lines.append("## Query-level case studies where ESA beats TF-IDF and LSA")
        lines.append("")
        lines.append(
            "Cell format: `AP@10 / first relevant rank / relevant docs in top 5`. The same query is shown across all methods."
        )
        lines.append("")
        lines.append("| QID | Query | TF-IDF | LSA | Hybrid | ESA | Note |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in esa_wins:
            note = "ESA surfaces the strongest semantic match"
            if row["methods"]["hybrid"]["ap10"] > row["methods"]["esa"]["ap10"]:
                note = "Hybrid still edges ESA on AP, but ESA beats TF-IDF and LSA"
            lines.append(
                "| {qid} | {query} | {tfidf} | {lsa} | {hybrid} | {esa} | {note} |".format(
                    qid=row["query_id"],
                    query=row["query"].replace("|", "/"),
                    tfidf=self._format_method_cell(row["methods"]["tfidf"]),
                    lsa=self._format_method_cell(row["methods"]["lsa"]),
                    hybrid=self._format_method_cell(row["methods"]["hybrid"]),
                    esa=self._format_method_cell(row["methods"]["esa"]),
                    note=note,
                )
            )
        lines.append("")
        lines.append("## Queries where hybrid still beats ESA")
        lines.append("")
        lines.append("| QID | Query | Hybrid | ESA | Comment |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in hybrid_wins:
            comment = "Hybrid keeps the best balance of lexical and latent evidence"
            lines.append(
                "| {qid} | {query} | {hybrid} | {esa} | {comment} |".format(
                    qid=row["query_id"],
                    query=row["query"].replace("|", "/"),
                    hybrid=self._format_method_cell(row["methods"]["hybrid"]),
                    esa=self._format_method_cell(row["methods"]["esa"]),
                    comment=comment,
                )
            )
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append(
            "ESA is strongest when the relevant documents are conceptually close but do not share the same surface wording."
        )
        lines.append(
            "The hybrid model still wins when direct lexical evidence matters enough that the TF-IDF branch can correct ESA drift."
        )
        lines.append("")
        lines.append("## Files written")
        lines.append(f"- `{self.out_folder / 'esa_vs_tfidf_overlay_all_metrics.png'}`")
        lines.append(f"- `{self.out_folder / 'esa_vs_all_methods_overlay_all_metrics.png'}`")
        lines.append(f"- `{self.out_folder / 'esa_summary.json'}`")
        lines.append(f"- `{self.out_folder / 'comparison_summary.json'}`")

        analysis_text = "\n".join(lines)
        analysis_path = self.out_folder.parent / "analysis.md"
        analysis_path.write_text(analysis_text, encoding="utf-8")
        (self.out_folder / "analysis.md").write_text(analysis_text, encoding="utf-8")
        return analysis_path

    def run_full_evaluation(self):
        queries_json, docs_json, qrels = self.load_dataset()
        if self.args.prebuilt_index_path:
            concept_docs = concept_ids = concept_source = None
        else:
            concept_docs, concept_ids, concept_source = self.load_concept_corpus()
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processed_queries = self.preprocess_texts(queries)
        processed_docs = self.preprocess_texts(docs)

        ranked_tfidf, tfidf_retriever = self.rank_tfidf(
            processed_docs, doc_ids, processed_queries
        )
        ranked_lsa, lsa_retriever = self.rank_lsa(processed_docs, doc_ids, processed_queries)
        ranked_hybrid, hybrid_retriever = self.rank_hybrid(
            processed_docs, doc_ids, processed_queries
        )
        ranked_esa, esa_retriever = self.rank_esa(
            processed_docs,
            doc_ids,
            processed_queries,
            concept_docs=concept_docs,
            concept_ids=concept_ids,
            concept_source=concept_source,
            prebuilt_index_path=self.args.prebuilt_index_path,
        )

        rankings = {
            "tfidf": ranked_tfidf,
            "lsa": ranked_lsa,
            "hybrid": ranked_hybrid,
            "esa": ranked_esa,
        }
        retrievers = {
            "tfidf": tfidf_retriever,
            "lsa": lsa_retriever,
            "hybrid": hybrid_retriever,
            "esa": esa_retriever,
        }

        all_metrics = {
            method_key: self.evaluate_rankings(rankings[method_key], query_ids, qrels)
            for method_key in METHOD_ORDER
        }

        self.save_overlay_plots(
            all_metrics,
            COMPARE_ORDER,
            "esa_vs_tfidf_overlay",
            "Cranfield Comparison: ESA vs Standard TF-IDF",
        )
        self.save_overlay_plots(
            all_metrics,
            METHOD_ORDER,
            "esa_vs_all_methods_overlay",
            "Cranfield Comparison: TF-IDF vs LSA vs Hybrid vs ESA",
        )
        self.save_per_metric_plots(
            all_metrics,
            COMPARE_ORDER,
            "esa_vs_tfidf_overlay",
            "ESA vs Standard TF-IDF",
        )

        summary = self.save_summary(all_metrics, retrievers)
        esa_summary = self.save_esa_summary(all_metrics["esa"], esa_retriever)
        query_rows = self.build_query_rows(rankings, query_ids, queries, qrels)
        analysis_path = self.write_analysis_markdown(
            summary,
            query_rows,
            all_metrics["tfidf"],
            all_metrics["lsa"],
            all_metrics["hybrid"],
            all_metrics["esa"],
        )

        return {
            "summary": summary,
            "esa_summary": esa_summary,
            "analysis_path": analysis_path,
            "rankings": rankings,
            "retrievers": retrievers,
            "metrics": all_metrics,
            "query_rows": query_rows,
        }
