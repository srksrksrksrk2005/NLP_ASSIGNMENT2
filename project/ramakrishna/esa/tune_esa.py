import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CURRENT_DIR = Path(__file__).resolve().parent
RAMAKRISHNA_ROOT = CURRENT_DIR.parent
ASSIGNMENT_ROOT = CURRENT_DIR.parents[2]
for path in [str(ASSIGNMENT_ROOT), str(RAMAKRISHNA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from esa_experiment import ESAExperiment
from esa_retrieval import ESARetrieval


def build_parser():
    parser = argparse.ArgumentParser(description="Sweep ESA hyperparameters for Cranfield")
    parser.add_argument(
        "-dataset",
        default=str(ASSIGNMENT_ROOT / "cranfield"),
        help="Path to the Cranfield dataset folder",
    )
    parser.add_argument(
        "-out_folder",
        default=str(CURRENT_DIR / "output_tuning"),
        help="Folder where tuning artifacts will be written",
    )
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])
    parser.add_argument("-random_state", type=int, default=42)
    return parser


def _make_namespace(base_args, **overrides):
    payload = {
        "dataset": base_args.dataset,
        "out_folder": base_args.out_folder,
        "segmenter": base_args.segmenter,
        "tokenizer": base_args.tokenizer,
        "lsa_components": 250,
        "hybrid_lsa_components": 240,
        "tfidf_weight": 0.2,
        "esa_top_concepts": 25,
        "esa_min_similarity": 0.0,
        "random_state": base_args.random_state,
        "max_df": 0.9,
        "min_df": 1,
        "tfidf_norm": "l2",
        "ngram_max": 2,
        "disable_sublinear_tf": False,
        "custom": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def evaluate_esa_config(experiment, processed_docs, doc_ids, processed_queries, qrels, query_ids, config):
    retriever = ESARetrieval(
        top_concepts=config["esa_top_concepts"],
        min_similarity=config["esa_min_similarity"],
        random_state=config["random_state"],
        sublinear_tf=config["sublinear_tf"],
        max_df=config["max_df"],
        min_df=config["min_df"],
        norm=config["tfidf_norm"],
        ngram_range=(1, config["ngram_max"]),
    )
    retriever.build_index(processed_docs, doc_ids)
    ranked_docs = retriever.rank(processed_queries)
    metrics = experiment.evaluate_rankings(ranked_docs, query_ids, qrels)
    return retriever, ranked_docs, metrics


def run_tuning(base_args):
    output_dir = Path(base_args.out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = ESAExperiment(_make_namespace(base_args))
    queries_json, docs_json, qrels = experiment.load_dataset()
    query_ids = [item["query number"] for item in queries_json]
    queries = [item["query"] for item in queries_json]
    doc_ids = [item["id"] for item in docs_json]
    docs = [item["body"] for item in docs_json]

    processed_queries = experiment.preprocess_texts(queries)
    processed_docs = experiment.preprocess_texts(docs)

    vectorizer_grid = list(
        itertools.product(
            [True, False],
            [0.9, 0.95, 1.0],
            [1, 2],
            [1, 2],
        )
    )
    concept_grid = list(
        itertools.product(
            [25, 50, 75, 100, 150, 200, 300],
            [0.0, 0.02, 0.05],
        )
    )

    results = []
    best_entry = None

    print("Starting ESA tuning sweep")
    print(f"Vectorizer grid size: {len(vectorizer_grid)}")
    print(f"Concept grid size: {len(concept_grid)}")

    for sublinear_tf, max_df, min_df, ngram_max in vectorizer_grid:
        config = {
            "sublinear_tf": sublinear_tf,
            "max_df": max_df,
            "min_df": min_df,
            "ngram_max": ngram_max,
            "tfidf_norm": "l2",
            "esa_top_concepts": 100,
            "esa_min_similarity": 0.0,
            "random_state": base_args.random_state,
        }
        print(
            "Testing vectorizer config:",
            f"sublinear_tf={sublinear_tf}, max_df={max_df}, min_df={min_df}, ngram_max={ngram_max}",
        )
        retriever, _, metrics = evaluate_esa_config(
            experiment,
            processed_docs,
            doc_ids,
            processed_queries,
            qrels,
            query_ids,
            config,
        )
        entry = {
            **config,
            "stage": "vectorizer",
            "actual_concepts": retriever.actual_concepts,
            "avg_active_concepts": retriever.avg_active_concepts,
            "concept_density": retriever.concept_density,
            "precision@10": metrics["precision"][-1],
            "recall@10": metrics["recall"][-1],
            "fscore@10": metrics["fscore"][-1],
            "map@10": metrics["map"][-1],
            "ndcg@10": metrics["ndcg"][-1],
            "mrr@10": metrics["mrr"][-1],
        }
        results.append(entry)

    vectorizer_results = [row for row in results if row["stage"] == "vectorizer"]
    vectorizer_results.sort(
        key=lambda row: (row["map@10"], row["ndcg@10"], row["mrr@10"]),
        reverse=True,
    )
    best_vectorizer = vectorizer_results[0]
    print("Best vectorizer config:", best_vectorizer)

    for top_concepts, min_similarity in concept_grid:
        config = {
            "sublinear_tf": bool(best_vectorizer["sublinear_tf"]),
            "max_df": float(best_vectorizer["max_df"]),
            "min_df": int(best_vectorizer["min_df"]),
            "ngram_max": int(best_vectorizer["ngram_max"]),
            "tfidf_norm": "l2",
            "esa_top_concepts": top_concepts,
            "esa_min_similarity": min_similarity,
            "random_state": base_args.random_state,
        }
        print(
            "Testing concept config:",
            f"top_concepts={top_concepts}, min_similarity={min_similarity}",
        )
        retriever, _, metrics = evaluate_esa_config(
            experiment,
            processed_docs,
            doc_ids,
            processed_queries,
            qrels,
            query_ids,
            config,
        )
        entry = {
            **config,
            "stage": "concept",
            "actual_concepts": retriever.actual_concepts,
            "avg_active_concepts": retriever.avg_active_concepts,
            "concept_density": retriever.concept_density,
            "precision@10": metrics["precision"][-1],
            "recall@10": metrics["recall"][-1],
            "fscore@10": metrics["fscore"][-1],
            "map@10": metrics["map"][-1],
            "ndcg@10": metrics["ndcg"][-1],
            "mrr@10": metrics["mrr"][-1],
        }
        results.append(entry)

    concept_results = [row for row in results if row["stage"] == "concept"]
    concept_results.sort(
        key=lambda row: (row["map@10"], row["ndcg@10"], row["mrr@10"]),
        reverse=True,
    )
    best_entry = concept_results[0]

    results_path = output_dir / "esa_tuning_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "stage",
            "sublinear_tf",
            "max_df",
            "min_df",
            "ngram_max",
            "tfidf_norm",
            "esa_top_concepts",
            "esa_min_similarity",
            "random_state",
            "actual_concepts",
            "avg_active_concepts",
            "concept_density",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_config = {
        "sublinear_tf": bool(best_entry["sublinear_tf"]),
        "max_df": float(best_entry["max_df"]),
        "min_df": int(best_entry["min_df"]),
        "ngram_max": int(best_entry["ngram_max"]),
        "tfidf_norm": "l2",
        "esa_top_concepts": int(best_entry["esa_top_concepts"]),
        "esa_min_similarity": float(best_entry["esa_min_similarity"]),
        "random_state": base_args.random_state,
        "map@10": float(best_entry["map@10"]),
        "ndcg@10": float(best_entry["ndcg@10"]),
        "mrr@10": float(best_entry["mrr@10"]),
        "precision@10": float(best_entry["precision@10"]),
        "recall@10": float(best_entry["recall@10"]),
    }

    best_path = output_dir / "esa_best_config.json"
    best_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print("\nBest ESA configuration")
    print(best_config)
    print("Tuning results written to:", results_path)
    print("Best config written to:", best_path)

    return best_config, results_path, best_path


def main():
    start_time = time.time()
    args = build_parser().parse_args()
    run_tuning(args)
    end_time = time.time()
    print(f"Total tuning time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
