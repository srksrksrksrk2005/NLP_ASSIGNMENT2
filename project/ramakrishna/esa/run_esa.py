import argparse
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
RAMAKRISHNA_ROOT = CURRENT_DIR.parent
ASSIGNMENT_ROOT = CURRENT_DIR.parents[2]
for path in [str(ASSIGNMENT_ROOT), str(RAMAKRISHNA_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from esa_experiment import ESAExperiment


def build_parser():
    parser = argparse.ArgumentParser(description="ESA retrieval for Cranfield")
    parser.add_argument(
        "-dataset",
        default=str(ASSIGNMENT_ROOT / "cranfield"),
        help="Path to the Cranfield dataset folder",
    )
    parser.add_argument(
        "-out_folder",
        default=str(CURRENT_DIR / "output_esa"),
        help="Folder where plots and summaries will be written",
    )
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])

    parser.add_argument("-lsa_components", type=int, default=250)
    parser.add_argument("-hybrid_lsa_components", type=int, default=240)
    parser.add_argument("-tfidf_weight", type=float, default=0.2)
    parser.add_argument("-esa_top_concepts", type=int, default=25)
    parser.add_argument("-esa_min_similarity", type=float, default=0.0)
    parser.add_argument("-random_state", type=int, default=42)

    parser.add_argument("-max_df", type=float, default=0.9)
    parser.add_argument("-min_df", type=int, default=1)
    parser.add_argument("-tfidf_norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("-ngram_max", type=int, default=2, choices=[1, 2])
    parser.add_argument("-disable_sublinear_tf", action="store_true")
    parser.add_argument("-custom", action="store_true")
    return parser


def main():
    start_time = time.time()
    args = build_parser().parse_args()
    if args.tfidf_norm == "none":
        args.tfidf_norm = None

    experiment = ESAExperiment(args)

    if args.custom:
        _, docs_json, _ = experiment.load_dataset()
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]
        processed_docs = experiment.preprocess_texts(docs)
        query = input("Enter query below\n")
        processed_query = experiment.preprocess_texts([query])
        ranked_docs, _ = experiment.rank_esa(processed_docs, doc_ids, processed_query)
        ranked_docs = ranked_docs[0]

        print("\nTop five document IDs:")
        for doc_id in ranked_docs[:5]:
            print(doc_id)
    else:
        results = experiment.run_full_evaluation()
        print("ESA comparison plots saved in:", args.out_folder)
        print("K=10 summary")
        for method_key in ["tfidf", "lsa", "hybrid", "esa"]:
            metrics = results["metrics"][method_key]
            print(f"\n{method_key.upper()}")
            print(f"Precision@10: {metrics['precision'][-1]:.4f}")
            print(f"Recall@10: {metrics['recall'][-1]:.4f}")
            print(f"Fscore@10: {metrics['fscore'][-1]:.4f}")
            print(f"MAP@10: {metrics['map'][-1]:.4f}")
            print(f"nDCG@10: {metrics['ndcg'][-1]:.4f}")
            print(f"MRR@10: {metrics['mrr'][-1]:.4f}")
        print("Analysis written to:", results["analysis_path"])

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
