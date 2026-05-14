"""Run the original ESA experiment and keep the outputs inside `humanised/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, k10_row, load_json, metric_rows, run_script, write_csv


ORIGINAL_SCRIPT = VENDOR_DIR / "ramakrishna" / "esa" / "run_esa.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_esa"


def build_tables(output_dir: Path) -> None:
    summary = load_json(output_dir / "esa_metrics_summary.json")
    rows = []
    summary_rows = []
    for method_name, metrics in summary.items():
        metrics_by_k = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fscore": metrics["fscore"],
            "map": metrics["map"],
            "ndcg": metrics["ndcg"],
            "mrr": metrics["mrr"],
        }
        rows.extend(metric_rows(metrics_by_k, method_name))
        summary_rows.append(k10_row(method_name, metrics_by_k))

    write_csv(
        output_dir / "esa_metrics.csv",
        rows,
        ["method", "k", "precision", "recall", "fscore", "map", "ndcg", "mrr"],
    )
    write_csv(
        output_dir / "summary_k10.csv",
        summary_rows,
        ["method", "precision@10", "recall@10", "fscore@10", "map@10", "ndcg@10", "mrr@10"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised ESA wrapper")
    parser.add_argument("--dataset", default=str(CRANFIELD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lsa-components", type=int, default=250)
    parser.add_argument("--hybrid-lsa-components", type=int, default=240)
    parser.add_argument("--tfidf-weight", type=float, default=0.2)
    parser.add_argument("--esa-top-concepts", type=int, default=25)
    parser.add_argument("--esa-min-similarity", type=float, default=0.0)
    parser.add_argument("--concept-source", default="cranfield", choices=["cranfield", "20news-technical"])
    parser.add_argument("--concept-limit", type=int, default=0)
    parser.add_argument("--prebuilt-index-path", default="")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--tfidf-norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2])
    parser.add_argument("--disable-sublinear-tf", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = [
        "-dataset",
        str(Path(args.dataset)),
        "-out_folder",
        str(output_dir),
        "-lsa_components",
        str(args.lsa_components),
        "-hybrid_lsa_components",
        str(args.hybrid_lsa_components),
        "-tfidf_weight",
        str(args.tfidf_weight),
        "-esa_top_concepts",
        str(args.esa_top_concepts),
        "-esa_min_similarity",
        str(args.esa_min_similarity),
        "-concept_source",
        args.concept_source,
        "-concept_limit",
        str(args.concept_limit),
        "-prebuilt_index_path",
        str(args.prebuilt_index_path),
        "-random_state",
        str(args.random_state),
        "-max_df",
        str(args.max_df),
        "-min_df",
        str(args.min_df),
        "-tfidf_norm",
        args.tfidf_norm,
        "-ngram_max",
        str(args.ngram_max),
    ]
    if args.disable_sublinear_tf:
        run_args.append("-disable_sublinear_tf")

    run_script(ORIGINAL_SCRIPT, run_args)
    build_tables(output_dir)
    print("ESA outputs written to:", output_dir)


if __name__ == "__main__":
    main()
