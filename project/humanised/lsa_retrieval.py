"""Run the original LSA experiment and keep the outputs inside `humanised/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, k10_row, load_json, metric_rows, run_script, write_csv


ORIGINAL_SCRIPT = VENDOR_DIR / "ramakrishna" / "lsa" / "run_lsa.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_lsa"


def build_tables(output_dir: Path) -> None:
    summary = load_json(output_dir / "lsa_summary.json")
    metrics_by_k = {
        "precision": summary["precision"],
        "recall": summary["recall"],
        "fscore": summary["fscore"],
        "map": summary["map"],
        "ndcg": summary["ndcg"],
        "mrr": summary["mrr"],
    }

    write_csv(
        output_dir / "lsa_metrics.csv",
        metric_rows(metrics_by_k, "lsa"),
        ["method", "k", "precision", "recall", "fscore", "map", "ndcg", "mrr"],
    )

    write_csv(
        output_dir / "summary_k10.csv",
        [
            {
                **k10_row("lsa", metrics_by_k),
                "lsa_components_requested": summary["lsa_components_requested"],
                "lsa_components_used": summary["lsa_components_used"],
                "explained_variance_ratio_sum": summary["explained_variance_ratio_sum"],
            }
        ],
        [
            "method",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
            "lsa_components_requested",
            "lsa_components_used",
            "explained_variance_ratio_sum",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised LSA wrapper")
    parser.add_argument("--dataset", default=str(CRANFIELD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lsa-components", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--tfidf-norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("--ngram-max", type=int, default=1, choices=[1, 2])
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
    print("LSA outputs written to:", output_dir)


if __name__ == "__main__":
    main()
