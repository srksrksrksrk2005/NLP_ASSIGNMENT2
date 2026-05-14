"""Run the original BM25 experiment and keep the outputs inside `humanised/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, k10_row, load_json, metric_rows, run_script, write_csv


ORIGINAL_SCRIPT = VENDOR_DIR / "ritisha" / "bm25" / "run_bm25.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_bm25"


def build_tables(output_dir: Path) -> None:
    results = load_json(output_dir / "bm25_vs_tfidf_results.json")

    curve_rows = metric_rows(results["tfidf"], "tfidf") + metric_rows(results["bm25"], "bm25")
    write_csv(
        output_dir / "comparison_metrics.csv",
        curve_rows,
        ["method", "k", "precision", "recall", "fscore", "map", "ndcg", "mrr"],
    )

    summary_rows = [
        k10_row("tfidf", results["tfidf"]),
        k10_row("bm25", results["bm25"]),
    ]
    write_csv(
        output_dir / "summary_k10.csv",
        summary_rows,
        ["method", "precision@10", "recall@10", "fscore@10", "map@10", "ndcg@10", "mrr@10"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised BM25 wrapper")
    parser.add_argument("--dataset", default=str(CRANFIELD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--segmenter", default="punkt", choices=["punkt", "naive"])
    parser.add_argument("--tokenizer", default="ptb", choices=["ptb", "naive"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        ORIGINAL_SCRIPT,
        [
            "-dataset",
            str(Path(args.dataset)),
            "-out_folder",
            str(output_dir),
            "-segmenter",
            args.segmenter,
            "-tokenizer",
            args.tokenizer,
        ],
    )
    build_tables(output_dir)
    print("BM25 outputs written to:", output_dir)


if __name__ == "__main__":
    main()
