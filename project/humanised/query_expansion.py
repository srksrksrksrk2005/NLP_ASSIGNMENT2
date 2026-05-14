"""Run the local, vendored query-expansion experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, run_script


ORIGINAL_SCRIPT = VENDOR_DIR / "Nikhil" / "query_expansion" / "run_experiments.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_query_expansion"


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised query-expansion runner")
    parser.add_argument("--dataset", default=str(CRANFIELD_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--methods",
        default="baseline_tfidf,wordnet,embedding_tfidf,embedding_lsa,embedding_esa,embedding_word2vec",
        help="Comma-separated methods to evaluate in the local vendor runner.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        ORIGINAL_SCRIPT,
        [
            "--dataset",
            str(Path(args.dataset)),
            "--output",
            str(output_dir),
            "--methods",
            args.methods,
        ],
    )
    print("Query-expansion outputs written to:", output_dir)


if __name__ == "__main__":
    main()
