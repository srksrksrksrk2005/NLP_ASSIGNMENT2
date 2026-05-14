"""Run the original unordered local-context BoW experiment inside `humanised/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, run_script


ORIGINAL_SCRIPT = VENDOR_DIR / "chandan" / "unordered_local_context_bow" / "run_experiments.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_local_bow"
DEFAULT_PROCESSED_DOCS = VENDOR_DIR / "output" / "stopword_removed_docs.txt"
DEFAULT_PROCESSED_QUERIES = VENDOR_DIR / "output" / "stopword_removed_queries.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised local-context BoW wrapper")
    parser.add_argument("--dataset-dir", default=str(CRANFIELD_DIR))
    parser.add_argument("--processed-docs", default=str(DEFAULT_PROCESSED_DOCS))
    parser.add_argument("--processed-queries", default=str(DEFAULT_PROCESSED_QUERIES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        ORIGINAL_SCRIPT,
        [
            "--dataset-dir",
            str(Path(args.dataset_dir)),
            "--processed-docs",
            str(Path(args.processed_docs)),
            "--processed-queries",
            str(Path(args.processed_queries)),
            "--output-dir",
            str(output_dir),
        ],
    )
    print("Local-context BoW outputs written to:", output_dir)


if __name__ == "__main__":
    main()
