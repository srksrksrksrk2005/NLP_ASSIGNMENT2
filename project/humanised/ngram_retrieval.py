"""Run the original n-gram experiment and keep the outputs inside `humanised/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, run_script


ORIGINAL_SCRIPT = VENDOR_DIR / "ritisha" / "ngram" / "run_ngram.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_ngram"


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised n-gram wrapper")
    parser.add_argument("--dataset", default=str(CRANFIELD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--segmenter", default="punkt", choices=["punkt", "naive"])
    parser.add_argument("--tokenizer", default="ptb", choices=["ptb", "naive"])
    parser.add_argument("--n-values", default="2,3,4,5,6,7,8")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        ORIGINAL_SCRIPT,
        [
            "--dataset",
            str(Path(args.dataset)),
            "--out-folder",
            str(output_dir),
            "--segmenter",
            args.segmenter,
            "--tokenizer",
            args.tokenizer,
            "--n-values",
            args.n_values,
        ],
    )
    print("N-gram outputs written to:", output_dir)


if __name__ == "__main__":
    main()
