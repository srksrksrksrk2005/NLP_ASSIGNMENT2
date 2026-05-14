"""Run the original SIF transfer experiment with a clean entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent / "vendor" / "chandan" / "sif_embeddings" / "run_transfer_experiments.py"
    output_dir = Path(__file__).resolve().parent / "output_sif_transfer"
    command = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(output_dir),
        "--backbones",
        "tfidf_term,lsa_term,esa_term",
        "--a-values",
        "0.01,0.001,0.0001",
        "--n-components",
        "0,1,2",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
