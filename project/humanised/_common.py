"""Shared helpers for the humanised experiment wrappers."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
ASSIGNMENT_ROOT = PROJECT_DIR.parent
VENDOR_DIR = SCRIPT_DIR / "vendor"
CRANFIELD_DIR = VENDOR_DIR / "cranfield"


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_series(metrics_by_k: Mapping[str, Sequence[float]], metric_name: str) -> Sequence[float]:
    aliases = {
        "precision": ["precision", "Precision"],
        "recall": ["recall", "Recall"],
        "fscore": ["fscore", "Fscore", "F-score", "F-Score"],
        "map": ["map", "MAP"],
        "ndcg": ["ndcg", "nDCG", "NDCG"],
        "mrr": ["mrr", "MRR"],
    }
    for candidate in aliases.get(metric_name, [metric_name, metric_name.upper(), metric_name.capitalize()]):
        if candidate in metrics_by_k:
            return metrics_by_k[candidate]
    raise KeyError(metric_name)


def metric_rows(metrics_by_k: Mapping[str, Sequence[float]], method_name: str) -> list[dict[str, object]]:
    rows = []
    for index in range(10):
        rows.append(
            {
                "method": method_name,
                "k": index + 1,
                "precision": _metric_series(metrics_by_k, "precision")[index],
                "recall": _metric_series(metrics_by_k, "recall")[index],
                "fscore": _metric_series(metrics_by_k, "fscore")[index],
                "map": _metric_series(metrics_by_k, "map")[index],
                "ndcg": _metric_series(metrics_by_k, "ndcg")[index],
                "mrr": _metric_series(metrics_by_k, "mrr")[index],
            }
        )
    return rows


def k10_row(method_name: str, metrics_by_k: Mapping[str, Sequence[float]], runtime_seconds: float | None = None) -> dict[str, object]:
    row = {
        "method": method_name,
        "precision@10": _metric_series(metrics_by_k, "precision")[9],
        "recall@10": _metric_series(metrics_by_k, "recall")[9],
        "fscore@10": _metric_series(metrics_by_k, "fscore")[9],
        "map@10": _metric_series(metrics_by_k, "map")[9],
        "ndcg@10": _metric_series(metrics_by_k, "ndcg")[9],
        "mrr@10": _metric_series(metrics_by_k, "mrr")[9],
    }
    if runtime_seconds is not None:
        row["runtime_seconds"] = runtime_seconds
    return row


def run_script(script_path: Path, args: Sequence[str]) -> None:
    command = [sys.executable, str(script_path), *args]
    subprocess.run(command, check=True, cwd=str(Path(script_path).parent))
