"""Rebuild the query expansion overlay plot from the saved summary output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output_query_expansion"
SUMMARY_CACHE = Path(__file__).resolve().with_name(".query_expansion_summary.json")
SUMMARY_PATH = SUMMARY_CACHE
SOURCE_SUMMARY = OUTPUT_DIR / "summary.json"


def _prune_output_dir(keep_files: set[str]) -> None:
    for child in OUTPUT_DIR.iterdir():
        if child.name in keep_files:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    source_summary = SUMMARY_PATH if SUMMARY_PATH.exists() else SOURCE_SUMMARY
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if not SUMMARY_CACHE.exists():
        SUMMARY_CACHE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ks = list(range(1, 11))
    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    axes = axes.flatten()
    methods = [name for name in summary.keys() if name != "baseline_tfidf"]

    for axis, metric_name in zip(axes, metric_names):
        for method_name in methods:
            axis.plot(
                ks,
                [summary[method_name][str(k)][metric_name] for k in ks],
                linewidth=2.2,
                label=method_name,
            )
        axis.plot(
            ks,
            [summary["baseline_tfidf"][str(k)][metric_name] for k in ks],
            color="black",
            linestyle="--",
            linewidth=3.0,
            label="baseline_tfidf",
        )
        axis.set_title(metric_name.upper() if metric_name != "fscore" else "F-score")
        axis.set_xlabel("k")
        axis.set_ylabel("Score")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=12)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, frameon=False, fontsize=13)
    fig.suptitle("Query Expansion Methods vs Baseline", y=0.995, fontsize=22)
    fig.subplots_adjust(top=0.87, hspace=0.38, wspace=0.22)
    fig.savefig(OUTPUT_DIR / "eval_overlay.png", dpi=260)
    plt.close(fig)

    _prune_output_dir({"eval_overlay.png"})


if __name__ == "__main__":
    main()
