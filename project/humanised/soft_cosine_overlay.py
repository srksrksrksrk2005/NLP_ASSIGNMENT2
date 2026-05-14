"""Rebuild the soft cosine source overlay plots from the saved comparison summary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


LOCAL_SUMMARY = Path(__file__).resolve().parent / "output_soft_cosine" / "comparison_summary.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_soft_cosine"
SUMMARY_CACHE = Path(__file__).resolve().with_name(".soft_cosine_summary.json")


def _prune_output_dir(keep_files: set[str]) -> None:
    for child in OUTPUT_DIR.iterdir():
        if child.name in keep_files:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    summary_path = SUMMARY_CACHE if SUMMARY_CACHE.exists() else LOCAL_SUMMARY
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing soft cosine summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = summary["sources"]
    baseline = sources["tfidf"]["baseline"]["metrics_by_k"]
    baseline_k10 = sources["tfidf"]["baseline"]["k10"]
    ks = list(range(1, 11))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_CACHE.exists():
        SUMMARY_CACHE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    axes = axes.flatten()
    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]

    for axis, metric_name in zip(axes, metric_names):
        for source_name, source_payload in sources.items():
            axis.plot(
                ks,
                source_payload["best_metrics_by_k"][metric_name],
                linewidth=2.2,
                label=source_name,
            )
        axis.plot(
            ks,
            baseline[metric_name],
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
    fig.suptitle("Soft Cosine Sources vs Baseline", y=0.995, fontsize=22)
    fig.subplots_adjust(top=0.87, hspace=0.38, wspace=0.22)
    fig.savefig(OUTPUT_DIR / "eval_overlay.png", dpi=260)
    plt.close(fig)

    _prune_output_dir({"eval_overlay.png"})


if __name__ == "__main__":
    main()
