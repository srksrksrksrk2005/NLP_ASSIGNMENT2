#!/usr/bin/env python3
"""
Combine all Sudheer soft-cosine variants into one comparison plot/table.

Variants included:
1. soft cosine with TF-IDF-derived S
2. soft cosine with LSA-derived S
3. soft cosine with ESA-derived S
4. soft cosine with WordNet-derived S
5. soft cosine with Word2Vec-derived S
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_json, save_overlay_plot, write_json, write_summary_csv  # noqa: E402


def main() -> None:
    soft_cosine_sources_summary = load_json(SCRIPT_DIR / "soft_cosine_sources" / "output" / "summary.json")
    soft_cosine_word2vec_summary = load_json(SCRIPT_DIR / "soft_cosine_word2vec" / "output" / "summary.json")

    output_dir = SCRIPT_DIR / "soft_cosine_comparison" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = soft_cosine_word2vec_summary["methods"]["baseline_tfidf"]["metrics_by_k"]

    grouped_metrics = {
        "tfidf_s": soft_cosine_sources_summary["sources"]["tfidf"]["best_metrics_by_k"],
        "lsa_s": soft_cosine_sources_summary["sources"]["lsa"]["best_metrics_by_k"],
        "esa_s": soft_cosine_sources_summary["sources"]["esa"]["best_metrics_by_k"],
        "wordnet_s": soft_cosine_sources_summary["sources"]["wordnet"]["best_metrics_by_k"],
        "word2vec_s": soft_cosine_word2vec_summary["methods"]["soft_cosine_word2vec"]["metrics_by_k"],
    }

    method_labels = {
        "tfidf_s": "soft cosine (S from TF-IDF)",
        "lsa_s": "soft cosine (S from LSA)",
        "esa_s": "soft cosine (S from ESA)",
        "wordnet_s": "soft cosine (S from WordNet)",
        "word2vec_s": "soft cosine (S from Word2Vec)",
    }

    ordered_keys = ["tfidf_s", "lsa_s", "esa_s", "wordnet_s", "word2vec_s"]

    save_overlay_plot(
        output_dir / "all_soft_cosine_overlay.png",
        baseline_metrics,
        grouped_metrics,
        method_labels,
        ordered_keys,
        "Baseline vs All Soft Cosine Variants",
    )

    rows = [
        {
            "method": "soft cosine (S from TF-IDF)",
            **{f"{metric}@10": soft_cosine_sources_summary["sources"]["tfidf"]["best_k10"][metric] for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]},
            "runtime_seconds": soft_cosine_sources_summary["sources"]["tfidf"]["best_runtime_seconds"],
        },
        {
            "method": "soft cosine (S from LSA)",
            **{f"{metric}@10": soft_cosine_sources_summary["sources"]["lsa"]["best_k10"][metric] for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]},
            "runtime_seconds": soft_cosine_sources_summary["sources"]["lsa"]["best_runtime_seconds"],
        },
        {
            "method": "soft cosine (S from ESA)",
            **{f"{metric}@10": soft_cosine_sources_summary["sources"]["esa"]["best_k10"][metric] for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]},
            "runtime_seconds": soft_cosine_sources_summary["sources"]["esa"]["best_runtime_seconds"],
        },
        {
            "method": "soft cosine (S from WordNet)",
            **{f"{metric}@10": soft_cosine_sources_summary["sources"]["wordnet"]["best_k10"][metric] for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]},
            "runtime_seconds": soft_cosine_sources_summary["sources"]["wordnet"]["best_runtime_seconds"],
        },
        {
            "method": "soft cosine (S from Word2Vec)",
            **{f"{metric}@10": soft_cosine_word2vec_summary["methods"]["soft_cosine_word2vec"]["k10"][metric] for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]},
            "runtime_seconds": soft_cosine_word2vec_summary["methods"]["soft_cosine_word2vec"]["runtime_seconds"],
        },
    ]

    write_summary_csv(
        output_dir / "all_soft_cosine_summary_k10.csv",
        rows,
        fieldnames=[
            "method",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
            "runtime_seconds",
        ],
    )

    write_json(
        output_dir / "all_soft_cosine_summary.json",
        {
            "baseline_metrics": baseline_metrics,
            "methods": rows,
            "best_method_by_map": max(rows, key=lambda row: row["map@10"])["method"],
            "plot": str(output_dir / "all_soft_cosine_overlay.png"),
            "csv": str(output_dir / "all_soft_cosine_summary_k10.csv"),
        },
    )

    print("Wrote:", output_dir / "all_soft_cosine_overlay.png")
    print("Wrote:", output_dir / "all_soft_cosine_summary_k10.csv")


if __name__ == "__main__":
    main()
