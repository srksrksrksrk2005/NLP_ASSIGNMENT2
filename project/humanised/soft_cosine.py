"""Run the original soft-cosine source comparison without WordNet."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from _common import CRANFIELD_DIR, SCRIPT_DIR, VENDOR_DIR, load_json, write_csv, write_json


ORIGINAL_DIR = VENDOR_DIR / "sudheer" / "soft_cosine_sources"
ORIGINAL_SCRIPT = ORIGINAL_DIR / "run_experiments.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_soft_cosine"
DEFAULT_PROCESSED_DOCS = VENDOR_DIR / "output" / "stopword_removed_docs.txt"
DEFAULT_PROCESSED_QUERIES = VENDOR_DIR / "output" / "stopword_removed_queries.txt"


def load_original_module():
    module_name = "humanised_soft_cosine_sources_original"
    spec = importlib.util.spec_from_file_location(module_name, ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load soft cosine runner from {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_tables(output_dir: Path, source_results: list[dict[str, object]]) -> None:
    baseline_source = next((row for row in source_results if row["source"] == "tfidf"), source_results[0])
    baseline_metrics = baseline_source["baseline"]["metrics_by_k"]
    baseline_k10 = baseline_source["baseline"]["k10"]

    summary_rows = [
        {
            "method": "baseline_tfidf",
            "source": "baseline",
            "source_label": "baseline_tfidf",
            "primary_param": "-",
            "primary_value": "-",
            "precision@10": baseline_k10["precision"],
            "recall@10": baseline_k10["recall"],
            "fscore@10": baseline_k10["fscore"],
            "map@10": baseline_k10["map"],
            "ndcg@10": baseline_k10["ndcg"],
            "mrr@10": baseline_k10["mrr"],
            "runtime_seconds": baseline_source["baseline"]["runtime_seconds"],
        }
    ]
    for row in source_results:
        best_config = row["best_config"]
        primary_param = row["primary_param"]
        summary_rows.append(
            {
                "method": row["source"],
                "source": row["source"],
                "source_label": row["label"],
                "primary_param": primary_param,
                "primary_value": best_config[primary_param],
                "precision@10": row["best_k10"]["precision"],
                "recall@10": row["best_k10"]["recall"],
                "fscore@10": row["best_k10"]["fscore"],
                "map@10": row["best_k10"]["map"],
                "ndcg@10": row["best_k10"]["ndcg"],
                "mrr@10": row["best_k10"]["mrr"],
                "runtime_seconds": row["best_runtime_seconds"],
            }
        )

    write_csv(
        output_dir / "summary_k10.csv",
        summary_rows,
        [
            "method",
            "source",
            "source_label",
            "primary_param",
            "primary_value",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
            "runtime_seconds",
        ],
    )

    comparison_summary = {
        "dataset": str(CRANFIELD_DIR),
        "sources": {
            row["source"]: {
                "label": row["label"],
                "vectorizer": row["vectorizer"],
                "baseline": row["baseline"],
                "best_config": row["best_config"],
                "best_k10": row["best_k10"],
                "best_metrics_by_k": row["best_metrics_by_k"],
                "best_runtime_seconds": row["best_runtime_seconds"],
                "best_by_primary": row["best_by_primary"],
                "significance": row["significance"],
                "paths": row["paths"],
            }
            for row in source_results
        },
        "best_source": max(
            source_results,
            key=lambda row: (row["best_k10"]["map"], row["best_k10"]["ndcg"], row["best_k10"]["mrr"]),
        )["source"],
        "paths": {
            "summary_json": str(output_dir / "summary.json"),
            "summary_csv": str(output_dir / "summary_k10.csv"),
            "comparison_summary_json": str(output_dir / "comparison_summary.json"),
            "overlay_plot": str(output_dir / "eval_overlay.png"),
        },
    }
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    write_json(
        output_dir / "summary.json",
        {
            "dataset": str(CRANFIELD_DIR),
            "baseline": {
                "metrics_by_k": baseline_metrics,
                "k10": baseline_k10,
                "runtime_seconds": baseline_source["baseline"]["runtime_seconds"],
            },
            "sources": comparison_summary["sources"],
            "best_source": comparison_summary["best_source"],
            "paths": comparison_summary["paths"],
        },
    )

    report_lines = [
        "# Soft Cosine Source Comparison Report",
        "",
        "This humanised runner keeps the tuned soft-cosine comparison from the original project and skips WordNet.",
        "",
        "## Best Results",
        "",
        "| Source | Primary Param | Best Value | MAP@10 | nDCG@10 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in source_results:
        report_lines.append(
            f"| {row['label']} | {row['primary_param']} | {row['best_config'][row['primary_param']]} | "
            f"{row['best_k10']['map']:.4f} | {row['best_k10']['ndcg']:.4f} | {row['best_k10']['mrr']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- TF-IDF, LSA, and ESA use the original sweep logic and the same Cranfield preprocessing.",
            "- WordNet is intentionally omitted in this humanised version.",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanised soft cosine wrapper")
    parser.add_argument("--dataset-dir", default=str(CRANFIELD_DIR))
    parser.add_argument("--processed-docs", default=str(DEFAULT_PROCESSED_DOCS))
    parser.add_argument("--processed-queries", default=str(DEFAULT_PROCESSED_QUERIES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sources", default="tfidf,lsa,esa", help="Comma-separated soft-cosine sources to run")
    args = parser.parse_args()

    module = load_original_module()
    source_keys = {piece.strip() for piece in args.sources.split(",") if piece.strip()}
    allowed_specs = [spec for spec in module.SOURCE_SPECS if spec["key"] in source_keys]
    if not allowed_specs:
        raise ValueError("No soft-cosine sources selected.")

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")
    processed_docs = load_json(Path(args.processed_docs))
    processed_queries = load_json(Path(args.processed_queries))

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    doc_texts = module.flatten_collection(processed_docs)
    query_texts = module.flatten_collection(processed_queries)

    source_results = []
    for source_spec in allowed_specs:
        print(f"[{source_spec['key']}] running tuned sweep", flush=True)
        source_results.append(
            {
                "primary_param": source_spec["primary_param"],
                **module.run_source_sweep(
                    source_spec=source_spec,
                    docs_texts=doc_texts,
                    queries_texts=query_texts,
                    queries_json=queries_json,
                    query_ids=query_ids,
                    qrels=qrels,
                    doc_ids=doc_ids,
                    output_dir=output_dir,
                ),
            }
        )

    baseline_source = next((row for row in source_results if row["source"] == "tfidf"), source_results[0])
    baseline_metrics = baseline_source["baseline"]["metrics_by_k"]
    grouped_metrics = {row["source"]: row["best_metrics_by_k"] for row in source_results}
    labels = {row["source"]: row["label"] for row in source_results}
    ordered_keys = [row["source"] for row in source_results]
    module.save_overlay_plot(
        output_dir / "eval_overlay.png",
        baseline_metrics,
        grouped_metrics,
        labels,
        ordered_keys,
        "Baseline vs Best Soft Cosine Sources",
    )

    build_tables(output_dir, source_results)
    print("Soft-cosine outputs written to:", output_dir)


if __name__ == "__main__":
    main()
