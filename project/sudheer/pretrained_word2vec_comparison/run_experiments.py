#!/usr/bin/env python3
"""
Compare scratch-trained Word2Vec backbones against pretrained and fine-tuned
embedding sources on the Cranfield collection.

The script reuses the previously generated scratch-trained results from the
existing Sudheer experiment folders, then evaluates a pretrained embedding
source and a fine-tuned variant that starts from the same pretrained vectors.
It produces per-method source comparisons for the centroid, IDF-weighted,
and soft-cosine Word2Vec backbones.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUDHEER_DIR = SCRIPT_DIR.parent
ASSIGNMENT_ROOT = SCRIPT_DIR.parents[2]
if str(SUDHEER_DIR) not in sys.path:
    sys.path.insert(0, str(SUDHEER_DIR))

from common import (  # noqa: E402
    ASSIGNMENT_ROOT as COMMON_ASSIGNMENT_ROOT,
    DenseCosineModel,
    Evaluation,
    SoftCosineModel,
    TfidfIndex,
    approximate_randomization_pvalue,
    build_average_embeddings,
    build_example_query_comparison,
    build_term_similarity_matrix,
    compute_metrics,
    compute_per_query_metrics,
    load_json,
    load_pretrained_word2vec_artifacts,
    normalize_collection,
    save_metric_plot,
    save_overlay_plot,
    save_rankings,
    train_word2vec_artifacts,
    unigram_feature_map,
    write_json,
    write_summary_csv,
)


PRETRAINED_MODEL = "glove-wiki-gigaword-100"


METHODS = [
    {
        "method_key": "word2vec_average",
        "label": "Average Word2Vec",
        "summary_dir": SUDHEER_DIR / "word2vec_average" / "output",
        "source_output_dir": SUDHEER_DIR / "word2vec_average" / "output" / "word2vec_average",
        "mode": "dense",
    },
    {
        "method_key": "word2vec_idf_weighted",
        "label": "IDF-weighted Word2Vec",
        "summary_dir": SUDHEER_DIR / "word2vec_idf_weighted" / "output",
        "source_output_dir": SUDHEER_DIR / "word2vec_idf_weighted" / "output" / "word2vec_idf_weighted",
        "mode": "dense",
    },
    {
        "method_key": "soft_cosine_word2vec",
        "label": "Word2Vec Soft Cosine",
        "summary_dir": SUDHEER_DIR / "soft_cosine_word2vec" / "output",
        "source_output_dir": SUDHEER_DIR / "soft_cosine_word2vec" / "output" / "soft_cosine_word2vec",
        "mode": "soft_cosine",
    },
]


SOURCE_ORDER = ["scratch", "pretrained", "finetuned"]
SOURCE_LABELS = {
    "scratch": "scratch-trained",
    "pretrained": f"pretrained:{PRETRAINED_MODEL}",
    "finetuned": f"finetuned:{PRETRAINED_MODEL}",
}


def load_rankings_by_query_id(rankings_path: Path, query_ids: list[int]) -> list[list[int]]:
    rankings_payload = load_json(rankings_path)
    return [list(rankings_payload[str(query_id)]) for query_id in query_ids]


def load_method_summary(summary_dir: Path, method_key: str) -> dict:
    summary = load_json(summary_dir / "summary.json")
    source_metrics = load_json(summary_dir / method_key / "metrics.json")
    source_rankings = load_json(summary_dir / method_key / "rankings_top20.json")
    return {
        "summary": summary,
        "source_metrics": source_metrics,
        "source_rankings": source_rankings,
        "source_output_dir": summary_dir / method_key,
    }


def k10_from_metrics(metrics_by_k: dict[str, list[float]]) -> dict[str, float]:
    return {metric_name: metric_values[9] for metric_name, metric_values in metrics_by_k.items()}


def copy_source_artifacts(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ["metrics.json", "rankings_top20.json", "eval_plot.png"]:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, destination_dir / file_name)


def evaluate_dense_variant(
    docs,
    queries,
    doc_ids,
    query_ids,
    qrels,
    word_vectors,
    vector_size: int,
    term_weights: dict[str, float] | None,
    vocab_size: int,
    method_key: str,
    source_key: str,
    source_label: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    doc_vectors, doc_stats = build_average_embeddings(
        collection=docs,
        word_vectors=word_vectors,
        vector_size=vector_size,
        term_weights=term_weights,
    )
    query_vectors, query_stats = build_average_embeddings(
        collection=queries,
        word_vectors=word_vectors,
        vector_size=vector_size,
        term_weights=term_weights,
    )
    model = DenseCosineModel.build(f"{method_key}:{source_key}", doc_vectors, doc_ids)
    rankings, _ = model.rank(query_vectors)
    metrics_by_k = compute_metrics(Evaluation(), rankings, query_ids, qrels)
    per_query = compute_per_query_metrics(Evaluation(), rankings, query_ids, qrels)
    runtime_seconds = time.time() - started

    save_metric_plot(
        output_dir / "eval_plot.png",
        metrics_by_k,
        f"{source_label} Metrics",
    )
    save_rankings(output_dir / "rankings_top20.json", query_ids, rankings, top_k=20)
    write_json(
        output_dir / "metrics.json",
        {
            "source": source_key,
            "source_label": source_label,
            "metrics_by_k": metrics_by_k,
            "k10": k10_from_metrics(metrics_by_k),
            "runtime_seconds": runtime_seconds,
            "vector_size": vector_size,
            "present_in_model": len(word_vectors),
            "model_coverage": float(len(word_vectors) / vocab_size) if vocab_size else 0.0,
            "avg_in_vocab_tokens_per_doc": doc_stats["avg_in_vocab_tokens_per_item"],
            "avg_in_vocab_tokens_per_query": query_stats["avg_in_vocab_tokens_per_item"],
            "zero_vector_docs": doc_stats["zero_vector_items"],
            "zero_vector_queries": query_stats["zero_vector_items"],
            "per_query": per_query,
        },
    )

    return {
        "source": source_key,
        "source_label": source_label,
        "metrics_by_k": metrics_by_k,
        "k10": k10_from_metrics(metrics_by_k),
        "runtime_seconds": runtime_seconds,
        "vector_size": vector_size,
        "present_in_model": len(word_vectors),
        "model_coverage": float(len(word_vectors) / vocab_size) if vocab_size else 0.0,
        "avg_in_vocab_tokens_per_doc": doc_stats["avg_in_vocab_tokens_per_item"],
        "avg_in_vocab_tokens_per_query": query_stats["avg_in_vocab_tokens_per_item"],
        "zero_vector_docs": doc_stats["zero_vector_items"],
        "zero_vector_queries": query_stats["zero_vector_items"],
        "rankings": rankings,
        "per_query": per_query,
    }


def evaluate_soft_cosine_variant(
    docs,
    doc_ids,
    query_tf,
    baseline_index: TfidfIndex,
    query_ids,
    qrels,
    word_vectors,
    vector_size: int,
    top_k_neighbors: int,
    min_similarity: float,
    similarity_power: float,
    method_key: str,
    source_key: str,
    source_label: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    similarity_matrix, similarity_stats = build_term_similarity_matrix(
        vocab=baseline_index.vocab,
        term_to_idx=baseline_index.term_to_idx,
        word_vectors=word_vectors,
        top_k_neighbors=top_k_neighbors,
        min_similarity=min_similarity,
        similarity_power=similarity_power,
    )
    model = SoftCosineModel.build(f"{method_key}:{source_key}", baseline_index, similarity_matrix)
    rankings, _ = model.rank_query_matrix(query_tf)
    metrics_by_k = compute_metrics(Evaluation(), rankings, query_ids, qrels)
    per_query = compute_per_query_metrics(Evaluation(), rankings, query_ids, qrels)
    runtime_seconds = time.time() - started

    save_metric_plot(
        output_dir / "eval_plot.png",
        metrics_by_k,
        f"{source_label} Metrics",
    )
    save_rankings(output_dir / "rankings_top20.json", query_ids, rankings, top_k=20)
    write_json(
        output_dir / "metrics.json",
        {
            "source": source_key,
            "source_label": source_label,
            "metrics_by_k": metrics_by_k,
            "k10": k10_from_metrics(metrics_by_k),
            "runtime_seconds": runtime_seconds,
            "vector_size": vector_size,
            "present_in_model": len(word_vectors),
            "model_coverage": float(len(word_vectors) / len(baseline_index.vocab)) if baseline_index.vocab else 0.0,
            "represented_terms": similarity_stats["represented_terms"],
            "representation_coverage": similarity_stats["representation_coverage"],
            "retained_neighbor_edges": similarity_stats["retained_neighbor_edges"],
            "avg_neighbors_per_term": similarity_stats["avg_neighbors_per_term"],
            "per_query": per_query,
        },
    )

    return {
        "source": source_key,
        "source_label": source_label,
        "metrics_by_k": metrics_by_k,
        "k10": k10_from_metrics(metrics_by_k),
        "runtime_seconds": runtime_seconds,
        "vector_size": vector_size,
        "present_in_model": len(word_vectors),
        "model_coverage": float(len(word_vectors) / len(baseline_index.vocab)) if baseline_index.vocab else 0.0,
        "represented_terms": similarity_stats["represented_terms"],
        "representation_coverage": similarity_stats["representation_coverage"],
        "retained_neighbor_edges": similarity_stats["retained_neighbor_edges"],
        "avg_neighbors_per_term": similarity_stats["avg_neighbors_per_term"],
        "rankings": rankings,
        "per_query": per_query,
    }


def write_method_report(
    output_path: Path,
    title: str,
    method_key: str,
    summary: dict,
    scratch_payload: dict,
    source_payloads: dict[str, dict],
    best_source_key: str,
    examples: list[dict[str, object]],
    significance: dict[str, float],
) -> None:
    baseline = summary["methods"]["baseline_tfidf"]
    scratch = scratch_payload
    best_source = source_payloads[best_source_key]

    lines = [f"# {title}", ""]
    lines.extend([
        "## Setup",
        "",
        f"- Scratch results are reused from the existing `{method_key}` experiment output.",
        f"- Pretrained source: `{PRETRAINED_MODEL}`.",
        "- Fine-tuned source: the same pretrained vectors, continued on Cranfield with the same Word2Vec hyperparameters.",
        "",
    ])

    lines.extend([
        "## Scratch Baseline",
        "",
        f"- MAP@10: {scratch['k10']['map']:.4f}",
        f"- nDCG@10: {scratch['k10']['ndcg']:.4f}",
        f"- MRR@10: {scratch['k10']['mrr']:.4f}",
        "",
    ])

    lines.extend([
        f"## Source Comparison for {title}",
        "",
        "| Source | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Model Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for source_key in SOURCE_ORDER:
        payload = source_payloads[source_key]
        lines.append(
            f"| {payload['source_label']} | {payload['k10']['precision']:.4f} | {payload['k10']['recall']:.4f} | "
            f"{payload['k10']['fscore']:.4f} | {payload['k10']['map']:.4f} | {payload['k10']['ndcg']:.4f} | "
            f"{payload['k10']['mrr']:.4f} | {payload['model_coverage']:.2%} |"
        )
    lines.append("")

    lines.extend([
        f"- Best source by MAP@10: {source_payloads[best_source_key]['source_label']}",
        f"- MAP@10 delta vs scratch: {source_payloads[best_source_key]['k10']['map'] - scratch['k10']['map']:+.4f}",
        f"- nDCG@10 delta vs scratch: {source_payloads[best_source_key]['k10']['ndcg'] - scratch['k10']['ndcg']:+.4f}",
        f"- Approx. randomization p-value (AP@10): {significance['ap_at_10_pvalue']:.4f}",
        f"- Approx. randomization p-value (nDCG@10): {significance['ndcg_at_10_pvalue']:.4f}",
        "",
    ])

    if examples:
        lines.extend(["## Example Query Comparison", ""])
        lines.append("| Query ID | Baseline Hits@5 | Source Hits@5 | Baseline AP@10 | Source AP@10 | Delta AP@10 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in examples:
            lines.append(
                f"| {row['query_id']} | {row['baseline_hits_at_5']} | {row['method_hits_at_5']} | "
                f"{row['baseline_ap_at_10']:.4f} | {row['method_ap_at_10']:.4f} | {row['delta_ap_at_10']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- The scratch backbone is the current in-domain Word2Vec model trained from scratch on Cranfield.",
        "- The pretrained source adds external semantic information and usually improves vocabulary coverage.",
        "- The fine-tuned source keeps the pretrained initialization but adapts it to the Cranfield collection.",
        "",
        "## References",
        "",
        "- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)",
        "- Pennington et al., GloVe: Global Vectors for Word Representation (2014)",
        "- Sidorov et al., Soft Similarity and Soft Cosine Measure (2014)",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scratch-trained and pretrained Word2Vec backbones.")
    parser.add_argument("--dataset-dir", default=str(COMMON_ASSIGNMENT_ROOT / "cranfield"))
    parser.add_argument("--processed-docs", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_docs.txt"))
    parser.add_argument("--processed-queries", default=str(COMMON_ASSIGNMENT_ROOT / "output" / "stopword_removed_queries.txt"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument("--pretrained-model", default=PRETRAINED_MODEL)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_json = load_json(dataset_dir / "cran_docs.json")
    queries_json = load_json(dataset_dir / "cran_queries.json")
    qrels = load_json(dataset_dir / "cran_qrels.json")

    processed_docs = normalize_collection(load_json(Path(args.processed_docs)))
    processed_queries = normalize_collection(load_json(Path(args.processed_queries)))

    doc_ids = [item["id"] for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]

    doc_feature_maps = [unigram_feature_map(doc) for doc in processed_docs]
    query_feature_maps = [unigram_feature_map(query) for query in processed_queries]

    baseline_dir = output_dir / "baseline_tfidf"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_index = TfidfIndex.build("baseline_tfidf", doc_feature_maps, doc_ids)
    baseline_rankings, _ = baseline_index.rank_feature_maps(query_feature_maps)
    baseline_metrics = compute_metrics(Evaluation(), baseline_rankings, query_ids, qrels)
    baseline_per_query = compute_per_query_metrics(Evaluation(), baseline_rankings, query_ids, qrels)
    save_metric_plot(baseline_dir / "eval_plot.png", baseline_metrics, "Baseline TF-IDF Evaluation Metrics")
    save_rankings(baseline_dir / "rankings_top20.json", query_ids, baseline_rankings, top_k=20)
    write_json(
        baseline_dir / "metrics.json",
        {
            "metrics_by_k": baseline_metrics,
            "k10": k10_from_metrics(baseline_metrics),
            "runtime_seconds": 0.0,
        },
    )

    pretrained_artifacts = load_pretrained_word2vec_artifacts(args.pretrained_model, baseline_index.vocab)
    finetuned_artifacts = train_word2vec_artifacts(
        docs_tokens=processed_docs,
        vocab=baseline_index.vocab,
        vector_size=pretrained_artifacts.vector_size,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=10,
        initial_word_vectors=pretrained_artifacts.word_vectors,
        embedding_source="finetuned",
        pretrained_model_name=args.pretrained_model,
    )

    source_artifacts = {
        "pretrained": pretrained_artifacts,
        "finetuned": finetuned_artifacts,
    }

    source_payloads_by_method: dict[str, dict[str, dict]] = {}
    example_payloads_by_method: dict[str, list[dict[str, object]]] = {}
    summary_rows = []
    comparison_summary = {
        "dataset": str(dataset_dir),
        "pretrained_model": args.pretrained_model,
        "baseline": {
            "metrics_by_k": baseline_metrics,
            "k10": k10_from_metrics(baseline_metrics),
            "runtime_seconds": 0.0,
        },
        "methods": {},
    }

    for method_spec in METHODS:
        method_key = method_spec["method_key"]
        method_label = method_spec["label"]
        method_dir = output_dir / method_key
        method_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{method_key}] comparing scratch, pretrained, and finetuned sources", flush=True)

        scratch_source_dir = method_spec["source_output_dir"]
        scratch_dir = method_dir / "scratch"
        copy_source_artifacts(scratch_source_dir, scratch_dir)

        method_summary = load_method_summary(method_spec["summary_dir"], method_key)
        scratch_metrics = method_summary["source_metrics"]
        scratch_rankings = [
            list(method_summary["source_rankings"][str(query_id)])
            for query_id in query_ids
        ]

        source_payloads: dict[str, dict] = {
            "scratch": {
                "source": "scratch",
                "source_label": SOURCE_LABELS["scratch"],
                "metrics_by_k": scratch_metrics["metrics_by_k"],
                "k10": scratch_metrics["k10"],
                "runtime_seconds": scratch_metrics["runtime_seconds"],
                "vector_size": scratch_metrics.get("best_config", {}).get("vector_size", 100),
                "present_in_model": scratch_metrics.get("present_in_model", 0),
                "model_coverage": scratch_metrics.get("model_coverage", 0.0),
                "rankings": scratch_rankings,
                "per_query": compute_per_query_metrics(Evaluation(), scratch_rankings, query_ids, qrels),
            },
        }

        if method_spec["mode"] == "dense":
            term_weights = baseline_index.idf_by_term() if method_key == "word2vec_idf_weighted" else None
            source_payloads["pretrained"] = evaluate_dense_variant(
                docs=processed_docs,
                queries=processed_queries,
                doc_ids=doc_ids,
                query_ids=query_ids,
                qrels=qrels,
                word_vectors=source_artifacts["pretrained"].word_vectors,
                vector_size=source_artifacts["pretrained"].vector_size,
                term_weights=term_weights,
                vocab_size=len(baseline_index.vocab),
                method_key=method_key,
                source_key="pretrained",
                source_label=SOURCE_LABELS["pretrained"],
                output_dir=method_dir / "pretrained",
            )
            print(
                f"  pretrained MAP@10={source_payloads['pretrained']['k10']['map']:.4f} "
                f"nDCG@10={source_payloads['pretrained']['k10']['ndcg']:.4f}",
                flush=True,
            )
            source_payloads["finetuned"] = evaluate_dense_variant(
                docs=processed_docs,
                queries=processed_queries,
                doc_ids=doc_ids,
                query_ids=query_ids,
                qrels=qrels,
                word_vectors=source_artifacts["finetuned"].word_vectors,
                vector_size=source_artifacts["finetuned"].vector_size,
                term_weights=term_weights,
                vocab_size=len(baseline_index.vocab),
                method_key=method_key,
                source_key="finetuned",
                source_label=SOURCE_LABELS["finetuned"],
                output_dir=method_dir / "finetuned",
            )
            print(
                f"  finetuned MAP@10={source_payloads['finetuned']['k10']['map']:.4f} "
                f"nDCG@10={source_payloads['finetuned']['k10']['ndcg']:.4f}",
                flush=True,
            )
        else:
            query_tf = baseline_index.feature_maps_to_matrix(query_feature_maps)
            best_config = method_summary["summary"]["best_config"]
            source_payloads["pretrained"] = evaluate_soft_cosine_variant(
                docs=processed_docs,
                doc_ids=doc_ids,
                query_tf=query_tf,
                baseline_index=baseline_index,
                query_ids=query_ids,
                qrels=qrels,
                word_vectors=source_artifacts["pretrained"].word_vectors,
                vector_size=source_artifacts["pretrained"].vector_size,
                top_k_neighbors=int(best_config["top_k_neighbors"]),
                min_similarity=float(best_config["min_similarity"]),
                similarity_power=float(best_config["similarity_power"]),
                method_key=method_key,
                source_key="pretrained",
                source_label=SOURCE_LABELS["pretrained"],
                output_dir=method_dir / "pretrained",
            )
            print(
                f"  pretrained MAP@10={source_payloads['pretrained']['k10']['map']:.4f} "
                f"nDCG@10={source_payloads['pretrained']['k10']['ndcg']:.4f}",
                flush=True,
            )
            source_payloads["finetuned"] = evaluate_soft_cosine_variant(
                docs=processed_docs,
                doc_ids=doc_ids,
                query_tf=query_tf,
                baseline_index=baseline_index,
                query_ids=query_ids,
                qrels=qrels,
                word_vectors=source_artifacts["finetuned"].word_vectors,
                vector_size=source_artifacts["finetuned"].vector_size,
                top_k_neighbors=int(best_config["top_k_neighbors"]),
                min_similarity=float(best_config["min_similarity"]),
                similarity_power=float(best_config["similarity_power"]),
                method_key=method_key,
                source_key="finetuned",
                source_label=SOURCE_LABELS["finetuned"],
                output_dir=method_dir / "finetuned",
            )
            print(
                f"  finetuned MAP@10={source_payloads['finetuned']['k10']['map']:.4f} "
                f"nDCG@10={source_payloads['finetuned']['k10']['ndcg']:.4f}",
                flush=True,
            )

        best_source_key = max(
            SOURCE_ORDER,
            key=lambda source_key: (
                source_payloads[source_key]["k10"]["map"],
                source_payloads[source_key]["k10"]["ndcg"],
                source_payloads[source_key]["k10"]["mrr"],
            ),
        )
        best_source = source_payloads[best_source_key]

        method_labels = {source_key: source_payloads[source_key]["source_label"] for source_key in SOURCE_ORDER}
        grouped_metrics = {source_key: source_payloads[source_key]["metrics_by_k"] for source_key in SOURCE_ORDER}
        ordered_keys = SOURCE_ORDER
        save_overlay_plot(
            method_dir / "source_overlay.png",
            baseline_metrics,
            grouped_metrics,
            method_labels,
            ordered_keys,
            f"{method_label}: Baseline vs Scratch vs Pretrained vs Finetuned",
        )

        delta_vs_scratch = {
            metric: best_source["k10"][metric] - source_payloads["scratch"]["k10"][metric]
            for metric in ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
        }
        significance = {
            "ap_at_10_pvalue": approximate_randomization_pvalue(
                source_payloads["scratch"]["per_query"]["average_precision"],
                best_source["per_query"]["average_precision"],
            ),
            "ndcg_at_10_pvalue": approximate_randomization_pvalue(
                source_payloads["scratch"]["per_query"]["ndcg"],
                best_source["per_query"]["ndcg"],
            ),
        }

        examples = build_example_query_comparison(
            query_ids=query_ids,
            queries_json=queries_json,
            qrels=qrels,
            baseline_rankings=source_payloads["scratch"]["rankings"],
            method_rankings=best_source["rankings"],
            baseline_per_query=source_payloads["scratch"]["per_query"],
            method_per_query=best_source["per_query"],
            output_dir=method_dir,
        )
        example_payloads_by_method[method_key] = examples

        write_method_report(
            output_path=method_dir / "experiment_report.md",
            title=f"{method_label} Pretrained Comparison Report",
            method_key=method_key,
            summary=method_summary["summary"],
            scratch_payload=source_payloads["scratch"],
            source_payloads=source_payloads,
            best_source_key=best_source_key,
            examples=examples,
            significance=significance,
        )

        source_payloads_by_method[method_key] = source_payloads

        comparison_summary["methods"][method_key] = {
            "label": method_label,
            "best_config": method_summary["summary"]["best_config"],
            "best_source": best_source_key,
            "delta_vs_scratch": delta_vs_scratch,
            "significance": significance,
            "sources": {
                source_key: {
                    "source_label": source_payloads[source_key]["source_label"],
                    "k10": source_payloads[source_key]["k10"],
                    "metrics_by_k": source_payloads[source_key]["metrics_by_k"],
                    "runtime_seconds": source_payloads[source_key]["runtime_seconds"],
                    "present_in_model": source_payloads[source_key]["present_in_model"],
                    "model_coverage": source_payloads[source_key]["model_coverage"],
                }
                for source_key in SOURCE_ORDER
            },
            "paths": {
                "method_report": str(method_dir / "experiment_report.md"),
                "source_overlay": str(method_dir / "source_overlay.png"),
                "example_markdown": str(method_dir / "example_query_comparison.md"),
            },
        }

        for source_key in SOURCE_ORDER:
            row = {
                "method": method_key,
                "source": source_key,
                "source_label": source_payloads[source_key]["source_label"],
                "precision@10": source_payloads[source_key]["k10"]["precision"],
                "recall@10": source_payloads[source_key]["k10"]["recall"],
                "fscore@10": source_payloads[source_key]["k10"]["fscore"],
                "map@10": source_payloads[source_key]["k10"]["map"],
                "ndcg@10": source_payloads[source_key]["k10"]["ndcg"],
                "mrr@10": source_payloads[source_key]["k10"]["mrr"],
                "runtime_seconds": source_payloads[source_key]["runtime_seconds"],
                "model_coverage": source_payloads[source_key]["model_coverage"],
                "baseline_map@10": baseline_metrics["map"][9],
                "gain_vs_scratch_map@10": source_payloads[source_key]["k10"]["map"] - source_payloads["scratch"]["k10"]["map"],
                "gain_vs_scratch_ndcg@10": source_payloads[source_key]["k10"]["ndcg"] - source_payloads["scratch"]["k10"]["ndcg"],
            }
            summary_rows.append(row)

    write_summary_csv(
        output_dir / "summary_k10.csv",
        summary_rows,
        fieldnames=[
            "method",
            "source",
            "source_label",
            "precision@10",
            "recall@10",
            "fscore@10",
            "map@10",
            "ndcg@10",
            "mrr@10",
            "runtime_seconds",
            "model_coverage",
            "baseline_map@10",
            "gain_vs_scratch_map@10",
            "gain_vs_scratch_ndcg@10",
        ],
    )
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    write_json(
        output_dir / "summary.json",
        {
            "dataset": str(dataset_dir),
            "pretrained_model": args.pretrained_model,
            "baseline": {
                "metrics_by_k": baseline_metrics,
                "k10": k10_from_metrics(baseline_metrics),
            },
            "methods": comparison_summary["methods"],
            "paths": {
                "summary_json": str(output_dir / "summary.json"),
                "summary_csv": str(output_dir / "summary_k10.csv"),
                "comparison_summary_json": str(output_dir / "comparison_summary.json"),
                "baseline_dir": str(baseline_dir),
            },
        },
    )

    lines = ["# Pretrained Word2Vec Comparison", ""]
    lines.extend([
        "This comparison reuses the existing scratch-trained results and adds two external embedding sources:",
        f"- pretrained: `{PRETRAINED_MODEL}`",
        f"- finetuned: `{PRETRAINED_MODEL}` continued on Cranfield",
        "",
    ])
    for method_key, method_result in comparison_summary["methods"].items():
        lines.extend([f"## {method_result['label']}", ""])
        lines.append(
            "| Source | MAP@10 | nDCG@10 | MRR@10 | Coverage |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for source_key in SOURCE_ORDER:
            payload = method_result["sources"][source_key]
            lines.append(
                f"| {payload['source_label']} | {payload['k10']['map']:.4f} | {payload['k10']['ndcg']:.4f} | "
                f"{payload['k10']['mrr']:.4f} | {payload['model_coverage']:.2%} |"
            )
        lines.append("")
        lines.append(
            f"- Best source: {method_result['sources'][method_result['best_source']]['source_label']}"
        )
        lines.append(
            f"- MAP@10 delta vs scratch: {method_result['delta_vs_scratch']['map']:+.4f}"
        )
        lines.append("")

    lines.extend([
        "## Scratch vs Pretrained Training",
        "",
        "- The original Sudheer Word2Vec runs train skip-gram/backbone vectors from scratch on Cranfield.",
        "- The comparison folder adds a pretrained embedding source and a fine-tuned variant, so you can see how much external semantic knowledge helps.",
        "- If the fine-tuned source wins, that is the strongest answer to the performance gap in the original scratch-trained plots.",
        "",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Comparison written to:", output_dir)
    for method_key, method_result in comparison_summary["methods"].items():
        print(
            f"{method_key}: best source={method_result['sources'][method_result['best_source']]['source_label']} "
            f"MAP@10={method_result['sources'][method_result['best_source']]['k10']['map']:.4f}"
        )


if __name__ == "__main__":
    main()