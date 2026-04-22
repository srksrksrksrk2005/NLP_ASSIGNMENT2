import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from core.data import get_docs_ids_and_texts, get_query_ids_and_texts, load_cranfield_dataset
from core.evaluation import CranfieldEvaluation
from core.preprocessing import Preprocessor
from core.retrieval import VectorSpaceRetrieval, flatten_query_tokens
from expansion.base import MatrixQueryExpander
from expansion.embedding_matrices import (
    build_esa_neighbor_map,
    build_lsa_neighbor_map,
    build_tfidf_neighbor_map,
    build_word2vec_neighbor_map,
)
from expansion.wordnet_matrix import WordNetOOVResolver, build_wordnet_neighbor_map


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[3]
    default_dataset = base_dir / "cranfield"
    default_output = Path(__file__).resolve().parent / "output"

    parser = argparse.ArgumentParser(description="Query replacement + expansion experiments")
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to cranfield directory")
    parser.add_argument("--output", default=str(default_output), help="Directory to store outputs")

    parser.add_argument(
        "--methods",
        default="wordnet,embedding_tfidf,embedding_lsa,embedding_esa,embedding_word2vec",
        help=(
            "Comma-separated methods from: wordnet, embedding_tfidf, embedding_lsa, "
            "embedding_esa, embedding_word2vec"
        ),
    )

    parser.add_argument("--top-k-neighbors", type=int, default=10)
    parser.add_argument("--min-similarity", type=float, default=0.05)

    parser.add_argument("--self-weight", type=float, default=1.0)
    parser.add_argument("--expansion-weight", type=float, default=0.35)
    parser.add_argument("--replacement-weight", type=float, default=1.0)
    parser.add_argument("--replacement-expansion-weight", type=float, default=0.35)
    parser.add_argument("--max-oov-candidates", type=int, default=5)

    parser.add_argument("--lsa-components", type=int, default=128)
    parser.add_argument("--esa-top-concepts", type=int, default=100)

    parser.add_argument("--w2v-vector-size", type=int, default=100)
    parser.add_argument("--w2v-window", type=int, default=5)
    parser.add_argument("--w2v-min-count", type=int, default=1)
    parser.add_argument("--w2v-workers", type=int, default=1)
    parser.add_argument("--w2v-sg", type=int, default=1)
    parser.add_argument("--w2v-epochs", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print periodic live logs every N processed terms where applicable",
    )

    return parser.parse_args()



def _query_vectors_for_expander(
    processed_queries: List[List[List[str]]],
    expander: MatrixQueryExpander,
    show_progress: bool = True,
    logger: Callable[[str], None] | None = None,
    log_every: int = 25,
) -> np.ndarray:
    vectors = np.zeros((len(processed_queries), len(expander.vocab)), dtype=np.float64)
    total_queries = len(processed_queries)
    start = time.time()
    if logger is not None:
        logger(f"Starting query expansion for {total_queries} queries")

    query_iter = tqdm(
        processed_queries,
        desc="Expanding queries",
        unit="query",
        disable=not show_progress,
    )
    for idx, query in enumerate(query_iter):
        counts = Counter(flatten_query_tokens(query))
        vectors[idx] = expander.build_query_vector_from_counts(counts)
        done = idx + 1
        if logger is not None and (done == 1 or done % max(1, log_every) == 0 or done == total_queries):
            elapsed = time.time() - start
            logger(f"Expanded queries: {done}/{total_queries} (elapsed {elapsed:.1f}s)")

    if logger is not None:
        logger(f"Finished query expansion in {time.time() - start:.2f}s")
    return vectors



def _preview_query_expansion(
    query_id: str,
    query_tokens: List[List[str]],
    vector: np.ndarray,
    vocab: List[str],
    top_n: int = 12,
) -> Dict:
    token_counts = Counter(flatten_query_tokens(query_tokens))
    nonzero = np.where(vector > 0)[0]
    top_indices = sorted(nonzero, key=lambda i: vector[i], reverse=True)[:top_n]
    top_terms = [{"term": vocab[i], "weight": float(vector[i])} for i in top_indices]
    return {
        "query_id": str(query_id),
        "original_terms": dict(token_counts),
        "expanded_top_terms": top_terms,
    }



def _plot_method_metrics(method_name: str, metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    ks = sorted(int(k) for k in metrics.keys())
    series = {
        "Precision": [metrics[str(k)]["precision"] for k in ks],
        "Recall": [metrics[str(k)]["recall"] for k in ks],
        "F-Score": [metrics[str(k)]["fscore"] for k in ks],
        "MAP": [metrics[str(k)]["map"] for k in ks],
        "nDCG": [metrics[str(k)]["ndcg"] for k in ks],
        "MRR": [metrics[str(k)]["mrr"] for k in ks],
    }

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(ks, values, marker="o", linewidth=2, label=label)
    plt.title(f"Evaluation Metrics - {method_name}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def _plot_overlay_metrics(all_results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    if not all_results:
        return

    metric_names = ["precision", "recall", "fscore", "map", "ndcg", "mrr"]
    pretty_names = {
        "precision": "Precision",
        "recall": "Recall",
        "fscore": "F-Score",
        "map": "MAP",
        "ndcg": "nDCG",
        "mrr": "MRR",
    }
    ks = sorted(int(k) for k in next(iter(all_results.values())).keys())
    methods = list(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for axis, metric_name in zip(axes, metric_names):
        for method in methods:
            values = [all_results[method][str(k)][metric_name] for k in ks]
            if method == "embedding_tfidf":
                axis.plot(ks, values, marker="o", linewidth=3, linestyle="--", label=f"{method} baseline")
            else:
                axis.plot(ks, values, marker="o", linewidth=1.8, label=method)
        axis.set_title(pretty_names[metric_name])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("k")
        axis.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.suptitle("Cranfield Query Expansion Comparison Across Methods", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def run() -> None:
    args = parse_args()
    start = time.time()

    def log(message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress
    log_every = max(1, args.log_every)
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    log(f"Loading Cranfield dataset from: {args.dataset}")
    docs_json, queries_json, qrels = load_cranfield_dataset(args.dataset)
    doc_ids, doc_texts = get_docs_ids_and_texts(docs_json)
    query_ids, query_texts = get_query_ids_and_texts(queries_json)
    log(f"Loaded raw data: {len(doc_ids)} docs, {len(query_ids)} queries, {len(qrels)} qrels")

    stage_start = time.time()
    log("Preprocessing document corpus")
    preprocessor = Preprocessor(use_lemmatization=True)
    processed_docs = preprocessor.preprocess_corpus(doc_texts)
    log(f"Document preprocessing done in {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    log("Preprocessing query corpus")
    processed_queries = preprocessor.preprocess_corpus(query_texts)
    log(f"Query preprocessing done in {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    log("Building TF-IDF retrieval index")
    retriever = VectorSpaceRetrieval()
    retriever.build(processed_docs, doc_ids)
    if retriever.doc_tfidf is None:
        raise RuntimeError("Document TF-IDF matrix was not created")
    doc_tfidf = retriever.doc_tfidf
    log(f"Retrieval index built in {time.time() - stage_start:.2f}s")

    evaluator = CranfieldEvaluation(qrels)

    log(f"Vocabulary size after preprocessing: {len(retriever.vocab)}")

    wordnet_neighbors: Dict[str, List[tuple[str, float]]] | None = None
    if "wordnet" in requested_methods:
        stage_start = time.time()
        log("Building WordNet in-vocabulary similarity map")
        wordnet_neighbors = build_wordnet_neighbor_map(
            retriever.vocab,
            top_k=args.top_k_neighbors,
            min_similarity=args.min_similarity,
            progress=show_progress,
            logger=log,
            log_every=log_every,
        )
        log(f"WordNet similarity map built in {time.time() - stage_start:.2f}s")
    else:
        log("Skipping WordNet in-vocabulary matrix (wordnet method not requested)")

    stage_start = time.time()
    log("Building WordNet OOV replacement index")
    oov_resolver = WordNetOOVResolver(
        retriever.vocab,
        progress=show_progress,
        logger=log,
        log_every=log_every,
    )
    log(f"WordNet OOV index built in {time.time() - stage_start:.2f}s")

    all_results: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}
    log(f"Running methods: {', '.join(requested_methods)}")

    methods_iter = tqdm(
        requested_methods,
        desc="Methods",
        unit="method",
        disable=not show_progress,
    )
    for method in methods_iter:
        method_start = time.time()
        print(f"\n=== Running method: {method} ===", flush=True)
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log(f"[{method}] Preparing neighbor map")

        try:
            if method == "wordnet":
                if wordnet_neighbors is None:
                    raise RuntimeError("WordNet neighbors unavailable for wordnet method")
                neighbors = wordnet_neighbors
            elif method == "embedding_tfidf":
                neighbors = build_tfidf_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                    progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
            elif method == "embedding_lsa":
                neighbors = build_lsa_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    n_components=args.lsa_components,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                    progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
            elif method == "embedding_esa":
                neighbors = build_esa_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    top_concepts=args.esa_top_concepts,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                    progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
            elif method == "embedding_word2vec":
                neighbors = build_word2vec_neighbor_map(
                    processed_docs,
                    retriever.vocab,
                    vector_size=args.w2v_vector_size,
                    window=args.w2v_window,
                    min_count=args.w2v_min_count,
                    workers=args.w2v_workers,
                    sg=args.w2v_sg,
                    epochs=args.w2v_epochs,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                    progress=show_progress,
                    logger=log,
                    log_every=log_every,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            log(f"[{method}] Neighbor map ready. Expanding queries")

            expander = MatrixQueryExpander(
                vocab=retriever.vocab,
                neighbors=neighbors,
                oov_resolver=oov_resolver.resolve,
                self_weight=args.self_weight,
                expansion_weight=args.expansion_weight,
                replacement_weight=args.replacement_weight,
                replacement_expansion_weight=args.replacement_expansion_weight,
                max_oov_candidates=args.max_oov_candidates,
                min_similarity=args.min_similarity,
                expand_replacements=True,
            )

            query_vectors = _query_vectors_for_expander(
                processed_queries,
                expander,
                show_progress=show_progress,
                logger=log,
            )
            log(f"[{method}] Ranking documents")
            rankings = retriever.query_vectors_to_rankings(query_vectors)
            rankings_by_query = {
                str(qid): [str(doc_id) for doc_id in ranked]
                for qid, ranked in zip(query_ids, rankings)
            }

            log(f"[{method}] Computing evaluation metrics")
            metrics = evaluator.evaluate(rankings_by_query, query_ids, k_values=range(1, 11))

            previews = []
            for idx in range(min(8, len(query_ids))):
                previews.append(
                    _preview_query_expansion(
                        query_ids[idx],
                        processed_queries[idx],
                        query_vectors[idx],
                        retriever.vocab,
                    )
                )

            (method_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            (method_dir / "rankings_top20.json").write_text(
                json.dumps({qid: docs[:20] for qid, docs in rankings_by_query.items()}, indent=2),
                encoding="utf-8",
            )
            (method_dir / "query_expansion_preview.json").write_text(
                json.dumps(previews, indent=2),
                encoding="utf-8",
            )
            _plot_method_metrics(method, metrics, method_dir / "eval_plot.png")

            all_results[method] = metrics
            print(
                f"Finished {method}. P@10={metrics['10']['precision']:.4f}, MAP@10={metrics['10']['map']:.4f}",
                flush=True,
            )
            log(f"[{method}] Completed in {time.time() - method_start:.2f}s")

        except Exception as exc:
            failures[method] = str(exc)
            print(f"Method {method} failed: {exc}", flush=True)
            log(f"[{method}] Failed after {time.time() - method_start:.2f}s")

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    csv_file = output_dir / "summary_k10.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "precision@10", "recall@10", "fscore@10", "map@10", "ndcg@10", "mrr@10"])
        for method, metrics in all_results.items():
            m10 = metrics["10"]
            writer.writerow(
                [
                    method,
                    m10["precision"],
                    m10["recall"],
                    m10["fscore"],
                    m10["map"],
                    m10["ndcg"],
                    m10["mrr"],
                ]
            )

    if failures:
        (output_dir / "failed_methods.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )

    if all_results:
        _plot_overlay_metrics(all_results, output_dir / "eval_overlay.png")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f} seconds", flush=True)
    print(f"Summary: {summary_file}", flush=True)
    print(f"k=10 table: {csv_file}", flush=True)
    if failures:
        print("Some methods failed. See failed_methods.json", flush=True)


if __name__ == "__main__":
    run()
