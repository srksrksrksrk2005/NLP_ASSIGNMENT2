import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

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

    return parser.parse_args()



def _query_vectors_for_expander(
    processed_queries: List[List[List[str]]],
    expander: MatrixQueryExpander,
) -> np.ndarray:
    vectors = np.zeros((len(processed_queries), len(expander.vocab)), dtype=np.float64)
    for idx, query in enumerate(processed_queries):
        counts = Counter(flatten_query_tokens(query))
        vectors[idx] = expander.build_query_vector_from_counts(counts)
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



def run() -> None:
    args = parse_args()
    start = time.time()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_json, queries_json, qrels = load_cranfield_dataset(args.dataset)
    doc_ids, doc_texts = get_docs_ids_and_texts(docs_json)
    query_ids, query_texts = get_query_ids_and_texts(queries_json)

    preprocessor = Preprocessor(use_lemmatization=True)
    processed_docs = preprocessor.preprocess_corpus(doc_texts)
    processed_queries = preprocessor.preprocess_corpus(query_texts)

    retriever = VectorSpaceRetrieval()
    retriever.build(processed_docs, doc_ids)
    if retriever.doc_tfidf is None:
        raise RuntimeError("Document TF-IDF matrix was not created")
    doc_tfidf = retriever.doc_tfidf

    evaluator = CranfieldEvaluation(qrels)

    print(f"Loaded {len(doc_ids)} docs, {len(query_ids)} queries, vocab size {len(retriever.vocab)}")

    wordnet_neighbors = build_wordnet_neighbor_map(
        retriever.vocab,
        top_k=args.top_k_neighbors,
        min_similarity=args.min_similarity,
    )
    oov_resolver = WordNetOOVResolver(retriever.vocab)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}

    for method in requested_methods:
        print(f"\n=== Running method: {method} ===")
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            if method == "wordnet":
                neighbors = wordnet_neighbors
            elif method == "embedding_tfidf":
                neighbors = build_tfidf_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                )
            elif method == "embedding_lsa":
                neighbors = build_lsa_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    n_components=args.lsa_components,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
                )
            elif method == "embedding_esa":
                neighbors = build_esa_neighbor_map(
                    doc_tfidf,
                    retriever.vocab,
                    top_concepts=args.esa_top_concepts,
                    top_k=args.top_k_neighbors,
                    min_similarity=args.min_similarity,
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
                )
            else:
                raise ValueError(f"Unknown method: {method}")

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

            query_vectors = _query_vectors_for_expander(processed_queries, expander)
            rankings = retriever.query_vectors_to_rankings(query_vectors)
            rankings_by_query = {
                str(qid): [str(doc_id) for doc_id in ranked]
                for qid, ranked in zip(query_ids, rankings)
            }

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

            all_results[method] = metrics
            print(f"Finished {method}. P@10={metrics['10']['precision']:.4f}, MAP@10={metrics['10']['map']:.4f}")

        except Exception as exc:
            failures[method] = str(exc)
            print(f"Method {method} failed: {exc}")

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

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Summary: {summary_file}")
    print(f"k=10 table: {csv_file}")
    if failures:
        print("Some methods failed. See failed_methods.json")


if __name__ == "__main__":
    run()
