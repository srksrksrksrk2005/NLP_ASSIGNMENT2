import argparse, json, os, sys, time
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import Evaluation  # type: ignore
from inflectionReduction import InflectionReduction  # type: ignore
from sentenceSegmentation import SentenceSegmentation  # type: ignore
from stopwordRemoval import StopwordRemoval  # type: ignore
from tokenization import Tokenization  # type: ignore
from hybrid_retrieval import HybridRetrieval


def preprocess(texts):
    seg = SentenceSegmentation()
    tok = Tokenization()
    inf = InflectionReduction()
    stp = StopwordRemoval()

    out = [seg.punkt(t) for t in texts]
    out = [tok.pennTreeBank(t) for t in out]
    out = [inf.reduce(t) for t in out]
    out = [stp.fromList(t) for t in out]
    return out


def load_cranfield(path):
    queries = json.load(open(path / "cran_queries.json"))
    docs = json.load(open(path / "cran_docs.json"))
    qrels = json.load(open(path / "cran_qrels.json"))
    return queries, docs, qrels


def run_evaluation(args):
    queries_json, docs_json, qrels = load_cranfield(Path(args.dataset))

    query_ids = [q["query number"] for q in queries_json]
    doc_ids = [d["id"] for d in docs_json]
    proc_queries = preprocess([q["query"] for q in queries_json])
    proc_docs = preprocess([d["body"] for d in docs_json])

    # build hybrid index (combines tfidf + lsa)
    retriever = HybridRetrieval(
        n_components=args.lsa_components,
        tfidf_weight=args.tfidf_weight,
        random_state=args.random_state,
        sublinear_tf=not args.disable_sublinear_tf,
        max_df=args.max_df, min_df=args.min_df,
        norm=args.tfidf_norm,
        ngram_range=(1, args.ngram_max),
    )
    retriever.build_index(proc_docs, doc_ids)
    ranked = retriever.rank(proc_queries)

    # evaluate at k = 1..10
    ev = Evaluation()
    precisions, recalls, fscores = [], [], []
    maps, ndcgs, mrrs = [], [], []

    for k in range(1, 11):
        precisions.append(ev.meanPrecision(ranked, query_ids, qrels, k))
        recalls.append(ev.meanRecall(ranked, query_ids, qrels, k))
        fscores.append(ev.meanFscore(ranked, query_ids, qrels, k))
        maps.append(ev.meanAveragePrecision(ranked, query_ids, qrels, k))
        ndcgs.append(ev.meanNDCG(ranked, query_ids, qrels, k))
        mrrs.append(ev.meanReciprocalRank(ranked, query_ids, qrels, k))

        print(f"P, R, F @ {k}: {precisions[-1]:.4f}, {recalls[-1]:.4f}, {fscores[-1]:.4f}")
        print(f"MAP, nDCG, MRR @ {k}: {maps[-1]:.4f}, {ndcgs[-1]:.4f}, {mrrs[-1]:.4f}")

    # plot
    os.makedirs(args.out_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for vals, label in [(precisions, "Precision"), (recalls, "Recall"),
                        (fscores, "F-Score"), (maps, "MAP"),
                        (ndcgs, "nDCG"), (mrrs, "MRR")]:
        plt.plot(range(1, 11), vals, label=label)
    plt.legend()
    plt.title("Hybrid (TF-IDF + LSA) Evaluation - Cranfield")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(Path(args.out_folder) / "hybrid_eval_plot.png")

    # save results
    summary = {
        "lsa_components_requested": args.lsa_components,
        "lsa_components_used": retriever.actual_components,
        "tfidf_weight": retriever.tfidf_weight,
        "explained_variance_ratio_sum": retriever.explained_variance,
        "vectorizer": {
            "sublinear_tf": retriever.sublinear_tf,
            "max_df": retriever.max_df, "min_df": retriever.min_df,
            "norm": retriever.norm,
            "ngram_range": list(retriever.ngram_range),
            "vocab_size": retriever.vocab_size,
        },
        "precision": precisions, "recall": recalls, "fscore": fscores,
        "map": maps, "ndcg": ndcgs, "mrr": mrrs,
    }
    with open(Path(args.out_folder) / "hybrid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nHybrid: {retriever.actual_components} components, "
          f"tfidf_weight={retriever.tfidf_weight}, "
          f"explained var = {retriever.explained_variance:.4f}")


def run_custom_query(args):
    _, docs_json, _ = load_cranfield(Path(args.dataset))
    doc_ids = [d["id"] for d in docs_json]
    proc_docs = preprocess([d["body"] for d in docs_json])

    retriever = HybridRetrieval(
        n_components=args.lsa_components,
        tfidf_weight=args.tfidf_weight,
        random_state=args.random_state,
        sublinear_tf=not args.disable_sublinear_tf,
        max_df=args.max_df, min_df=args.min_df,
        norm=args.tfidf_norm,
        ngram_range=(1, args.ngram_max),
    )
    retriever.build_index(proc_docs, doc_ids)

    query = input("Enter query:\n")
    proc_q = preprocess([query])
    ranked = retriever.rank(proc_q)[0]
    print("\nTop 5 docs:", ranked[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid TF-IDF + LSA retrieval")
    parser.add_argument("-dataset", default=str(PROJECT_ROOT / "cranfield"))
    parser.add_argument("-out_folder", default=str(CURRENT_DIR / "output_hybrid"))
    parser.add_argument("-segmenter", default="punkt", choices=["naive", "punkt"])
    parser.add_argument("-tokenizer", default="ptb", choices=["naive", "ptb"])
    parser.add_argument("-lsa_components", type=int, default=240)
    parser.add_argument("-tfidf_weight", type=float, default=0.2)
    parser.add_argument("-random_state", type=int, default=42)
    parser.add_argument("-max_df", type=float, default=0.95)
    parser.add_argument("-min_df", type=int, default=2)
    parser.add_argument("-tfidf_norm", default="l2", choices=["l1", "l2", "none"])
    parser.add_argument("-ngram_max", type=int, default=1, choices=[1, 2])
    parser.add_argument("-disable_sublinear_tf", action="store_true")
    parser.add_argument("-custom", action="store_true")
    args = parser.parse_args()

    if args.tfidf_norm == "none":
        args.tfidf_norm = None

    t0 = time.time()
    if args.custom:
        run_custom_query(args)
    else:
        run_evaluation(args)
    print(f"Done in {time.time() - t0:.1f}s")
