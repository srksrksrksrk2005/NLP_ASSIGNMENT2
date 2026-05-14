
import sys, os, argparse, json, time
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT)

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval   # baseline TF-IDF
from evaluation import Evaluation

sys.path.insert(0, os.path.dirname(__file__))
from bm25_retrieval import BM25Retrieval

def preprocess(texts, args):
    segmenter  = SentenceSegmentation()
    tokenizer  = Tokenization()
    reducer    = InflectionReduction()
    sw_remover = StopwordRemoval()
    processed  = []
    for text in texts:
        segs    = segmenter.punkt(text)   if args.segmenter == "punkt" else segmenter.naive(text)
        toks    = tokenizer.pennTreeBank(segs) if args.tokenizer == "ptb" else tokenizer.naive(segs)
        reduced = reducer.reduce(toks)
        cleaned = sw_remover.fromList(reduced)
        processed.append(cleaned)
    return processed

def evaluate(retriever, doc_IDs_ordered, query_ids, qrels):
    """Return dict of lists: precision, recall, fscore, MAP, nDCG, MRR over k=1..10."""
    ev = Evaluation()
    out = {m: [] for m in ["precision", "recall", "fscore", "MAP", "nDCG", "MRR"]}
    for k in range(1, 11):
        out["precision"].append(ev.meanPrecision(doc_IDs_ordered, query_ids, qrels, k))
        out["recall"].append(   ev.meanRecall(   doc_IDs_ordered, query_ids, qrels, k))
        out["fscore"].append(   ev.meanFscore(   doc_IDs_ordered, query_ids, qrels, k))
        out["MAP"].append(      ev.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k))
        out["nDCG"].append(     ev.meanNDCG(     doc_IDs_ordered, query_ids, qrels, k))
        out["MRR"].append(      ev.meanReciprocalRank(  doc_IDs_ordered, query_ids, qrels, k))
    return out

def main(args):
    os.makedirs(args.out_folder, exist_ok=True)

    # Load data
    queries_json = json.load(open(os.path.join(args.dataset, "cran_queries.json")))
    query_ids    = [item["query number"] for item in queries_json]
    queries      = [item["query"]        for item in queries_json]

    docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json")))
    doc_ids   = [item["id"]   for item in docs_json]
    docs      = [item["body"] for item in docs_json]

    qrels = json.load(open(os.path.join(args.dataset, "cran_qrels.json")))

    print("Preprocessing …")
    proc_docs    = preprocess(docs,    args)
    proc_queries = preprocess(queries, args)

    print("\nBuilding TF-IDF index …")
    t0 = time.time()
    tfidf = InformationRetrieval()
    tfidf.buildIndex(proc_docs, doc_ids)
    tfidf_ranked = tfidf.rank(proc_queries)
    t_tfidf = time.time() - t0
    print(f"TF-IDF done in {t_tfidf:.2f}s")

    print("\nBuilding BM25 index …")
    t0 = time.time()
    bm25 = BM25Retrieval(k1=1.5, b=0.75)
    bm25.buildIndex(proc_docs, doc_ids)
    bm25_ranked = bm25.rank(proc_queries)
    t_bm25 = time.time() - t0
    print(f"BM25 done in {t_bm25:.2f}s")

    print("\nEvaluating …")
    tfidf_res = evaluate(tfidf, tfidf_ranked, query_ids, qrels)
    bm25_res  = evaluate(bm25,  bm25_ranked,  query_ids, qrels)

    header = (f"{'k':>3}  {'P_tf':>6} {'P_bm':>6}  "
              f"{'R_tf':>6} {'R_bm':>6}  "
              f"{'MAP_tf':>7} {'MAP_bm':>7}  "
              f"{'nDCG_tf':>8} {'nDCG_bm':>8}  "
              f"{'MRR_tf':>7} {'MRR_bm':>7}")
    print("\n" + header)
    print("-" * len(header))
    table_lines = [header, "-" * len(header)]
    for k in range(10):
        line = (f"{k+1:>3}  "
                f"{tfidf_res['precision'][k]:.4f} {bm25_res['precision'][k]:.4f}  "
                f"{tfidf_res['recall'][k]:.4f} {bm25_res['recall'][k]:.4f}  "
                f"{tfidf_res['MAP'][k]:.5f} {bm25_res['MAP'][k]:.5f}  "
                f"{tfidf_res['nDCG'][k]:.6f} {bm25_res['nDCG'][k]:.6f}  "
                f"{tfidf_res['MRR'][k]:.5f} {bm25_res['MRR'][k]:.5f}")
        print(line)
        table_lines.append(line)

    table_lines.append(f"\nTF-IDF total time: {t_tfidf:.2f}s")
    table_lines.append(f"BM25   total time: {t_bm25:.2f}s")

    with open(os.path.join(args.out_folder, "comparison_table.txt"), "w") as f:
        f.write("\n".join(table_lines))

    json.dump({"tfidf": tfidf_res, "bm25": bm25_res},
              open(os.path.join(args.out_folder, "bm25_vs_tfidf_results.json"), "w"),
              indent=2)

    ks      = list(range(1, 11))
    metrics = ["precision", "recall", "MAP", "nDCG", "MRR"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))

    for ax, metric in zip(axes, metrics):
        ax.plot(ks, tfidf_res[metric], marker='o', label="TF-IDF")
        ax.plot(ks, bm25_res[metric],  marker='s', label="BM25")
        ax.set_title(metric)
        ax.set_xlabel("k")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("BM25 vs TF-IDF — Cranfield Dataset", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.out_folder, "eval_plot_bm25_vs_tfidf.png")
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot saved to {plot_path}")

    print("\n── Summary at k=10 ──────────────────────────────────────")
    for m in metrics:
        tf_v  = tfidf_res[m][-1]
        bm_v  = bm25_res[m][-1]
        delta = bm_v - tf_v
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {m:<12} TF-IDF={tf_v:.4f}  BM25={bm_v:.4f}  Δ={delta:+.4f} {arrow}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset",    default="cranfield/")
    parser.add_argument("-out_folder", default="project/ritisha/bm25/output_bm25/")
    parser.add_argument("-segmenter",  default="punkt")
    parser.add_argument("-tokenizer",  default="ptb")
    args = parser.parse_args()
    main(args)
