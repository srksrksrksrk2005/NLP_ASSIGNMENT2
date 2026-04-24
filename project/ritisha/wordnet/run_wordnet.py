"""
run_wordnet.py  –  Evaluate the WordNet local-context IR system on Cranfield.

Usage (from repo root):
    python project/ritisha/wordnet/run_wordnet.py \
        -dataset cranfield/ \
        -out_folder project/ritisha/wordnet/output_wordnet/

NOTE: WordNet expansion makes indexing slower than the base system (~5-10 min
for all 1400 Cranfield docs). This is expected — no GPU or external API needed.
"""

import sys
import os
import argparse
import json
import time
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT)

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from evaluation import Evaluation

from wordnet_retrieval import WordNetRetrieval


def preprocess(texts, args):
    segmenter  = SentenceSegmentation()
    tokenizer  = Tokenization()
    reducer    = InflectionReduction()
    sw_remover = StopwordRemoval()

    processed = []
    for text in texts:
        if args.segmenter == "punkt":
            segs = segmenter.punkt(text)
        else:
            segs = segmenter.naive(text)

        if args.tokenizer == "ptb":
            toks = tokenizer.pennTreeBank(segs)
        else:
            toks = tokenizer.naive(segs)

        reduced = reducer.reduce(toks)
        cleaned = sw_remover.fromList(reduced)
        processed.append(cleaned)

    return processed


def main(args):
    os.makedirs(args.out_folder, exist_ok=True)

    queries_json = json.load(open(os.path.join(args.dataset, "cran_queries.json")))
    query_ids    = [item["query number"] for item in queries_json]
    queries      = [item["query"] for item in queries_json]

    docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json")))
    doc_ids   = [item["id"]   for item in docs_json]
    docs      = [item["body"] for item in docs_json]

    qrels = json.load(open(os.path.join(args.dataset, "cran_qrels.json")))
    print("Preprocessing documents …")
    processed_docs    = preprocess(docs,    args)

    print("Preprocessing queries …")
    processed_queries = preprocess(queries, args)

    retriever = WordNetRetrieval(max_synonyms=3, context_window=10)

    print("\nBuilding WordNet-expanded index …  (this may take several minutes)")
    t0 = time.time()
    retriever.buildIndex(processed_docs, doc_ids)
    t1 = time.time()

    print("Ranking …")
    doc_IDs_ordered = retriever.rank(processed_queries)
    t2 = time.time()

    print(f"\nIndex build time : {t1 - t0:.2f}s")
    print(f"Ranking time     : {t2 - t1:.2f}s")
    print(f"Total time       : {t2 - t0:.2f}s\n")

    evaluator = Evaluation()
    precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

    for k in range(1, 11):
        precision = evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
        recall    = evaluator.meanRecall   (doc_IDs_ordered, query_ids, qrels, k)
        fscore    = evaluator.meanFscore   (doc_IDs_ordered, query_ids, qrels, k)
        MAP       = evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
        nDCG      = evaluator.meanNDCG     (doc_IDs_ordered, query_ids, qrels, k)
        MRR       = evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k)

        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        MAPs.append(MAP)
        nDCGs.append(nDCG)
        MRRs.append(MRR)

        print(f"k={k:2d}  P={precision:.4f}  R={recall:.4f}  F={fscore:.4f}  "
              f"MAP={MAP:.4f}  nDCG={nDCG:.4f}  MRR={MRR:.4f}")
        
    ks = range(1, 11)
    plt.figure(figsize=(9, 5))
    plt.plot(ks, precisions, marker='o', label="Precision")
    plt.plot(ks, recalls,    marker='s', label="Recall")
    plt.plot(ks, fscores,    marker='^', label="F-Score")
    plt.plot(ks, MAPs,       marker='D', label="MAP")
    plt.plot(ks, nDCGs,      marker='x', label="nDCG")
    plt.plot(ks, MRRs,       marker='*', label="MRR")
    plt.xlabel("k")
    plt.title("WordNet Local-Context IR System – Evaluation Metrics (Cranfield)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out_folder, "eval_plot_wordnet.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WordNet local-context IR evaluation")
    parser.add_argument("-dataset",    default="cranfield/")
    parser.add_argument("-out_folder", default="project/ritisha/wordnet/output_wordnet/")
    parser.add_argument("-segmenter",  default="punkt")
    parser.add_argument("-tokenizer",  default="ptb")
    args = parser.parse_args()
    main(args)