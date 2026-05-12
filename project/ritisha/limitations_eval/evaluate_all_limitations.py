"""
Usage (from repo root):
    python project/ritisha/limitations_eval/evaluate_all_limitations.py \
        --dataset cranfield/ \
        --test_set project/limitation_test_set_expanded.json \
        --out_folder project/ritisha/limitations_eval/output_limitations_all/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "project/ritisha/ngram"))
sys.path.insert(0, str(ROOT / "project/ritisha/wordnet"))
sys.path.insert(0, str(ROOT / "project/ritisha/bm25"))

sys.path.insert(0, str(ROOT / "project/ramakrishna/lsa"))
sys.path.insert(0, str(ROOT / "project/ramakrishna/esa"))

EXPANSION_ROOT = ROOT / "project/Nikhil/query_expansion"
sys.path.insert(0, str(EXPANSION_ROOT))
sys.path.insert(0, str(EXPANSION_ROOT / "core"))
sys.path.insert(0, str(EXPANSION_ROOT / "expansion"))

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval

from ngram_retrieval   import NgramRetrieval
from wordnet_retrieval import WordNetRetrieval
from bm25_retrieval    import BM25Retrieval
from lsa_retrieval     import LSARetrieval
from esa_retrieval     import ESARetrieval

try:
    from core.preprocessing import Preprocessor as ExpPreprocessor
    from core.retrieval import VectorSpaceRetrieval, flatten_query_tokens
    from expansion.base import MatrixQueryExpander
    from expansion.wordnet_matrix import build_wordnet_neighbor_map, WordNetOOVResolver
    from expansion.embedding_matrices import build_lsa_neighbor_map, build_esa_neighbor_map
    NIKHIL_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"Query expansion modules unavailable ({exc}). Expansion methods will be skipped.")
    NIKHIL_AVAILABLE = False


_seg = SentenceSegmentation()
_tok = Tokenization()
_red = InflectionReduction()
_sw  = StopwordRemoval()

def preprocess_base(text: str) -> List[List[str]]:
    return _sw.fromList(_red.reduce(_tok.pennTreeBank(_seg.punkt(text))))

def preprocess_corpus_base(texts: List[str]) -> List[List[List[str]]]:
    return [preprocess_base(t) for t in texts]


def map_at_k(ranked: List, relevant: List, k: int = 10) -> Optional[float]:
    if not relevant:
        return None
    rel  = set(str(r) for r in relevant)
    hits = 0
    total = 0.0
    for i, doc in enumerate(ranked[:k]):
        if str(doc) in rel:
            hits  += 1
            total += hits / (i + 1)
    return total / min(len(rel), k)

def eval_limitation(system, queries: List[Tuple], k: int = 10) -> float:
    """system.rank([proc_q]) must return [[doc_id, ...]]"""
    scores = []
    for proc_q, rel_ids in queries:
        if not rel_ids:
            continue
        s = map_at_k(system.rank([proc_q])[0], rel_ids, k)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class _SparseTfidf:
    doc_ids: List
    idf: Dict[str, float]
    postings: Dict

    @classmethod
    def build(cls, feature_maps: List[Dict], doc_ids: List) -> "_SparseTfidf":
        doc_ids = list(doc_ids)
        df = Counter()
        for fm in feature_maps:
            df.update(fm.keys())
        N   = float(len(feature_maps))
        idf = {f: math.log(N / d) for f, d in df.items() if d}

        postings: Dict = defaultdict(list)
        for di, fm in enumerate(feature_maps):
            raw, nsq = {}, 0.0
            for feat, tf in fm.items():
                w = tf * idf.get(feat, 0.0)
                if w > 0:
                    raw[feat] = w
                    nsq += w * w
            if nsq == 0:
                continue
            n = math.sqrt(nsq)
            for feat, w in raw.items():
                postings[feat].append((di, w / n))
        return cls(doc_ids=doc_ids, idf=idf, postings=dict(postings))

    def score(self, qmap: Dict) -> np.ndarray:
        sc = np.zeros(len(self.doc_ids), dtype=np.float64)
        rq, nsq = {}, 0.0
        for f, tf in qmap.items():
            w = tf * self.idf.get(f, 0.0)
            if w > 0:
                rq[f] = w
                nsq  += w * w
        if nsq == 0:
            return sc
        n = math.sqrt(nsq)
        for f, w in rq.items():
            qw = w / n
            for di, dw in self.postings.get(f, ()):
                sc[di] += qw * dw
        return sc

    def rank(self, qmaps: List[Dict]) -> List[List]:
        out = []
        for qm in qmaps:
            sc    = self.score(qm)
            order = sorted(range(len(self.doc_ids)),
                           key=lambda i: (-sc[i], self.doc_ids[i]))
            out.append([self.doc_ids[i] for i in order])
        return out


def _unigram_map(doc: List[List[str]]) -> Dict[str, float]:
    c = Counter()
    for sent in doc:
        c.update(sent)
    return dict(c)


def _ctx_map(doc: List[List[str]], radius: int, orders: tuple) -> Dict[str, float]:
    feats: set = set()
    for sent in doc:
        n = len(sent)
        for i in range(n):
            bag = sorted(set(sent[max(0, i - radius): min(n, i + radius + 1)]))
            for order in orders:
                if len(bag) >= order:
                    for combo in combinations(bag, order):
                        feats.add(f"ctx{order}:" + "__".join(combo))
    return {f: 1.0 for f in feats}


class LocalContextBOW:
    def __init__(self, proc_docs: List, doc_ids: List,
                 radius: int = 4, orders: tuple = (2, 3),
                 min_df: int = 2, alpha: float = 0.35):
        self.doc_ids = list(doc_ids)
        self.alpha   = alpha
        self.radius  = radius
        self.orders  = orders
        self.min_df  = min_df

        self.baseline = _SparseTfidf.build(
            [_unigram_map(d) for d in proc_docs], doc_ids)

        raw_ctx    = [_ctx_map(d, radius, orders) for d in proc_docs]
        self._df   = Counter()
        for cm in raw_ctx:
            self._df.update(cm.keys())
        filtered   = [{f: v for f, v in cm.items() if self._df[f] >= min_df}
                      for cm in raw_ctx]
        self.ctx   = _SparseTfidf.build(filtered, doc_ids)

    def rank(self, queries: List[List[List[str]]]) -> List[List]:
        out = []
        for q in queries:
            umap     = _unigram_map(q)
            cmap     = {f: v for f, v in _ctx_map(q, self.radius, self.orders).items()
                        if self._df.get(f, 0) >= self.min_df}
            combined = (self.baseline.score(umap)
                        + self.alpha * self.ctx.score(cmap))
            order    = sorted(range(len(self.doc_ids)),
                              key=lambda i: (-combined[i], self.doc_ids[i]))
            out.append([self.doc_ids[i] for i in order])
        return out


class QueryReductionSystem:
    def __init__(self, proc_docs: List, doc_ids: List, keep_k: int = 5):
        self.keep_k = keep_k
        self.model  = _SparseTfidf.build(
            [_unigram_map(d) for d in proc_docs], doc_ids)

    def _reduce(self, q: List[List[str]]) -> Dict[str, float]:
        tokens = [t for sent in q for t in sent]
        kept   = sorted(tokens,
                        key=lambda t: self.model.idf.get(t, 0.0),
                        reverse=True)[:max(1, self.keep_k)]
        return dict(Counter(kept))

    def rank(self, queries: List[List[List[str]]]) -> List[List]:
        return self.model.rank([self._reduce(q) for q in queries])


class ExpansionWrapper:
    def __init__(self, retriever, expander=None):
        self.retriever = retriever
        self.expander  = expander

    def rank(self, queries: List[List[List[str]]]) -> List[List]:
        n       = len(queries)
        vectors = np.zeros((n, len(self.retriever.vocab)), dtype=np.float64)
        for i, query in enumerate(queries):
            counts = Counter(flatten_query_tokens(query))
            if self.expander is not None:
                vectors[i] = self.expander.build_query_vector_from_counts(counts)
            else:
                for term, cnt in counts.items():
                    idx = self.retriever.term_to_idx.get(term)
                    if idx is not None:
                        vectors[i, idx] += float(cnt)
        return self.retriever.query_vectors_to_rankings(vectors)


def build_base_systems(proc_docs, doc_ids) -> Dict:
    systems = {}
    print("  [1/6] TF-IDF …")
    ir = InformationRetrieval()
    ir.buildIndex(proc_docs, doc_ids)
    systems["TF-IDF"] = ir

    print("  [2/6] N-gram (n=2) …")
    ng = NgramRetrieval(n=2)
    ng.buildIndex(proc_docs, doc_ids)
    systems["N-gram"] = ng

    print("  [3/6] WordNet local context …")
    wn = WordNetRetrieval(max_synonyms=1, context_window=5)
    wn.buildIndex(proc_docs, doc_ids)
    systems["WordNet-LC"] = wn

    print("  [4/6] BM25 …")
    bm = BM25Retrieval(k1=1.5, b=0.75)
    bm.buildIndex(proc_docs, doc_ids)
    systems["BM25"] = bm

    print("  [5/6] LSA (128 components) …")
    lsa = LSARetrieval(n_components=128)
    lsa.build_index(proc_docs, doc_ids)
    systems["LSA"] = lsa

    print("  [6/6] ESA (top 50 concepts) …")
    esa = ESARetrieval(top_concepts=50)
    esa.build_index(proc_docs, doc_ids)
    systems["ESA"] = esa

    return systems


def build_local_systems(proc_docs, doc_ids) -> Dict:
    systems = {}
    try:
        print("  [C1] LocalContext-BOW ...")
        systems["LocalContext-BOW"] = LocalContextBOW(proc_docs, doc_ids)
        print("  [C2] QueryReduction ...")
        systems["QueryReduction"] = QueryReductionSystem(proc_docs, doc_ids)
    except Exception as exc:
        warnings.warn(f"LocalContext/QueryReduction build failed: {exc}")
    return systems


def build_expansion_systems(doc_texts, doc_ids) -> Dict:
    if not NIKHIL_AVAILABLE:
        return {}
    systems = {}
    try:
        print("  [N1] Expansion preprocessing ...")
        pre       = ExpPreprocessor(use_lemmatization=True)
        proc_docs = pre.preprocess_corpus(doc_texts)
        str_ids   = [str(d) for d in doc_ids]

        print("  [N2] Expansion retriever ...")
        retriever = VectorSpaceRetrieval()
        retriever.build(proc_docs, str_ids)
        doc_tfidf = retriever.doc_tfidf
        systems["Exp-TF-IDF"] = ExpansionWrapper(retriever)

        print("  [N3] WordNet expansion ...")
        wn_map = build_wordnet_neighbor_map(
            retriever.vocab, top_k=10, min_similarity=0.08, progress=False)
        oov = WordNetOOVResolver(retriever.vocab, progress=False)
        systems["Exp-WordNet"] = ExpansionWrapper(
            retriever,
            MatrixQueryExpander(vocab=retriever.vocab, neighbors=wn_map,
                                oov_resolver=oov.resolve,
                                self_weight=1.0, expansion_weight=0.20))

        print("  [N4] LSA expansion ...")
        lsa_map = build_lsa_neighbor_map(
            doc_tfidf, retriever.vocab, top_k=10, min_similarity=0.08)
        systems["Exp-LSA"] = ExpansionWrapper(
            retriever,
            MatrixQueryExpander(vocab=retriever.vocab, neighbors=lsa_map,
                                oov_resolver=oov.resolve,
                                self_weight=1.0, expansion_weight=0.20))

        print("  [N5] ESA expansion ...")
        esa_map = build_esa_neighbor_map(
            doc_tfidf, retriever.vocab,
            top_concepts=100, top_k=10, min_similarity=0.08)
        systems["Exp-ESA"] = ExpansionWrapper(
            retriever,
            MatrixQueryExpander(vocab=retriever.vocab, neighbors=esa_map,
                                oov_resolver=oov.resolve,
                                self_weight=1.0, expansion_weight=0.20))

    except Exception as exc:
        warnings.warn(f"Expansion system build failed: {exc}")
    return systems


def build_lim_queries(limitations: dict,
                      preprocessor) -> Dict[str, List[Tuple]]:
    """preprocessor: text (str) -> List[List[str]]"""
    out = {}
    for lim_name, lim_data in limitations.items():
        qs = []
        for q in lim_data.get("cranfield_queries", []):
            qs.append((preprocessor(q["query"]), q.get("relevant_docs", [])))
        for q in lim_data.get("custom_queries", []):
            qs.append((preprocessor(q["query"]),
                       q.get("expected_relevant_docs", [])))
        out[lim_name] = qs
    return out


LIM_ABBREV = {
    "lack_of_semantic_understanding":             "Semantic",
    "high_dimensionality_and_computational_cost": "Dim/Cost",
    "poor_scalability":                           "Scale",
    "word_sense_ambiguity":                       "Ambiguity",
    "out_of_vocabulary":                          "OOV",
    "lack_of_contextual_representation":          "Context",
    "sparse_representations":                     "Sparse",
}

def plot_heatmap(results, lim_names, methods, out_path):
    data = np.array([[results[l].get(m, 0.0) for m in methods]
                     for l in lim_names])
    fig, ax = plt.subplots(
        figsize=(max(10, len(methods) * 1.3), max(5, len(lim_names) * 0.85)))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(lim_names)))
    ax.set_yticklabels([LIM_ABBREV.get(l, l) for l in lim_names], fontsize=9)
    for i in range(len(lim_names)):
        for j in range(len(methods)):
            ax.text(j, i, f"{data[i,j]:.3f}",
                    ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="MAP@10")
    ax.set_title("MAP@10 — All Methods × All Limitations", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Heatmap saved → {out_path}")


def plot_bars(results, lim_names, methods, out_path):
    ncols = math.ceil(len(lim_names) / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(max(14, ncols * 2.5), 8))
    axes      = axes.flatten()
    colors    = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    for i, lim in enumerate(lim_names):
        ax     = axes[i]
        scores = [results[lim].get(m, 0.0) for m in methods]
        ax.bar(range(len(methods)), scores, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=55, ha="right", fontsize=6)
        ax.set_title(LIM_ABBREV.get(lim, lim), fontsize=8)
        ax.set_ylim(0, max(0.5, max(scores) * 1.15) if scores else 0.5)
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("MAP@10 per Limitation — All Methods", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Bar chart saved → {out_path}")


def build_table(results, lim_names, methods) -> str:
    CW, LW = 9, 45
    lines  = []

    hdr = f"{'Limitation':<{LW}}" + "".join(f"{m:>{CW}}" for m in methods)
    lines += ["", "MAP@10 — All Methods x All Limitations",
              "=" * len(hdr), hdr, "-" * len(hdr)]

    for lim in lim_names:
        best = max(results[lim].values(), default=0.0)
        row  = f"{LIM_ABBREV.get(lim, lim):<{LW}}"
        for m in methods:
            sc  = results[lim].get(m, float("nan"))
            mrk = ("*" if (not math.isnan(sc) and
                           abs(sc - best) < 1e-6 and best > 0)
                   else " ")
            row += f"{sc:>{CW-1}.4f}{mrk}"
        lines.append(row)

    lines += ["", "* = best for that limitation", ""]

    lines.append("Best method per limitation:")
    for lim in lim_names:
        if not results[lim]:
            continue
        bm = max(results[lim], key=lambda m: results[lim].get(m, 0))
        lines.append(f"  {LIM_ABBREV.get(lim, lim):<20} "
                     f"-> {bm:<28} ({results[lim][bm]:.4f})")

    nb = [m for m in methods if m != "TF-IDF"]
    lines += ["", "Improvement over TF-IDF (check = delta MAP@10 >= 0.005):"]
    hdr2 = f"{'Limitation':<{LW}}" + "".join(f"{m[:CW-1]:>{CW}}" for m in nb)
    lines += [hdr2, "-" * len(hdr2)]
    for lim in lim_names:
        base = results[lim].get("TF-IDF", 0.0)
        row  = f"{LIM_ABBREV.get(lim, lim):<{LW}}"
        for m in nb:
            row += f"{'YES' if results[lim].get(m,0)-base>=0.005 else 'no':>{CW}}"
        lines.append(row)

    lines += ["", "Average MAP@10 across all limitations:"]
    for m in methods:
        vals = [results[l].get(m, 0.0) for l in lim_names if m in results[l]]
        avg  = float(np.mean(vals)) if vals else float("nan")
        lines.append(f"  {m:<32} {avg:.4f}")

    return "\n".join(lines)


def main(args):
    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Cranfield corpus ...")
    docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json")))
    doc_ids   = [item["id"]   for item in docs_json]
    doc_texts = [item["body"] for item in docs_json]

    print("Loading limitation test set ...")
    test_set    = json.load(open(args.test_set))
    limitations = test_set["limitations"]
    lim_names   = list(limitations.keys())

    # Preprocess corpus
    print("\nPreprocessing corpus (base pipeline) ...")
    t0 = time.time()
    proc_docs_base = preprocess_corpus_base(doc_texts)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Build systems
    print("\nBuilding base systems (TF-IDF, N-gram, WordNet-LC, BM25, LSA, ESA) ...")
    t0 = time.time()
    base_systems = build_base_systems(proc_docs_base, doc_ids)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nBuilding local-context and query-reduction systems ...")
    t0 = time.time()
    local_systems = build_local_systems(proc_docs_base, doc_ids)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nBuilding query expansion systems ...")
    t0 = time.time()
    expansion_systems = build_expansion_systems(doc_texts, doc_ids)
    print(f"  Done in {time.time()-t0:.1f}s" if expansion_systems else "  Skipped.")

    # Preprocess limitation test queries
    print("\nPreprocessing limitation test queries ...")
    lim_q_base   = build_lim_queries(limitations, preprocess_base)
    lim_q_exp = {}
    if expansion_systems and NIKHIL_AVAILABLE:
        exp_pre = ExpPreprocessor(use_lemmatization=True)
        lim_q_exp = build_lim_queries(limitations, exp_pre.preprocess_text)

    # Evaluate
    results: Dict[str, Dict[str, float]] = {l: {} for l in lim_names}

    print("\nEvaluating base systems ...")
    for mname, sys_ in base_systems.items():
        print(f"  {mname} ...")
        for lim in lim_names:
            results[lim][mname] = eval_limitation(sys_, lim_q_base[lim])

    print("\nEvaluating local-context and query-reduction systems (per-limitation) ...")
    for mname, sys_ in local_systems.items():
        print(f"  {mname} ...")
        for lim in lim_names:
            results[lim][mname] = eval_limitation(sys_, lim_q_base[lim])

    if expansion_systems:
        print("\nEvaluating query expansion systems ...")
        for mname, sys_ in expansion_systems.items():
            print(f"  {mname} ...")
            for lim in lim_names:
                results[lim][mname] = eval_limitation(sys_, lim_q_exp[lim])

    # Method order for table/plots
    all_methods = (list(base_systems.keys())
                   + list(local_systems.keys())
                   + list(expansion_systems.keys()))

    # Save JSON
    json.dump(results,
              open(out_dir / "all_limitations_results.json", "w"), indent=2)

    # Table
    table = build_table(results, lim_names, all_methods)
    print("\n" + table)
    (out_dir / "all_limitations_table.txt").write_text(table, encoding="utf-8")
    print(f"\nTable saved -> {out_dir / 'all_limitations_table.txt'}")

    # Plots
    print("\nGenerating plots ...")
    plot_heatmap(results, lim_names, all_methods,
                 out_dir / "heatmap_all_methods.png")
    plot_bars(results, lim_names, all_methods,
              out_dir / "bar_per_limitation.png")

    print(f"\nAll done. Outputs in: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Unified per-limitation evaluation — all methods")
    ap.add_argument("--dataset",    default="cranfield/")
    ap.add_argument("--test_set",
                    default="project/limitation_test_set_expanded.json")
    ap.add_argument("--out_folder",
                    default="project/ritisha/limitations_eval/output_limitations_all/")
    main(ap.parse_args())
