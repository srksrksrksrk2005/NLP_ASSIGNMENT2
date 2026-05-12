# BM25 Retrieval System

## Approach

BM25 (Best Match 25) is a probabilistic ranking function that extends TF-IDF
with two key improvements:

**1. Term-frequency saturation** — in TF-IDF, a term appearing 100× in a
document scores 100× more than a term appearing once. BM25 saturates this
effect via parameter `k1`: after a few occurrences, extra repetitions give
diminishing returns.

**2. Document length normalisation** — longer documents naturally accumulate
higher raw TF counts. BM25 normalises by document length relative to the
average, controlled by parameter `b`.

### Formula

```
BM25(d,q) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1 × (1 − b + b × |d|/avgdl)]

IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

Default hyperparameters: `k1 = 1.5`, `b = 0.75`.

### Reference
Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.

## Files

| File | Purpose |
|------|---------|
| `bm25_retrieval.py` | `BM25Retrieval` class — no external IR library used |
| `run_bm25.py` | Evaluation script comparing BM25 vs TF-IDF on Cranfield |
| `output_bm25/` | Generated plots, tables, and JSON results |

## Running

```bash
# from repo root
python project/ritisha/bm25/run_bm25.py \
    -dataset cranfield/ \
    -out_folder project/ritisha/bm25/output_bm25/
```

## Outputs

- `eval_plot_bm25_vs_tfidf.png` — metric curves for both methods
- `comparison_table.txt` — k=1..10 table of all metrics
- `bm25_vs_tfidf_results.json` — raw numbers

## Expected behaviour

BM25 typically outperforms TF-IDF on:
- **Short queries** — length normalisation helps avoid bias toward long documents
- **Repeated-term queries** — TF saturation prevents over-weighting of frequent terms
- **Precision-oriented metrics** (MAP, MRR) — more discriminative ranking

## Dependencies

`numpy`, `matplotlib` — both shared with the base assignment.
