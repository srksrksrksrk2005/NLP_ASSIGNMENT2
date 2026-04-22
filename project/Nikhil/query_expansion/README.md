# Query Expansion Experiments (Nikhil)

This folder contains a standalone implementation for:

1. WordNet-based query replacement and expansion
2. Embedding-based query replacement and expansion using similarity matrices from:
   - TF-IDF term vectors
   - LSA term vectors
   - ESA-style concept vectors
   - Word2Vec term vectors

The implementation is self-contained inside this folder and does not require editing files outside `Nikhil`.

## Design

- Shared expansion engine: `expansion/base.py`
  - Performs replacement for OOV query terms.
  - Spreads query term-frequency mass to similar in-vocabulary words.
- WordNet matrix and OOV resolver: `expansion/wordnet_matrix.py`
- Embedding matrix builders: `expansion/embedding_matrices.py`
- Cranfield runner: `run_experiments.py`
- Shared core modules:
  - `core/preprocessing.py`
  - `core/retrieval.py`
  - `core/evaluation.py`
  - `core/data.py`

## How to run

From `NLP_ASSIGNMENT2/project/Nikhil/query_expansion`:

```bash
python run_experiments.py
```

Default methods run:

- `wordnet`
- `embedding_tfidf`
- `embedding_lsa`
- `embedding_esa`
- `embedding_word2vec`

## Useful options

```bash
python run_experiments.py \
  --methods wordnet,embedding_tfidf,embedding_lsa,embedding_esa,embedding_word2vec \
  --top-k-neighbors 10 \
  --min-similarity 0.05 \
  --expansion-weight 0.35 \
  --replacement-weight 1.0
```

## Output

Outputs are saved under `output/`:

- `summary.json`
- `summary_k10.csv`
- Per-method folders with:
  - `metrics.json`
  - `rankings_top20.json`
  - `query_expansion_preview.json`

## Notes

- OOV terms are always handled through WordNet replacement into vocabulary terms.
- Embedding methods differ only in how the in-vocabulary word-word similarity matrix is built.
