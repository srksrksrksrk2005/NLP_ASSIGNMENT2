# N-gram IR System

## Approach

The base VSM (TF-IDF unigrams) treats every word independently, so it misses
phrase-level evidence. For example, a document containing "boundary layer" as a
compound concept scores the same as one that mentions "boundary" and "layer" in
completely different contexts.

This module extends the vocabulary with **bigrams** (consecutive word pairs),
giving the retriever access to local word-order information without any
structural change to the cosine-similarity ranking pipeline.

### How it works

1. During indexing, every document sentence is scanned for unigrams **and**
   bigrams. A bigram `(w1, w2)` is stored as the token `"w1_w2"`.
2. TF-IDF weights are computed over the combined unigram + bigram vocabulary
   exactly as in the base system.
3. At query time, the same n-gram extraction is applied to the query before
   computing cosine similarity.

## Files

| File | Purpose |
|------|---------|
| `ngram_retrieval.py` | `NgramRetrieval` class (drop-in replacement for `InformationRetrieval`) |
| `run_ngram.py` | End-to-end evaluation script (same metrics as base system) |
| `output_ngram/` | Generated plots and result files |

## Running

```bash
# from the repo root
python project/ritisha/ngram/run_ngram.py \
    -dataset cranfield/ \
    -out_folder project/ritisha/ngram/output_ngram/
```

## Dependencies

All dependencies are shared with the base assignment:
`numpy`, `nltk`, `matplotlib`

## Expected improvement

Bigrams help most for queries that contain domain-specific phrases common in
aeronautics literature (e.g., "heat transfer", "boundary layer", "mach number").
The MAP and nDCG scores are expected to improve over the unigram baseline for
such phrase queries.
