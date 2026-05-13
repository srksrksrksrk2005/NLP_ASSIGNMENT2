# Word2Vec Soft Cosine Backbone

This experiment keeps explicit TF-IDF term vectors, but changes the scoring function so semantically related terms can partially match each other through Word2Vec similarity.

It now also includes an internal comparison against a TF-IDF-derived term-similarity matrix, so the folder can compare:

1. TF-IDF vectors + Word2Vec-derived `S`
2. TF-IDF vectors + TF-IDF-derived `S`

## Retrieval idea

1. Build the baseline unigram TF-IDF representation.
2. Train Word2Vec on the stopword-removed Cranfield documents.
3. Build a sparse term-similarity matrix from top Word2Vec neighbors.
4. Score documents with soft cosine instead of ordinary cosine.
5. Compare the Word2Vec-derived similarity matrix against a TF-IDF-derived similarity matrix built on the same term space.

## Paper grounding

- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013)
- Sidorov et al., *Soft Similarity and Soft Cosine Measure* (2014)

## Run

```bash
python NLP_ASSIGNMENT2/project/sudheer/soft_cosine_word2vec/run_experiments.py
```

## Main outputs

- `output/summary.json`
- `output/summary_k10.csv`
- `output/config_sweep.json`
- `output/neighbor_sweep_best.json`
- `output/neighbor_sweep_best.csv`
- `output/neighbor_sweep_best.png`
- `output/eval_overlay.png`
- `output/all_tuned_combinations_overlay.png`
- `experiment_report.md`
- `output/soft_cosine_tfidf_source/metrics.json`
- `output/soft_cosine_tfidf_source/sweep_results.json`

## Notes

- This is the cleanest Word2Vec-informed bridge between lexical retrieval and semantic retrieval.
- It is more interpretable than a pure dense centroid method because the representation stays term-based.
- The main experiment still uses `S` from Word2Vec neighbors, but the folder now records whether a TF-IDF-derived `S` performs better or worse on the same TF-IDF backbone.
