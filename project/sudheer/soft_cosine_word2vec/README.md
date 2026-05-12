# Word2Vec Soft Cosine Backbone

This experiment keeps explicit TF-IDF term vectors, but changes the scoring function so semantically related terms can partially match each other through Word2Vec similarity.

## Retrieval idea

1. Build the baseline unigram TF-IDF representation.
2. Train Word2Vec on the stopword-removed Cranfield documents.
3. Build a sparse term-similarity matrix from top Word2Vec neighbors.
4. Score documents with soft cosine instead of ordinary cosine.

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

## Notes

- This is the cleanest Word2Vec-informed bridge between lexical retrieval and semantic retrieval.
- It is more interpretable than a pure dense centroid method because the representation stays term-based.
