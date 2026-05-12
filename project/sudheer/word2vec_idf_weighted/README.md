# Word2Vec IDF-Weighted Backbone

This experiment uses Word2Vec token embeddings, but instead of plain averaging it weights each token vector by TF-IDF-style importance before computing the document/query centroid.

## Retrieval idea

1. Train Word2Vec on the stopword-removed Cranfield documents.
2. Weight each token vector by `tf * idf`.
3. Average the weighted vectors to get one dense vector per document/query.
4. Rank documents with cosine similarity.

## Paper grounding

- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013)
- Arora, Liang, and Ma, *A Simple but Tough-to-Beat Baseline for Sentence Embeddings* (ICLR 2017)

## Run

```bash
python NLP_ASSIGNMENT2/project/sudheer/word2vec_idf_weighted/run_experiments.py
```

## Main outputs

- `output/summary.json`
- `output/summary_k10.csv`
- `output/config_sweep.json`
- `output/vector_size_sweep_best.json`
- `output/vector_size_sweep_best.csv`
- `output/vector_size_sweep_best.png`
- `output/eval_overlay.png`
- `output/all_tuned_combinations_overlay.png`
- `experiment_report.md`

## Notes

- This is a stronger retrieval backbone than plain averaging when rare technical terms matter.
- It still compresses each item into one vector, so phrase structure is not explicit.
