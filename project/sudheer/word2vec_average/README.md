# Word2Vec Average Backbone

This experiment treats Word2Vec as a true retrieval backbone by converting each document and query into a single centroid embedding.

## Retrieval idea

1. Train Word2Vec on the stopword-removed Cranfield documents.
2. Represent each document by the mean of its in-vocabulary token vectors.
3. Represent each query the same way.
4. Rank documents with cosine similarity in the dense embedding space.

## Paper grounding

- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013)
- Arora, Liang, and Ma, *A Simple but Tough-to-Beat Baseline for Sentence Embeddings* (ICLR 2017)

## Run

```bash
python NLP_ASSIGNMENT2/project/sudheer/word2vec_average/run_experiments.py
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

- This is the simplest Word2Vec backbone.
- It is fast and easy to justify.
- It does not preserve phrase order or explicit local context.
