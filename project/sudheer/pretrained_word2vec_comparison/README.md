# Pretrained Word2Vec Comparison

This folder compares the existing scratch-trained Sudheer Word2Vec backbones against a pretrained embedding source and a fine-tuned variant.

## What it compares

The comparison is run for the three Word2Vec-backed retrieval ideas already implemented under `sudheer`:

1. `word2vec_average`
2. `word2vec_idf_weighted`
3. `soft_cosine_word2vec`

For each method, the runner compares:

- scratch-trained Word2Vec on Cranfield
- pretrained embeddings from `glove-wiki-gigaword-100`
- the same pretrained vectors fine-tuned on Cranfield

## Run

```bash
py -3 NLP_ASSIGNMENT2/project/sudheer/pretrained_word2vec_comparison/run_experiments.py
```

## Main outputs

- `output/baseline_tfidf/`
- `output/<method>/scratch/`
- `output/<method>/pretrained/`
- `output/<method>/finetuned/`
- `output/<method>/source_overlay.png`
- `output/<method>/example_query_comparison.md`
- `output/summary.json`
- `output/summary_k10.csv`
- `output/comparison_summary.json`
- `experiment_report.md`

## Notes

- The scratch results are reused from the earlier Word2Vec experiments.
- The pretrained and fine-tuned variants use the same downstream retrieval code, so the comparison stays method-consistent.