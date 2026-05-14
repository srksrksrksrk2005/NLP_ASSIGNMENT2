# Smooth Inverse Frequency Embeddings

This folder contains the `project/chandan` implementation of pure SIF-based dense retrieval for Cranfield.

## What the method does

- Start from the same stopword-removed Cranfield tokens used by the other project experiments.
- Load pre-trained 50-dimensional GloVe vectors.
- Estimate corpus token probabilities `p(w)` from the Cranfield document collection.
- Build one dense vector per document/query using the SIF weight `a / (a + p(w))`.
- Optionally remove the top principal components from the dense vectors.
- Rank documents with cosine similarity in the dense space.

## Tuned parameters

- `a`: SIF smoothing parameter.
- `n_components`: number of top principal components removed after embedding.

The default sweep matches the report setup:

- `a in {0.01, 0.001, 0.0001}`
- `n_components in {0, 1, 2}`

## Run

```bash
python NLP_ASSIGNMENT2/project/chandan/sif_embeddings/run_experiments.py
```

For the adapted-backbone transfer experiment requested after the base SIF run:

```bash
python NLP_ASSIGNMENT2/project/chandan/sif_embeddings/run_transfer_experiments.py
```

If you already have a local converted GloVe file, place it at:

```text
NLP_ASSIGNMENT2/project/pretrained_models/glove.6B.50d.w2v.txt
```

Otherwise the runner will try the gensim downloader fallback for `glove-wiki-gigaword-50`.

## Main outputs

- `output/summary.json`
- `output/summary_k10.csv`
- `output/config_sweep.json`
- `output/config_sweep_k10.csv`
- `output/a_sweep_best.json`
- `output/a_sweep_best.csv`
- `output/a_sweep_best.png`
- `output/eval_overlay.png`
- `output/all_tuned_combinations_overlay.png`
- `experiment_report.md`

Transfer outputs are written separately so they do not overwrite the pure-SIF run:

- `output_transferability/summary.json`
- `output_transferability/summary_k10.csv`
- `output_transferability/eval_overlay.png`
- `output_transferability/k10_backbone_bars.png`
- `transferability_report.md`

## Notes

- This is a pure dense retrieval method with no lexical fusion.
- It is useful as an isolated test of whether weighting and PC removal can rescue centroid-style semantic retrieval on a technical IR benchmark.
- The transfer runner applies the same SIF recipe to token vectors adapted from `TF-IDF`, `LSA`, and `ESA`.
