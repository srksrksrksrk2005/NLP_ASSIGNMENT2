# LSA Implementation

This folder contains a standalone implementation of `LSA` for the Cranfield retrieval project.

## Files

- `lsa_retrieval.py`: LSA retriever built on top of TF-IDF + Truncated SVD
- `run_lsa.py`: runner script for evaluation mode and custom-query mode

## Run evaluation

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py
```

This now uses the tuned defaults:

- `lsa_components = 250`
- `sublinear_tf = True`
- `max_df = 0.95`
- `min_df = 2`
- `ngram_range = (1, 1)`

## Run with custom number of latent dimensions

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -lsa_components 150
```

## Try other vectorizer settings

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -max_df 1.0 -min_df 1
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -ngram_max 2
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -disable_sublinear_tf
```

## Run custom-query mode

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -custom
```

## Outputs

The script writes these files to `output_lsa/` by default:

- `lsa_eval_plot.png`
- `lsa_summary.json`
