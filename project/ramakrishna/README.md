# LSA and Hybrid Implementation

This folder contains standalone implementations of:

- `LSA`
- `Hybrid TF-IDF + LSA`

## Files

- `lsa_retrieval.py`: LSA retriever built on top of TF-IDF + Truncated SVD
- `run_lsa.py`: runner script for evaluation mode and custom-query mode
- `hybrid_retrieval.py`: weighted score fusion of TF-IDF and LSA
- `run_hybrid.py`: runner script for the hybrid retriever

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

## Run hybrid evaluation

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_hybrid.py
```

This now uses the tuned hybrid defaults:

- `lsa_components = 240`
- `tfidf_weight = 0.2`
- `sublinear_tf = True`
- `max_df = 0.95`
- `min_df = 2`
- `ngram_range = (1, 1)`

## Try other vectorizer settings

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -max_df 1.0 -min_df 1
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -ngram_max 2
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -disable_sublinear_tf
python NLP_ASSIGNMENT2/project/ramakrishna/run_hybrid.py -tfidf_weight 0.3 -lsa_components 200
```

## Run custom-query mode

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/run_lsa.py -custom
```

## Outputs

The script writes these files to `output_lsa/` by default:

- `lsa_eval_plot.png`
- `lsa_summary.json`

The hybrid script writes these files to `output_hybrid/` by default:

- `hybrid_eval_plot.png`
- `hybrid_summary.json`
