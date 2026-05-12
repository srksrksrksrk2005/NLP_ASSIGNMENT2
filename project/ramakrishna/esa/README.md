# ESA Retrieval

This folder contains a self-contained Explicit Semantic Analysis style experiment for the Cranfield dataset.

## What is implemented

- `esa_retrieval.py`: document-concept ESA retriever
- `run_esa.py`: full evaluation, overlay plots, and analysis export
- `tune_esa.py`: hyperparameter sweep for ESA
- `output_esa/`: generated plots and summaries
- `output_tuning/`: tuning CSV and best-config JSON

## How to run

Run the ESA evaluation:

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/esa/run_esa.py
```

Run ESA with an external technical concept corpus:

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/esa/run_esa.py -concept_source 20news-technical -concept_limit 1500
```

Run ESA from a prebuilt bundle:

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/esa/run_esa.py -prebuilt_index_path C:\path\to\esa_bundle.joblib
```

Run the ESA tuning sweep:

```powershell
python NLP_ASSIGNMENT2/project/ramakrishna/esa/tune_esa.py
```

## Notes

- The ESA variant here treats Cranfield documents as explicit concepts.
- You can also switch ESA to an external concept corpus based on technical 20 Newsgroups categories.
- A prebuilt ESA bundle can be loaded with `-prebuilt_index_path`; it should contain the serialized vectorizer plus the document and concept matrices.
- Queries are projected into concept space using TF-IDF similarity.
- The tuning sweep searches over the TF-IDF preprocessing stage and the number of retained concepts.
- `run_esa.py` writes an overlay plot for `ESA vs TF-IDF` and a second overlay plot for `TF-IDF vs LSA vs Hybrid vs ESA`.
- The tuned default configuration is `esa_top_concepts=25`, `sublinear_tf=True`, `max_df=0.9`, `min_df=1`, and `ngram_range=(1, 2)`.
- The report-style analysis lives in [`analysis.md`](./analysis.md), and the generated copies are written under `output_esa/`.
