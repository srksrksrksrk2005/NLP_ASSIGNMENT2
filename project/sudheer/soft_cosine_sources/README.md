# Soft Cosine Source Comparison

This folder runs a source-specific soft-cosine comparison over the Cranfield collection using four similarity sources:

- TF-IDF term similarity
- LSA term similarity
- ESA term similarity
- WordNet term similarity

Each source gets its own tuned sweep and its own output subfolder under `output/`.

## Run

From `NLP_ASSIGNMENT2/project/sudheer/soft_cosine_sources`:

```bash
python run_experiments.py
```

## Outputs

The runner writes:

- `output/summary.json`
- `output/summary_k10.csv`
- `output/comparison_summary.json`
- `output/eval_overlay.png`
- `output/experiment_report.md`
- one folder per source:
  - `output/tfidf/`
  - `output/lsa/`
  - `output/esa/`
  - `output/wordnet/`

Each source folder contains:

- `metrics.json`
- `baseline_metrics.json`
- `sweep_results.json`
- `sweep_best.json`
- `sweep_best.csv`
- `sweep_best.png`
- `baseline_eval_plot.png`
- `eval_plot.png`
- `source_overlay.png`
- `all_tuned_combinations_overlay.png`
- `rankings_top20.json`
- `example_query_comparison.json`
- `example_query_comparison.md`
- `experiment_report.md`

## Tuning ranges

The current sweep uses compact grids so the outputs stay readable:

- TF-IDF: `top_k_neighbors` 5/10/20, `min_similarity` 0.05/0.10, `similarity_power` 1.0/2.0
- LSA: `n_components` 150/250, `top_k_neighbors` 10/20, `min_similarity` 0.05/0.10, `similarity_power` 1.0/2.0
- ESA: `top_concepts` 25/50, `top_k_neighbors` 10/20, `min_similarity` 0.00/0.02, `similarity_power` 1.0/2.0
- WordNet: `top_k_neighbors` 5/10/20, `min_similarity` 0.05, `similarity_power` 1.0/2.0

## Notes

- The runner reuses the same Cranfield preprocessing used by the other Sudheer experiments.
- The source-specific TF-IDF baseline is evaluated with the same vectorizer setup used for that source.
- WordNet uses unigram features because its neighbor graph is lexical rather than phrasal.
