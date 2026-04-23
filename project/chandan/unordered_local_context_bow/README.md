# Unordered Local-Context Bag-of-Words

This folder contains the `project/chandan` implementation for the unordered local-context bag-of-words method.

## What the method does

- Start from the same stopword-removed Cranfield tokens used by the baseline.
- Build the baseline unigram TF-IDF retriever.
- Build extra unordered local-context features from a window around each token.
- Inside each local window, token presence matters but order does not.
- Combine baseline cosine scores with local-context cosine scores.

## What `n` means

- `n` is the local context radius.
- For each token position, the method looks up to `n` words to the left and `n` words to the right.
- So `n = 1` means a maximum local window of 3 tokens, `n = 4` means up to 9 tokens.

## Run

```bash
python NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/run_experiments.py
```

## Main outputs

- `output/summary.json`: overall experiment summary
- `output/summary_k10.csv`: baseline vs method comparison at `k=10`
- `output/config_sweep.json`: full sweep across `n`, context orders, `min_df`, and `alpha`
- `output/n_sweep_best_by_radius.csv`: best result for each local context size `n`
- `output/n_sweep_best_by_radius.png`: plot of best metric values by `n`
- `output/eval_overlay.png`: baseline vs method overlay across `k=1..10`
- `experiment_report.md`: write-up for the method and results
