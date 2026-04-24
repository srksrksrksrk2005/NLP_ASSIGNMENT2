# N-gram Tuning Report

## Method Summary

The n-gram model augments unigram TF-IDF with contiguous multi-word tokens. This sweep tunes the maximum n and compares each tuned configuration against the baseline unigram TF-IDF run.

## Tuning Grid

- `tested_n_values = [2, 3, 4, 5, 6, 7, 8]`
- ranking uses cosine similarity over TF-IDF vectors

## Best Configuration

- `best_n = 2`
- `MAP@10 = 0.3044`
- `nDCG@10 = 0.4638`
- `MRR@10 = 0.7479`

## n Sweep (k=10)

| n | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2760 | 0.3994 | 0.3022 | 0.3044 | 0.4638 | 0.7479 | 22.61 |
| 3 | 0.2724 | 0.3956 | 0.2987 | 0.2954 | 0.4565 | 0.7304 | 53.02 |
| 4 | 0.2720 | 0.3944 | 0.2980 | 0.2924 | 0.4548 | 0.7295 | 97.08 |
| 5 | 0.2716 | 0.3941 | 0.2976 | 0.2902 | 0.4515 | 0.7242 | 164.50 |
| 6 | 0.2707 | 0.3945 | 0.2971 | 0.2903 | 0.4521 | 0.7249 | 186.65 |
| 7 | 0.2724 | 0.3965 | 0.2989 | 0.2906 | 0.4523 | 0.7226 | 235.29 |
| 8 | 0.2724 | 0.3968 | 0.2990 | 0.2906 | 0.4527 | 0.7226 | 263.40 |

## Baseline vs Best at k=10

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2836 | 0.4100 | 0.3103 | 0.3026 | 0.4589 | 0.7368 |
| ngram_best (n=2) | 0.2760 | 0.3994 | 0.3022 | 0.3044 | 0.4638 | 0.7479 |

## Output Files

- Summary JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/summary.json`
- Summary CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/summary_k10.csv`
- n sweep JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/n_sweep_results.json`
- n sweep CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/n_sweep_results.csv`
- Overlay plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/eval_overlay.png`
- n sweep plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/ngram/output_ngram/n_sweep_k10.png`