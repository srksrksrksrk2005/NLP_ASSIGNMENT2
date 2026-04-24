# WordNet Local-Context Tuning Report

## Method Summary

This run tunes WordNet local context size and compares all tuned combinations against baseline unigram TF-IDF.

## Tuning Grid

- `tested_context_windows = [1, 2, 3, 4, 5, 6, 8, 10, 12]`
- `max_synonyms = 3`

## Best Configuration

- `best_context_window = 1`
- `MAP@10 = 0.2641`
- `nDCG@10 = 0.4118`
- `MRR@10 = 0.6672`

## Context Window Sweep (k=10)

| context_window | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2591 | 0.3694 | 0.2815 | 0.2641 | 0.4118 | 0.6672 | 30.64 |
| 2 | 0.2578 | 0.3697 | 0.2806 | 0.2617 | 0.4101 | 0.6647 | 28.79 |
| 3 | 0.2533 | 0.3641 | 0.2758 | 0.2532 | 0.4018 | 0.6473 | 28.79 |
| 4 | 0.2507 | 0.3618 | 0.2736 | 0.2504 | 0.3984 | 0.6447 | 28.77 |
| 5 | 0.2529 | 0.3629 | 0.2751 | 0.2505 | 0.3993 | 0.6447 | 28.80 |
| 6 | 0.2551 | 0.3654 | 0.2773 | 0.2521 | 0.4010 | 0.6477 | 28.93 |
| 8 | 0.2529 | 0.3640 | 0.2756 | 0.2521 | 0.4003 | 0.6451 | 29.01 |
| 10 | 0.2547 | 0.3651 | 0.2770 | 0.2527 | 0.4017 | 0.6487 | 29.74 |
| 12 | 0.2551 | 0.3649 | 0.2772 | 0.2538 | 0.4020 | 0.6486 | 28.85 |

## Baseline vs Best at k=10

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2836 | 0.4100 | 0.3103 | 0.3026 | 0.4589 | 0.7368 |
| wordnet_best (context_window=1) | 0.2591 | 0.3694 | 0.2815 | 0.2641 | 0.4118 | 0.6672 |

## Output Files

- Summary JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/summary.json`
- Summary CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/summary_k10.csv`
- Sweep JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/context_window_sweep_results.json`
- Sweep CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/context_window_sweep_results.csv`
- Overlay plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/eval_overlay.png`
- Sweep plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/ritisha/wordnet/output_wordnet/context_window_sweep_k10.png`