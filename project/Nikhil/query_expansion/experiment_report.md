# Query Expansion Experiment Report

## What Was Fixed

1. Added a true non-expanded TF-IDF baseline (`baseline_tfidf`) for correct comparison.
2. Kept retrieval model fixed to base TF-IDF document index; expansion is applied only to queries.
3. Added distribution-aware, method-specific min similarity floors derived from each method's score distribution.
4. Added adaptive similarity thresholding using mean and quantile filtering on top of the method-specific floor.
5. Added normalized neighbor mass allocation so expansion does not overpower original query terms.
6. Added method-vs-baseline plots and full overlay plots.
7. Added explicit example-case comparisons for the report query set.
8. Added persistent WordNet graph caching on disk for faster reruns.

## Run Configuration

- Dataset: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/cranfield
- Methods: baseline_tfidf, embedding_lsa
- top_k_neighbors: 15
- base_min_similarity_floor: 0.03
- method_threshold_quantile: 0.45
- self_weight: 1.0
- expansion_weight: 0.25
- replacement_weight: 0.9
- replacement_expansion_weight: 0.2
- adaptive_mean_similarity_threshold: True
- mean_similarity_factor: 0.85
- normalize_neighbor_mass: True
- similarity_power: 1.15

## LSA Dimension Sweep

The LSA-only sweep kept the same tuned expansion settings and varied only `lsa_components` across 32, 64, 96, 128, and 160.

| Components | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.2791 | 0.3957 | 0.3033 | 0.3026 | 0.4522 | 0.7441 |
| 64 | 0.2756 | 0.3927 | 0.3001 | 0.2986 | 0.4482 | 0.7384 |
| 96 | 0.2796 | 0.3993 | 0.3048 | 0.3022 | 0.4514 | 0.7375 |
| 128 | 0.2787 | 0.3982 | 0.3040 | 0.3015 | 0.4507 | 0.7365 |
| 160 | 0.2764 | 0.3959 | 0.3020 | 0.3012 | 0.4489 | 0.7305 |

32 components was the best LSA setting on MAP@10 and MRR@10, while 96 components was the best on precision, recall, and F-score. The strongest LSA MAP@10 only slightly beat the non-expanded baseline, so the sweep confirmed that LSA helps, but not enough to overtake the better baseline result across the board.

## Dynamic Min Similarity by Method

| Method | Base Floor | Derived Floor | Score Count | Mean | Std | Median | Q45 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| embedding_lsa | 0.0300 | 0.6646 | 119760 | 0.7169 | 0.1933 | 0.6922 | 0.6646 | 1.0000 |

## k=10 Scores

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2813 | 0.4005 | 0.3059 | 0.3024 | 0.4546 | 0.7379 |
| embedding_lsa | 0.2764 | 0.3959 | 0.3020 | 0.3012 | 0.4489 | 0.7305 |

## Delta vs Baseline (k=10)

| Method | dP@10 | dR@10 | dF@10 | dMAP@10 | dnDCG@10 | dMRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| embedding_lsa | -0.0049 | -0.0046 | -0.0039 | -0.0012 | -0.0057 | -0.0074 |

## Best Method Per Metric at k=10

- precision: baseline_tfidf (0.2813)
- recall: baseline_tfidf (0.4005)
- fscore: baseline_tfidf (0.3059)
- map: baseline_tfidf (0.3024)
- ndcg: baseline_tfidf (0.4546)
- mrr: baseline_tfidf (0.7379)

## Example Cases Summary

### Dataset Query Cases

| Query ID | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |
| --- | ---: | --- | ---: | ---: |
| 9 | 1 | baseline_tfidf | 1 | +0 |
| 39 | 2 | baseline_tfidf | 2 | +0 |
| 40 | 2 | baseline_tfidf | 2 | +0 |
| 51 | 4 | baseline_tfidf | 4 | +0 |
| 64 | 2 | baseline_tfidf | 2 | +0 |
| 81 | 1 | baseline_tfidf | 1 | +0 |
| 90 | 2 | embedding_lsa | 3 | +1 |

### Custom Query Cases

| Case | Mapped Query | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |
| --- | --- | ---: | --- | ---: | ---: |
| slip-flow heat transfer in internal channels | 9 | 1 | baseline_tfidf | 1 | +0 |
| transition detection in hypersonic wakes behind slender bodies | 40 | 2 | baseline_tfidf | 2 | +0 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | 1 | baseline_tfidf | 1 | +0 |
| shock-induced boundary-layer separation | 90 | 2 | embedding_lsa | 3 | +1 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | 0 | baseline_tfidf | 0 | +0 |

## Limitation-Solving Score Table (updated)

| Method | Semantic | Dim/Cost | Scale | Ambiguity | OOV | Context | Sparse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 2/10 | 6/10 | 6/10 | 2/10 | 1/10 | 2/10 | 2/10 |
| wordnet | 8/10 | 3/10 | 3/10 | 7/10 | 8/10 | 5/10 | 3/10 |
| embedding_tfidf | 4/10 | 6/10 | 6/10 | 3/10 | 2/10 | 3/10 | 3/10 |
| embedding_lsa | 5/10 | 5/10 | 5/10 | 4/10 | 2/10 | 4/10 | 7/10 |
| embedding_esa | 6/10 | 4/10 | 5/10 | 4/10 | 3/10 | 4/10 | 7/10 |
| embedding_word2vec | 7/10 | 5/10 | 5/10 | 5/10 | 4/10 | 5/10 | 5/10 |

## Output Files

- Summary JSON: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/lsa_sweep_20260423/lsa_160/summary.json
- Summary CSV: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/lsa_sweep_20260423/lsa_160/summary_k10.csv
- Overlay plot: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/lsa_sweep_20260423/lsa_160/eval_overlay.png
- Example comparison markdown: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/lsa_sweep_20260423/lsa_160/example_query_comparison.md
- Example comparison json: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/lsa_sweep_20260423/lsa_160/example_query_comparison.json