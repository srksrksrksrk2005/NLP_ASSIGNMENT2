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
- Methods: baseline_tfidf, wordnet, embedding_tfidf, embedding_lsa, embedding_esa, embedding_word2vec
- top_k_neighbors: 10
- base_min_similarity_floor: 0.08
- method_threshold_quantile: 0.6
- self_weight: 1.0
- expansion_weight: 0.2
- replacement_weight: 0.85
- replacement_expansion_weight: 0.15
- adaptive_mean_similarity_threshold: False
- mean_similarity_factor: 1.0
- normalize_neighbor_mass: True
- similarity_power: 1.0

## Dynamic Min Similarity by Method

| Method | Base Floor | Derived Floor | Score Count | Mean | Std | Median | Q60 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wordnet | 0.0800 | 1.0000 | 29826 | 0.8969 | 0.1495 | 0.9412 | 1.0000 | 1.0000 |
| embedding_tfidf | 0.0800 | 0.6882 | 79805 | 0.6021 | 0.2616 | 0.5774 | 0.6882 | 1.0000 |
| embedding_lsa | 0.0800 | 0.8521 | 79840 | 0.7795 | 0.1775 | 0.7801 | 0.8521 | 1.0000 |
| embedding_esa | 0.0800 | 0.6882 | 79810 | 0.6015 | 0.2622 | 0.5774 | 0.6882 | 1.0000 |
| embedding_word2vec | 0.0800 | 0.8942 | 79840 | 0.8416 | 0.1040 | 0.8793 | 0.8942 | 0.9942 |

## k=10 Scores

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2813 | 0.4005 | 0.3059 | 0.3024 | 0.4546 | 0.7379 |
| wordnet | 0.2800 | 0.3991 | 0.3047 | 0.3047 | 0.4577 | 0.7482 |
| embedding_tfidf | 0.2804 | 0.3985 | 0.3046 | 0.3010 | 0.4530 | 0.7372 |
| embedding_lsa | 0.2791 | 0.3986 | 0.3043 | 0.3009 | 0.4510 | 0.7335 |
| embedding_esa | 0.2804 | 0.3985 | 0.3046 | 0.3009 | 0.4531 | 0.7375 |
| embedding_word2vec | 0.2796 | 0.3974 | 0.3037 | 0.3011 | 0.4532 | 0.7371 |

## Delta vs Baseline (k=10)

| Method | dP@10 | dR@10 | dF@10 | dMAP@10 | dnDCG@10 | dMRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wordnet | -0.0013 | -0.0014 | -0.0012 | +0.0024 | +0.0031 | +0.0103 |
| embedding_tfidf | -0.0009 | -0.0020 | -0.0013 | -0.0013 | -0.0017 | -0.0007 |
| embedding_lsa | -0.0022 | -0.0019 | -0.0016 | -0.0015 | -0.0037 | -0.0044 |
| embedding_esa | -0.0009 | -0.0019 | -0.0013 | -0.0014 | -0.0015 | -0.0004 |
| embedding_word2vec | -0.0018 | -0.0030 | -0.0022 | -0.0013 | -0.0014 | -0.0007 |

## Best Method Per Metric at k=10

- precision: baseline_tfidf (0.2813)
- recall: baseline_tfidf (0.4005)
- fscore: baseline_tfidf (0.3059)
- map: wordnet (0.3047)
- ndcg: wordnet (0.4577)
- mrr: wordnet (0.7482)

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
| 90 | 2 | baseline_tfidf | 2 | +0 |

### Custom Query Cases

| Case | Mapped Query | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |
| --- | --- | ---: | --- | ---: | ---: |
| slip-flow heat transfer in internal channels | 9 | 1 | baseline_tfidf | 1 | +0 |
| transition detection in hypersonic wakes behind slender bodies | 40 | 2 | baseline_tfidf | 2 | +0 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | 1 | baseline_tfidf | 1 | +0 |
| shock-induced boundary-layer separation | 90 | 2 | baseline_tfidf | 2 | +0 |
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

- Summary JSON: NLP_ASSIGNMENT2/project/humanised/output_query_expansion/summary.json
- Summary CSV: NLP_ASSIGNMENT2/project/humanised/output_query_expansion/summary_k10.csv
- Overlay plot: NLP_ASSIGNMENT2/project/humanised/output_query_expansion/eval_overlay.png
- Example comparison markdown: NLP_ASSIGNMENT2/project/humanised/output_query_expansion/example_query_comparison.md
- Example comparison json: NLP_ASSIGNMENT2/project/humanised/output_query_expansion/example_query_comparison.json