# Query Expansion Experiment Report

## What Was Fixed

1. Added a true non-expanded TF-IDF baseline (`baseline_tfidf`) for correct comparison.
2. Kept retrieval model fixed to base TF-IDF document index; expansion is applied only to queries.
3. Added adaptive similarity thresholding using mean similarity filtering.
4. Added normalized neighbor mass allocation so expansion does not overpower original query terms.
5. Added method-vs-baseline plots and full overlay plots.
6. Added explicit example-case comparisons for the report query set.
7. Added persistent WordNet graph caching on disk for faster reruns.

## Run Configuration

- Dataset: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/cranfield
- Methods: baseline_tfidf, wordnet, embedding_tfidf, embedding_lsa, embedding_esa, embedding_word2vec
- top_k_neighbors: 10
- min_similarity: 0.05
- self_weight: 1.0
- expansion_weight: 0.2
- replacement_weight: 0.85
- replacement_expansion_weight: 0.15
- adaptive_mean_similarity_threshold: True
- mean_similarity_factor: 0.9
- normalize_neighbor_mass: True
- similarity_power: 1.0

## k=10 Scores

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2813 | 0.4005 | 0.3059 | 0.3024 | 0.4546 | 0.7379 |
| wordnet | 0.2818 | 0.4033 | 0.3071 | 0.3059 | 0.4596 | 0.7506 |
| embedding_tfidf | 0.2804 | 0.3999 | 0.3054 | 0.3026 | 0.4512 | 0.7403 |
| embedding_lsa | 0.2782 | 0.3971 | 0.3031 | 0.3023 | 0.4520 | 0.7441 |
| embedding_esa | 0.2800 | 0.3993 | 0.3049 | 0.3032 | 0.4523 | 0.7419 |
| embedding_word2vec | 0.2796 | 0.4004 | 0.3051 | 0.3043 | 0.4550 | 0.7405 |

## Delta vs Baseline (k=10)

| Method | dP@10 | dR@10 | dF@10 | dMAP@10 | dnDCG@10 | dMRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wordnet | +0.0004 | +0.0028 | +0.0012 | +0.0036 | +0.0050 | +0.0127 |
| embedding_tfidf | -0.0009 | -0.0006 | -0.0005 | +0.0002 | -0.0034 | +0.0024 |
| embedding_lsa | -0.0031 | -0.0034 | -0.0028 | -0.0001 | -0.0026 | +0.0062 |
| embedding_esa | -0.0013 | -0.0011 | -0.0010 | +0.0008 | -0.0023 | +0.0041 |
| embedding_word2vec | -0.0018 | -0.0001 | -0.0008 | +0.0019 | +0.0004 | +0.0027 |

## Best Method Per Metric at k=10

- precision: wordnet (0.2818)
- recall: wordnet (0.4033)
- fscore: wordnet (0.3071)
- map: wordnet (0.3059)
- ndcg: wordnet (0.4596)
- mrr: wordnet (0.7506)

## Example Cases Summary

### Dataset Query Cases

| Query ID | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |
| --- | ---: | --- | ---: | ---: |
| 9 | 1 | embedding_lsa | 2 | +1 |
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

- Summary JSON: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/summary.json
- Summary CSV: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/summary_k10.csv
- Overlay plot: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/eval_overlay.png
- Example comparison markdown: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/example_query_comparison.md
- Example comparison json: /home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/Nikhil/query_expansion/output/example_query_comparison.json