# Query Reduction Experiment Report

## Method Summary

Query reduction removes noisy or overly broad terms before ranking. This is a lightweight, non-deep-learning way to lower query-time cost and sometimes reduce ambiguity by keeping only the most informative parts of the query.

## Implemented Methods

- `idf_topk`: keep the top-k highest-IDF query terms.
- `prf_term_pruning`: run baseline retrieval once, score original query terms by how strongly they are supported in the top retrieved documents, then keep only the strongest original terms.

## Best Query Reduction Method

- `method = prf_term_pruning_extended`
- `best_parameter = {'top_docs': 15, 'keep_k': 12, 'alpha': 0.2}`
- `result = best query-reduction variant is above baseline on MAP@10`

## k=10 Comparison

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 |
| unordered_local_context_bow | 0.2938 | 0.4251 | 0.3215 | 0.3297 | 0.4883 | 0.7779 |
| idf_topk_legacy | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 |
| idf_topk_extended | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 |
| prf_term_pruning_legacy | 0.2751 | 0.3923 | 0.2993 | 0.2943 | 0.4436 | 0.7254 |
| prf_term_pruning_extended | 0.2840 | 0.4098 | 0.3107 | 0.3066 | 0.4617 | 0.7466 |

## Best-Method Delta vs Baseline at k=10

- `dMAP@10 = +0.0006`
- `dnDCG@10 = +0.0005`
- `dMRR@10 = +0.0005`

The query-reduction variants reduce some noisy cases, but in Cranfield many long query terms are actually useful technical clues rather than filler. So aggressive pruning tends to remove signal along with noise, which hurts the final ranking. In the current `chandan` results, `unordered_local_context_bow` remains the strongest method overall.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 2 | 2 | 0.1538 | 0.1538 | 0.0000 |
| 64 | 1 | 1 | 0.2619 | 0.2778 | 0.0159 |
| 81 | 1 | 1 | 0.2869 | 0.2869 | 0.0000 |
| 90 | 2 | 2 | 0.1048 | 0.1048 | 0.0000 |
| 190 | 1 | 1 | 0.1532 | 0.2472 | 0.0940 |
| 169 | 2 | 2 | 0.2800 | 0.3400 | 0.0600 |
| 92 | 4 | 4 | 0.4459 | 0.5030 | 0.0571 |
| 208 | 3 | 4 | 0.4881 | 0.5429 | 0.0548 |

## Unified Overlay

The unified overlay compares the baseline, the earlier unordered local-context method, and all query-reduction methods in one figure.

## Output Files

- Summary JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/summary.json`
- Summary CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/summary_k10.csv`
- Unified overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/unified_eval_overlay.png`
- IDF top-k all-combinations overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/idf_topk/output/all_tuned_combinations_overlay.png`
- IDF top-k extended all-combinations overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/idf_topk_extended/output/all_tuned_combinations_overlay.png`
- PRF pruning all-combinations overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/prf_term_pruning/output/all_tuned_combinations_overlay.png`
- PRF pruning extended all-combinations overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/prf_term_pruning_extended/output/all_tuned_combinations_overlay.png`
- Comparison summary: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/comparison_summary.json`
- Example comparison markdown: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/example_query_comparison.md`