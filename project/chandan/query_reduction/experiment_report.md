# Query Reduction Experiment Report

## Method Summary

Query reduction removes noisy or overly broad terms before ranking. This is a lightweight, non-deep-learning way to lower query-time cost and sometimes reduce ambiguity by keeping only the most informative parts of the query.

## Implemented Methods

- `idf_topk`: keep the top-k highest-IDF query terms.
- `prf_term_pruning`: run baseline retrieval once, score original query terms by how strongly they are supported in the top retrieved documents, then keep only the strongest original terms.

## Best Query Reduction Method

- `method = idf_topk`
- `best_parameter = 18`
- `result = best query-reduction variant, but still below the full baseline TF-IDF system on this dataset`

## k=10 Comparison

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 |
| unordered_local_context_bow | 0.2902 | 0.4210 | 0.3177 | 0.3240 | 0.4829 | 0.7661 |
| idf_topk | 0.2831 | 0.4094 | 0.3099 | 0.3058 | 0.4611 | 0.7461 |
| prf_term_pruning | 0.2667 | 0.3798 | 0.2903 | 0.2791 | 0.4244 | 0.6860 |

## Best-Method Delta vs Baseline at k=10

- `dMAP@10 = -0.0002`
- `dnDCG@10 = -0.0000`
- `dMRR@10 = +0.0000`

The query-reduction variants reduce some noisy cases, but in Cranfield many long query terms are actually useful technical clues rather than filler. So aggressive pruning tends to remove signal along with noise, which hurts the final ranking. In the current `chandan` results, `unordered_local_context_bow` remains the strongest method overall.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 2 | 2 | 0.1538 | 0.1538 | 0.0000 |
| 64 | 1 | 1 | 0.2619 | 0.2619 | 0.0000 |
| 81 | 1 | 1 | 0.2869 | 0.2869 | 0.0000 |
| 90 | 2 | 2 | 0.1048 | 0.1048 | 0.0000 |
| 225 | 2 | 2 | 0.0971 | 0.0971 | 0.0000 |
| 224 | 0 | 0 | 0.0346 | 0.0346 | 0.0000 |
| 223 | 2 | 2 | 0.3333 | 0.3333 | 0.0000 |
| 222 | 4 | 4 | 0.5542 | 0.5542 | 0.0000 |

## Unified Overlay

The unified overlay compares the baseline, the earlier unordered local-context method, and all query-reduction methods in one figure.

## Output Files

- Summary JSON: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/summary.json`
- Summary CSV: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/summary_k10.csv`
- Unified overlay: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/unified_eval_overlay.png`
- Comparison summary: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/comparison_summary.json`
- Example comparison markdown: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/query_reduction/output/example_query_comparison.md`