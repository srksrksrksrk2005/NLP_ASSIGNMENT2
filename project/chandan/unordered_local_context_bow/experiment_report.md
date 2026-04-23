# Unordered Local-Context Bag-of-Words Experiment Report

## Method Summary

The baseline system uses unigram TF-IDF with cosine similarity. The proposed method adds unordered local-context features extracted from nearby-word windows. Inside each local window, only token presence matters and word order is ignored, so a phrase such as `heat transfer` and the same two words appearing in swapped order still activate the same context feature.

## Hypothesis

A local unordered context signal should help when the baseline retrieves documents with the right words but the wrong nearby context. This is especially relevant for ambiguous technical terms and for queries whose important words should appear close together in relevant documents.

## What n Means

Here, `n` is the local context radius. For each token position, we inspect up to `n` neighboring words on the left and `n` neighboring words on the right, then build unordered context features from the unique tokens in that local window. So `n = 1` means a maximum window of 3 tokens, `n = 2` means up to 5 tokens, and so on.

## Baseline Limitations Addressed

- The assignment report highlights that the baseline VSM lacks contextual representation and ignores local proximity information.
- The project brief explicitly asks us to fix factual retrieval failures observed in the baseline; local context is a direct response to those failures.
- For ambiguous terms, plain TF-IDF gives the same credit to a shared token even when its nearby words indicate a different sense.

## Best Configuration

- `n = 4`
- `context_orders = [2]`
- `min_context_df = 2`
- `context_weight_alpha = 0.75`
- `context_feature_vocab_size = 85835`

## Local Context Size Sweep

| n | Best orders | Best min_df | Best alpha | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [2, 3] | 2 | 0.75 | 0.2933 | 0.4259 | 0.3215 | 0.3222 | 0.4829 | 0.7580 |
| 2 | [2] | 2 | 0.75 | 0.2898 | 0.4185 | 0.3171 | 0.3189 | 0.4783 | 0.7618 |
| 3 | [2] | 2 | 0.75 | 0.2920 | 0.4233 | 0.3197 | 0.3216 | 0.4804 | 0.7642 |
| 4 | [2] | 2 | 0.75 | 0.2902 | 0.4210 | 0.3177 | 0.3240 | 0.4829 | 0.7661 |
| 5 | [2, 3] | 2 | 0.75 | 0.2924 | 0.4233 | 0.3202 | 0.3237 | 0.4814 | 0.7652 |

The sweep shows a trade-off rather than a single monotonic pattern. `n = 1` is strongest on precision, recall, F-score, and tied nDCG, while `n = 4` gives the best MAP@10 and MRR@10. In this dataset, a moderately larger window captures useful technical co-occurrence without drifting too far from the local topic.

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.26 |
| unordered_local_context_bow | 0.2902 | 0.4210 | 0.3177 | 0.3240 | 0.4829 | 0.7661 | 65.50 |

## Delta vs Baseline at k=10

- `dP@10 = +0.0071`
- `dR@10 = +0.0116`
- `dF@10 = +0.0079`
- `dMAP@10 = +0.0179`
- `dnDCG@10 = +0.0218`
- `dMRR@10 = +0.0200`

## Significance Checks

Approximate randomization over per-query scores:

- `AP@10 p-value = 0.0000`
- `nDCG@10 p-value = 0.0000`

## Interpretation

The local-context method helps when relevance depends on nearby co-occurrence instead of isolated term overlap. Because the context features are unordered, the model gains proximity-sensitive evidence without committing to exact word order, which is useful for technical queries where terminology may appear in slightly different surface forms.

At the same time, the method is still limited by lexical overlap: if the right concept uses entirely different vocabulary, unordered context alone cannot fix that. It is better viewed as a context-aware extension of TF-IDF, not a full semantic model.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 2 | 0.1538 | 0.1484 | -0.0055 |
| 64 | 1 | 2 | 0.2619 | 0.3333 | 0.0714 |
| 81 | 1 | 1 | 0.2869 | 0.3036 | 0.0167 |
| 90 | 2 | 2 | 0.1048 | 0.1071 | 0.0024 |
| 119 | 0 | 2 | 0.1944 | 0.7000 | 0.5056 |
| 142 | 1 | 1 | 0.2500 | 0.6111 | 0.3611 |
| 200 | 2 | 3 | 0.5833 | 0.8167 | 0.2333 |

## Output Files

- Summary JSON: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/summary.json`
- Summary CSV: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/summary_k10.csv`
- Overlay plot: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/eval_overlay.png`
- n-sweep JSON: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.json`
- n-sweep CSV: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.csv`
- n-sweep plot: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.png`
- Example comparison markdown: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/example_query_comparison.md`
- Config sweep JSON: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/config_sweep.json`