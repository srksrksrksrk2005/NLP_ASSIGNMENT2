# Query Expansion Analysis Notes

## Setup

We evaluated query replacement and expansion on the Cranfield collection using a fixed TF-IDF document index.
Expansion is applied only to queries, not to the retrieval corpus.

Methods compared:

- `baseline_tfidf`
- `wordnet`
- `embedding_tfidf`
- `embedding_lsa`
- `embedding_esa`
- `embedding_word2vec`

Expansion controls used in the latest run:

- method-specific min similarity floors derived from each method's similarity distribution
- adaptive mean-threshold filtering
- adaptive quantile-threshold filtering
- normalized neighbor-mass allocation
- scaled similarity scores before mass allocation

## Best Configuration

The best run so far is:

- `trial_label = tuned_more_expansion`
- `top_k_neighbors = 15`
- `base_min_similarity_floor = 0.03`
- `method_threshold_quantile = 0.45`
- `mean_similarity_factor = 0.85`
- `expansion_weight = 0.25`
- `replacement_weight = 0.90`
- `replacement_expansion_weight = 0.20`
- `similarity_power = 1.15`

Why this run mattered:

- it improved WordNet over the previous two trials
- it kept the baseline intact while testing broader expansion
- it exposed how each method's similarity distribution forces a different effective floor

## Comparison at k=10

| Method | Precision | Recall | F-score | MAP | nDCG | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard TF-IDF | 0.2813 | 0.4005 | 0.3059 | 0.3024 | 0.4546 | 0.7379 |
| WordNet | 0.2818 | 0.4035 | 0.3071 | 0.3062 | 0.4602 | 0.7511 |
| TF-IDF neighbors | 0.2791 | 0.3985 | 0.3045 | 0.3037 | 0.4525 | 0.7361 |
| LSA neighbors | 0.2787 | 0.3982 | 0.3040 | 0.3015 | 0.4508 | 0.7365 |
| ESA neighbors | 0.2800 | 0.3998 | 0.3055 | 0.3040 | 0.4531 | 0.7349 |
| Word2Vec neighbors | 0.2796 | 0.3990 | 0.3040 | 0.3002 | 0.4534 | 0.7372 |

## Delta vs Baseline at k=10

| Method | dP@10 | dR@10 | dF@10 | dMAP@10 | dnDCG@10 | dMRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WordNet | +0.0004 | +0.0030 | +0.0012 | +0.0038 | +0.0056 | +0.0132 |
| TF-IDF neighbors | -0.0022 | -0.0019 | -0.0014 | +0.0013 | -0.0021 | -0.0018 |
| LSA neighbors | -0.0027 | -0.0023 | -0.0019 | -0.0009 | -0.0039 | -0.0014 |
| ESA neighbors | -0.0013 | -0.0007 | -0.0005 | +0.0016 | -0.0015 | -0.0030 |
| Word2Vec neighbors | -0.0018 | -0.0015 | -0.0019 | -0.0022 | -0.0013 | -0.0006 |

## Dynamic Min Similarity by Method

These are the method-specific floors derived from each method's own similarity-score distribution in the latest run.

| Method | Base Floor | Derived Floor | Score Count | Mean | Std | Median | Q45 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WordNet | 0.0300 | 0.9231 | 37336 | 0.8818 | 0.1534 | 0.9333 | 0.9231 | 1.0000 |
| TF-IDF neighbors | 0.0300 | 0.4472 | 119755 | 0.5332 | 0.2606 | 0.4824 | 0.4472 | 1.0000 |
| LSA neighbors | 0.0300 | 0.6932 | 119760 | 0.7387 | 0.1812 | 0.7203 | 0.6932 | 1.0000 |
| ESA neighbors | 0.0300 | 0.4472 | 119714 | 0.5329 | 0.2609 | 0.4830 | 0.4472 | 1.0000 |
| Word2Vec neighbors | 0.0300 | 0.8629 | 119760 | 0.8351 | 0.1053 | 0.8732 | 0.8629 | 0.9942 |

## ESA and Word2Vec Coverage

Both methods cover the full Cranfield vocabulary in this setup.

- ESA coverage: `7984/7984` terms, `100.00%`
- Word2Vec coverage: `7984/7984` terms, `100.00%`

That means every vocabulary term had either an ESA concept vector or a Word2Vec embedding in the current run. The difference is not coverage, but how the similarity structure is built.

## Run Timeline

| Run | Trial Label | Key Thresholds | WordNet MAP@10 | WordNet MRR@10 | Change |
| --- | --- | --- | ---: | ---: | --- |
| 1 | seed_current | floor=0.08, q=0.60, mean=1.00 | 0.3058 | 0.7506 | Baseline comparison with conservative thresholds |
| 2 | low_thresh_q50 | floor=0.05, q=0.50, mean=0.90 | 0.3059 | 0.7506 | Slight gain from broader expansion |
| 3 | tuned_more_expansion | floor=0.03, q=0.45, mean=0.85, top_k=15 | 0.3062 | 0.7511 | Best run so far |

What changed in the last step:

- lowered the base floor again
- widened the neighbor budget from 10 to 15
- increased similarity power to favor stronger neighbors
- kept normalized mass allocation so original query terms still matter

## Example Cases

### Dataset Query Cases

| Query ID | Baseline Hits@5 | Best Method | Best Hits@5 | Delta |
| --- | ---: | --- | ---: | ---: |
| 9 | 1 | baseline_tfidf | 1 | +0 |
| 39 | 2 | baseline_tfidf | 2 | +0 |
| 40 | 2 | embedding_tfidf | 3 | +1 |
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

## Interpretation

WordNet remains the best overall expansion method in the current tuning sweep.
The more permissive thresholds helped slightly, but gains are modest because the baseline TF-IDF model is already strong on many Cranfield queries.

The important behavior is method-specific:

- WordNet has the highest effective floor because its similarity distribution is tight and strong.
- Word2Vec also has a high floor, which means only very close neighbors survive.
- TF-IDF and ESA are looser, so they admit more candidates but also drift more easily.

The example queries show the same pattern:

- query 40 and query 90 benefit from expansion
- several other queries do not improve over baseline
- the OOV liftbody/propwash case stays hard even after expansion, which is the main remaining limitation

## Files Written

- [summary.json](output/summary.json)
- [summary_k10.csv](output/summary_k10.csv)
- [eval_overlay.png](output/eval_overlay.png)
- [example_query_comparison.md](output/example_query_comparison.md)
- [example_query_comparison.json](output/example_query_comparison.json)
- [run_timeline.json](output/run_timeline.json)
