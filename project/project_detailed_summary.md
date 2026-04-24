# NLP Assignment 2 - Detailed Project Summary

## 1. Project Objective

The project extends the Assignment 2 information retrieval baseline on the Cranfield dataset.

Baseline system:
- Vector Space Model with TF-IDF and cosine similarity.
- Standard preprocessing: sentence segmentation, tokenization, inflection reduction, and stopword removal.

Main goal:
- Improve retrieval quality on technical aerospace queries, especially where exact lexical overlap is weak.

Primary evaluation metrics:
- Precision@k, Recall@k, F-score@k, MAP@k, nDCG@k, MRR@k (reported mostly at k=10).


## 2. Project Structure and Experimental Tracks

The project evolved into two complementary tracks:

### A) Query Expansion Track (Nikhil)

Folder:
- `NLP_ASSIGNMENT2/project/Nikhil/query_expansion`

Implemented methods:
- `baseline_tfidf` (non-expanded baseline for fair comparison)
- `wordnet`
- `embedding_tfidf`
- `embedding_lsa`
- `embedding_esa`
- `embedding_word2vec`

Core idea:
- Keep document index fixed.
- Expand/rewrite only queries.
- Handle OOV terms via WordNet replacement.
- Allocate expansion mass to nearest vocabulary neighbors using similarity matrices.

Important engineering fixes in this track:
- True non-expanded baseline added.
- Method-specific dynamic similarity floors.
- Adaptive thresholding (quantile + mean-based filtering).
- Neighbor mass normalization to prevent over-expansion.
- Disk cache for WordNet graph/similarity artifacts.


### B) Representation and Hybrid Retrieval Track (ramakrishna)

Folder:
- `NLP_ASSIGNMENT2/project/ramakrishna`

Subtracks:
- LSA retrieval (`lsa/`)
- Hybrid TF-IDF + LSA (`lsa/`)
- ESA-style concept retrieval (`esa/`)

Core idea:
- Improve semantic matching at retrieval stage (not only query reformulation).
- Compare pure lexical, latent semantic, hybrid fusion, and concept-space methods.


## 3. Nikhil Track - Detailed Findings

### 3.1 Full multi-method query expansion run (k=10)

From `output/summary.json`, k=10 summary:

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2813 | 0.4005 | 0.3059 | 0.3024 | 0.4546 | 0.7379 |
| wordnet | 0.2818 | 0.4035 | 0.3071 | 0.3062 | 0.4602 | 0.7511 |
| embedding_tfidf | 0.2791 | 0.3985 | 0.3045 | 0.3037 | 0.4525 | 0.7361 |
| embedding_lsa | 0.2787 | 0.3982 | 0.3040 | 0.3015 | 0.4508 | 0.7365 |
| embedding_esa | 0.2800 | 0.3998 | 0.3055 | 0.3040 | 0.4531 | 0.7349 |
| embedding_word2vec | 0.2796 | 0.3990 | 0.3040 | 0.3002 | 0.4534 | 0.7372 |

Interpretation:
- `wordnet` is the best query-expansion method in this run across MAP@10, nDCG@10, and MRR@10.
- Embedding-based expansion variants are close to baseline but mostly give marginal gains or slight drops depending on metric.


### 3.2 LSA-only query expansion sweep (special run)

From `experiment_report.md`:
- LSA dimensions tested: 32, 64, 96, 128, 160.
- Best MAP@10 in that sweep: 32 components (0.3026).
- Best precision/recall/F-score in that sweep: 96 components.

Observation:
- LSA-based query expansion helped specific queries but did not consistently beat the non-expanded baseline in this isolated sweep configuration.


### 3.3 Query-level behavior (Nikhil report examples)

From `experiment_report.md`:
- Most custom failure-type queries remained tied with baseline.
- Clear improvement seen for the shock-induced boundary-layer separation case (mapped query 90), where LSA expansion improved Hits@5 by +1.


## 4. ramakrishna Track - Detailed Findings

### 4.1 Tuned LSA retrieval

Best tuned config (from `lsa/analysis.md`):
- `lsa_components=250`
- `sublinear_tf=True`
- `max_df=0.95`
- `min_df=2`
- `norm=l2`

Reported result:
- MAP@10 = 0.3665
- nDCG@10 = 0.5169
- MRR@10 = 0.7952
- P@1 = 0.7289

Compared with TF-IDF baseline:
- Large gain across all primary ranking metrics.


### 4.2 Tuned Hybrid (TF-IDF + LSA)

Best tuned config:
- `lsa_components=240`
- `tfidf_weight=0.2`
- `sublinear_tf=True`
- `max_df=0.95`
- `min_df=2`

From `lsa/output_compare/comparison_summary.json` at k=10:
- TF-IDF: MAP 0.3026, nDCG 0.4589, MRR 0.7368
- LSA: MAP 0.3665, nDCG 0.5169, MRR 0.7952
- Hybrid: MAP 0.3695, nDCG 0.5220, MRR 0.8053

Interpretation:
- Hybrid is strongest in this run, indicating lexical + latent fusion is better than either alone.


### 4.3 ESA-style retrieval

Best ESA config (from `esa/analysis.md`):
- `esa_top_concepts=25`
- `esa_min_similarity=0.0`
- `sublinear_tf=True`
- `max_df=0.9`
- `min_df=1`
- `ngram_range=(1,2)`

From `esa/output_esa/comparison_summary.json` at k=10:
- TF-IDF: MAP 0.3026, nDCG 0.4589, MRR 0.7368
- LSA: MAP 0.3703, nDCG 0.5210, MRR 0.7804
- Hybrid: MAP 0.3691, nDCG 0.5221, MRR 0.7896
- ESA: MAP 0.3431, nDCG 0.4980, MRR 0.7654

Interpretation:
- ESA beats baseline clearly.
- ESA is not best overall, but it wins for selected concept-heavy queries where wording mismatch is strong.


## 5. Consolidated Technical Interpretation

### What consistently worked

- Retrieval-space semantic modeling (LSA and Hybrid) delivered the strongest global metric gains.
- TF-IDF preprocessing choices had major impact (`sublinear_tf`, `max_df`, `min_df`).
- Hybrid fusion improved robustness by combining exact lexical evidence with latent semantic evidence.


### What helped selectively

- Query expansion methods gave modest aggregate gains, but improved certain difficult semantic/OOV cases.
- ESA retrieval provided targeted wins on concept-level matching, though global metrics remained below best LSA/Hybrid runs.


### Why results differ across runs

Different branches used different preprocessing/search spaces:
- LSA/Hybrid comparison run used unigram-focused settings in `lsa/output_compare`.
- ESA comparison run used `ngram_range=(1,2)` and different document-term statistics.

So absolute values vary slightly between report blocks, but the trend is stable:
- Baseline < ESA < tuned LSA/Hybrid for overall performance.


## 6. Final Ranking of Approaches (Overall)

Based on k=10 metrics and analysis across both tracks:

1. **Hybrid TF-IDF + LSA** (best overall balance and top rank behavior in main LSA/Hybrid run)
2. **Tuned LSA retrieval** (very close to hybrid, consistently strong)
3. **ESA retrieval** (strong concept matching, clear baseline improvement)
4. **WordNet query expansion** (best among query-expansion-only variants)
5. **Other embedding-based query expansion methods** (near-baseline, case-dependent gains)
6. **Plain TF-IDF baseline** (reference point)


## 7. Practical Conclusion

The project shows that the largest improvements came from **retrieval representation learning and hybrid fusion**, not from query expansion alone.

If deploying one final system from these experiments:
- Choose **Hybrid TF-IDF + LSA** as the primary retriever.
- Use **ESA** as a complementary retriever or reranking signal for concept-heavy queries.
- Keep **WordNet expansion** as an optional fallback for OOV-heavy query rewriting.


## 8. Suggested Next Extensions

- Fuse Hybrid + ESA scores with calibrated weights.
- Add per-query adaptive routing (lexical-heavy vs concept-heavy queries).
- Use significance testing (paired query-level tests) on AP and nDCG deltas.
- Add error taxonomy tags to automate method selection per query type.