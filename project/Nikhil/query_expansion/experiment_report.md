# Query Expansion Experiment Report

## Scope
This experiment evaluates the two target query expansion families on the Cranfield collection:

- WordNet-based query replacement and expansion
- Embedding-based query replacement and expansion using similarity matrices built from TF-IDF, LSA, ESA, and Word2Vec

The implementation lives entirely inside the `query_expansion` folder. The run logs are saved in `output/live_run.log`.

## Run Setup

- Dataset: Cranfield
- Preprocessing: sentence split, tokenization, lemmatization, stopword removal
- Retrieval model: TF-IDF cosine ranking over the processed corpus
- Expansion mechanism: query term-frequency spreading over a word-word similarity graph
- OOV handling: WordNet-based replacement into in-vocabulary terms
- Main hyperparameters used in this run:
  - `top_k_neighbors = 10`
  - `min_similarity = 0.05`
  - `expansion_weight = 0.35`
  - `replacement_weight = 1.0`
  - `replacement_expansion_weight = 0.35`

## Output Artifacts

- Per-method plots: `output/<method>/eval_plot.png`
- Overlay comparison plot: `output/eval_overlay.png`
- Per-method metrics: `output/<method>/metrics.json`
- Full summary: `output/summary.json`
- Fixed k=10 score table: `output/summary_k10.csv`
- Live log: `output/live_run.log`

The earlier empty `output/wordnet/` folder was due to the run not having finished yet. After the full run completed, it now contains the WordNet method outputs.

## Overall Results at k=10

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wordnet | 0.2684 | 0.3897 | 0.2949 | 0.2887 | 0.4395 | 0.7171 |
| embedding_tfidf | 0.2644 | 0.3843 | 0.2906 | 0.2711 | 0.4154 | 0.6771 |
| embedding_lsa | 0.2636 | 0.3828 | 0.2898 | 0.2471 | 0.3842 | 0.5882 |
| embedding_esa | 0.2662 | 0.3858 | 0.2921 | 0.2696 | 0.4139 | 0.6676 |
| embedding_word2vec | 0.2716 | 0.3993 | 0.2995 | 0.2865 | 0.4318 | 0.6915 |

## What Improved And What Decreased

### WordNet-based expansion
WordNet is the strongest method for ranked retrieval quality in this run. Compared with embedding_tfidf, it improves every metric at k=10:

- Precision@10: +0.0040
- Recall@10: +0.0055
- F@10: +0.0043
- MAP@10: +0.0176
- nDCG@10: +0.0241
- MRR@10: +0.0400

This means WordNet helped get relevant documents earlier in the ranking and improved overall ranking quality. The likely cause is that lexical synonym replacement and term spreading recover useful variants for technical words that appear in the Cranfield collection with slightly different surface forms.

Compared with Word2Vec, WordNet is a little weaker on Precision@10, Recall@10, and F@10, but still better on MAP, nDCG, and MRR. That suggests WordNet is slightly more conservative and better at ordering the top relevant documents, while Word2Vec retrieves a slightly broader set of relevant documents in the top 10.

### TF-IDF similarity expansion
TF-IDF expansion is the baseline embedding-style method in this run. It is simple and stable, but it mostly preserves lexical overlap rather than adding much semantic generalization. It performs below WordNet and Word2Vec on almost every metric.

### LSA expansion
LSA is the weakest method in this run. It loses to embedding_tfidf on all reported metrics at k=10, especially MAP, nDCG, and MRR. The main reason is that low-rank compression smooths away fine-grained technical distinctions in Cranfield. That helps broad topic similarity, but it hurts early ranking of exact relevant papers.

### ESA expansion
ESA gives a small improvement over TF-IDF on Precision@10, Recall@10, and F@10, but it is slightly worse on MAP, nDCG, and MRR. This looks like a mixed effect: concept mapping helps retrieve more relevant material, but it also introduces broader concept overlap that can disturb the very top of the ranking.

### Word2Vec expansion
Word2Vec is the best method on Precision@10, Recall@10, and F@10. It also performs very well on MAP and nDCG, though WordNet still wins those ranking-sensitive metrics. The likely cause is that Word2Vec captures corpus-specific similarity better than a general lexicon and is better at finding technical near-synonyms inside the Cranfield vocabulary.

## Best Hyperparameter Setting

No formal hyperparameter sweep was run, so there is no statistically validated best setting. The best observed configuration in this experiment is the default one used for all methods:

- `top_k_neighbors = 10`
- `min_similarity = 0.05`
- `expansion_weight = 0.35`
- `replacement_weight = 1.0`
- `replacement_expansion_weight = 0.35`

If a later tuning pass is done, the most likely useful knobs are:

- lower `expansion_weight` for WordNet if precision needs to be protected
- higher `top_k_neighbors` for broader recall-oriented queries
- smaller `min_similarity` only if the vocabulary is very sparse and synonym coverage is weak

## Fixed Limitation Score Table

These scores are relative, empirical ratings on a 10-point scale based on this Cranfield run. Higher is better.

| Method | Semantic understanding | Ambiguity handling | OOV handling | Context handling | Sparse representation | Cost / scalability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wordnet | 9 | 8 | 9 | 6 | 2 | 3 |
| embedding_tfidf | 4 | 3 | 2 | 3 | 2 | 8 |
| embedding_lsa | 6 | 5 | 2 | 4 | 8 | 5 |
| embedding_esa | 7 | 5 | 3 | 5 | 8 | 4 |
| embedding_word2vec | 8 | 6 | 4 | 6 | 4 | 5 |

### Interpretation of the scores

- WordNet is strongest for semantic matching, ambiguity, and OOV replacement, but it is expensive to build and not good at dense representation.
- TF-IDF expansion is cheap and scalable, but it adds very little true semantic coverage.
- LSA and ESA both address sparsity well, with ESA slightly more interpretable and LSA slightly simpler, but LSA underperforms in ranking quality on this dataset.
- Word2Vec gives the best overall balance for ranking quality and semantic recall, but it is still weaker than WordNet on exact lexical ambiguity control and early ranking of the very best documents.

## Comparison Summary

- Best MAP@10, nDCG@10, and MRR@10: WordNet
- Best Precision@10, Recall@10, and F@10: Word2Vec
- Weakest overall method in this run: LSA
- Best tradeoff between semantic recall and ranking quality: WordNet

## Notes On Caching And Speed

WordNet vocab similarity caching is now implemented in code and will be saved under `cache/wordnet/` inside the experiment folder on the next run that hits the cache path. This is intended to avoid rebuilding the WordNet similarity graph every time.

The current run that generated these results completed before the persistent cache was first populated, so the cache will help the next execution rather than this one.

## Recommended Figure To Use In The Writeup

Use the overlay plot for comparison across methods:

- [overlay comparison](output/eval_overlay.png)

For method-specific curves, use the per-method plots inside each output folder, for example:

- [WordNet plot](output/wordnet/eval_plot.png)
- [TF-IDF plot](output/embedding_tfidf/eval_plot.png)
- [LSA plot](output/embedding_lsa/eval_plot.png)
- [ESA plot](output/embedding_esa/eval_plot.png)
- [Word2Vec plot](output/embedding_word2vec/eval_plot.png)

## Final Takeaway

For this dataset and this setup, WordNet is the best choice if the goal is stronger ranking quality and better handling of synonymy and ambiguity. Word2Vec is the best choice if the goal is raw top-10 precision and recall. LSA is not competitive here, and ESA is a moderate compromise between sparse concept matching and retrieval quality.
