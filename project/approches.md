# Approach Comparison

## Current methods

1. n-gram
2. unordered local-context bag-of-words
3. WordNet-based query replacement and expansion
4. Word-embedding-based query replacement and expansion
5. LSA
6. ESA
7. Query reduction
8. WordNet-based local context retrieval

Note:
- Here, `unordered local-context bag-of-words` means we look at words appearing within a local window or neighborhood, but we do not preserve their order.

## Limitation coverage table

Legend:
- `X` = strongly addresses the limitation
- `1/10` to `9/10` = partial help strength out of 10
- blank = does not directly address the limitation

Abbreviations:
- `Sem.` = lack of semantic understanding
- `Dim./Cost` = high dimensionality and computational cost
- `Scale` = poor scalability
- `Ambig.` = word sense ambiguity
- `OOV` = out-of-vocabulary problem
- `Context` = lack of contextual representation
- `Sparse` = sparse representations

| Method | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n-gram |  |  |  | 3/10 |  | X |  | Captures phrase/order information and can weakly help when nearby words disambiguate meaning. |
| unordered local-context bag-of-words |  |  |  | X |  | X |  | Local context helps disambiguate terms and provides local co-occurrence evidence without preserving word order. |
| WordNet-based query replacement and expansion | X |  |  | 4/10 | X |  |  | Adds synonym and lexical-semantic variants; can help OOV if a related in-vocabulary term exists. |
| Word-embedding-based query replacement and expansion | X |  |  |  | 5/10 |  |  | Helps semantic matching through similar words; OOV help is partial because it depends on embedding coverage. |
| LSA | X | X | 4/10 | 4/10 |  |  | X | Projects documents into a lower-dimensional latent space, reducing sparsity and capturing broad term co-occurrence patterns. |
| ESA | X |  | 4/10 | 4/10 | 5/10 | 4/10 | X | Maps text into an explicit concept space, which can improve semantic matching and sometimes help rare terms through concept overlap. |
| Query reduction |  | X |  | 4/10 |  |  |  | Removes noisy query terms, which can lower query-time cost and sometimes reduce ambiguity. |
| WordNet-based local context retrieval | X |  |  | X | X | X |  | Uses lexical semantics together with nearby context, so it is stronger on ambiguity and context than plain expansion. |

## Mergeable-method tables

Legend:
- `High` = natural merge, strong complementarity
- `Medium` = useful merge, but needs tuning
- `Low` = possible, but overlap/risk is higher
- `X` = strongly addresses the limitation
- `1/10` to `9/10` = partial help strength out of 10
- blank = does not directly address the limitation

Note:
- These tables list the practically meaningful merges among your current ideas.
- They are grouped by the number of methods combined at one time.
- The numeric scores are rough comparative strengths, not exact measured values.

## 2-method merges

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phrase-aware semantic expansion | n-gram + WordNet-based query replacement/expansion | High | X |  |  | 5/10 | X | X |  | n-grams preserve technical phrases, while WordNet adds synonymy and lexical variation. | Phrase mismatch + synonymy |
| Context-aware local matching | n-gram + unordered local-context bag-of-words | Medium |  |  |  | X |  | X |  | Ordered n-grams catch exact matches; unordered local context adds softer neighborhood evidence. | Exact phrases plus nearby-term evidence |
| Reduced semantic expansion | Query reduction + WordNet-based query replacement/expansion | High | X | X |  | 4/10 | X |  |  | First remove noisy terms, then expand only the core concepts. | Long or verbose queries |
| Reduced distributional expansion | Query reduction + word-embedding-based query replacement/expansion | High | X | X |  | 4/10 | 5/10 |  |  | Query reduction increases precision before embedding-based broadening adds recall. | Noisy queries with semantic mismatch |
| Distributional phrase retrieval | n-gram + word-embedding-based query replacement/expansion | High | X |  |  | 4/10 | 5/10 | X |  | Phrase clues preserve precision while embeddings add semantic recall. | Low lexical overlap in multi-word queries |
| Latent plus phrase retrieval | LSA + n-gram | High | X | X | 4/10 | 4/10 |  | X | X | LSA adds broad latent-topic similarity, while n-grams preserve exact technical phrases. | Semantic mismatch with technical phrases |
| Latent plus local-context retrieval | LSA + unordered local-context bag-of-words | High | X | X | 4/10 | X |  | X | X | Latent semantics improve topic match; unordered local context keeps nearby-term evidence. | Ambiguous technical queries |
| Concept plus phrase retrieval | ESA + n-gram | High | X |  | 4/10 | 4/10 | 5/10 | X | X | ESA contributes explicit concept matching, while n-grams protect precise phrase intent. | Concept mismatch with phrase sensitivity |
| Concept plus local-context retrieval | ESA + unordered local-context bag-of-words | High | X |  | 4/10 | X | 5/10 | X | X | ESA broadens concept-level matching, while unordered local context gives softer contextual constraints. | Queries mixing concepts and local context |
| Sense-aware lexical retrieval | WordNet-based query expansion + WordNet-based local context retrieval | High | X |  |  | X | X | X |  | Expansion improves recall, and local context reduces wrong-sense matches. | Ambiguity + OOV |
| Multi-source semantic expansion | WordNet-based expansion + word-embedding-based expansion | Medium | X |  |  | 4/10 | X |  |  | WordNet gives clean lexical relations; embeddings add corpus-level similarity not present in WordNet. | Broader recall |
| Reduced latent retrieval | Query reduction + LSA | Medium | X | X | 4/10 | 4/10 |  |  | X | Cleaner queries often map better into latent space than noisy full queries. | Noisy or over-specific queries |
| Reduced concept retrieval | Query reduction + ESA | Medium | X | X | 4/10 | 4/10 | 4/10 | 4/10 | X | Removing noisy terms can make concept activation more focused and interpretable. | Verbose queries with mixed concepts |

## 3-method merges

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reduced phrase-aware semantic expansion | Query reduction + n-gram + WordNet-based query replacement/expansion | High | X | X |  | 5/10 | X | X |  | Reduction removes noise, n-grams preserve phrase intent, and WordNet adds lexical-semantic coverage. | Long technical queries with phrase mismatch |
| Reduced distributional phrase retrieval | Query reduction + n-gram + word-embedding-based query replacement/expansion | High | X | X |  | 5/10 | 5/10 | X |  | Query cleanup improves precision, phrases keep exactness, and embeddings recover semantic neighbors. | Low-overlap technical queries |
| Sense-aware phrase retrieval | n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | High | X |  |  | X | X | X |  | Phrase evidence improves precision while WordNet handles synonymy and local context helps disambiguation. | Ambiguity inside multi-word technical queries |
| Latent sense-aware retrieval | LSA + WordNet-based query replacement/expansion + WordNet-based local context retrieval | High | X | X | 4/10 | X | X | X | X | LSA captures broad latent similarity, WordNet adds lexical expansion, and local context controls sense drift. | Semantic mismatch plus ambiguity |
| Concept sense-aware retrieval | ESA + WordNet-based query replacement/expansion + WordNet-based local context retrieval | High | X |  | 4/10 | X | X | X | X | ESA adds explicit concept matching while WordNet expansion and local context make the retrieval more sense-sensitive. | Concept overlap with ambiguous query terms |
| Hybrid semantic retrieval | WordNet-based expansion + word-embedding-based expansion + unordered local-context bag-of-words | Medium | X |  |  | X | X | X |  | Combines symbolic and distributional semantics with local co-occurrence evidence, but needs careful weighting. | Broad semantic recall with some context control |
| Reduced concept phrase retrieval | Query reduction + ESA + n-gram | Medium | X | X | 4/10 | 4/10 | 4/10 | X | X | The reduction stage removes noisy terms, ESA gives concept-level matching, and n-grams preserve phrase precision. | Verbose queries with strong phrase structure |
| Reduced latent local-context retrieval | Query reduction + LSA + unordered local-context bag-of-words | Medium | X | X | 4/10 | X |  | X | X | Cleaner queries help LSA and unordered local context together, though tuning is more important than in simpler hybrids. | Overlong ambiguous queries |

## 4-method merges

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reduced sense-aware phrase semantic retrieval | Query reduction + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | High | X | X |  | X | X | X |  | This gives a clean pipeline: reduce noise, preserve phrases, expand semantically, then disambiguate with local context. | Strong all-around lexical-semantic retrieval |
| Reduced distributional phrase semantic retrieval | Query reduction + n-gram + word-embedding-based query replacement/expansion + unordered local-context bag-of-words | High | X | X |  | X | 6/10 | X |  | Phrase features preserve exactness while embeddings and unordered local context improve semantic and contextual matching. | Technical queries needing both precision and recall |
| Latent hybrid phrase-semantic retrieval | LSA + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | Medium | X | X | 4/10 | X | X | X | X | Strong coverage of phrase, semantic, and ambiguity issues, but tuning becomes more delicate. | Research-oriented hybrid with symbolic support |
| Concept hybrid phrase-semantic retrieval | ESA + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | Medium | X |  | 4/10 | X | X | X | X | Explicit concepts plus phrase and WordNet signals can be powerful, though interactions can become complex. | Interpretable semantic hybrid |
| Dual-semantic reduced hybrid | Query reduction + WordNet-based expansion + word-embedding-based expansion + n-gram | Medium | X | X |  | 5/10 | X | X |  | Gives both symbolic and distributional semantics while keeping phrase awareness. | Queries with synonymy plus phrase specificity |

## 5-method merges and larger hybrids

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full lexical-semantic latent hybrid | Query reduction + n-gram + unordered local-context bag-of-words + WordNet-based query replacement/expansion + LSA | Medium | X | X | 4/10 | X | X | X | X | Covers noise reduction, phrase structure, local co-occurrence, lexical semantics, and latent semantics in one pipeline. | Broad-coverage final system with ablations |
| Full lexical-semantic concept hybrid | Query reduction + n-gram + unordered local-context bag-of-words + WordNet-based query replacement/expansion + ESA | Medium | X | X | 4/10 | X | X | X | X | Similar to the latent hybrid, but uses explicit concepts instead of latent dimensions. | Interpretable final system |
| Symbolic + distributional hybrid | Query reduction + n-gram + WordNet-based expansion + word-embedding-based expansion + WordNet-based local context retrieval | Medium | X | X |  | X | X | X |  | Strong on synonymy, phrase precision, ambiguity, and semantic recall, but can become ad hoc without careful design. | Strong experimental hybrid for comparison |
| Maximal hybrid pipeline | Query reduction + n-gram + unordered local-context bag-of-words + WordNet-based expansion + word-embedding-based expansion + LSA or ESA + WordNet-based local context retrieval | Low | X | X | 4/10 | X | X | X | X | Covers nearly every limitation, but the system may be hard to justify, tune, and analyze cleanly. | Only useful as a final exploratory ablation, not as the main proposed method |

## Practical note

If you want a clean project story, the strongest merges are:

1. `Query reduction + semantic expansion`
2. `n-gram/local context + semantic method`
3. `WordNet expansion + WordNet local context retrieval`

These combinations are easier to justify than a very large hybrid built from everything at once.

## Next backbones to try: BM25 and Word2Vec

Your current methods already separate nicely into:

- `backbones`: TF-IDF, LSA, ESA, WordNet-style lexical semantics
- `add-ons`: n-grams, unordered local context, query expansion, query reduction

That same structure can be reused for new backbones.

### A) BM25 as a backbone

BM25 should be treated as the strongest lexical baseline/backbone to add next.

Natural variants:

1. `BM25 unigram`
   - Direct replacement for TF-IDF as the lexical backbone.
   - Best first comparison because it is simple and standard.

2. `BM25 + n-gram indexing`
   - Use unigram + bigram, or unigram + bigram + trigram.
   - Good when phrase structure matters.

3. `BM25 + unordered local-context bag-of-words`
   - Retrieve with BM25, then add a reranking score based on local window overlap or local co-occurrence.
   - This keeps BM25 as the main retriever and uses local context as a second signal.

4. `BM25 + query reduction`
   - Reduce noisy terms before retrieval.
   - Especially useful for long technical queries.

5. `BM25 + WordNet query expansion`
   - Strong portable version of your current WordNet expansion idea.

6. `BM25 + embedding-based query expansion`
   - Expand the query with nearest neighbors from embedding space, but score with BM25.
   - This is one of the cleanest "same implementation on a new backbone" experiments.

7. `BM25 + semantic reranking`
   - First-stage retrieval with BM25, second-stage rerank with Word2Vec similarity, WMD, or document embeddings.
   - Very strong design because BM25 gives precision and semantic reranking improves recall/order.

### B) Word2Vec as a backbone

If you want Word2Vec to be a true backbone, the key question is:

- how do we convert word embeddings into a single query/document representation or score?

The most practical methods are below.

#### 1) Average Word2Vec / centroid embedding

- Represent a query or document by the mean of its word vectors.
- Score with cosine similarity.
- This is the simplest Word2Vec backbone.

Good:
- very easy to implement
- fast
- reusable across query expansion and reranking

Weakness:
- ignores word order
- common words can dominate unless weighted

#### 2) IDF-weighted average Word2Vec

- Average word vectors, but weight each token by IDF or TF-IDF-like importance.
- Better than plain averaging for retrieval because important terms matter more.

This is probably the best first Word2Vec backbone to try.

#### 3) SIF-style sentence/document embeddings

- Use weighted averaging of word vectors with stronger downweighting of frequent words.
- Then remove the top principal component.
- This often improves sentence/document embeddings over plain averaging.

This is still simple, but more principled than raw averaging.

#### 4) Paragraph Vector / Doc2Vec

- Learn document-level dense vectors directly rather than averaging word vectors.
- This is the classic "word2vec-family document embedding" method.

Good:
- genuine document embedding backbone
- better than raw averaging when trained well

Risk:
- needs careful training and may be unstable on small datasets like Cranfield

#### 5) Word Mover's Distance (WMD)

- Compare query and document by the minimum transport cost between their word embeddings.
- Strong semantic matching method for short texts and vocabulary mismatch.

Good:
- semantically powerful
- handles partial lexical mismatch well

Weakness:
- expensive
- better as reranker on top of BM25 than as full first-stage retrieval

#### 6) Soft Cosine with embedding-based term similarity

- Keep a bag-of-words style representation, but when scoring, allow similar words to partially match.
- This is a nice bridge between lexical backbones and embeddings.

Why it is useful:
- can be applied on top of TF-IDF or BM25-style term weighting
- gives a Word2Vec-informed similarity without fully switching to dense retrieval

### Best Word2Vec-specific experiment order

If you want a clean and defensible progression, try them in this order:

1. `IDF-weighted average Word2Vec`
2. `SIF-style embeddings`
3. `BM25 + Word2Vec reranking`
4. `BM25 + WMD reranking`
5. `Doc2Vec`

This order goes from easiest and most stable to more expensive or harder-to-tune methods.

## Which of your current ideas transfer to the new backbones?

### Cleanly transferable to BM25, TF-IDF, LSA, ESA, and Word2Vec-style systems

1. `Query reduction`
   - Works everywhere.

2. `WordNet query expansion`
   - Works everywhere.

3. `Embedding-based query expansion`
   - Works everywhere, as long as scoring is done by the backbone afterward.

4. `n-gram features`
   - Natural for TF-IDF and BM25.
   - Can also be adapted to Word2Vec if phrase embeddings are built or phrase tokens are learned.

5. `unordered local-context bag-of-words`
   - Best used as an extra feature or reranking signal on top of any backbone.

### Less directly transferable

1. `LSA`
   - Already a document-space embedding method, so it does not combine as naturally with another dense document embedding backbone like Doc2Vec unless used in score fusion.

2. `ESA`
   - Works best as its own concept-space backbone or as a fused semantic signal.

## Recommended experiment matrix

To keep the project organized, the cleanest experiment matrix is:

### Lexical backbones

1. `TF-IDF unigram`
2. `TF-IDF unigram + bigram`
3. `BM25 unigram`
4. `BM25 unigram + bigram`

For each lexical backbone, try:

- `baseline`
- `+ query reduction`
- `+ WordNet expansion`
- `+ embedding expansion`
- `+ local-context reranking`

### Dense / semantic backbones

1. `LSA`
2. `ESA`
3. `Word2Vec average`
4. `Word2Vec IDF-weighted average`
5. `Word2Vec SIF`
6. `Doc2Vec`

For each dense backbone, try:

- `baseline`
- `+ query reduction`
- `+ WordNet expansion`
- `+ embedding expansion`
- optional `+ n-gram phrase handling` if phrase tokens are learned

### Hybrid systems

1. `BM25 + LSA`
2. `BM25 + ESA`
3. `BM25 + Word2Vec average`
4. `BM25 + Word2Vec SIF`
5. `BM25 + WMD reranking`

These hybrids may be your best-performing final systems, because lexical and semantic signals usually complement each other.

## Suggested project story

If you want the experiments to read clearly in the report, frame them as:

1. `Stage 1: lexical backbones`
   - TF-IDF vs BM25

2. `Stage 2: semantic backbones`
   - LSA, ESA, Word2Vec-based document/query embeddings

3. `Stage 3: reusable query processing`
   - query reduction, WordNet expansion, embedding expansion

4. `Stage 4: hybrid retrieval`
   - lexical backbone + semantic reranking/fusion

This gives you one clean narrative:
- first improve lexical retrieval,
- then test semantic backbones,
- then apply the same query-side methods across them,
- then combine the strongest lexical and semantic systems.

## Research-backed methods to cite

- `Word2Vec`: Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013)
- `SIF sentence embeddings`: Arora, Liang, and Ma, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" (ICLR 2017)
- `Doc2Vec / Paragraph Vector`: Le and Mikolov, "Distributed Representations of Sentences and Documents" (2014)
- `WMD`: Kusner et al., "From Word Embeddings to Document Distances" (ICML 2015)
- `BM25`: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
- `Soft Cosine`: Sidorov et al., "Soft Similarity and Soft Cosine Measure" (2014)



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



# Merging Pipeline - Implementation Summary

## Project Structure

```
merging/
├── Main Scripts
│   ├── main.py                          # Main orchestration (links all blocks)
│   ├── block1_query_processing.py       # Block 1: Query expansion & reduction
│   ├── blocks23_retrieval_ranking.py    # Blocks 2 & 3: Retrieval & ranking
│   ├── run_experiments.py               # Experiment runner with grid search
│   ├── validate.py                      # Validation/testing script
│   └── quickstart.bat                   # Quick-start menu (Windows)
│
├── Utilities (utils/)
│   ├── __init__.py
│   ├── logger.py                        # WandB-compatible logging
│   ├── data_loader.py                   # Dataset loading and management
│   └── preprocessing.py                 # Text preprocessing utilities
│
├── Configuration (configs/)
│   ├── default_config.json              # Default configuration template
│   ├── lsa_lsa_config.json              # LSA expansion + LSA ranking preset
│   └── wordnet_esa_config.json          # WordNet expansion + ESA ranking preset
│
├── Output (output/)
│   └── [Generated during runs]
│       ├── results.json                 # Complete results with metrics
│       ├── metrics_plot.png             # Metrics visualization
│       ├── config.json                  # Saved config for reproducibility
│       ├── pipeline_*.log               # Execution logs
│       └── grid_search_summary_*.json   # Grid search results
│
├── Dependencies & Docs
│   ├── README.md                        # Full documentation
│   ├── requirements.txt                 # Python package requirements
│   └── IMPLEMENTATION_SUMMARY.md        # This file
```

## What Was Implemented

### Block 1: Query Processing (block1_query_processing.py)

**Functionality:**
- Query expansion modes: none, LSA, ESA, WordNet, Word2Vec, TF-IDF
- Query reduction: enabled/disabled with configurable parameters
- IDF-weighted query vectorization
- Batch processing support
- Outputs: numpy arrays (IDF-weighted vectors) or token lists

**Key Classes:**
- `QueryProcessor`: Main class handling all query processing operations

**Features:**
- Lazy loading of models (LSA, TF-IDF, etc.)
- Flexible preprocessing pipeline
- Configurable keyword extraction for reduction
- Supports both single queries and batch processing

**Usage:**
```python
processor = QueryProcessor(config)
query_vector = processor.process_query("information retrieval")
batch_vectors = processor.process_batch(queries, docs_tokens)
```

### Blocks 2 & 3: Retrieval & Ranking (blocks23_retrieval_ranking.py)

**Functionality:**
- Block 2 Retrieval types: TF-IDF, N-gram, Local Bag-of-Words
- Block 3 Ranking types: TF-IDF, LSA, ESA
- Integrated single pipeline for efficiency
- Document indexing and ranking with scores

**Key Classes:**
- `RetrievalRankingPipeline`: Combined retrieval and ranking

**Features:**
- Modular design: easily switch between retrieval and ranking methods
- Sparse and dense matrix handling
- Concept-based (ESA) and latent (LSA) ranking methods
- Returns ranked documents with similarity scores

**Usage:**
```python
pipeline = RetrievalRankingPipeline(config)
pipeline.build_retrieval_index(docs, doc_ids)
rankings = pipeline.rank(query_vectors)
```

### Main Orchestration Script (main.py)

**Functionality:**
- Ties all three blocks together
- Handles data loading and preprocessing
- Performs IR evaluation (Precision, Recall, MAP, NDCG, MRR)
- Generates visualizations
- Integrates with WandB for experiment tracking

**Key Features:**
- Complete end-to-end pipeline
- Automatic metric calculation
- Plot generation with matplotlib
- JSON output for reproducibility
- Command-line interface for easy parameter tuning

**Evaluation Metrics:**
- Precision@10, Recall@10, F-score@10
- MAP@10 (Mean Average Precision)
- NDCG@10 (Normalized Discounted Cumulative Gain)
- MRR@10 (Mean Reciprocal Rank)

### Utilities Module (utils/)

**Logger (logger.py):**
- PipelineLogger class with WandB integration
- Structured logging to file and console
- Metrics tracking and artifact logging
- Compatible with WandB dashboards

**Data Loader (data_loader.py):**
- DataLoader class for managing datasets
- JSON-based data loading
- Results saving/loading functionality
- Handles Cranfield dataset format

**Preprocessing (preprocessing.py):**
- TextPreprocessor for standardized text processing
- Supports: tokenization, lowercasing, stopword removal, stemming, lemmatization
- Batch processing capabilities
- NLTK-based implementation

### Experiment Runner (run_experiments.py)

**Functionality:**
- Single experiment runner
- Grid search over multiple combinations
- Systematic testing of all block combinations
- Results aggregation and summary generation

**Features:**
- Easy parameter specification
- Automatic execution of multiple configurations
- Success/failure tracking
- Summary statistics with success rates
- JSON export of results

### Validation Script (validate.py)

**Tests:**
1. Import testing (all required packages)
2. Utility modules loading
3. Block 1 functionality
4. Blocks 2 & 3 functionality
5. Configuration validation
6. Data loading (if available)

**Purpose:** Verify the entire pipeline is correctly installed and functional

## Configuration System

### Config Structure

Each config file contains:

```json
{
  "block1_query_processing": {
    "expansion_mode": "none|lsa|esa|wordnet|word2vec|tfidf",
    "reduction_enabled": true|false,
    "expansion_params": { /* mode-specific params */ }
  },
  "block2_retrieval_mode": {
    "retrieval_type": "tfidf|ngram|local_bow",
    "retrieval_params": { /* retrieval-specific params */ }
  },
  "block3_ranking_mode": {
    "ranking_type": "tfidf|lsa|esa",
    "ranking_params": { /* ranking-specific params */ }
  },
  "dataset": { /* dataset paths */ },
  "output": { /* output directory and options */ },
  "logging": { /* logging and WandB settings */ }
}
```

### Pre-configured Presets

1. **default_config.json**: Basic configuration template
2. **lsa_lsa_config.json**: LSA expansion + LSA ranking
3. **wordnet_esa_config.json**: WordNet expansion + ESA ranking

## WandB Integration

### Features
- Automatic configuration logging
- Metrics tracking and visualization
- Artifact saving (results, plots)
- Experiment comparison dashboard

### Enabling WandB
```bash
# Command line
python main.py --wandb

# Or in config:
# "logging": { "use_wandb": true, "wandb_project": "nlp-merging" }
```

## Usage Examples

### Quick Start
```bash
# Using the quick-start menu (Windows)
quickstart.bat

# Or directly with Python
python main.py
```

### Command Line Examples
```bash
# Baseline (TF-IDF for all blocks)
python main.py --block1-mode none --block2-retrieval tfidf --block3-ranking tfidf

# LSA-based pipeline
python main.py --block1-mode lsa --block3-ranking lsa

# WordNet expansion + ESA ranking
python main.py --block1-mode wordnet --block3-ranking esa

# With query reduction
python main.py --block1-reduce --block3-ranking lsa

# Custom config file
python main.py --config configs/lsa_lsa_config.json

# With WandB logging
python main.py --wandb
```

### Grid Search Examples
```bash
# Grid search with all defaults
python run_experiments.py --mode grid

# Grid search with specific blocks
python run_experiments.py --mode grid \
  --grid-block1 none lsa wordnet \
  --grid-block2 tfidf ngram \
  --grid-block3 lsa esa

# Include reduction in grid search
python run_experiments.py --mode grid --grid-reduce
```

### Testing & Validation
```bash
# Validate entire installation
python validate.py

# Run single Block 1 query
python block1_query_processing.py --query "your query" --mode lsa
```

## Supported Method Combinations

### Query Expansion Modes
- **none**: Basic tokenization only
- **lsa**: Latent Semantic Analysis-based expansion
- **esa**: Explicit Semantic Analysis (document similarity-based)
- **wordnet**: WordNet synonym expansion
- **word2vec**: Word embeddings (framework provided)
- **tfidf**: TF-IDF weighted term expansion

### Retrieval Methods (Block 2)
- **tfidf**: Standard TF-IDF with cosine similarity
- **ngram**: Character/word n-gram based retrieval
- **local_bow**: Windowed bag-of-words representation

### Ranking Methods (Block 3)
- **tfidf**: TF-IDF cosine similarity ranking
- **lsa**: Latent Semantic Analysis ranking
- **esa**: Explicit Semantic Analysis ranking

### Total Combinations
- 6 query expansion modes × 2 retrieval methods × 3 ranking methods = **36 combinations**
- With optional query reduction: **72 total configurations**

## Output Files

All outputs are saved in the configured `output_dir`:

### results.json
```json
{
  "config": { /* full configuration used */ },
  "metrics": {
    "precision@10": 0.45,
    "recall@10": 0.32,
    "map@10": 0.38,
    "ndcg@10": 0.42,
    "mrr@10": 0.65
  },
  "rankings": {
    "query_id": ["doc1", "doc2", ...],
    ...
  },
  "timestamp": "2024-01-15T10:30:45"
}
```

### Visualization Files
- `metrics_plot.png`: Bar chart of IR metrics
- `config.json`: Saved configuration for reproducibility

### Log Files
- `pipeline_YYYYMMDD_HHMMSS.log`: Detailed execution log

### Grid Search Results
- `grid_search_summary_YYYYMMDD_HHMMSS.json`: Summary of all grid search runs

## Key Features

✓ **Modular Design**: Easily switch between different methods in each block
✓ **Batch Processing**: Process multiple queries efficiently
✓ **Comprehensive Evaluation**: Standard IR metrics (P@10, R@10, MAP, NDCG, MRR)
✓ **WandB Integration**: Track experiments and compare results
✓ **Configuration-Based**: All settings in JSON config files
✓ **Reproducibility**: Save configs and results for easy reproduction
✓ **Grid Search**: Automatically test multiple configurations
✓ **Extensible**: Easy to add new query expansion, retrieval, or ranking methods
✓ **Logging**: Detailed logging with file and console output
✓ **Visualization**: Automatic plot generation

## Performance Characteristics

- **TF-IDF**: Fastest (~seconds for full pipeline)
- **N-gram**: Slower than TF-IDF (~seconds)
- **LSA**: Slower than TF-IDF, captures latent semantics (~seconds)
- **ESA**: Slowest, concept-based ranking (~seconds for moderate corpus)
- **WordNet expansion**: Memory-intensive for large vocabularies
- **Query reduction**: Minimal overhead (~milliseconds)

## Extensibility Points

To add new methods:

1. **New Query Expansion Mode**: Add method to `QueryProcessor` class
2. **New Retrieval Type**: Add method to `RetrievalRankingPipeline` class
3. **New Ranking Type**: Add ranking method to `RetrievalRankingPipeline` class
4. **Custom Evaluation**: Extend `evaluate_rankings` in `main.py`
5. **New Preprocessing**: Extend `TextPreprocessor` in `utils/preprocessing.py`

## System Requirements

- Python 3.7+
- scikit-learn >= 0.24.0
- numpy >= 1.20.0
- nltk >= 3.6.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- wandb >= 0.12.0 (optional)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Validate installation
python validate.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Config not found | Check path with `--config` flag |
| Dataset not found | Update dataset path in config |
| ImportError | Run `pip install -r requirements.txt` |
| NLTK data missing | Run NLTK download commands |
| WandB issues | Install with `pip install wandb` or set `use_wandb: false` |

## Next Steps

1. **Run validation**: `python validate.py`
2. **Try quickstart**: `python main.py` or `quickstart.bat`
3. **Explore configs**: Check `configs/` for different presets
4. **Run experiments**: `python run_experiments.py --mode grid`
5. **Check results**: Look in `output/` directory

## File Checksums

| File | Purpose | Lines |
|------|---------|-------|
| block1_query_processing.py | Query processing | ~300 |
| blocks23_retrieval_ranking.py | Retrieval & ranking | ~350 |
| main.py | Main orchestration | ~250 |
| utils/logger.py | Logging utilities | ~100 |
| utils/data_loader.py | Data management | ~70 |
| utils/preprocessing.py | Text preprocessing | ~120 |
| run_experiments.py | Experiment runner | ~200 |
| validate.py | Validation tests | ~200 |

**Total: ~1500 lines of code**

---

## Support & Documentation

- **Main README**: See `README.md` for detailed usage guide
- **Config Templates**: Check `configs/` directory
- **Validation**: Run `python validate.py` to verify setup
- **Examples**: Use `quickstart.bat` for interactive menu

---

Created: 2024
Part of: NLP Assignment 2 - Information Retrieval Method Merging
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
- `min_context_df = 3`
- `context_weight_alpha = 0.8`
- `context_feature_vocab_size = 45535`

## Local Context Size Sweep

| n | Best orders | Best min_df | Best alpha | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [2] | 3 | 0.50 | 0.2978 | 0.4303 | 0.3258 | 0.3255 | 0.4843 | 0.7655 |
| 2 | [2] | 3 | 0.80 | 0.2947 | 0.4253 | 0.3224 | 0.3261 | 0.4840 | 0.7709 |
| 3 | [2, 3] | 3 | 0.80 | 0.2960 | 0.4279 | 0.3238 | 0.3277 | 0.4889 | 0.7758 |
| 4 | [2] | 3 | 0.80 | 0.2938 | 0.4251 | 0.3215 | 0.3297 | 0.4883 | 0.7779 |
| 5 | [2] | 3 | 0.80 | 0.2911 | 0.4224 | 0.3188 | 0.3258 | 0.4827 | 0.7711 |
| 6 | [2] | 3 | 0.80 | 0.2933 | 0.4255 | 0.3212 | 0.3274 | 0.4856 | 0.7720 |

The sweep shows a trade-off rather than a single monotonic pattern. `n = 1` is strongest on precision, recall, F-score, and tied nDCG, while `n = 4` gives the best MAP@10 and MRR@10. In this dataset, a moderately larger window captures useful technical co-occurrence without drifting too far from the local topic.

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.20 |
| unordered_local_context_bow | 0.2938 | 0.4251 | 0.3215 | 0.3297 | 0.4883 | 0.7779 | 379.85 |

## Delta vs Baseline at k=10

- `dP@10 = +0.0107`
- `dR@10 = +0.0158`
- `dF@10 = +0.0116`
- `dMAP@10 = +0.0236`
- `dnDCG@10 = +0.0272`
- `dMRR@10 = +0.0318`

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
| 40 | 2 | 2 | 0.1538 | 0.1538 | 0.0000 |
| 64 | 1 | 2 | 0.2619 | 0.3333 | 0.0714 |
| 81 | 1 | 1 | 0.2869 | 0.3869 | 0.1000 |
| 90 | 2 | 2 | 0.1048 | 0.1071 | 0.0024 |
| 119 | 0 | 2 | 0.1944 | 0.8333 | 0.6389 |
| 142 | 1 | 1 | 0.2500 | 0.6250 | 0.3750 |
| 61 | 3 | 3 | 0.4611 | 0.7153 | 0.2542 |

## Output Files

- Summary JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/summary.json`
- Summary CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/summary_k10.csv`
- Overlay plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/eval_overlay.png`
- All-combinations overlay: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/all_tuned_combinations_overlay.png`
- n-sweep JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.json`
- n-sweep CSV: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.csv`
- n-sweep plot: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/n_sweep_best_by_radius.png`
- Example comparison markdown: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/example_query_comparison.md`
- Config sweep JSON: `/home/crimson/Projects/Acads/NLP/Project/NLP_ASSIGNMENT2/project/chandan/unordered_local_context_bow/output/config_sweep.json`