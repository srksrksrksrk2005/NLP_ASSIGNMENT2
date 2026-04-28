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
