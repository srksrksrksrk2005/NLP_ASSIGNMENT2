# Approach Comparison

## Current methods

1. n-gram
2. n-gram without order (local context)
3. WordNet-based query replacement and expansion
4. Word-embedding-based query replacement and expansion
5. LSA
6. ESA
7. Query reduction
8. WordNet-based local context retrieval

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
| n-gram without order (local context) |  |  |  | X |  | X |  | Local context helps disambiguate terms and gives softer phrase-level evidence even without strict order. |
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
| Context-aware phrase matching | n-gram + n-gram without order (local context) | Medium |  |  |  | X |  | X |  | Ordered phrases catch exact matches; unordered local context adds softer neighborhood evidence. | Phrase variants and nearby-term evidence |
| Reduced semantic expansion | Query reduction + WordNet-based query replacement/expansion | High | X | X |  | 4/10 | X |  |  | First remove noisy terms, then expand only the core concepts. | Long or verbose queries |
| Reduced distributional expansion | Query reduction + word-embedding-based query replacement/expansion | High | X | X |  | 4/10 | 5/10 |  |  | Query reduction increases precision before embedding-based broadening adds recall. | Noisy queries with semantic mismatch |
| Distributional phrase retrieval | n-gram + word-embedding-based query replacement/expansion | High | X |  |  | 4/10 | 5/10 | X |  | Phrase clues preserve precision while embeddings add semantic recall. | Low lexical overlap in multi-word queries |
| Latent plus phrase retrieval | LSA + n-gram | High | X | X | 4/10 | 4/10 |  | X | X | LSA adds broad latent-topic similarity, while n-grams preserve exact technical phrases. | Semantic mismatch with technical phrases |
| Latent plus local-context retrieval | LSA + n-gram without order (local context) | High | X | X | 4/10 | X |  | X | X | Latent semantics improve topic match; local context keeps nearby-term evidence. | Ambiguous technical queries |
| Concept plus phrase retrieval | ESA + n-gram | High | X |  | 4/10 | 4/10 | 5/10 | X | X | ESA contributes explicit concept matching, while n-grams protect precise phrase intent. | Concept mismatch with phrase sensitivity |
| Concept plus local-context retrieval | ESA + n-gram without order (local context) | High | X |  | 4/10 | X | 5/10 | X | X | ESA broadens concept-level matching, while local context gives softer contextual constraints. | Queries mixing concepts and local context |
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
| Hybrid semantic retrieval | WordNet-based expansion + word-embedding-based expansion + n-gram without order (local context) | Medium | X |  |  | X | X | X |  | Combines symbolic and distributional semantics with local evidence, but needs careful weighting. | Broad semantic recall with some context control |
| Reduced concept phrase retrieval | Query reduction + ESA + n-gram | Medium | X | X | 4/10 | 4/10 | 4/10 | X | X | The reduction stage removes noisy terms, ESA gives concept-level matching, and n-grams preserve phrase precision. | Verbose queries with strong phrase structure |
| Reduced latent local-context retrieval | Query reduction + LSA + n-gram without order (local context) | Medium | X | X | 4/10 | X |  | X | X | Cleaner queries help LSA and local context together, though tuning is more important than in simpler hybrids. | Overlong ambiguous queries |

## 4-method merges

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reduced sense-aware phrase semantic retrieval | Query reduction + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | High | X | X |  | X | X | X |  | This gives a clean pipeline: reduce noise, preserve phrases, expand semantically, then disambiguate with local context. | Strong all-around lexical-semantic retrieval |
| Reduced distributional phrase semantic retrieval | Query reduction + n-gram + word-embedding-based query replacement/expansion + n-gram without order (local context) | High | X | X |  | X | 6/10 | X |  | Phrase features preserve exactness while embeddings and local context improve semantic and contextual matching. | Technical queries needing both precision and recall |
| Latent hybrid phrase-semantic retrieval | LSA + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | Medium | X | X | 4/10 | X | X | X | X | Strong coverage of phrase, semantic, and ambiguity issues, but tuning becomes more delicate. | Research-oriented hybrid with symbolic support |
| Concept hybrid phrase-semantic retrieval | ESA + n-gram + WordNet-based query replacement/expansion + WordNet-based local context retrieval | Medium | X |  | 4/10 | X | X | X | X | Explicit concepts plus phrase and WordNet signals can be powerful, though interactions can become complex. | Interpretable semantic hybrid |
| Dual-semantic reduced hybrid | Query reduction + WordNet-based expansion + word-embedding-based expansion + n-gram | Medium | X | X |  | 5/10 | X | X |  | Gives both symbolic and distributional semantics while keeping phrase awareness. | Queries with synonymy plus phrase specificity |

## 5-method merges and larger hybrids

| Combined approach | Methods to merge | Merge quality | Sem. | Dim./Cost | Scale | Ambig. | OOV | Context | Sparse | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full lexical-semantic latent hybrid | Query reduction + n-gram + n-gram without order (local context) + WordNet-based query replacement/expansion + LSA | Medium | X | X | 4/10 | X | X | X | X | Covers noise reduction, phrase structure, local context, lexical semantics, and latent semantics in one pipeline. | Broad-coverage final system with ablations |
| Full lexical-semantic concept hybrid | Query reduction + n-gram + n-gram without order (local context) + WordNet-based query replacement/expansion + ESA | Medium | X | X | 4/10 | X | X | X | X | Similar to the latent hybrid, but uses explicit concepts instead of latent dimensions. | Interpretable final system |
| Symbolic + distributional hybrid | Query reduction + n-gram + WordNet-based expansion + word-embedding-based expansion + WordNet-based local context retrieval | Medium | X | X |  | X | X | X |  | Strong on synonymy, phrase precision, ambiguity, and semantic recall, but can become ad hoc without careful design. | Strong experimental hybrid for comparison |
| Maximal hybrid pipeline | Query reduction + n-gram + n-gram without order (local context) + WordNet-based expansion + word-embedding-based expansion + LSA or ESA + WordNet-based local context retrieval | Low | X | X | 4/10 | X | X | X | X | Covers nearly every limitation, but the system may be hard to justify, tune, and analyze cleanly. | Only useful as a final exploratory ablation, not as the main proposed method |

## Practical note

If you want a clean project story, the strongest merges are:

1. `Query reduction + semantic expansion`
2. `n-gram/local context + semantic method`
3. `WordNet expansion + WordNet local context retrieval`

These combinations are easier to justify than a very large hybrid built from everything at once.
