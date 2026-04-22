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
- `[x]` = directly helps with the limitation
- `(~)` = partially helps
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
| n-gram |  |  |  | (~) |  | [x] |  | Captures phrase/order information and can weakly help when nearby words disambiguate meaning. |
| n-gram without order (local context) |  |  |  | [x] |  | [x] |  | Local context helps disambiguate terms and gives softer phrase-level evidence even without strict order. |
| WordNet-based query replacement and expansion | [x] |  |  | (~) | [x] |  |  | Adds synonym and lexical-semantic variants; can help OOV if a related in-vocabulary term exists. |
| Word-embedding-based query replacement and expansion | [x] |  |  |  | (~) |  |  | Helps semantic matching through similar words; OOV help is partial because it depends on embedding coverage. |
| LSA | [x] | [x] | (~) | (~) |  |  | [x] | Projects documents into a lower-dimensional latent space, reducing sparsity and capturing broad term co-occurrence patterns. |
| ESA | [x] |  | (~) | (~) | (~) | (~) | [x] | Maps text into an explicit concept space, which can improve semantic matching and sometimes help rare terms through concept overlap. |
| Query reduction |  | [x] |  | (~) |  |  |  | Removes noisy query terms, which can lower query-time cost and sometimes reduce ambiguity. |
| WordNet-based local context retrieval | [x] |  |  | [x] | [x] | [x] |  | Uses lexical semantics together with nearby context, so it is stronger on ambiguity and context than plain expansion. |

## Mergeable-method table

Legend:
- `High` = natural merge, strong complementarity
- `Medium` = useful merge, but needs tuning
- `Low` = possible, but overlap/risk is higher

| Combined approach | Methods to merge | Merge quality | Why the merge makes sense | Good for |
| --- | --- | --- | --- | --- |
| Phrase-aware semantic expansion | n-gram + WordNet-based query replacement/expansion | High | n-grams preserve technical phrases, while WordNet adds synonymy and lexical variation. | Phrase mismatch + synonymy |
| Context-aware phrase matching | n-gram + n-gram without order (local context) | Medium | Ordered phrases catch exact matches; unordered local context adds softer neighborhood evidence. | Phrase variants and nearby-term evidence |
| Reduced semantic expansion | Query reduction + WordNet-based or embedding-based expansion | High | First remove noisy words, then expand only the core concepts. | Long or verbose queries |
| Distributional phrase retrieval | n-gram + embedding-based query replacement/expansion | High | Phrase clues preserve precision while embeddings add semantic recall. | Low lexical overlap in multi-word queries |
| Latent plus local-context hybrid | LSA + n-gram or local-context retrieval | High | LSA gives broad latent-topic similarity, while local context preserves phrase-level cues that latent methods may blur. | Semantic mismatch with technical phrases |
| Concept plus local-context hybrid | ESA + n-gram or local-context retrieval | High | ESA contributes interpretable concept matching, while local context protects exact technical intent. | Concept mismatch with phrase sensitivity |
| Sense-aware lexical retrieval | WordNet-based query expansion + WordNet-based local context retrieval | High | Expansion improves recall, and local context reduces wrong-sense matches. | Ambiguity + OOV |
| Multi-source semantic expansion | WordNet-based expansion + embedding-based expansion | Medium | WordNet gives clean lexical relations; embeddings add corpus-level similarity not present in WordNet. | Broader recall |
| Reduced latent retrieval | Query reduction + LSA | Medium | Cleaner queries often map better into latent space than noisy full queries. | Noisy or over-specific queries |
| Reduced concept retrieval | Query reduction + ESA | Medium | Removing noisy terms can make concept activation more focused and interpretable. | Verbose queries with mixed concepts |
| Full hybrid pipeline | Query reduction + local context/n-gram + semantic expansion + LSA or ESA | Medium | Covers the most limitations, but tuning becomes harder and the system can become ad hoc if not justified carefully. | Final combined system with ablations |

## Practical note

If you want a clean project story, the strongest merges are:

1. `Query reduction + semantic expansion`
2. `n-gram/local context + semantic method`
3. `WordNet expansion + WordNet local context retrieval`

These combinations are easier to justify than a very large hybrid built from everything at once.
