# IDF-weighted Word2Vec Backbone Experiment Report

## Method Summary

The baseline system uses unigram TF-IDF with cosine similarity. The proposed backbone still uses Word2Vec token embeddings, but instead of a plain centroid it weights each token vector by TF * IDF before averaging.

This gives more influence to technically informative terms and reduces the dominance of very common words, while still keeping the retrieval model simple and fast.

## Hypothesis

Compared with plain averaging, IDF-weighted centroids should preserve more query intent because rare technical terms contribute more strongly to the final dense representation.

The method still compresses each item into a single vector, so it is expected to help semantic matching without fully solving ambiguity, phrase structure, or local context problems.

## Baseline Limitations Addressed

- The baseline VSM depends strongly on exact lexical overlap.
- Plain centroid averaging can overemphasize frequent words.
- IDF weighting is meant to keep semantically important technical tokens more visible in the dense representation.

## Best Configuration

- `vector_size = 150`
- `window = 5`
- `training_objective = skipgram`
- `epochs = 20`
- `model_coverage = 100.00%`
- `avg_in_vocab_tokens_per_doc = 93.21`
- `avg_in_vocab_tokens_per_query = 10.03`

## Embedding Size Sweep

| Vector size | Window | Objective | Epochs | Coverage | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 5 | skipgram | 20 | 100.00% | 0.2249 | 0.3393 | 0.2503 | 0.2294 | 0.3674 | 0.6174 |
| 100 | 5 | skipgram | 20 | 100.00% | 0.2413 | 0.3615 | 0.2678 | 0.2531 | 0.3961 | 0.6567 |
| 150 | 5 | skipgram | 20 | 100.00% | 0.2484 | 0.3715 | 0.2757 | 0.2624 | 0.4084 | 0.6723 |

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.07 |
| word2vec_idf_weighted | 0.2484 | 0.3715 | 0.2757 | 0.2624 | 0.4084 | 0.6723 | 93.08 |

## Delta vs Baseline at k=10

- `dP@10 = -0.0347`
- `dR@10 = -0.0379`
- `dF@10 = -0.0342`
- `dMAP@10 = -0.0436`
- `dnDCG@10 = -0.0527`
- `dMRR@10 = -0.0738`

## Significance Checks

Approximate randomization over per-query scores:

- `AP@10 p-value = 0.0001`
- `nDCG@10 p-value = 0.0001`

## Interpretation

This method is a stronger first Word2Vec backbone than plain averaging because it respects corpus-level term importance. If a query contains a few decisive technical words, weighting helps those words shape the centroid instead of being diluted by generic context terms.

Its limitation is structural rather than lexical: the representation is still one dense vector per item, so compositional detail, token order, and explicit local matching remain weak compared with lexical or hybrid models.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0417 | 0.0417 |
| 40 | 2 | 2 | 0.1538 | 0.0769 | -0.0769 |
| 64 | 1 | 1 | 0.2619 | 0.1786 | -0.0833 |
| 81 | 1 | 0 | 0.2869 | 0.0417 | -0.2452 |
| 90 | 2 | 1 | 0.1048 | 0.0952 | -0.0095 |
| 119 | 0 | 1 | 0.1944 | 0.6667 | 0.4722 |
| 12 | 0 | 2 | 0.0167 | 0.4167 | 0.4000 |
| 180 | 3 | 5 | 0.4152 | 0.7958 | 0.3807 |

## Research References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (NeurIPS 2013): https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html
- Arora, Liang, and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (ICLR 2017): https://openreview.net/forum?id=SyK00v5xx

## Output Files

- summary_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\summary.json`
- summary_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\summary_k10.csv`
- overlay_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\eval_overlay.png`
- all_configs_overlay: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\all_tuned_combinations_overlay.png`
- vector_size_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\vector_size_sweep_best.json`
- vector_size_sweep_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\vector_size_sweep_best.csv`
- vector_size_sweep_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\vector_size_sweep_best.png`
- example_markdown: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\example_query_comparison.md`
- config_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_idf_weighted\output\config_sweep.json`