# Average Word2Vec Backbone Experiment Report

## Method Summary

The baseline system uses unigram TF-IDF with cosine similarity. The proposed backbone trains Word2Vec on the Cranfield document corpus and represents each document and query by the average of its in-vocabulary word vectors.

This follows the simplest compositional Word2Vec strategy: convert token embeddings into a single document/query centroid, then rank with cosine similarity in the dense vector space.

## Hypothesis

A centroid embedding should recover some semantic similarity that TF-IDF misses, especially when a relevant document uses related technical wording instead of the exact same surface form.

Because the method compresses the full token sequence into one dense vector, it should be fast at query time, but it may blur fine-grained distinctions and phrase structure.

## Baseline Limitations Addressed

- The baseline VSM depends strongly on exact lexical overlap.
- Dense centroids can partially address semantic mismatch between queries and documents.
- Averaging is still context-light, so ambiguity and phrase ordering are only weakly handled.

## Best Configuration

- `vector_size = 100`
- `window = 5`
- `training_objective = skipgram`
- `epochs = 20`
- `model_coverage = 100.00%`
- `avg_in_vocab_tokens_per_doc = 93.21`
- `avg_in_vocab_tokens_per_query = 10.03`

## Embedding Size Sweep

| Vector size | Window | Objective | Epochs | Coverage | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 5 | skipgram | 20 | 100.00% | 0.2080 | 0.2999 | 0.2276 | 0.1984 | 0.3284 | 0.5781 |
| 100 | 5 | skipgram | 20 | 100.00% | 0.2298 | 0.3333 | 0.2519 | 0.2230 | 0.3649 | 0.6127 |
| 150 | 5 | skipgram | 20 | 100.00% | 0.2284 | 0.3318 | 0.2508 | 0.2213 | 0.3660 | 0.6181 |

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.08 |
| word2vec_average | 0.2298 | 0.3333 | 0.2519 | 0.2230 | 0.3649 | 0.6127 | 98.69 |

## Delta vs Baseline at k=10

- `dP@10 = -0.0533`
- `dR@10 = -0.0761`
- `dF@10 = -0.0579`
- `dMAP@10 = -0.0831`
- `dnDCG@10 = -0.0962`
- `dMRR@10 = -0.1334`

## Significance Checks

Approximate randomization over per-query scores:

- `AP@10 p-value = 0.0000`
- `nDCG@10 p-value = 0.0000`

## Interpretation

Average Word2Vec is the cleanest way to turn word embeddings into a retrieval backbone. Its main strength is simplicity: once the embedding space is trained, both documents and queries can be embedded and compared very cheaply.

Its main weakness is that every item is compressed into a single centroid. Important rare words can be washed out by more frequent words, and the method has no direct representation of local order or phrase boundaries.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 1 | 0.0000 | 0.0500 | 0.0500 |
| 40 | 2 | 1 | 0.1538 | 0.1355 | -0.0183 |
| 64 | 1 | 0 | 0.2619 | 0.0000 | -0.2619 |
| 81 | 1 | 0 | 0.2869 | 0.0000 | -0.2869 |
| 90 | 2 | 1 | 0.1048 | 0.0857 | -0.0190 |
| 167 | 0 | 2 | 0.0333 | 0.5556 | 0.5222 |
| 184 | 1 | 4 | 0.0982 | 0.5792 | 0.4810 |
| 14 | 0 | 1 | 0.0556 | 0.4444 | 0.3889 |

## Research References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (NeurIPS 2013): https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html
- Arora, Liang, and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (ICLR 2017): https://openreview.net/forum?id=SyK00v5xx

## Output Files

- summary_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\summary.json`
- summary_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\summary_k10.csv`
- overlay_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\eval_overlay.png`
- all_configs_overlay: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\all_tuned_combinations_overlay.png`
- vector_size_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\vector_size_sweep_best.json`
- vector_size_sweep_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\vector_size_sweep_best.csv`
- vector_size_sweep_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\vector_size_sweep_best.png`
- example_markdown: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\example_query_comparison.md`
- config_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\word2vec_average\output\config_sweep.json`