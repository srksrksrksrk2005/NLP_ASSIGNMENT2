# Word2Vec Soft Cosine Backbone Experiment Report

## Method Summary

The baseline system uses unigram TF-IDF with cosine similarity. The proposed method keeps the TF-IDF representation, but replaces exact lexical matching with soft matching through a sparse term similarity matrix built from Word2Vec neighbors.

This follows the soft cosine idea: two different words can contribute partial similarity if their embeddings place them near each other, so the method behaves like a lexical-semantic bridge rather than a pure dense retriever.

## Hypothesis

Soft cosine should preserve the precision advantages of TF-IDF while relaxing the requirement that matching terms be literally identical.

Because the document representation remains sparse and explicit, the method should be more interpretable than dense centroid retrieval and more semantically tolerant than ordinary cosine over TF-IDF.

## Baseline Limitations Addressed

- The baseline only rewards exact term overlap.
- Word2Vec neighbors can inject graded similarity between related technical terms.
- Soft cosine still depends on the quality and coverage of the embedding space used to define term similarity.

## Best Configuration

- `vector_size = 100`
- `window = 5`
- `training_objective = skipgram`
- `epochs = 20`
- `top_k_neighbors = 10`
- `min_similarity = 0.10`
- `similarity_power = 2.00`
- `representation_coverage = 100.00%`
- `avg_neighbors_per_term = 10.00`

## Neighbor Budget Sweep

| Top-k neighbors | Similarity floor | Similarity power | Vector size | Coverage | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.10 | 2.00 | 100 | 100.00% | 0.2787 | 0.4038 | 0.3055 | 0.2910 | 0.4394 | 0.6943 |
| 10 | 0.10 | 2.00 | 100 | 100.00% | 0.2756 | 0.3990 | 0.3021 | 0.2917 | 0.4372 | 0.7018 |
| 20 | 0.10 | 2.00 | 100 | 100.00% | 0.2733 | 0.3944 | 0.2992 | 0.2869 | 0.4307 | 0.6974 |

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.08 |
| soft_cosine_word2vec | 0.2756 | 0.3990 | 0.3021 | 0.2917 | 0.4372 | 0.7018 | 211.45 |

## Delta vs Baseline at k=10

- `dP@10 = -0.0076`
- `dR@10 = -0.0104`
- `dF@10 = -0.0078`
- `dMAP@10 = -0.0143`
- `dnDCG@10 = -0.0239`
- `dMRR@10 = -0.0443`

## Significance Checks

Approximate randomization over per-query scores:

- `AP@10 p-value = 0.0618`
- `nDCG@10 p-value = 0.0029`

## Interpretation

This method is a strong compromise between lexical and semantic retrieval. It keeps explicit TF-IDF term weights, which helps precision, but it no longer treats all non-identical terms as completely unrelated.

Its main tuning burden is the similarity matrix itself: if too many neighbors are kept, the model drifts; if too few are kept, the method collapses back toward ordinary TF-IDF.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 1 | 0.0000 | 0.2500 | 0.2500 |
| 40 | 2 | 1 | 0.1538 | 0.0256 | -0.1282 |
| 64 | 1 | 1 | 0.2619 | 0.2619 | 0.0000 |
| 81 | 1 | 0 | 0.2869 | 0.1310 | -0.1560 |
| 90 | 2 | 3 | 0.1048 | 0.1024 | -0.0024 |
| 208 | 3 | 5 | 0.4881 | 0.9821 | 0.4940 |
| 78 | 2 | 4 | 0.5750 | 1.0000 | 0.4250 |
| 129 | 4 | 5 | 0.5625 | 0.9861 | 0.4236 |

## Research References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (NeurIPS 2013): https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html
- Sidorov et al., Soft Similarity and Soft Cosine Measure (2014): https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462014000300007

## Output Files

- summary_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\summary.json`
- summary_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\summary_k10.csv`
- overlay_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\eval_overlay.png`
- all_configs_overlay: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\all_tuned_combinations_overlay.png`
- neighbor_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\neighbor_sweep_best.json`
- neighbor_sweep_csv: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\neighbor_sweep_best.csv`
- neighbor_sweep_plot: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\neighbor_sweep_best.png`
- example_markdown: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\example_query_comparison.md`
- config_sweep_json: `C:\Users\sudhi\OneDrive\Desktop\CS6370-NLP\project\NLP_ASSIGNMENT2\project\sudheer\soft_cosine_word2vec\output\config_sweep.json`