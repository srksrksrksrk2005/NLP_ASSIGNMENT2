# Smooth Inverse Frequency Embeddings Experiment Report

## Method Summary

The baseline system uses unigram TF-IDF with cosine similarity. The proposed SIF backbone replaces sparse lexical vectors with one dense vector per query/document.

Each token vector is downweighted by the smooth inverse frequency term `a / (a + p(w))`, where `p(w)` is estimated from the Cranfield document collection. The resulting dense vectors are optionally purified by subtracting the top principal components learned from document embeddings.

## Hypothesis

Plain embedding averages often let frequent words dominate the final centroid. SIF should help by suppressing common terms while keeping rare technical vocabulary more visible.

Principal-component removal should further improve retrieval by subtracting broad discourse directions that are shared across many documents but are not specific to the current query intent.

## Baseline Limitations Addressed

- The TF-IDF baseline depends heavily on exact lexical overlap.
- SIF tries to address vocabulary mismatch without falling back to direct token matching.
- Because the final representation is still a single dense vector, rare multi-term technical intent can still be blurred.

## Best Configuration

- `smoothing_a = 0.01`
- `removed_principal_components = 2`
- `vector_size = 50`
- `embedding_source = glove-wiki-gigaword-50`
- `model_coverage = 45.65%`
- `avg_in_vocab_tokens_per_doc = 60.60`
- `avg_in_vocab_tokens_per_query = 6.34`

## SIF Configuration Sweep

| Smoothing a | Removed PCs | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-02 | 0 | 0.0809 | 0.1179 | 0.0891 | 0.0651 | 0.1346 | 0.2718 |
| 1e-02 | 1 | 0.0831 | 0.1198 | 0.0915 | 0.0700 | 0.1397 | 0.2866 |
| 1e-02 | 2 | 0.0871 | 0.1285 | 0.0960 | 0.0718 | 0.1447 | 0.2881 |
| 1e-03 | 0 | 0.0702 | 0.1009 | 0.0771 | 0.0597 | 0.1201 | 0.2520 |
| 1e-03 | 1 | 0.0729 | 0.1067 | 0.0806 | 0.0640 | 0.1242 | 0.2563 |
| 1e-03 | 2 | 0.0724 | 0.1049 | 0.0798 | 0.0658 | 0.1263 | 0.2664 |
| 1e-04 | 0 | 0.0489 | 0.0750 | 0.0557 | 0.0470 | 0.0860 | 0.1702 |
| 1e-04 | 1 | 0.0542 | 0.0862 | 0.0622 | 0.0525 | 0.0963 | 0.1874 |
| 1e-04 | 2 | 0.0493 | 0.0784 | 0.0564 | 0.0501 | 0.0906 | 0.1833 |

## k=10 Results

| Method | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_tfidf | 0.2831 | 0.4094 | 0.3099 | 0.3061 | 0.4611 | 0.7461 | 0.05 |
| sif_embeddings | 0.0871 | 0.1285 | 0.0960 | 0.0718 | 0.1447 | 0.2881 | 10.54 |

## Delta vs Baseline at k=10

- `dP@10 = -0.1960`
- `dR@10 = -0.2808`
- `dF@10 = -0.2138`
- `dMAP@10 = -0.2342`
- `dnDCG@10 = -0.3164`
- `dMRR@10 = -0.4580`

## Significance Checks

Approximate randomization over per-query scores:

- `AP@10 p-value = 0.0000`
- `nDCG@10 p-value = 0.0000`

## Interpretation

SIF is a stronger dense baseline than plain average embeddings because it addresses two known failure modes: dominance by frequent tokens and over-shared corpus directions.

Even with those corrections, the method still compresses each item into one centroid. That compression is especially costly on Cranfield, where exact rare terminology often carries most of the relevance signal.

## Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 0 | 0.1538 | 0.0000 | -0.1538 |
| 64 | 1 | 0 | 0.2619 | 0.0000 | -0.2619 |
| 81 | 1 | 1 | 0.2869 | 0.0833 | -0.2036 |
| 90 | 2 | 1 | 0.1048 | 0.0357 | -0.0690 |
| 131 | 1 | 3 | 0.0873 | 0.4015 | 0.3142 |
| 135 | 3 | 4 | 0.4586 | 0.6204 | 0.1618 |
| 75 | 0 | 2 | 0.0000 | 0.1389 | 0.1389 |

## Research References

- Arora, Liang, and Ma, A Simple but Tough-to-Beat Baseline for Sentence Embeddings (ICLR 2017): https://openreview.net/forum?id=SyK00v5xx
- Pennington, Socher, and Manning, GloVe: Global Vectors for Word Representation (EMNLP 2014): https://aclanthology.org/D14-1162/

## Output Files

- summary_json: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/summary.json`
- summary_csv: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/summary_k10.csv`
- overlay_plot: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/eval_overlay.png`
- all_configs_overlay: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/all_tuned_combinations_overlay.png`
- a_sweep_json: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/a_sweep_best.json`
- a_sweep_csv: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/a_sweep_best.csv`
- a_sweep_plot: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/a_sweep_best.png`
- config_sweep_json: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/config_sweep.json`
- config_sweep_csv: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/config_sweep_k10.csv`
- example_markdown: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output/example_query_comparison.md`