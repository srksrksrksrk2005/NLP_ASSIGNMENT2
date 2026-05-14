# SIF Transferability Report

## What this experiment does

Instead of using only pre-trained word vectors, this experiment adapts `TF-IDF`, `LSA`, and `ESA` into token-level vector spaces and then runs the same SIF pipeline on top of those token vectors.

## Best backbone per requested family

| Backbone | Coverage (docs) | Coverage (queries) | Best a | Removed PCs | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSA term space | 100.00% | 97.75% | 1e-04 | 1 | 0.3311 | 0.4785 | 0.3629 | 0.3748 | 0.5160 | 0.7469 |
| TF-IDF term space | 100.00% | 97.75% | 1e-03 | 2 | 0.3133 | 0.4513 | 0.3429 | 0.3528 | 0.5034 | 0.7769 |
| ESA term space | 100.00% | 97.75% | 1e-03 | 1 | 0.3102 | 0.4442 | 0.3388 | 0.3516 | 0.5028 | 0.7864 |

## Baseline reference

- `baseline_tfidf MAP@10 = 0.3061`
- `best transferred SIF backbone = LSA term space`
- `best transferred SIF MAP@10 = 0.3748`

## Notes

- TF-IDF term space uses each term's document-distribution profile as its token vector.
- LSA term space uses each term's latent SVD embedding as its token vector.
- ESA term space uses each term's pruned explicit concept activation vector as its token vector.
- SIF itself is unchanged: weighted averaging plus optional principal-component removal.

## Example Query Comparison For Best Backbone

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 4 | 0.1538 | 0.3350 | 0.1812 |
| 64 | 1 | 3 | 0.2619 | 0.7556 | 0.4937 |
| 81 | 1 | 3 | 0.2869 | 0.5889 | 0.3020 |
| 90 | 2 | 3 | 0.1048 | 0.2095 | 0.1048 |
| 35 | 0 | 3 | 0.0000 | 0.6875 | 0.6875 |
| 5 | 1 | 3 | 0.1067 | 0.6643 | 0.5576 |
| 119 | 0 | 2 | 0.1944 | 0.7500 | 0.5556 |

## Output Files

- `summary.json`: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output_transferability/summary.json`
- `summary_k10.csv`: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output_transferability/summary_k10.csv`
- `eval_overlay.png`: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output_transferability/eval_overlay.png`
- `k10_backbone_bars.png`: `/home/chandanmannepula/NLP PROJECT/NLP_ASSIGNMENT2/project/chandan/sif_embeddings/output_transferability/k10_backbone_bars.png`
