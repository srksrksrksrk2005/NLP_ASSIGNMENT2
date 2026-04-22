# ESA Analysis Notes

## ESA Setup
We used a Cranfield document-concept ESA variant: each document is treated as an explicit concept, queries are projected into that concept space through TF-IDF similarity, and the top concepts are retained before ranking.

## Best ESA Configuration
- `esa_top_concepts = 25`
- `esa_min_similarity = 0.0`
- `sublinear_tf = True`
- `max_df = 0.9`
- `min_df = 1`
- `ngram_range = (1, 2)`

## Comparison at k=10

| Method | Precision | Recall | F-score | MAP | nDCG | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Standard TF-IDF | 0.2836 | 0.4100 | 0.3103 | 0.3026 | 0.4589 | 0.7368 |
| Tuned LSA | 0.3284 | 0.4681 | 0.3582 | 0.3703 | 0.5210 | 0.7804 |
| Tuned Hybrid (TF-IDF + LSA) | 0.3240 | 0.4669 | 0.3551 | 0.3691 | 0.5221 | 0.7896 |
| Tuned ESA | 0.3133 | 0.4456 | 0.3413 | 0.3431 | 0.4980 | 0.7654 |

## Query-level case studies where ESA beats TF-IDF and LSA

Cell format: `AP@10 / first relevant rank / relevant docs in top 5`. The same query is shown across all methods.

| QID | Query | TF-IDF | LSA | Hybrid | ESA | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 34 | have wind tunnel interference effects been investigated on a systematic basis . | 0.356 / 3 / 3 | 0.291 / 2 / 2 | 0.395 / 1 / 3 | 0.587 / 1 / 4 | ESA surfaces the strongest semantic match |
| 75 | do the discrepancies among current analyses of the vorticity effect on stagnation-point heat transfer result primarily from the differences in the viscosity-temperature law assumed . | 0.000 / 12 / 0 | 0.000 / 14 / 0 | 0.000 / 17 / 0 | 0.214 / 1 / 1 | ESA surfaces the strongest semantic match |
| 155 | technical report on measurement of ablation during flight . | 0.276 / 2 / 2 | 0.339 / 1 / 2 | 0.286 / 1 / 2 | 0.500 / 1 / 3 | ESA surfaces the strongest semantic match |
| 106 | experimental techniques in shell vibration . | 0.305 / 1 / 2 | 0.406 / 2 / 3 | 0.394 / 2 / 3 | 0.554 / 1 / 3 | ESA surfaces the strongest semantic match |
| 149 | has anyone developed an analysis which accurately establishes the large deflection behaviour of conical shells . | 0.175 / 1 / 2 | 0.338 / 1 / 3 | 0.340 / 1 / 3 | 0.484 / 1 / 3 | ESA surfaces the strongest semantic match |

## Queries where hybrid still beats ESA

| QID | Query | Hybrid | ESA | Comment |
| --- | --- | --- | --- | --- |
| 118 | what are the aerodynamic interference effects on the fin lift and body lift of a fin-body combination . | 0.500 / 1 / 2 | 0.121 / 5 / 1 | Hybrid keeps the best balance of lexical and latent evidence |
| 184 | work on small-oscillation re-entry motions . | 0.491 / 1 / 4 | 0.153 / 1 / 1 | Hybrid keeps the best balance of lexical and latent evidence |
| 102 | basic dynamic characteristics of structures continuous over many spans . | 0.600 / 1 / 3 | 0.286 / 2 / 2 | Hybrid keeps the best balance of lexical and latent evidence |
| 180 | how does scale height vary with altitude in an atmosphere . | 0.975 / 1 / 5 | 0.673 / 1 / 4 | Hybrid keeps the best balance of lexical and latent evidence |

## Interpretation

ESA is strongest when the relevant documents are conceptually close but do not share the same surface wording.
The hybrid model still wins when direct lexical evidence matters enough that the TF-IDF branch can correct ESA drift.

## Files written
- `C:\Users\srksr\NLP\NLP_project\NLP_ASSIGNMENT2\project\ramakrishna\esa\output_esa\esa_vs_tfidf_overlay_all_metrics.png`
- `C:\Users\srksr\NLP\NLP_project\NLP_ASSIGNMENT2\project\ramakrishna\esa\output_esa\esa_vs_all_methods_overlay_all_metrics.png`
- `C:\Users\srksr\NLP\NLP_project\NLP_ASSIGNMENT2\project\ramakrishna\esa\output_esa\esa_summary.json`
- `C:\Users\srksr\NLP\NLP_project\NLP_ASSIGNMENT2\project\ramakrishna\esa\output_esa\comparison_summary.json`