# LSA Tuning Notes

## Best LSA dimensionality

The best standalone choice for latent dimensionality was:

- `lsa_components = 250`

Why `250`:

- it gave the best overall `MAP@10` among the component sweep
- it also gave the best `P@1`
- higher values like `300` or `400` only gave metric-specific gains, not a better overall trade-off

Component sweep summary:

- `100`: too much compression, weaker ranking
- `150`: clear improvement over `100`
- `200`: strong jump
- `250`: best overall balance
- `300`: slightly better `nDCG@10`, but slightly worse `MAP@10`
- `400`: slightly better `MRR@10`, but weaker overall than `250`

## Additional parameter tuning

After fixing `lsa_components = 250`, extra tuning on the TF-IDF stage improved LSA even more.

Best tuned vectorizer settings:

- `sublinear_tf = True`
- `max_df = 0.95`
- `min_df = 2`
- `norm = l2`
- `ngram_range = (1, 1)`

Reason:

- `sublinear_tf=True` gave the largest gain by reducing the effect of very high term frequencies
- `min_df=2` helped remove very rare noisy terms
- `max_df=0.95` slightly improved overall robustness by filtering extremely frequent terms

## Final tuned LSA result

Final tuned configuration:

- `lsa_components = 250`
- `sublinear_tf = True`
- `max_df = 0.95`
- `min_df = 2`

Final scores:

- `MAP@10 = 0.3665`
- `nDCG@10 = 0.5169`
- `MRR@10 = 0.7952`
- `P@1 = 0.7289`
- explained variance ratio sum `= 0.5534`

## Comparison to baseline TF-IDF

Compared to the original TF-IDF baseline:

- `MAP@10`: `0.3665` vs `0.3026`
- `nDCG@10`: `0.5169` vs `0.4589`
- `MRR@10`: `0.7952` vs `0.7368`
- `P@1`: `0.7289` vs `0.6489`

So the tuned LSA system is now clearly better than the baseline on the key ranking metrics.

## Query-level case studies where LSA beats TF-IDF

The following Cranfield queries are good examples where the normal TF-IDF system failed to retrieve good documents near the top, but tuned LSA retrieved clearly better results.

| QID | Query | TF-IDF result | LSA result | Relevant docs promoted by LSA |
| --- | --- | --- | --- | --- |
| `167` | exact solution methods for calculating the ablative mass loss of a material ablating at high temperatures in a hypersonic flight environment | `0/5` relevant in top 5, first relevant at rank `10`, `AP@10 = 0.033` | `2/5` relevant in top 5, first relevant at rank `1`, `AP@10 = 0.667` | `274`, `82` |
| `184` | work on small-oscillation re-entry motions | `1/5` relevant in top 5, first relevant at rank `2`, `AP@10 = 0.098` | `4/5` relevant in top 5, first relevant at rank `1`, `AP@10 = 0.658` | `716`, `639`, `67`, `715` |
| `81` | what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel | `1/5` relevant in top 5, first relevant at rank `5`, `AP@10 = 0.287` | `3/5` relevant in top 5, first relevant at rank `1`, `AP@10 = 0.756` | `799`, `631`, `672` |
| `118` | what are the aerodynamic interference effects on the fin lift and body lift of a fin-body combination | `0/5` relevant in top 5, first relevant at rank `8`, `AP@10 = 0.087` | `2/5` relevant in top 5, first relevant at rank `1`, `AP@10 = 0.500` | `923`, `924` |
| `12` | how can the aerodynamic performance of channel flow ground effect machines be calculated | `0/5` relevant in top 5, first relevant at rank `10`, `AP@10 = 0.017` | `2/5` relevant in top 5, first relevant at rank `2`, `AP@10 = 0.266` | `652`, `624` |
| `119` | what is the effect of initial axisymmetric deviations from circularity on the non linear load-deflection response of cylinders under hydrostatic pressure | `0/5` relevant in top 5, first relevant at rank `6`, `AP@10 = 0.194` | first relevant at rank `1`, `AP@10 = 0.500` | `897` |

Short qualitative observations:

- `Query 167`: TF-IDF retrieved only broadly related ablation papers, while LSA moved the actually relevant re-entry and heat-shield documents to the top.
- `Query 184`: TF-IDF found only one relevant document in the top 5, but LSA retrieved four relevant re-entry motion papers near the top.
- `Query 81`: TF-IDF mostly returned generic aerofoil or tunnel documents, while LSA surfaced the exact tunnel-interference correction papers.
- `Query 118`: TF-IDF missed the main fin-body lift interference papers, but LSA ranked them first and second.
- `Query 12`: TF-IDF overfocused on generic channel-flow documents, while LSA recovered ground-effect-related documents.
- `Query 119`: TF-IDF failed to prioritize the correct shell-buckling paper, while LSA placed the most relevant document at rank 1.

Overall pattern:

- LSA helps most when exact lexical overlap is weak.
- It is especially useful when the relevant concept is distributed across related technical terms.
- It is less dependent on exact wording than plain TF-IDF.

## Good report sentence

We first tuned the latent dimensionality and found `250` components to provide the best overall trade-off. We then tuned the TF-IDF stage used before SVD, and the final configuration with `sublinear_tf=True`, `max_df=0.95`, and `min_df=2` gave the best overall LSA performance, improving over the baseline TF-IDF system on `MAP@10`, `nDCG@10`, `MRR@10`, and `P@1`.

## Hybrid TF-IDF + LSA

We implemented a hybrid retriever that combines:

- cosine similarity in the TF-IDF space
- cosine similarity in the LSA latent space

using a weighted score fusion:

`hybrid_score = alpha * tfidf_score + (1 - alpha) * lsa_score`

## Hybrid tuning

We tuned the following parameters for the hybrid system:

- `tfidf_weight (alpha)`
- `lsa_components`
- `sublinear_tf`
- `max_df`
- `min_df`
- `ngram_range`

Best tuned hybrid configuration:

- `lsa_components = 240`
- `tfidf_weight = 0.2`
- `sublinear_tf = True`
- `max_df = 0.95`
- `min_df = 2`
- `ngram_range = (1, 1)`

Key tuning observations:

- A pure average was not best; the best hybrid gave slightly more weight to the LSA branch than to the TF-IDF branch.
- `sublinear_tf=True` remained very important for hybrid performance.
- `max_df=0.95` and `min_df=2` continued to help by filtering extremely common and very rare terms.
- Adding bigrams increased vocabulary size substantially but reduced overall retrieval quality in the hybrid setup.

## Final tuned hybrid result

Final scores for the tuned hybrid system:

- `MAP@10 = 0.3695`
- `nDCG@10 = 0.5220`
- `MRR@10 = 0.8053`
- `P@1 = 0.7422`
- explained variance ratio sum `= 0.5415`

## Comparison of hybrid with baseline TF-IDF and tuned LSA

Compared to the original TF-IDF baseline:

- `MAP@10`: `0.3695` vs `0.3026`
- `nDCG@10`: `0.5220` vs `0.4589`
- `MRR@10`: `0.8053` vs `0.7368`
- `P@1`: `0.7422` vs `0.6489`

Compared to the tuned standalone LSA system:

- `MAP@10`: `0.3695` vs `0.3665`
- `nDCG@10`: `0.5220` vs `0.5169`
- `MRR@10`: `0.8053` vs `0.7952`
- `P@1`: `0.7422` vs `0.7289`

So the hybrid is better than both the original TF-IDF baseline and the tuned standalone LSA system on the main ranking metrics.

## Query-level case studies where the hybrid helps when both standalone methods are weak

The following queries are useful examples where both standalone methods were weak, and the hybrid system gave a clearer improvement in at least one important retrieval behavior such as `AP@10`, first relevant rank, or the number of relevant documents in the top 5.

| QID | Query | TF-IDF result | LSA result | Hybrid result | Hybrid gain |
| --- | --- | --- | --- | --- | --- |
| `162` | what accurate or exact solutions of the laminar separation point for various incompressible and compressible boundary layers with zero heat transfer are available | `1/5` relevant in top 5, `AP@10 = 0.059` | `1/5` relevant in top 5, `AP@10 = 0.148` | `2/5` relevant in top 5, `AP@10 = 0.203` | promoted relevant doc `54` into top 5 |
| `80` | are methods of measuring aerodynamic derivatives available which could be adopted for use in short running time facilities | `1/5` relevant in top 5, `AP@10 = 0.200` | `1/5` relevant in top 5, `AP@10 = 0.267` | `2/5` relevant in top 5, `AP@10 = 0.280` | added relevant doc `597` to top 5 |
| `95` | what is the theoretical heat transfer distribution around a hemisphere | `1/5` relevant in top 5, `AP@10 = 0.333` | `0/5` relevant in top 5, `AP@10 = 0.122` | `2/5` relevant in top 5, `AP@10 = 0.300` | improved top-5 relevant coverage by recovering `662` and `635` together |
| `158` | what are the available properties of high-temperature air | `1/5` relevant in top 5, `AP@10 = 0.111` | `1/5` relevant in top 5, `AP@10 = 0.093` | `2/5` relevant in top 5, `AP@10 = 0.100` | improved top-5 relevant coverage by bringing `1011` into the top 5 |
| `113` | what data exists on oscillatory aerodynamic forces on control surfaces at transonic mach numbers | `1/5` relevant in top 5, `AP@10 = 0.090` | `1/5` relevant in top 5, `AP@10 = 0.100` | first relevant at rank `1`, `AP@10 = 0.267` | moved relevant doc `748` to rank 1 |

Short qualitative observations:

- `Query 162`: both standalone systems focused on nearby but not exact laminar boundary layer papers; the hybrid recovered a directly relevant zero-heat-transfer document.
- `Query 80`: both standalone methods found only one clearly relevant result, while the hybrid added another derivatives-measurement paper to the top 5.
- `Query 95`: TF-IDF captured one relevant hemisphere paper and LSA drifted toward generic heat-transfer topics, while the hybrid recovered two relevant hemisphere-related documents together in the top 5.
- `Query 158`: both standalone systems returned only one clearly relevant document in the top 5, while the hybrid pulled in another relevant high-temperature-air document.
- `Query 113`: both standalone methods were weak, but the hybrid placed the most relevant control-surface flutter document at rank 1.

Overall pattern for the hybrid:

- TF-IDF contributes exact lexical precision.
- LSA contributes broader semantic similarity.
- The hybrid helps most when one method finds part of the signal and the other method supplies the missing complementary evidence.
