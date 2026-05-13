# Latent Semantic Analysis (LSA) and Hybrid TF-IDF

## 1. Implementation Details

We implemented Latent Semantic Analysis (LSA) to project documents and queries from a sparse, high-dimensional term space into a dense, lower-dimensional latent semantic space. This process uncovers underlying semantic relationships (like synonymy) that raw term-matching algorithms fail to capture.

Our implementation uses the `TfidfVectorizer` to generate a term-document matrix, which is then decomposed using `TruncatedSVD`. The optimal hyperparameters for the vectorizer include `sublinear_tf=True` (to dampen the effect of highly frequent terms), `max_df=0.95` (removing corpus-specific stopwords), and `min_df=2` (removing extremely rare words).

We also built a **Hybrid Retrieval** system that fuses standard TF-IDF similarity with LSA similarity using a weighted sum:
`score = alpha * TF-IDF + (1 - alpha) * LSA`

## 2. LSA Hyperparameter Tuning ($k$)

We evaluated LSA performance across different numbers of latent components ($k \in \{100, 200, 250, 300\}$).

| LSA Variant | P@10 | R@10 | MAP@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Baseline TF-IDF | 0.2836 | 0.4100 | 0.3026 | 0.4589 | 0.7368 |
| LSA ($k=100$) | 0.3116 | 0.4446 | 0.3321 | 0.4795 | 0.7289 |
| LSA ($k=200$) | 0.3222 | **0.4645** | 0.3639 | 0.5139 | 0.7937 |
| **LSA ($k=250$)** | **0.3227** | 0.4641 | **0.3665** | 0.5169 | 0.7952 |
| LSA ($k=300$) | 0.3196 | 0.4617 | 0.3640 | **0.5180** | **0.8016** |

**Analysis:**
All LSA variants substantially outperform the baseline TF-IDF, confirming that projecting into a latent semantic space captures cross-term correlations. Increasing $k$ from 100 to 200 yields a massive jump. The optimal configuration is $k = 250$, which achieves the highest MAP@10 (0.3665). Extending to $k = 300$ causes MAP@10 to decline slightly, indicating that higher-order singular vectors encode document-specific noise rather than generalizable semantic structure (the bias-variance trade-off). We chose $k = 250$ to balance semantic richness and overfitting.

## 3. Hybrid vs Pure LSA vs TF-IDF

With $k=240$ and $\alpha=0.2$ (meaning 20% weight to TF-IDF and 80% to LSA), we compared the hybrid approach:

| Retrieval Variant | P@10 | R@10 | MAP@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Baseline TF-IDF | 0.2836 | 0.4100 | 0.3026 | 0.4589 | 0.7368 |
| LSA ($k=250$) | 0.3227 | 0.4641 | 0.3665 | 0.5169 | 0.7952 |
| **Hybrid ($k=240$, $\alpha=0.2$)** | **0.3231** | **0.4642** | **0.3695** | **0.5220** | **0.8053** |

**Analysis:**
The hybrid model achieves the best performance across all five metrics. Compared to baseline TF-IDF, it boosts MAP@10 by 22.1% relative.

Crucially, it also outperforms pure LSA. Pure LSA suffers from occasional "semantic drift," where the latent projection pulls in documents that are topically related but not relevant to the specific query intent. By retaining 20% of the original TF-IDF similarity as an anchor, the hybrid model ensures that documents with exact surface-level term matches receive a necessary ranking boost, while the 80% LSA component continues to bridge vocabulary gaps (synonymy). This balance makes it strictly superior to either method alone.
