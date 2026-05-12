# Soft Cosine Source Comparison Report

This comparison reuses the same Cranfield preprocessing and tunes four soft-cosine sources: TF-IDF, LSA, ESA, and WordNet.

## Best Results

| Source | Primary Param | Best Value | MAP@10 | nDCG@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF soft cosine | top_k_neighbors | 5 | 0.3031 | 0.4500 | 0.7113 |
| LSA soft cosine | n_components | 150 | 0.3025 | 0.4446 | 0.6849 |
| ESA soft cosine | top_concepts | 50 | 0.3075 | 0.4549 | 0.7167 |
| WordNet soft cosine | top_k_neighbors | 5 | 0.2616 | 0.4138 | 0.6807 |

Best source overall: ESA soft cosine with MAP@10=0.3075

## Notes
- Each source has its own tuned sweep plot under its output folder.
- WordNet uses unigram vocabulary because its synonym graph is lexical rather than phrasal.
- TF-IDF, LSA, and ESA use the same pretokenized Cranfield texts with their own tuned vectorizer settings.
