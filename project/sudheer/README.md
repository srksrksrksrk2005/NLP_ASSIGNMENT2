# Sudheer Word2Vec Experiments

This folder contains three separate retrieval experiments built in the same style as `project/chandan/unordered_local_context_bow`.

## Subfolders

1. `word2vec_average`
   - Plain centroid retrieval using the mean of Word2Vec token vectors.

2. `word2vec_idf_weighted`
   - TF-IDF/IDF-weighted centroid retrieval using Word2Vec token vectors.

3. `soft_cosine_word2vec`
   - TF-IDF retrieval scored with soft cosine, where term similarity comes from Word2Vec neighbors.

4. `pretrained_word2vec_comparison`
   - Compares the scratch-trained Word2Vec backbones against pretrained and fine-tuned embedding sources.

5. `soft_cosine_sources`
   - Compares soft cosine retrieval using TF-IDF, LSA, ESA, and WordNet similarity sources.

## Research grounding

- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013)
- Arora, Liang, and Ma, *A Simple but Tough-to-Beat Baseline for Sentence Embeddings* (ICLR 2017)
- Sidorov et al., *Soft Similarity and Soft Cosine Measure* (2014)

## Expected outputs per method

Each method folder is designed to generate:

- `output/baseline_tfidf/*`
- `output/<method_name>/*`
- `output/summary.json`
- `output/summary_k10.csv`
- `output/comparison_summary.json`
- `output/config_sweep.json`
- `output/eval_overlay.png`
- `output/all_tuned_combinations_overlay.png`
- `output/example_query_comparison.json`
- `output/example_query_comparison.md`
- one method-specific sweep JSON/CSV/PNG
- `experiment_report.md`

The soft cosine source-comparison folder also produces a separate output subfolder for each source: `tfidf`, `lsa`, `esa`, and `wordnet`.

## Run order

1. Run `word2vec_average/run_experiments.py`
2. Run `word2vec_idf_weighted/run_experiments.py`
3. Run `soft_cosine_word2vec/run_experiments.py`
4. Run `pretrained_word2vec_comparison/run_experiments.py`

This progression mirrors the project story:

1. start with the simplest Word2Vec backbone,
2. add IDF weighting,
3. then test the Word2Vec-informed soft lexical matcher.

The first three experiments train Word2Vec from scratch on the Cranfield corpus.
The pretrained comparison reuses those scratch results and evaluates an external embedding source (`glove-wiki-gigaword-100`) plus a fine-tuned variant on the same downstream backbones.
