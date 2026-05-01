# Limitations Evaluation — All Methods

This folder contains the unified evaluation of **all IR methods** implemented
across the team against the 7 limitations of the baseline TF-IDF system,
using the curated limitation test set (`limitation_test_set_expanded.json`).

---

## Methods Evaluated

| Method | Folder | Approach |
|--------|-------|----------|
| `TF-IDF` | Baseline | Standard unigram TF-IDF with cosine similarity |
| `N-gram` | Ritisha | Unigram + bigram TF-IDF |
| `WordNet-LC` | Ritisha | WordNet synonym expansion with Lesk disambiguation |
| `BM25` | Ritisha | Probabilistic ranking (no library) |
| `LSA` | Ramakrishna | Latent Semantic Analysis (128 components) |
| `ESA` | Ramakrishna | Explicit Semantic Analysis (top 50 concepts) |
| `TF-IDF` | Nikhil | TF-IDF with preprocessing pipeline |
| `WordNet-Exp` | Nikhil | Query expansion via WordNet similarity map |
| `LSA-Exp` | Nikhil | Query expansion via LSA embedding neighbors |
| `ESA-Exp` | Nikhil | Query expansion via ESA embedding neighbors |
| `LocalBOW` | Chandan | Unordered local-context bag-of-words |
| `QueryRed` | Chandan | Query reduction (IDF top-k / PRF pruning) |

---

## Limitations Tested

| Abbreviation | Full Name |
|---|---|
| Semantic | Lack of semantic understanding |
| Dim/Cost | High dimensionality and computational cost |
| Scale | Poor scalability |
| Ambiguity | Word sense ambiguity |
| OOV | Out-of-vocabulary terms |
| Context | Lack of contextual representation |
| Sparse | Sparse representations |

---

## Files

| File | Description |
|------|-------------|
| `evaluate_all_limitations.py` | Main evaluation script |
| `output_limitations_all/all_limitations_table.txt` | MAP@10 table with ✓/✗ per method per limitation |
| `output_limitations_all/all_limitations_results.json` | Raw MAP@10 scores |
| `output_limitations_all/heatmap_all_methods.png` | Colour heatmap — methods × limitations |
| `output_limitations_all/bar_per_limitation.png` | Bar charts per limitation |

---

## How to Run

```bash
python project/ritisha/limitations_eval/evaluate_all_limitations.py \
    --dataset cranfield/ \
    --test_set project/limitation_test_set_expanded.json \
    --out_folder project/ritisha/limitations_eval/output_limitations_all/
```

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cranfield/` | Path to Cranfield dataset folder |
| `--test_set` | `project/limitation_test_set_expanded.json` | Path to limitation test set |
| `--out_folder` | `project/ritisha/limitations_eval/output_limitations_all/` | Output directory |

---

## Architecture Notes

Each member used a different preprocessing pipeline:

- **Ritisha + Ramakrishna** — use the base pipeline (`SentenceSegmentation` →
  `Tokenization` → `InflectionReduction` → `StopwordRemoval`)
- **Nikhil** — uses his own `Preprocessor` (regex tokenisation + WordNet
  lemmatisation). His systems are evaluated with his pipeline so results
  are faithful to his implementation.
- **Chandan** — runs methods by re-implementing his SparseTfidfModel and
  context feature logic directly.

---

## Expected Runtime

| Phase | Approximate time |
|-------|-----------------|
| Base systems (TF-IDF, N-gram, BM25) | ~2 min |
| WordNet-LC | ~10 min |
| LSA + ESA| ~3 min |
| WordNet + LSA + ESA expansion | ~15 min |
| Evaluation across all limitations | ~5 min |
| **Total** | **~35 min** |

---

## Dependencies

All dependencies are shared with the base assignment plus each team member's
own requirements. Nikhil's Word2Vec expansion is **not included** by default
(it requires `gensim`). Install if needed:

```bash
pip install gensim
```
