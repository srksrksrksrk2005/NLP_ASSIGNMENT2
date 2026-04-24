# WordNet Local-Context IR System

## Approach

The base VSM fails when a query and a relevant document use different but
synonymous words — the **vocabulary mismatch** problem. For example, a query
asking about "velocity" may miss documents that only say "speed".

This module addresses that by expanding every token with synonyms from
**WordNet**, disambiguated using the surrounding **local context** (a
simplified Lesk algorithm).

### Disambiguation — why it matters

Naive WordNet expansion (add ALL synonyms of ALL senses) introduces noise.
For example, "star" has senses for celebrity, celestial body, and shape — 
adding all of them pollutes the index. Instead:

1. For each word `w` in a sentence, gather all its WordNet synsets.
2. For each synset, count how many words in its **definition + examples**
   also appear in the surrounding sentence window.
3. Keep only the synset with the **highest overlap** (most contextually
   appropriate sense).
4. Add up to `max_synonyms` lemmas from that synset.

This is performed **identically at query time**, so both sides of the
cosine similarity share the same vocabulary enrichment.

## Files

| File | Purpose |
|------|---------|
| `wordnet_retrieval.py` | `WordNetRetrieval` class |
| `run_wordnet.py` | End-to-end evaluation script |
| `output_wordnet/` | Generated plots and result files |

## Running

```bash
# from the repo root
python project/ritisha/wordnet/run_wordnet.py \
    -dataset cranfield/ \
    -out_folder project/ritisha/wordnet/output_wordnet/
```

> **Note:** Indexing ~1400 Cranfield documents with WordNet expansion takes
> roughly 5–10 minutes on a standard laptop. This is a one-time cost.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_synonyms` | 3 | More synonyms → higher recall, potentially lower precision |
| `context_window` | 10 | Larger window → more context for disambiguation |

## Dependencies

`nltk` (wordnet, averaged_perceptron_tagger, punkt), `numpy`, `matplotlib`

All are downloaded automatically on first run via `nltk.download(quiet=True)`.

## Expected improvement

WordNet expansion improves recall for queries with vocabulary mismatch.
nDCG and MAP are expected to improve over the unigram VSM baseline,
especially for shorter, single-concept queries.
