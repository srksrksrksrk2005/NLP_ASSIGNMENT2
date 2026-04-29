# Merging Pipeline - Implementation Summary

## Project Structure

```
merging/
├── Main Scripts
│   ├── main.py                          # Main orchestration (links all blocks)
│   ├── block1_query_processing.py       # Block 1: Query expansion & reduction
│   ├── blocks23_retrieval_ranking.py    # Blocks 2 & 3: Retrieval & ranking
│   ├── run_experiments.py               # Experiment runner with grid search
│   ├── validate.py                      # Validation/testing script
│   └── quickstart.bat                   # Quick-start menu (Windows)
│
├── Utilities (utils/)
│   ├── __init__.py
│   ├── logger.py                        # WandB-compatible logging
│   ├── data_loader.py                   # Dataset loading and management
│   └── preprocessing.py                 # Text preprocessing utilities
│
├── Configuration (configs/)
│   ├── default_config.json              # Default configuration template
│   ├── lsa_lsa_config.json              # LSA expansion + LSA ranking preset
│   └── wordnet_esa_config.json          # WordNet expansion + ESA ranking preset
│
├── Output (output/)
│   └── [Generated during runs]
│       ├── results.json                 # Complete results with metrics
│       ├── metrics_plot.png             # Metrics visualization
│       ├── config.json                  # Saved config for reproducibility
│       ├── pipeline_*.log               # Execution logs
│       └── grid_search_summary_*.json   # Grid search results
│
├── Dependencies & Docs
│   ├── README.md                        # Full documentation
│   ├── requirements.txt                 # Python package requirements
│   └── IMPLEMENTATION_SUMMARY.md        # This file
```

## What Was Implemented

### Block 1: Query Processing (block1_query_processing.py)

**Functionality:**
- Query expansion modes: none, LSA, ESA, WordNet, Word2Vec, TF-IDF
- Query reduction: enabled/disabled with configurable parameters
- IDF-weighted query vectorization
- Batch processing support
- Outputs: numpy arrays (IDF-weighted vectors) or token lists

**Key Classes:**
- `QueryProcessor`: Main class handling all query processing operations

**Features:**
- Lazy loading of models (LSA, TF-IDF, etc.)
- Flexible preprocessing pipeline
- Configurable keyword extraction for reduction
- Supports both single queries and batch processing

**Usage:**
```python
processor = QueryProcessor(config)
query_vector = processor.process_query("information retrieval")
batch_vectors = processor.process_batch(queries, docs_tokens)
```

### Blocks 2 & 3: Retrieval & Ranking (blocks23_retrieval_ranking.py)

**Functionality:**
- Block 2 Retrieval types: TF-IDF, N-gram, Local Bag-of-Words
- Block 3 Ranking types: TF-IDF, LSA, ESA
- Integrated single pipeline for efficiency
- Document indexing and ranking with scores

**Key Classes:**
- `RetrievalRankingPipeline`: Combined retrieval and ranking

**Features:**
- Modular design: easily switch between retrieval and ranking methods
- Sparse and dense matrix handling
- Concept-based (ESA) and latent (LSA) ranking methods
- Returns ranked documents with similarity scores

**Usage:**
```python
pipeline = RetrievalRankingPipeline(config)
pipeline.build_retrieval_index(docs, doc_ids)
rankings = pipeline.rank(query_vectors)
```

### Main Orchestration Script (main.py)

**Functionality:**
- Ties all three blocks together
- Handles data loading and preprocessing
- Performs IR evaluation (Precision, Recall, MAP, NDCG, MRR)
- Generates visualizations
- Integrates with WandB for experiment tracking

**Key Features:**
- Complete end-to-end pipeline
- Automatic metric calculation
- Plot generation with matplotlib
- JSON output for reproducibility
- Command-line interface for easy parameter tuning

**Evaluation Metrics:**
- Precision@10, Recall@10, F-score@10
- MAP@10 (Mean Average Precision)
- NDCG@10 (Normalized Discounted Cumulative Gain)
- MRR@10 (Mean Reciprocal Rank)

### Utilities Module (utils/)

**Logger (logger.py):**
- PipelineLogger class with WandB integration
- Structured logging to file and console
- Metrics tracking and artifact logging
- Compatible with WandB dashboards

**Data Loader (data_loader.py):**
- DataLoader class for managing datasets
- JSON-based data loading
- Results saving/loading functionality
- Handles Cranfield dataset format

**Preprocessing (preprocessing.py):**
- TextPreprocessor for standardized text processing
- Supports: tokenization, lowercasing, stopword removal, stemming, lemmatization
- Batch processing capabilities
- NLTK-based implementation

### Experiment Runner (run_experiments.py)

**Functionality:**
- Single experiment runner
- Grid search over multiple combinations
- Systematic testing of all block combinations
- Results aggregation and summary generation

**Features:**
- Easy parameter specification
- Automatic execution of multiple configurations
- Success/failure tracking
- Summary statistics with success rates
- JSON export of results

### Validation Script (validate.py)

**Tests:**
1. Import testing (all required packages)
2. Utility modules loading
3. Block 1 functionality
4. Blocks 2 & 3 functionality
5. Configuration validation
6. Data loading (if available)

**Purpose:** Verify the entire pipeline is correctly installed and functional

## Configuration System

### Config Structure

Each config file contains:

```json
{
  "block1_query_processing": {
    "expansion_mode": "none|lsa|esa|wordnet|word2vec|tfidf",
    "reduction_enabled": true|false,
    "expansion_params": { /* mode-specific params */ }
  },
  "block2_retrieval_mode": {
    "retrieval_type": "tfidf|ngram|local_bow",
    "retrieval_params": { /* retrieval-specific params */ }
  },
  "block3_ranking_mode": {
    "ranking_type": "tfidf|lsa|esa",
    "ranking_params": { /* ranking-specific params */ }
  },
  "dataset": { /* dataset paths */ },
  "output": { /* output directory and options */ },
  "logging": { /* logging and WandB settings */ }
}
```

### Pre-configured Presets

1. **default_config.json**: Basic configuration template
2. **lsa_lsa_config.json**: LSA expansion + LSA ranking
3. **wordnet_esa_config.json**: WordNet expansion + ESA ranking

## WandB Integration

### Features
- Automatic configuration logging
- Metrics tracking and visualization
- Artifact saving (results, plots)
- Experiment comparison dashboard

### Enabling WandB
```bash
# Command line
python main.py --wandb

# Or in config:
# "logging": { "use_wandb": true, "wandb_project": "nlp-merging" }
```

## Usage Examples

### Quick Start
```bash
# Using the quick-start menu (Windows)
quickstart.bat

# Or directly with Python
python main.py
```

### Command Line Examples
```bash
# Baseline (TF-IDF for all blocks)
python main.py --block1-mode none --block2-retrieval tfidf --block3-ranking tfidf

# LSA-based pipeline
python main.py --block1-mode lsa --block3-ranking lsa

# WordNet expansion + ESA ranking
python main.py --block1-mode wordnet --block3-ranking esa

# With query reduction
python main.py --block1-reduce --block3-ranking lsa

# Custom config file
python main.py --config configs/lsa_lsa_config.json

# With WandB logging
python main.py --wandb
```

### Grid Search Examples
```bash
# Grid search with all defaults
python run_experiments.py --mode grid

# Grid search with specific blocks
python run_experiments.py --mode grid \
  --grid-block1 none lsa wordnet \
  --grid-block2 tfidf ngram \
  --grid-block3 lsa esa

# Include reduction in grid search
python run_experiments.py --mode grid --grid-reduce
```

### Testing & Validation
```bash
# Validate entire installation
python validate.py

# Run single Block 1 query
python block1_query_processing.py --query "your query" --mode lsa
```

## Supported Method Combinations

### Query Expansion Modes
- **none**: Basic tokenization only
- **lsa**: Latent Semantic Analysis-based expansion
- **esa**: Explicit Semantic Analysis (document similarity-based)
- **wordnet**: WordNet synonym expansion
- **word2vec**: Word embeddings (framework provided)
- **tfidf**: TF-IDF weighted term expansion

### Retrieval Methods (Block 2)
- **tfidf**: Standard TF-IDF with cosine similarity
- **ngram**: Character/word n-gram based retrieval
- **local_bow**: Windowed bag-of-words representation

### Ranking Methods (Block 3)
- **tfidf**: TF-IDF cosine similarity ranking
- **lsa**: Latent Semantic Analysis ranking
- **esa**: Explicit Semantic Analysis ranking

### Total Combinations
- 6 query expansion modes × 2 retrieval methods × 3 ranking methods = **36 combinations**
- With optional query reduction: **72 total configurations**

## Output Files

All outputs are saved in the configured `output_dir`:

### results.json
```json
{
  "config": { /* full configuration used */ },
  "metrics": {
    "precision@10": 0.45,
    "recall@10": 0.32,
    "map@10": 0.38,
    "ndcg@10": 0.42,
    "mrr@10": 0.65
  },
  "rankings": {
    "query_id": ["doc1", "doc2", ...],
    ...
  },
  "timestamp": "2024-01-15T10:30:45"
}
```

### Visualization Files
- `metrics_plot.png`: Bar chart of IR metrics
- `config.json`: Saved configuration for reproducibility

### Log Files
- `pipeline_YYYYMMDD_HHMMSS.log`: Detailed execution log

### Grid Search Results
- `grid_search_summary_YYYYMMDD_HHMMSS.json`: Summary of all grid search runs

## Key Features

✓ **Modular Design**: Easily switch between different methods in each block
✓ **Batch Processing**: Process multiple queries efficiently
✓ **Comprehensive Evaluation**: Standard IR metrics (P@10, R@10, MAP, NDCG, MRR)
✓ **WandB Integration**: Track experiments and compare results
✓ **Configuration-Based**: All settings in JSON config files
✓ **Reproducibility**: Save configs and results for easy reproduction
✓ **Grid Search**: Automatically test multiple configurations
✓ **Extensible**: Easy to add new query expansion, retrieval, or ranking methods
✓ **Logging**: Detailed logging with file and console output
✓ **Visualization**: Automatic plot generation

## Performance Characteristics

- **TF-IDF**: Fastest (~seconds for full pipeline)
- **N-gram**: Slower than TF-IDF (~seconds)
- **LSA**: Slower than TF-IDF, captures latent semantics (~seconds)
- **ESA**: Slowest, concept-based ranking (~seconds for moderate corpus)
- **WordNet expansion**: Memory-intensive for large vocabularies
- **Query reduction**: Minimal overhead (~milliseconds)

## Extensibility Points

To add new methods:

1. **New Query Expansion Mode**: Add method to `QueryProcessor` class
2. **New Retrieval Type**: Add method to `RetrievalRankingPipeline` class
3. **New Ranking Type**: Add ranking method to `RetrievalRankingPipeline` class
4. **Custom Evaluation**: Extend `evaluate_rankings` in `main.py`
5. **New Preprocessing**: Extend `TextPreprocessor` in `utils/preprocessing.py`

## System Requirements

- Python 3.7+
- scikit-learn >= 0.24.0
- numpy >= 1.20.0
- nltk >= 3.6.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- wandb >= 0.12.0 (optional)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Validate installation
python validate.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Config not found | Check path with `--config` flag |
| Dataset not found | Update dataset path in config |
| ImportError | Run `pip install -r requirements.txt` |
| NLTK data missing | Run NLTK download commands |
| WandB issues | Install with `pip install wandb` or set `use_wandb: false` |

## Next Steps

1. **Run validation**: `python validate.py`
2. **Try quickstart**: `python main.py` or `quickstart.bat`
3. **Explore configs**: Check `configs/` for different presets
4. **Run experiments**: `python run_experiments.py --mode grid`
5. **Check results**: Look in `output/` directory

## File Checksums

| File | Purpose | Lines |
|------|---------|-------|
| block1_query_processing.py | Query processing | ~300 |
| blocks23_retrieval_ranking.py | Retrieval & ranking | ~350 |
| main.py | Main orchestration | ~250 |
| utils/logger.py | Logging utilities | ~100 |
| utils/data_loader.py | Data management | ~70 |
| utils/preprocessing.py | Text preprocessing | ~120 |
| run_experiments.py | Experiment runner | ~200 |
| validate.py | Validation tests | ~200 |

**Total: ~1500 lines of code**

---

## Support & Documentation

- **Main README**: See `README.md` for detailed usage guide
- **Config Templates**: Check `configs/` directory
- **Validation**: Run `python validate.py` to verify setup
- **Examples**: Use `quickstart.bat` for interactive menu

---

Created: 2024
Part of: NLP Assignment 2 - Information Retrieval Method Merging
