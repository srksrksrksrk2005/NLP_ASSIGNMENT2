# Merging Pipeline: Query Processing + Retrieval + Ranking

A modular information retrieval pipeline combining three distinct blocks:
1. **Block 1**: Query Processing (expansion, reduction, IDF weighting)
2. **Block 2**: Document Retrieval (TF-IDF, N-gram, Local BOW)
3. **Block 3**: Document Ranking (TF-IDF, LSA, ESA)

## Architecture

```
Block 1: Query Processing
├── Expansion modes: none, LSA, ESA, WordNet, Word2Vec, TF-IDF
├── Query reduction: enabled/disabled
└── Output: IDF-weighted query vectors

       ↓

Block 2: Document Retrieval
├── Retrieval types: TF-IDF, N-gram, Local Bag-of-Words
└── Builds searchable document index

       ↓

Block 3: Document Ranking
├── Ranking types: TF-IDF, LSA, ESA
├── Scores queries against documents
└── Returns ranked document list

       ↓

Evaluation & Logging
├── Precision@10, Recall@10, MAP@10, NDCG@10, MRR@10
├── JSON results saving
├── Plot generation
└── WandB integration (optional)
```

## Directory Structure

```
merging/
├── block1_query_processing.py    # Query expansion/reduction
├── blocks23_retrieval_ranking.py # Retrieval and ranking
├── main.py                        # Main orchestration script
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # WandB-compatible logging
│   ├── data_loader.py            # Dataset loading
│   └── preprocessing.py          # Text preprocessing
├── configs/
│   └── default_config.json       # Default configuration
├── output/                        # Output directory
│   ├── results.json
│   ├── metrics_plot.png
│   └── logs/
└── README.md
```

## Configuration

### Config File Structure (configs/default_config.json)

```json
{
  "block1_query_processing": {
    "expansion_mode": "none|lsa|esa|wordnet|word2vec|tfidf",
    "reduction_enabled": true|false,
    "expansion_params": {
      "lsa": { "n_components": 100 },
      "esa": { "top_concepts": 25 },
      "wordnet": { "synset_limit": 3 }
    },
    "reduction_params": {
      "method": "keyword_extraction",
      "top_k": 10
    }
  },
  "block2_retrieval_mode": {
    "retrieval_type": "tfidf|ngram|local_bow",
    "retrieval_params": { ... }
  },
  "block3_ranking_mode": {
    "ranking_type": "tfidf|lsa|esa",
    "ranking_params": { ... }
  },
  "dataset": {
    "path": "path/to/cranfield"
  },
  "output": {
    "output_dir": "path/to/output"
  },
  "logging": {
    "use_wandb": false,
    "wandb_project": "nlp-merging",
    "log_level": "INFO"
  }
}
```

## Usage

### Quick Start

```bash
# Run with default configuration
python main.py

# Run with custom config
python main.py --config configs/custom_config.json
```

### Advanced Usage

```bash
# Test different Block 1 expansion modes
python main.py --block1-mode lsa --block2-retrieval tfidf --block3-ranking lsa
python main.py --block1-mode wordnet --block2-retrieval ngram --block3-ranking esa
python main.py --block1-mode none --block2-retrieval local_bow --block3-ranking tfidf

# Enable query reduction
python main.py --block1-mode lsa --block1-reduce

# Enable WandB logging
python main.py --wandb
```

### Block 1: Query Processing Only

```python
from block1_query_processing import QueryProcessor
import json

# Load config
with open("configs/default_config.json") as f:
    config = json.load(f)

# Create processor
processor = QueryProcessor(config)

# Process single query
query = "information retrieval methods"
vector = processor.process_query(query)

# Process batch of queries
queries = ["query 1", "query 2", "query 3"]
vectors = processor.process_batch(queries)
```

### Block 2 & 3: Retrieval & Ranking

```python
from blocks23_retrieval_ranking import RetrievalRankingPipeline
import json

# Load config
with open("configs/default_config.json") as f:
    config = json.load(f)

# Create pipeline
pipeline = RetrievalRankingPipeline(config)

# Build index
pipeline.build_retrieval_index(documents, doc_ids)

# Rank queries
query_vectors = ...  # From Block 1 or elsewhere
rankings = pipeline.rank(query_vectors)
```

### Using Different Configurations

#### LSA Expansion + LSA Ranking
```bash
python main.py --block1-mode lsa --block3-ranking lsa
```

#### WordNet Expansion + ESA Ranking
```bash
python main.py --block1-mode wordnet --block3-ranking esa
```

#### Query Reduction + TF-IDF Everything
```bash
python main.py --block1-reduce --block1-mode none --block2-retrieval tfidf --block3-ranking tfidf
```

## Supported Methods

### Block 1 Query Expansion Modes
- **none**: No expansion, just tokenization
- **lsa**: Latent Semantic Analysis expansion
- **esa**: Explicit Semantic Analysis (concept-based)
- **wordnet**: WordNet synonym expansion
- **word2vec**: Word embeddings (placeholder)
- **tfidf**: TF-IDF weighted expansion

### Block 2 Retrieval Types
- **tfidf**: Standard TF-IDF with cosine similarity
- **ngram**: Character/word n-gram based retrieval
- **local_bow**: Windowed bag-of-words representation

### Block 3 Ranking Types
- **tfidf**: TF-IDF cosine similarity ranking
- **lsa**: Latent Semantic Analysis ranking
- **esa**: Explicit Semantic Analysis ranking

## Output Files

After running the pipeline, check the output directory:

1. **results.json**: Complete results with metrics and rankings
   ```json
   {
     "config": { ... },
     "metrics": {
       "precision@10": 0.45,
       "recall@10": 0.32,
       "map@10": 0.38,
       "ndcg@10": 0.42,
       "mrr@10": 0.65
     },
     "rankings": { "query_id": [...], ... },
     "timestamp": "2024-01-15T10:30:45.123456"
   }
   ```

2. **metrics_plot.png**: Visualization of IR metrics

3. **config.json**: Saved configuration used for this run

4. **pipeline_YYYYMMDD_HHMMSS.log**: Detailed execution log

## WandB Integration

Enable WandB logging to track experiments:

```bash
# Enable WandB logging
python main.py --wandb

# Or set in config file:
# "logging": { "use_wandb": true, "wandb_project": "nlp-merging" }
```

Logged to WandB:
- Configuration for all three blocks
- All metrics (Precision, Recall, MAP, NDCG, MRR)
- Metrics plot visualization
- Execution logs
- Results artifacts

## Examples

### Example 1: Query Reduction with LSA Ranking
```bash
python main.py --block1-reduce --block1-mode none --block3-ranking lsa
```
**Description**: Reduces query to top keywords, retrieves with TF-IDF, ranks with LSA.

### Example 2: Full Semantic Stack
```bash
python main.py --block1-mode lsa --block2-retrieval ngram --block3-ranking esa
```
**Description**: LSA query expansion → N-gram retrieval → ESA ranking.

### Example 3: Lightweight Pipeline
```bash
python main.py --block1-mode none --block2-retrieval tfidf --block3-ranking tfidf
```
**Description**: Simple TF-IDF based pipeline for baseline comparison.

## Batch Processing

Block 1 supports batch query processing:

```python
processor = QueryProcessor(config)

# Process 1000 queries at once
queries = [...1000 queries...]
batch_vectors = processor.process_batch(queries, return_vectors=True)

# Rank all at once
rankings = pipeline.rank(batch_vectors)
```

## Performance Notes

- **TF-IDF retrieval**: Fast, baseline
- **N-gram retrieval**: Slower, good for typo-tolerance
- **Local BOW retrieval**: Medium speed, context-aware
- **LSA ranking**: Slower than TF-IDF, captures latent semantics
- **ESA ranking**: Slower, concept-based ranking

## Troubleshooting

**Config not found**: Check that config file path is correct
```bash
python main.py --config /path/to/config.json
```

**Dataset not found**: Ensure dataset path in config points to cranfield directory
```json
"dataset": {
  "path": "C:\\Users\\...\\cranfield"
}
```

**WandB not available**: Install with `pip install wandb` or set `use_wandb: false`

**Preprocessing issues**: Ensure NLTK data is downloaded
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Extension Points

To add new modes:

1. **Block 1 - New Expansion Mode**: Add method to `QueryProcessor` class
2. **Block 2 - New Retrieval Type**: Add method to `RetrievalRankingPipeline` class
3. **Block 3 - New Ranking Type**: Add ranking method to `RetrievalRankingPipeline` class
4. **Custom Evaluation**: Extend `evaluate_rankings` in `main.py`

## Requirements

```
scikit-learn>=0.24.0
numpy>=1.20.0
nltk>=3.6.0
matplotlib>=3.4.0
scipy>=1.7.0
wandb>=0.12.0 (optional)
```

Install with:
```bash
pip install scikit-learn numpy nltk matplotlib scipy wandb
```

## License

Part of NLP Assignment 2 - Merging Information Retrieval Methods
