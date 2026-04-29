# Quick Reference Guide

## Fastest Start

```bash
# Option 1: Interactive menu (Windows)
quickstart.bat

# Option 2: Validate everything works
python validate.py

# Option 3: Run with defaults
python main.py
```

## Common Commands

### Test Specific Blocks

```bash
# Test Block 1 only (Query Processing)
python -c "from block1_query_processing import *; help(QueryProcessor)"

# Test Block 2 & 3 (Retrieval & Ranking)
python -c "from blocks23_retrieval_ranking import *; help(RetrievalRankingPipeline)"
```

### Run Different Method Combinations

```bash
# Baseline (TF-IDF for everything)
python main.py

# LSA-focused
python main.py --block1-mode lsa --block3-ranking lsa

# WordNet + ESA
python main.py --block1-mode wordnet --block3-ranking esa

# With query reduction
python main.py --block1-reduce

# N-gram retrieval
python main.py --block2-retrieval ngram

# Full semantic pipeline
python main.py --block1-mode lsa --block2-retrieval local_bow --block3-ranking esa

# With custom config
python main.py --config configs/lsa_lsa_config.json

# With WandB tracking
python main.py --wandb
```

### Run Multiple Experiments

```bash
# Quick grid search (10 experiments)
python run_experiments.py --mode grid \
  --grid-block1 none lsa wordnet \
  --grid-block2 tfidf ngram \
  --grid-block3 tfidf lsa

# Full grid search with reduction
python run_experiments.py --mode grid --grid-reduce

# All possible combinations (72 total)
python run_experiments.py --mode grid \
  --grid-block1 none lsa esa wordnet tfidf \
  --grid-block2 tfidf ngram local_bow \
  --grid-block3 tfidf lsa esa \
  --grid-reduce
```

## Understanding Output

### results.json
- **metrics**: Precision@10, Recall@10, MAP@10, NDCG@10, MRR@10
- **rankings**: List of top-10 ranked documents per query
- **config**: Full configuration used (for reproducibility)
- **timestamp**: When the experiment ran

### metrics_plot.png
Visual bar chart showing all evaluation metrics

### pipeline_*.log
Detailed execution log with timestamps

## Configuration

### Minimal Config (all required)
```json
{
  "block1_query_processing": { "expansion_mode": "none" },
  "block2_retrieval_mode": { "retrieval_type": "tfidf" },
  "block3_ranking_mode": { "ranking_type": "tfidf" },
  "dataset": { "path": "path/to/cranfield" },
  "output": { "output_dir": "output" }
}
```

### Presets Available
- `default_config.json`: Template with all options
- `lsa_lsa_config.json`: LSA expansion + LSA ranking
- `wordnet_esa_config.json`: WordNet + ESA

## Method Meanings

### Query Expansion (Block 1)
- **none**: Just tokenize
- **lsa**: Expand with latent semantic terms
- **esa**: Expand with conceptually similar documents
- **wordnet**: Add synonyms
- **tfidf**: Weighted expansion based on term importance
- **word2vec**: (Placeholder for embeddings)

### Retrieval (Block 2)
- **tfidf**: Standard term weighting
- **ngram**: Substring/token sequences
- **local_bow**: Context-aware bag-of-words

### Ranking (Block 3)
- **tfidf**: Cosine similarity on terms
- **lsa**: Similarity in latent semantic space
- **esa**: Concept-based ranking

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| `No such file or directory: cranfield` | Update dataset path in config |
| NLTK errors | Run: `python -c "import nltk; nltk.download('punkt')"` |
| Permission denied | Make script executable or use `python script.py` |
| No output | Check `output/` directory |
| Slow execution | Try TF-IDF retrieval/ranking instead of LSA/ESA |

## Performance Tips

1. **Fastest**: Block1=none, Block2=tfidf, Block3=tfidf
2. **Balanced**: Block1=wordnet, Block2=ngram, Block3=lsa
3. **Best Quality**: Block1=lsa, Block2=local_bow, Block3=esa (slowest)

## Development

### Add New Query Expansion
1. Add method `expand_query_NEWMODE()` to `QueryProcessor` class
2. Add params to config under `block1_query_processing.expansion_params`
3. Update `process_query()` dispatcher

### Add New Retrieval Method
1. Add method `build_NEWMETHOD_index()` to `RetrievalRankingPipeline` class
2. Update config schema
3. Update `build_retrieval_index()` dispatcher

### Add New Ranking Method
1. Add method `rank_NEWMETHOD()` to `RetrievalRankingPipeline` class
2. Update `rank()` dispatcher

## File Organization

```
merging/
├── [Core Scripts] - main.py, block1_*.py, blocks23_*.py
├── [Utilities] - utils/
├── [Configs] - configs/
├── [Output] - output/
└── [Reference] - README.md, IMPLEMENTATION_SUMMARY.md, this file
```

## Key Files to Know

| File | Edit For |
|------|----------|
| `configs/default_config.json` | Change parameters |
| `main.py` | Change evaluation metrics |
| `block1_query_processing.py` | Add/modify query expansion |
| `blocks23_retrieval_ranking.py` | Add/modify retrieval/ranking |
| `utils/preprocessing.py` | Customize text processing |

## One-Liners

```bash
# Just test if everything works
python validate.py && echo "All systems GO!"

# Run and show top results
python main.py && head -20 output/results.json

# Compare two configurations
python run_experiments.py --mode single --block1 lsa --block3 lsa
python run_experiments.py --mode single --block1 none --block3 tfidf

# Enable WandB and run
python main.py --wandb

# Check what happens with different expansions
for mode in none lsa wordnet tfidf; do python main.py --block1-mode $mode; done
```

## Environment Variables

None configured, but you can add to config:
- `WANDB_PROJECT`: WandB project name
- `NLTK_DATA`: NLTK data path

## API Quick Usage

```python
# Block 1 only
from block1_query_processing import QueryProcessor
import json

config = json.load(open("configs/default_config.json"))
processor = QueryProcessor(config)
vector = processor.process_query("your query")

# Blocks 2 & 3 only
from blocks23_retrieval_ranking import RetrievalRankingPipeline

pipeline = RetrievalRankingPipeline(config)
pipeline.build_retrieval_index(docs, doc_ids)
rankings = pipeline.rank(query_vectors)

# Full pipeline
from main import MergingPipeline

pipeline = MergingPipeline(config)
results = pipeline.run()
```

## Common Issues & Solutions

**Q: How do I run a quick test?**
A: `python validate.py` - shows if everything works

**Q: How do I compare methods?**
A: `python run_experiments.py --mode grid` - runs multiple combos

**Q: How do I use WandB?**
A: `python main.py --wandb` - logs to dashboard

**Q: Where are results saved?**
A: `output/results.json` by default

**Q: Can I add a new query expansion method?**
A: Yes, edit `block1_query_processing.py` and add to config

**Q: How do I use a custom dataset?**
A: Update `dataset.path` in your config file

**Q: What's the difference between LSA and ESA?**
A: LSA uses math (SVD), ESA uses concept similarity

---

For detailed info, see: `README.md` and `IMPLEMENTATION_SUMMARY.md`
