# Pretrained Word2Vec Comparison

This comparison reuses the existing scratch-trained results and adds two external embedding sources:
- pretrained: `glove-wiki-gigaword-100`
- finetuned: `glove-wiki-gigaword-100` continued on Cranfield

## Average Word2Vec

| Source | MAP@10 | nDCG@10 | MRR@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2230 | 0.3649 | 0.6127 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.0941 | 0.1799 | 0.3653 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.1983 | 0.3368 | 0.5754 | 100.00% |

- Best source: scratch-trained
- MAP@10 delta vs scratch: +0.0000

## IDF-weighted Word2Vec

| Source | MAP@10 | nDCG@10 | MRR@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2624 | 0.4084 | 0.6723 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.1055 | 0.1945 | 0.3791 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.2105 | 0.3420 | 0.5759 | 100.00% |

- Best source: scratch-trained
- MAP@10 delta vs scratch: +0.0000

## Word2Vec Soft Cosine

| Source | MAP@10 | nDCG@10 | MRR@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2917 | 0.4372 | 0.7018 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.2908 | 0.4419 | 0.7268 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.2748 | 0.4241 | 0.6940 | 100.00% |

- Best source: scratch-trained
- MAP@10 delta vs scratch: +0.0000

## Scratch vs Pretrained Training

- The original Sudheer Word2Vec runs train skip-gram/backbone vectors from scratch on Cranfield.
- The comparison folder adds a pretrained embedding source and a fine-tuned variant, so you can see how much external semantic knowledge helps.
- If the fine-tuned source wins, that is the strongest answer to the performance gap in the original scratch-trained plots.
