# Word2Vec Soft Cosine Pretrained Comparison Report

## Setup

- Scratch results are reused from the existing `soft_cosine_word2vec` experiment output.
- Pretrained source: `glove-wiki-gigaword-100`.
- Fine-tuned source: the same pretrained vectors, continued on Cranfield with the same Word2Vec hyperparameters.

## Scratch Baseline

- MAP@10: 0.2917
- nDCG@10: 0.4372
- MRR@10: 0.7018

## Source Comparison for Word2Vec Soft Cosine Pretrained Comparison Report

| Source | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Model Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2756 | 0.3990 | 0.3021 | 0.2917 | 0.4372 | 0.7018 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.2684 | 0.3887 | 0.2939 | 0.2908 | 0.4419 | 0.7268 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.2627 | 0.3804 | 0.2883 | 0.2748 | 0.4241 | 0.6940 | 100.00% |

- Best source by MAP@10: scratch-trained
- MAP@10 delta vs scratch: +0.0000
- nDCG@10 delta vs scratch: +0.0000
- Approx. randomization p-value (AP@10): 1.0000
- Approx. randomization p-value (nDCG@10): 1.0000

## Example Query Comparison

| Query ID | Baseline Hits@5 | Source Hits@5 | Baseline AP@10 | Source AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 1 | 0.2500 | 0.2500 | 0.0000 |
| 40 | 1 | 1 | 0.0256 | 0.0256 | 0.0000 |
| 64 | 1 | 1 | 0.2619 | 0.2619 | 0.0000 |
| 81 | 0 | 0 | 0.1310 | 0.1310 | 0.0000 |
| 90 | 3 | 3 | 0.1024 | 0.1024 | 0.0000 |
| 225 | 2 | 2 | 0.0800 | 0.0800 | 0.0000 |
| 224 | 0 | 0 | 0.0185 | 0.0185 | 0.0000 |
| 223 | 1 | 1 | 0.1500 | 0.1500 | 0.0000 |

## Notes

- The scratch backbone is the current in-domain Word2Vec model trained from scratch on Cranfield.
- The pretrained source adds external semantic information and usually improves vocabulary coverage.
- The fine-tuned source keeps the pretrained initialization but adapts it to the Cranfield collection.

## References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)
- Pennington et al., GloVe: Global Vectors for Word Representation (2014)
- Sidorov et al., Soft Similarity and Soft Cosine Measure (2014)
