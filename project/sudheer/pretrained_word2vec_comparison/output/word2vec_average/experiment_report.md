# Average Word2Vec Pretrained Comparison Report

## Setup

- Scratch results are reused from the existing `word2vec_average` experiment output.
- Pretrained source: `glove-wiki-gigaword-100`.
- Fine-tuned source: the same pretrained vectors, continued on Cranfield with the same Word2Vec hyperparameters.

## Scratch Baseline

- MAP@10: 0.2230
- nDCG@10: 0.3649
- MRR@10: 0.6127

## Source Comparison for Average Word2Vec Pretrained Comparison Report

| Source | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Model Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2298 | 0.3333 | 0.2519 | 0.2230 | 0.3649 | 0.6127 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.1084 | 0.1525 | 0.1181 | 0.0941 | 0.1799 | 0.3653 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.2062 | 0.3042 | 0.2279 | 0.1983 | 0.3368 | 0.5754 | 100.00% |

- Best source by MAP@10: scratch-trained
- MAP@10 delta vs scratch: +0.0000
- nDCG@10 delta vs scratch: +0.0000
- Approx. randomization p-value (AP@10): 1.0000
- Approx. randomization p-value (nDCG@10): 1.0000

## Example Query Comparison

| Query ID | Baseline Hits@5 | Source Hits@5 | Baseline AP@10 | Source AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 1 | 0.0500 | 0.0500 | 0.0000 |
| 40 | 1 | 1 | 0.1355 | 0.1355 | 0.0000 |
| 64 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 81 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 90 | 1 | 1 | 0.0857 | 0.0857 | 0.0000 |
| 225 | 1 | 1 | 0.0314 | 0.0314 | 0.0000 |
| 224 | 1 | 1 | 0.1111 | 0.1111 | 0.0000 |
| 223 | 3 | 3 | 0.4722 | 0.4722 | 0.0000 |

## Notes

- The scratch backbone is the current in-domain Word2Vec model trained from scratch on Cranfield.
- The pretrained source adds external semantic information and usually improves vocabulary coverage.
- The fine-tuned source keeps the pretrained initialization but adapts it to the Cranfield collection.

## References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)
- Pennington et al., GloVe: Global Vectors for Word Representation (2014)
- Sidorov et al., Soft Similarity and Soft Cosine Measure (2014)
