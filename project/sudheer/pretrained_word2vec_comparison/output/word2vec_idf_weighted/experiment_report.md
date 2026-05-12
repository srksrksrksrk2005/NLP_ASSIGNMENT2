# IDF-weighted Word2Vec Pretrained Comparison Report

## Setup

- Scratch results are reused from the existing `word2vec_idf_weighted` experiment output.
- Pretrained source: `glove-wiki-gigaword-100`.
- Fine-tuned source: the same pretrained vectors, continued on Cranfield with the same Word2Vec hyperparameters.

## Scratch Baseline

- MAP@10: 0.2624
- nDCG@10: 0.4084
- MRR@10: 0.6723

## Source Comparison for IDF-weighted Word2Vec Pretrained Comparison Report

| Source | P@10 | R@10 | F@10 | MAP@10 | nDCG@10 | MRR@10 | Model Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch-trained | 0.2484 | 0.3715 | 0.2757 | 0.2624 | 0.4084 | 0.6723 | 100.00% |
| pretrained:glove-wiki-gigaword-100 | 0.1209 | 0.1723 | 0.1319 | 0.1055 | 0.1945 | 0.3791 | 45.73% |
| finetuned:glove-wiki-gigaword-100 | 0.2111 | 0.3135 | 0.2338 | 0.2105 | 0.3420 | 0.5759 | 100.00% |

- Best source by MAP@10: scratch-trained
- MAP@10 delta vs scratch: +0.0000
- nDCG@10 delta vs scratch: +0.0000
- Approx. randomization p-value (AP@10): 1.0000
- Approx. randomization p-value (nDCG@10): 1.0000

## Example Query Comparison

| Query ID | Baseline Hits@5 | Source Hits@5 | Baseline AP@10 | Source AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0417 | 0.0417 | 0.0000 |
| 40 | 2 | 2 | 0.0769 | 0.0769 | 0.0000 |
| 64 | 1 | 1 | 0.1786 | 0.1786 | 0.0000 |
| 81 | 0 | 0 | 0.0417 | 0.0417 | 0.0000 |
| 90 | 1 | 1 | 0.0952 | 0.0952 | 0.0000 |
| 225 | 2 | 2 | 0.0800 | 0.0800 | 0.0000 |
| 224 | 2 | 2 | 0.1000 | 0.1000 | 0.0000 |
| 223 | 2 | 2 | 0.3189 | 0.3189 | 0.0000 |

## Notes

- The scratch backbone is the current in-domain Word2Vec model trained from scratch on Cranfield.
- The pretrained source adds external semantic information and usually improves vocabulary coverage.
- The fine-tuned source keeps the pretrained initialization but adapts it to the Cranfield collection.

## References

- Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality (2013)
- Pennington et al., GloVe: Global Vectors for Word Representation (2014)
- Sidorov et al., Soft Similarity and Soft Cosine Measure (2014)
