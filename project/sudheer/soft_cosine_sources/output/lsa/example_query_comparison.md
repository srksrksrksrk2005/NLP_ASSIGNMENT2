# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 1 | 3 | 0.0940 | 0.1859 | 0.0919 |
| 64 | 1 | 1 | 0.4167 | 0.4074 | -0.0093 |
| 81 | 2 | 0 | 0.5000 | 0.1508 | -0.3492 |
| 90 | 2 | 3 | 0.1190 | 0.2174 | 0.0984 |
| 95 | 1 | 2 | 0.1667 | 0.6667 | 0.5000 |
| 34 | 2 | 4 | 0.1905 | 0.6321 | 0.4417 |
| 129 | 4 | 5 | 0.6042 | 1.0000 | 0.3958 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 398, 763, 5]
- Method top 5: [102, 398, 983, 559, 1161]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 1205, 1158]
- Method top 5: [558, 1205, 536, 272, 79]
- Relevant docs recovered by method in top 5: [558, 536, 272]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [390, 52, 719, 914, 1107]
- Method top 5: [390, 688, 1053, 1234, 52]
- Relevant docs recovered by method in top 5: [390]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [714, 799, 652, 672, 203]
- Method top 5: [42, 652, 78, 714, 100]
- Relevant docs recovered by method in top 5: []

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 1187, 311, 415, 1239]
- Method top 5: [265, 1187, 291, 797, 487]
- Relevant docs recovered by method in top 5: [1187, 291, 797]

## Query 95

what is the theoretical heat transfer distribution around a hemisphere .

- Baseline top 5: [553, 662, 535, 628, 1104]
- Method top 5: [662, 283, 1161, 689, 559]
- Relevant docs recovered by method in top 5: [662, 283]

## Query 34

have wind tunnel interference effects been investigated on a systematic basis .

- Baseline top 5: [795, 1153, 516, 431, 280]
- Method top 5: [516, 672, 714, 594, 431]
- Relevant docs recovered by method in top 5: [516, 672, 714, 431]

## Query 129

has anyone derived simplified pump design equation from the fundamental three-dimensional equations for incompressible nonviscous flow .

- Baseline top 5: [945, 988, 989, 987, 36]
- Method top 5: [989, 988, 945, 984, 985]
- Relevant docs recovered by method in top 5: [989, 988, 945, 984, 985]
