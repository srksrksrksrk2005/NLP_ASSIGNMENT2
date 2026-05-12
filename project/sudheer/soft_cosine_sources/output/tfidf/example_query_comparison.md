# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 1 | 2 | 0.0940 | 0.0500 | -0.0440 |
| 64 | 1 | 2 | 0.4167 | 0.2167 | -0.2000 |
| 81 | 2 | 2 | 0.5000 | 0.4206 | -0.0794 |
| 90 | 2 | 2 | 0.1190 | 0.1389 | 0.0198 |
| 20 | 3 | 5 | 0.3298 | 0.6367 | 0.3069 |
| 95 | 1 | 1 | 0.1667 | 0.4286 | 0.2619 |
| 129 | 4 | 5 | 0.6042 | 0.8375 | 0.2333 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 398, 763, 5]
- Method top 5: [102, 398, 983, 5, 559]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 1205, 1158]
- Method top 5: [113, 37, 1205, 536, 272]
- Relevant docs recovered by method in top 5: [536, 272]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [390, 52, 719, 914, 1107]
- Method top 5: [1053, 52, 15, 390, 391]
- Relevant docs recovered by method in top 5: [390, 391]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [714, 799, 652, 672, 203]
- Method top 5: [714, 42, 672, 799, 203]
- Relevant docs recovered by method in top 5: [672, 799]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 1187, 311, 415, 1239]
- Method top 5: [265, 1187, 1239, 311, 1228]
- Relevant docs recovered by method in top 5: [1187, 311]

## Query 20

has anyone formally determined the influence of joule heating,  produced by the induced current,  in magnetohydrodynamic free convection flows under general conditions .

- Baseline top 5: [450, 408, 407, 270, 584]
- Method top 5: [408, 267, 270, 407, 104]
- Relevant docs recovered by method in top 5: [408, 267, 270, 407, 104]

## Query 95

what is the theoretical heat transfer distribution around a hemisphere .

- Baseline top 5: [553, 662, 535, 628, 1104]
- Method top 5: [662, 628, 553, 1104, 559]
- Relevant docs recovered by method in top 5: [662]

## Query 129

has anyone derived simplified pump design equation from the fundamental three-dimensional equations for incompressible nonviscous flow .

- Baseline top 5: [945, 988, 989, 987, 36]
- Method top 5: [988, 989, 945, 987, 985]
- Relevant docs recovered by method in top 5: [988, 989, 945, 987, 985]
