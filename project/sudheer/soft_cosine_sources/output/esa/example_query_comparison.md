# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 1 | 2 | 0.0940 | 0.0897 | -0.0043 |
| 64 | 1 | 2 | 0.4167 | 0.6333 | 0.2167 |
| 81 | 2 | 2 | 0.5000 | 0.3778 | -0.1222 |
| 90 | 2 | 2 | 0.1190 | 0.1190 | 0.0000 |
| 129 | 4 | 5 | 0.6042 | 1.0000 | 0.3958 |
| 208 | 4 | 4 | 0.5429 | 0.8129 | 0.2701 |
| 20 | 3 | 4 | 0.3298 | 0.5578 | 0.2280 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 398, 763, 5]
- Method top 5: [102, 398, 5, 846, 763]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 1205, 1158]
- Method top 5: [113, 1205, 536, 272, 37]
- Relevant docs recovered by method in top 5: [536, 272]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [390, 52, 719, 914, 1107]
- Method top 5: [390, 52, 1053, 489, 627]
- Relevant docs recovered by method in top 5: [390, 627]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [714, 799, 652, 672, 203]
- Method top 5: [42, 714, 799, 672, 78]
- Relevant docs recovered by method in top 5: [799, 672]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 1187, 311, 415, 1239]
- Method top 5: [265, 1187, 311, 1239, 358]
- Relevant docs recovered by method in top 5: [1187, 311]

## Query 129

has anyone derived simplified pump design equation from the fundamental three-dimensional equations for incompressible nonviscous flow .

- Baseline top 5: [945, 988, 989, 987, 36]
- Method top 5: [988, 989, 945, 990, 984]
- Relevant docs recovered by method in top 5: [988, 989, 945, 990, 984]

## Query 208

what is the effect of the shape of the drag polar of a lifting spacecraft on the amount of reduction in maximum deceleration obtainable by continuously varying the aerodynamic coefficients during re-entry .

- Baseline top 5: [1291, 1344, 163, 566, 1345]
- Method top 5: [1291, 1344, 1346, 1347, 275]
- Relevant docs recovered by method in top 5: [1291, 1344, 1346, 1347]

## Query 20

has anyone formally determined the influence of joule heating,  produced by the induced current,  in magnetohydrodynamic free convection flows under general conditions .

- Baseline top 5: [450, 408, 407, 270, 584]
- Method top 5: [408, 450, 267, 407, 104]
- Relevant docs recovered by method in top 5: [408, 267, 407, 104]
