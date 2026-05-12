# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 1 | 0.0000 | 0.2500 | 0.2500 |
| 40 | 2 | 1 | 0.1538 | 0.0256 | -0.1282 |
| 64 | 1 | 1 | 0.2619 | 0.2619 | 0.0000 |
| 81 | 1 | 0 | 0.2869 | 0.1310 | -0.1560 |
| 90 | 2 | 3 | 0.1048 | 0.1024 | -0.0024 |
| 208 | 3 | 5 | 0.4881 | 0.9821 | 0.4940 |
| 78 | 2 | 4 | 0.5750 | 1.0000 | 0.4250 |
| 129 | 4 | 5 | 0.5625 | 0.9861 | 0.4236 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [21, 152, 102, 142, 496]
- Relevant docs recovered by method in top 5: [21]

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [113, 37, 536, 1141, 1196]
- Relevant docs recovered by method in top 5: [536]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [834, 390, 1006, 52, 1008]
- Relevant docs recovered by method in top 5: [390]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [652, 714, 445, 206, 249]
- Relevant docs recovered by method in top 5: []

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [265, 316, 311, 1187, 798]
- Relevant docs recovered by method in top 5: [311, 1187, 798]

## Query 208

what is the effect of the shape of the drag polar of a lifting spacecraft on the amount of reduction in maximum deceleration obtainable by continuously varying the aerodynamic coefficients during re-entry .

- Baseline top 5: [1291, 1344, 566, 1345, 1124]
- Method top 5: [1291, 1344, 163, 1346, 1347]
- Relevant docs recovered by method in top 5: [1291, 1344, 163, 1346, 1347]

## Query 78

has anyone explained the kink in the surge line of a multi-stage axial compressor .

- Baseline top 5: [589, 543, 216, 237, 578]
- Method top 5: [543, 589, 588, 590, 216]
- Relevant docs recovered by method in top 5: [543, 589, 588, 590]

## Query 129

has anyone derived simplified pump design equation from the fundamental three-dimensional equations for incompressible nonviscous flow .

- Baseline top 5: [945, 988, 989, 987, 1281]
- Method top 5: [988, 945, 987, 984, 985]
- Relevant docs recovered by method in top 5: [988, 945, 987, 984, 985]
