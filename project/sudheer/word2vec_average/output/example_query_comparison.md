# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 1 | 0.0000 | 0.0500 | 0.0500 |
| 40 | 2 | 1 | 0.1538 | 0.1355 | -0.0183 |
| 64 | 1 | 0 | 0.2619 | 0.0000 | -0.2619 |
| 81 | 1 | 0 | 0.2869 | 0.0000 | -0.2869 |
| 90 | 2 | 1 | 0.1048 | 0.0857 | -0.0190 |
| 167 | 0 | 2 | 0.0333 | 0.5556 | 0.5222 |
| 184 | 1 | 4 | 0.0982 | 0.5792 | 0.4810 |
| 14 | 0 | 1 | 0.0556 | 0.4444 | 0.3889 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [102, 623, 5, 378, 21]
- Relevant docs recovered by method in top 5: [21]

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [536, 1205, 293, 1391, 37]
- Relevant docs recovered by method in top 5: [536]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [544, 428, 719, 844, 52]
- Relevant docs recovered by method in top 5: []

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [795, 714, 598, 755, 594]
- Relevant docs recovered by method in top 5: []

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [291, 187, 455, 1228, 487]
- Relevant docs recovered by method in top 5: [291]

## Query 167

exact solution methods for calculating the ablative mass loss of a material ablating at high temperatures in a hypersonic flight environment .

- Baseline top 5: [553, 1099, 1279, 1097, 1100]
- Method top 5: [274, 553, 82, 1279, 981]
- Relevant docs recovered by method in top 5: [274, 82]

## Query 184

work on small-oscillation re-entry motions .

- Baseline top 5: [554, 716, 1217, 1219, 978]
- Method top 5: [716, 639, 67, 718, 1379]
- Relevant docs recovered by method in top 5: [716, 639, 67, 1379]

## Query 14

papers on shock-sound wave interaction .

- Baseline top 5: [170, 1364, 256, 798, 345]
- Method top 5: [64, 402, 132, 345, 190]
- Relevant docs recovered by method in top 5: [64]
