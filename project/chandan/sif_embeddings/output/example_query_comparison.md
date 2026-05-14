# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 0 | 0.1538 | 0.0000 | -0.1538 |
| 64 | 1 | 0 | 0.2619 | 0.0000 | -0.2619 |
| 81 | 1 | 1 | 0.2869 | 0.0833 | -0.2036 |
| 90 | 2 | 1 | 0.1048 | 0.0357 | -0.0690 |
| 131 | 1 | 3 | 0.0873 | 0.4015 | 0.3142 |
| 135 | 3 | 4 | 0.4586 | 0.6204 | 0.1618 |
| 75 | 0 | 2 | 0.0000 | 0.1389 | 0.1389 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [45, 270, 546, 489, 607]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [1158, 804, 64, 311, 969]
- Relevant docs recovered by method in top 5: []

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [482, 416, 928, 1215, 152]
- Relevant docs recovered by method in top 5: []

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [714, 598, 809, 672, 1353]
- Relevant docs recovered by method in top 5: [672]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [1160, 1187, 866, 892, 1045]
- Relevant docs recovered by method in top 5: [1187]

## Query 131

what agreement is found between theoretically predicted instability times and experimentally measured collapse times for compressed columns in creep .

- Baseline top 5: [1021, 1023, 1025, 1026, 950]
- Method top 5: [950, 1026, 1018, 1017, 1023]
- Relevant docs recovered by method in top 5: [950, 1018, 1017]

## Query 135

what are the experimental results for the creep buckling of columns .

- Baseline top 5: [1026, 1023, 1017, 1015, 1012]
- Method top 5: [1026, 1023, 1017, 1018, 1012]
- Relevant docs recovered by method in top 5: [1026, 1023, 1017, 1018]

## Query 75

do the discrepancies among current analyses of the vorticity effect on stagnation-point heat transfer result primarily from the differences in the viscosity-temperature law assumed .

- Baseline top 5: [565, 1146, 629, 180, 951]
- Method top 5: [1160, 658, 670, 630, 486]
- Relevant docs recovered by method in top 5: [670, 630]
