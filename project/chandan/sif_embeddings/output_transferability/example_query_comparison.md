# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 4 | 0.1538 | 0.3350 | 0.1812 |
| 64 | 1 | 3 | 0.2619 | 0.7556 | 0.4937 |
| 81 | 1 | 3 | 0.2869 | 0.5889 | 0.3020 |
| 90 | 2 | 3 | 0.1048 | 0.2095 | 0.1048 |
| 35 | 0 | 3 | 0.0000 | 0.6875 | 0.6875 |
| 5 | 1 | 3 | 0.1067 | 0.6643 | 0.5576 |
| 119 | 0 | 2 | 0.1944 | 0.7500 | 0.5556 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [102, 846, 45, 623, 847]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [536, 976, 272, 126, 558]
- Relevant docs recovered by method in top 5: [536, 976, 272, 558]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [390, 52, 627, 856, 391]
- Relevant docs recovered by method in top 5: [390, 627, 391]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [652, 799, 631, 203, 672]
- Relevant docs recovered by method in top 5: [799, 631, 672]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [265, 798, 291, 187, 311]
- Relevant docs recovered by method in top 5: [798, 291, 311]

## Query 35

are there any papers dealing with acoustic wave propagation in reacting gases .

- Baseline top 5: [724, 1244, 1208, 1203, 75]
- Method top 5: [517, 132, 319, 166, 1327]
- Relevant docs recovered by method in top 5: [517, 132, 166]

## Query 5

what chemical kinetic system is applicable to hypersonic aerodynamic problems .

- Baseline top 5: [103, 1032, 410, 943, 552]
- Method top 5: [1296, 552, 103, 401, 1061]
- Relevant docs recovered by method in top 5: [1296, 552, 401]

## Query 119

what is the effect of initial axisymmetric deviations from circularity on the non linear (large-deflection) load-deflection response of cylinders under hydrostatic pressure .

- Baseline top 5: [928, 781, 1116, 938, 1331]
- Method top 5: [897, 1133, 1129, 926, 1132]
- Relevant docs recovered by method in top 5: [897, 926]
