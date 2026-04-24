# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 2 | 2 | 0.1538 | 0.1538 | 0.0000 |
| 64 | 1 | 2 | 0.2619 | 0.3333 | 0.0714 |
| 81 | 1 | 1 | 0.2869 | 0.3869 | 0.1000 |
| 90 | 2 | 2 | 0.1048 | 0.1071 | 0.0024 |
| 119 | 0 | 2 | 0.1944 | 0.8333 | 0.6389 |
| 142 | 1 | 1 | 0.2500 | 0.6250 | 0.3750 |
| 61 | 3 | 3 | 0.4611 | 0.7153 | 0.2542 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [102, 398, 45, 983, 5]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [536, 37, 1205, 272, 1141]
- Relevant docs recovered by method in top 5: [536, 272]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [52, 391, 878, 390, 482]
- Relevant docs recovered by method in top 5: [391, 390]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [714, 799, 652, 206, 249]
- Relevant docs recovered by method in top 5: [799]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [265, 311, 1187, 187, 457]
- Relevant docs recovered by method in top 5: [311, 1187]

## Query 119

what is the effect of initial axisymmetric deviations from circularity on the non linear (large-deflection) load-deflection response of cylinders under hydrostatic pressure .

- Baseline top 5: [928, 781, 1116, 938, 1331]
- Method top 5: [897, 928, 926, 1116, 789]
- Relevant docs recovered by method in top 5: [897, 926]

## Query 142

what dome contours minimize discontinuity stresses when used as closures on cylindrical pressure vessels .

- Baseline top 5: [827, 954, 887, 922, 1134]
- Method top 5: [954, 827, 1136, 887, 1125]
- Relevant docs recovered by method in top 5: [954]

## Query 61

does there exist a closed-form expression for the local heat transfer around a yawed cylinder .

- Baseline top 5: [564, 333, 1213, 566, 539]
- Method top 5: [539, 564, 566, 333, 1213]
- Relevant docs recovered by method in top 5: [539, 564, 566]
