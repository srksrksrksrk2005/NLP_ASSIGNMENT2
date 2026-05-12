# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0417 | 0.0417 |
| 40 | 2 | 2 | 0.1538 | 0.0769 | -0.0769 |
| 64 | 1 | 1 | 0.2619 | 0.1786 | -0.0833 |
| 81 | 1 | 0 | 0.2869 | 0.0417 | -0.2452 |
| 90 | 2 | 1 | 0.1048 | 0.0952 | -0.0095 |
| 119 | 0 | 1 | 0.1944 | 0.6667 | 0.4722 |
| 12 | 0 | 2 | 0.0167 | 0.4167 | 0.4000 |
| 180 | 3 | 5 | 0.4152 | 0.7958 | 0.3807 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 846, 763, 398]
- Method top 5: [102, 623, 378, 509, 1279]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 272, 295]
- Method top 5: [1205, 536, 1196, 272, 126]
- Relevant docs recovered by method in top 5: [536, 272]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [52, 390, 878, 593, 834]
- Method top 5: [719, 52, 14, 390, 425]
- Relevant docs recovered by method in top 5: [390]

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [249, 206, 652, 714, 799]
- Method top 5: [795, 714, 598, 755, 594]
- Relevant docs recovered by method in top 5: []

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 311, 1187, 187, 996]
- Method top 5: [291, 187, 1228, 316, 487]
- Relevant docs recovered by method in top 5: [291]

## Query 119

what is the effect of initial axisymmetric deviations from circularity on the non linear (large-deflection) load-deflection response of cylinders under hydrostatic pressure .

- Baseline top 5: [928, 781, 1116, 938, 1331]
- Method top 5: [897, 1132, 1116, 1068, 928]
- Relevant docs recovered by method in top 5: [897]

## Query 12

how can the aerodynamic performance of channel flow ground effect machines be calculated .

- Baseline top 5: [941, 1221, 270, 1223, 966]
- Method top 5: [624, 650, 453, 253, 33]
- Relevant docs recovered by method in top 5: [624, 650]

## Query 180

how does scale height vary with altitude in an atmosphere .

- Baseline top 5: [548, 622, 314, 616, 218]
- Method top 5: [622, 616, 548, 621, 617]
- Relevant docs recovered by method in top 5: [622, 616, 548, 621, 617]
