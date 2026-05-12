# Example Query Comparison

| Query ID | Baseline Hits@5 | Method Hits@5 | Baseline AP@10 | Method AP@10 | Delta AP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 40 | 1 | 1 | 0.0940 | 0.0769 | -0.0171 |
| 64 | 1 | 0 | 0.4167 | 0.0476 | -0.3690 |
| 81 | 2 | 1 | 0.5000 | 0.1944 | -0.3056 |
| 90 | 2 | 2 | 0.1190 | 0.1048 | -0.0143 |
| 216 | 0 | 2 | 0.0000 | 0.5833 | 0.5833 |
| 139 | 1 | 1 | 0.0667 | 0.2000 | 0.1333 |
| 124 | 1 | 2 | 0.2000 | 0.3333 | 0.1333 |

## Query 9

papers on internal /slip flow/ heat transfer studies .

- Baseline top 5: [102, 45, 398, 763, 5]
- Method top 5: [102, 270, 45, 524, 405]
- Relevant docs recovered by method in top 5: []

## Query 40

how can one detect transition phenomena in hypersonic wakes .

- Baseline top 5: [536, 37, 1141, 1205, 1158]
- Method top 5: [536, 83, 655, 154, 1264]
- Relevant docs recovered by method in top 5: [536]

## Query 64

can't the static deflection shapes be used in predicting flutter in place of vibrational shapes . if so,  can we provide a justification by means of an example .

- Baseline top 5: [390, 52, 719, 914, 1107]
- Method top 5: [109, 345, 52, 1292, 531]
- Relevant docs recovered by method in top 5: []

## Query 81

what are wind-tunnel corrections for a two-dimensional aerofoil mounted off-centre in a tunnel .

- Baseline top 5: [714, 799, 652, 672, 203]
- Method top 5: [1163, 526, 631, 440, 358]
- Relevant docs recovered by method in top 5: [631]

## Query 90

recent data on shock-induced boundary-layer separation .

- Baseline top 5: [265, 1187, 311, 415, 1239]
- Method top 5: [265, 1187, 311, 3, 1171]
- Relevant docs recovered by method in top 5: [1187, 311]

## Query 216

what investigations have been made of the wave system created by a static pressure distribution over a liquid surface .

- Baseline top 5: [407, 1225, 958, 1269, 974]
- Method top 5: [1206, 506, 156, 153, 398]
- Relevant docs recovered by method in top 5: [506, 156]

## Query 139

has the effect of the change of initial pressure due to deformation,  on the frequencies of vibration of circular cylindrical shells been investigated .

- Baseline top 5: [847, 897, 953, 848, 844]
- Method top 5: [953, 847, 841, 897, 826]
- Relevant docs recovered by method in top 5: [953]

## Query 124

in what areas, other than low density wind tunnel flows, is viscous compressible flow in slender channels a problem . what analytical investigations have been made of the stability of conical shells . how do the results compare with experiment .

- Baseline top 5: [941, 898, 1070, 938, 1222]
- Method top 5: [941, 1222, 967, 112, 607]
- Relevant docs recovered by method in top 5: [941, 967]
