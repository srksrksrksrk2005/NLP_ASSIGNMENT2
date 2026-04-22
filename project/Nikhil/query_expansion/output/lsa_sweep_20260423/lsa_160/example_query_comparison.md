# Example Query Comparison

## Dataset Query Cases

| Query ID | Method | Hits@5 | Top-5 Docs |
| --- | --- | ---: | --- |
| 9 | baseline_tfidf | 1 | 21, 326, 306, 1215, 102 |
| 9 | embedding_lsa | 1 | 21, 306, 326, 1215, 102 |
| 39 | baseline_tfidf | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | embedding_lsa | 2 | 1257, 272, 1264, 43, 1278 |
| 40 | baseline_tfidf | 2 | 536, 1141, 1257, 37, 272 |
| 40 | embedding_lsa | 2 | 1141, 1257, 536, 272, 37 |
| 51 | baseline_tfidf | 4 | 326, 494, 23, 94, 1301 |
| 51 | embedding_lsa | 3 | 326, 494, 1301, 23, 133 |
| 64 | baseline_tfidf | 2 | 878, 391, 52, 390, 593 |
| 64 | embedding_lsa | 2 | 878, 1252, 391, 52, 390 |
| 81 | baseline_tfidf | 1 | 652, 206, 249, 799, 714 |
| 81 | embedding_lsa | 1 | 206, 652, 249, 714, 799 |
| 90 | baseline_tfidf | 2 | 265, 311, 1187, 996, 457 |
| 90 | embedding_lsa | 3 | 265, 311, 1367, 1187, 996 |

## Custom Query Cases (mapped to closest Cranfield query)

| Case | Mapped Query | Method | Hits@5 | Top-5 Docs |
| --- | --- | --- | ---: | --- |
| slip-flow heat transfer in internal channels | 9 | baseline_tfidf | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | embedding_lsa | 1 | 550, 941, 270, 571, 102 |
| transition detection in hypersonic wakes behind slender bodies | 40 | baseline_tfidf | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | embedding_lsa | 2 | 1141, 154, 976, 289, 536 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | baseline_tfidf | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | embedding_lsa | 1 | 878, 391, 1252, 593, 1111 |
| shock-induced boundary-layer separation | 90 | baseline_tfidf | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | embedding_lsa | 3 | 265, 311, 1367, 1187, 996 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | baseline_tfidf | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | embedding_lsa | 0 | 669, 450, 1270, 607, 653 |