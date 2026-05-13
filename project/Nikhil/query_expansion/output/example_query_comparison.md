# Example Query Comparison

## Dataset Query Cases

| Query ID | Method | Hits@5 | Top-5 Docs |
| --- | --- | ---: | --- |
| 9 | baseline_tfidf | 1 | 21, 326, 306, 1215, 102 |
| 9 | wordnet | 1 | 21, 326, 306, 1215, 102 |
| 9 | embedding_tfidf | 1 | 21, 326, 306, 1215, 102 |
| 9 | embedding_lsa | 1 | 21, 326, 306, 1215, 102 |
| 9 | embedding_esa | 1 | 21, 326, 306, 1215, 102 |
| 9 | embedding_word2vec | 1 | 21, 326, 306, 1215, 102 |
| 39 | baseline_tfidf | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | wordnet | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | embedding_tfidf | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | embedding_lsa | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | embedding_esa | 2 | 1257, 1264, 272, 43, 1278 |
| 39 | embedding_word2vec | 2 | 1257, 1264, 272, 43, 1278 |
| 40 | baseline_tfidf | 2 | 536, 1141, 1257, 37, 272 |
| 40 | wordnet | 2 | 536, 1141, 1257, 37, 272 |
| 40 | embedding_tfidf | 2 | 536, 1141, 1257, 37, 272 |
| 40 | embedding_lsa | 2 | 536, 1141, 1257, 37, 272 |
| 40 | embedding_esa | 2 | 536, 1141, 1257, 37, 272 |
| 40 | embedding_word2vec | 2 | 536, 1141, 1257, 37, 272 |
| 51 | baseline_tfidf | 4 | 326, 494, 23, 94, 1301 |
| 51 | wordnet | 4 | 326, 494, 23, 94, 1301 |
| 51 | embedding_tfidf | 4 | 326, 494, 23, 94, 1301 |
| 51 | embedding_lsa | 4 | 326, 494, 23, 94, 1301 |
| 51 | embedding_esa | 4 | 326, 494, 23, 94, 1301 |
| 51 | embedding_word2vec | 4 | 326, 494, 23, 94, 1301 |
| 64 | baseline_tfidf | 2 | 878, 391, 52, 390, 593 |
| 64 | wordnet | 2 | 878, 391, 52, 390, 593 |
| 64 | embedding_tfidf | 2 | 878, 391, 52, 390, 593 |
| 64 | embedding_lsa | 2 | 878, 391, 52, 390, 593 |
| 64 | embedding_esa | 2 | 878, 391, 52, 390, 593 |
| 64 | embedding_word2vec | 2 | 878, 391, 52, 390, 593 |
| 81 | baseline_tfidf | 1 | 652, 206, 249, 799, 714 |
| 81 | wordnet | 1 | 652, 206, 249, 799, 714 |
| 81 | embedding_tfidf | 1 | 652, 206, 249, 799, 714 |
| 81 | embedding_lsa | 1 | 652, 206, 249, 799, 714 |
| 81 | embedding_esa | 1 | 652, 206, 249, 799, 714 |
| 81 | embedding_word2vec | 1 | 652, 206, 249, 799, 714 |
| 90 | baseline_tfidf | 2 | 265, 311, 1187, 996, 457 |
| 90 | wordnet | 2 | 265, 311, 1187, 996, 457 |
| 90 | embedding_tfidf | 2 | 265, 311, 1187, 996, 457 |
| 90 | embedding_lsa | 2 | 265, 311, 1187, 996, 457 |
| 90 | embedding_esa | 2 | 265, 311, 1187, 996, 457 |
| 90 | embedding_word2vec | 2 | 265, 311, 1187, 996, 457 |

## Custom Query Cases (mapped to closest Cranfield query)

| Case | Mapped Query | Method | Hits@5 | Top-5 Docs |
| --- | --- | --- | ---: | --- |
| slip-flow heat transfer in internal channels | 9 | baseline_tfidf | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | wordnet | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | embedding_tfidf | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | embedding_lsa | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | embedding_esa | 1 | 550, 941, 270, 1221, 102 |
| slip-flow heat transfer in internal channels | 9 | embedding_word2vec | 1 | 550, 941, 270, 1221, 102 |
| transition detection in hypersonic wakes behind slender bodies | 40 | baseline_tfidf | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | wordnet | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | embedding_tfidf | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | embedding_lsa | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | embedding_esa | 2 | 1141, 976, 289, 154, 536 |
| transition detection in hypersonic wakes behind slender bodies | 40 | embedding_word2vec | 2 | 1141, 976, 289, 154, 536 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | baseline_tfidf | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | wordnet | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | embedding_tfidf | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | embedding_lsa | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | embedding_esa | 1 | 391, 878, 593, 1111, 202 |
| replace vibrational shapes with static deflection shapes for flutter prediction | 64 | embedding_word2vec | 1 | 391, 878, 593, 1111, 202 |
| shock-induced boundary-layer separation | 90 | baseline_tfidf | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | wordnet | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | embedding_tfidf | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | embedding_lsa | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | embedding_esa | 2 | 265, 311, 1187, 996, 457 |
| shock-induced boundary-layer separation | 90 | embedding_word2vec | 2 | 265, 311, 1187, 996, 457 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | baseline_tfidf | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | wordnet | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | embedding_tfidf | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | embedding_lsa | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | embedding_esa | 0 | 669, 450, 1270, 607, 653 |
| what corrections are needed for a liftbody in a propwash flowfield inside a test duct | 81 | embedding_word2vec | 0 | 669, 450, 1270, 607, 653 |