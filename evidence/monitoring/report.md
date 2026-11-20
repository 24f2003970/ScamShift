# ScamShift monitoring and retraining execution

MLflow run: `c6c7b5f78f4a4dc189c0c9bdca0e21fc`
Validated 30 incoming labelled rows; combined training rows: 3899.

| Metric | Existing model on new batch | Champion benchmark | Candidate benchmark |
|---|---:|---:|---:|
| accuracy | 0.4000 | 0.9907 | 0.9884 |
| precision | undefined | 0.9745 | 0.9679 |
| recall | 0.0000 | 0.9503 | 0.9379 |
| f1 | 0.0000 | 0.9623 | 0.9527 |

Decision: **retain_champion**. Champion unchanged.

Scam recall is primary because missed scams expose users to fraud. False-alarm limits protect genuine messages; benchmark F1 balances precision and recall on the imbalanced SMS corpus. Both models use identical untouched evaluation cohorts.

- PASS: Benchmark and campaign datasets match the active version (Compared available fingerprints)
- PASS: Benchmark recall at least 85% (93.8%)
- PASS: Benchmark false alarms at most 3% (0.4%)
- PASS: Benchmark recall loss at most 2 percentage points (+1.24 pp)
- PASS: Benchmark false-alarm increase at most 1 percentage point (+0.09 pp)
- PASS: Campaign false alarms at most 25% (25.0%)
- PASS: Campaign recall improves, or recall holds and false alarms fall (Recall 0.0% to 100.0%, false alarms 0.0% to 25.0%)
- FAIL: Benchmark F1 does not decrease ([0.9622641509433962, 0.9526813880126183])
