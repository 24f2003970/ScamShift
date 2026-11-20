# Model quality monitoring and retraining practical

Run from the repository root after installing `requirements.txt`:

```bash
python train.py
python monitor_retrain.py --batch data/incoming_demo.csv --champion-dir runtime/baseline
python -m unittest test_monitoring -v
```

For an already running deployment, omit `--champion-dir` to read its active model from the existing SQLite registry (`SCAMSHIFT_STATE` selects the runtime directory). The workflow never activates a model. The baseline command is only needed for a fresh checkout; do not recreate a deployed champion for monitoring.

Task A: The CSV requires `id,text,label`. Labels are 0 for genuine and 1 for spam/scam. Missing values, invalid labels, duplicate IDs/messages, original corpus overlaps and protected evaluation overlaps fail validation. `validation.json` records batch size, class distribution and SHA-256. Invalid batches save a failed validation report and stop before training.

Task B: Load the saved trusted champion and evaluate it on all incoming labelled rows. `existing_model_evaluation.json` records accuracy, precision, recall, F1 and the confusion matrix.

Precision is reported as `null` / `undefined` when a model flags no messages; it has no predicted positives to measure. This is preserved explicitly rather than inventing a precision value.

Task C: The existing `fit_artifact`/`train` functions combine the original SMS training split with the incoming batch. They refit the existing word and character TF-IDF pipeline and logistic regression, with the project's feedback weighting. Save a separate `candidate.pkl` and `candidate_evaluation.json`. The run, parameters, metrics, model and complete monitoring report are logged to the existing ScamShift MLflow experiment. A tracking failure fails execution rather than claiming successful tracking.

Task D: Evaluate both models on identical untouched SMS benchmark and campaign audit cohorts. Use the existing project gate plus non-decreasing benchmark F1. Scam recall is the primary business metric: missed scams expose users to fraud. False-alarm caps protect genuine messages, while F1 balances precision and recall under class imbalance. The exact checks and result are in `quality_gate.json`. Eligible means available for manual promotion, never automatic replacement. A rejected candidate remains saved for inspection.

Task E: Each run creates a unique directory under `runtime/monitoring` containing the artifacts above, `report.json` and readable `report.md`. Show the console output, report, candidate file and MLflow run during the demonstration. The [saved report](../evidence/monitoring/report.md) and [console transcript](../evidence/monitoring/execution.txt) come from a fresh local SQLite deployment. The transcript shows the same champion ID and checksum before and after the run. The Linux GitHub Actions test job also runs the workflow and uploads the reports and model as the `monitoring-assessment` artifact.

The supplied incoming CSV contains 30 authored review fixtures from the project. It is disjoint from the original SMS corpus and protected evaluation fixtures. For operational monitoring, use a newly labelled batch the champion has never trained on. Candidate scores on the incoming batch are training diagnostics. Promotion eligibility uses the untouched SMS benchmark and campaign audit. The CLI uses ScamShift's word and character TF-IDF preprocessing and logistic regression.

Viva explanation: “I validate the incoming labelled messages, load the previous champion, and measure its current quality. I then retrain the same preprocessing and classifier with original training data plus the new messages. I compare both models on untouched evaluation data, record every quality check, save the candidate separately and track the run in MLflow. The champion remains unchanged until manual promotion.”
