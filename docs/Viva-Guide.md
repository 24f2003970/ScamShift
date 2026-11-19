# ScamShift | Project report and technical viva guide

## Project description

**ScamShift is an MLOps app that flags suspicious messages and compares active and retrained models against emerging Hinglish scam patterns.**

ScamShift is a local research prototype for message-risk triage. A user pastes a message into a browser dashboard and gets a classifier score, a threshold-based flag and rule-based cues such as credential or payment keywords. A baseline learns from historical UCI SMS spam. A labeled synthetic Hinglish campaign exposes a performance gap. Reviewed examples feed a candidate training run, MLflow records the experiment and artifact, and an application registry supports comparison, quality-gated promotion and rollback.

**30-second pitch.** “We built ScamShift to show what happens after a text classifier is first deployed. The baseline missed the Hinglish scams in our demo batch. We labeled a separate set of examples, trained a candidate and tracked the run in MLflow. Then we compared both versions on identical inputs, checked false alarms as well as recall, and demonstrated promotion and rollback.”

## Architecture

```text
Browser → FastAPI
  Scan → active version in SQLite → fitted text pipeline → score and flag
  Review → SQLite text and labels
  Train → background worker → UCI base + labeled feedback
                            → evaluate on two fixed cohorts
                            → MLflow Tracking run and artifacts
                            → register candidate in SQLite with MLflow run ID
  Replay → score the same labeled batch with active and candidate versions
  Promote or roll back → transactional active-version update in SQLite
```

| Code | Responsibility |
|---|---|
| `web/` | Message scanner, obfuscated samples, campaign replay, review controls and registry |
| `scamshift/model.py` | Text normalization, cohort splits, features, training, metrics and cue rules |
| `scamshift/store.py` | SQLite storage, MLflow logging, quality gates and activation |
| `scamshift/app.py` | API input validation, inference, replay and background training |
| `data/` | UCI corpus and authored Hinglish cohorts |
| `smoke_test.py`, `.github/workflows/verify.yml` | End-to-end verification and GitHub Actions workflow |

## Data, model and evaluation

- The UCI benchmark uses **spam/ham** labels. After deduplication by normalized text hash, 5,159 rows are split into 3,869 training rows and 1,290 held-out rows using stratification, a 25% holdout and seed 42.
- The **review queue** contains 30 synthetic Hinglish examples, 18 scams and 12 genuine messages. The **evaluation batch** contains 24 separate synthetic examples, 12 scams and 12 genuine. Both cover account takeover, parcel fees and task scams. Exact normalized hashes reserve evaluation rows from review-based training.
- Training and inference share Unicode NFKC normalization, case folding and whitespace collapse. A feature union combines word unigrams/bigrams and within-word character 3–5 grams. Both use sublinear TF-IDF. Character features use minimum document frequency 2.
- The model is logistic regression with class balancing, C=4 and a maximum of 500 iterations. Review examples get sample weight 18, UCI training examples weight 1. A score of at least 0.50 is flagged. These settings were chosen for the demo.
- Scanner cue chips are separate regex matches for credentials, urgency, payment and links. They can appear in genuine messages. The displayed score is an uncalibrated positive-class output.

| Measure | v1 baseline | v2 with all 30 reference labels |
|---|---:|---:|
| UCI spam caught | 153/161 | 151/161 |
| UCI spam recall | 95.03% | 93.79% |
| UCI false flags on genuine messages | 4/1,129 | 5/1,129 |
| UCI false alarm rate | 0.35% | 0.44% |
| UCI average precision | 0.991 | 0.989 |
| Synthetic Hinglish scams caught | 0/12 | 12/12 |
| Synthetic Hinglish false flags | 0/12 | 3/12 |
| Synthetic Hinglish false alarm rate | 0% | 25% |

V2 caught more scams in the authored batch while adding false flags and losing about 1.24 percentage points of UCI recall. On that synthetic batch v2 precision was 12/15, or 80%. The batch is small, balanced by design and reused for promotion. Real-world performance needs an independent evaluation with unseen campaigns and representative scam prevalence.

**Formulas.** Recall = TP/(TP+FN). False alarm rate = FP/(FP+TN). Precision = TP/(TP+FP). Average precision summarizes precision-recall performance across scores. It can be positive even if no positives cross the fixed decision threshold.

## How MLOps is integrated

1. **Bootstrap and serve.** First launch trains and activates v1. Later launches reuse SQLite state. Scans use the version active at the time of model load.
2. **Collect feedback.** Reviewers save Scam, Genuine or Unsure. The same normalized message can be relabeled. Unsure entries are excluded from training.
3. **Train asynchronously.** An operator triggers one in-process background worker. The API returns HTTP 202 and the browser polls. The worker fits a fresh pipeline on base data plus saved labels while the active model remains available for inference.
4. **Track in MLflow.** `MlflowClient` creates a run in the **ScamShift** experiment. It logs classifier description, threshold, training and review counts, dataset and feedback SHA-256 fingerprints, benchmark and campaign metrics, `evaluation.json` and `classifier.pkl`. Default tracking files are under `runtime/mlruns`. An existing tracking server can be configured with `MLFLOW_TRACKING_URI`.
5. **Version and compare.** The app's SQLite registry stores each fitted pipeline, metrics and MLflow run ID. Campaign Replay evaluates both versions on identical labeled inputs. This is offline, shadow-style comparison.
6. **Gate and promote.** The server requires UCI recall ≥85%, UCI false alarm rate ≤3%, synthetic campaign false alarm rate ≤25%, and strictly higher campaign recall than the active version. The active pointer and deployment event change in one transaction. The operator requests the release.
7. **Roll back.** Restore a previously deployed version using its saved artifact. New requests use the restored version without retraining or restarting.
8. **Verify.** The smoke test checks API validation, review handling, MLflow artifacts, retraining, promotion rejection and success, rollback and restart persistence. GitHub Actions is configured to run it on pushes and pull requests. Local API and Edge UI checks passed.

**The distinction examiners often ask for.** MLflow Tracking records experiments. The app-owned SQLite registry controls model serving. GitHub Actions is the CI configuration. Candidate training and release are operator-triggered, and the app runs locally.

### Code and API walk-through

- Show `register()` and `log_run()` in `store.py`, then match the run ID on the registry card to a directory under `runtime/mlruns`.
- Show `quality()` and `activate()` in the same file. Point out the four checks, active pointer and deployment event.
- In `app.py`, show `POST /api/scan`, `POST /api/replay`, `POST /api/reviews`, `POST /api/train`, `GET /api/train` and `POST /api/models/{version}/{action}`. Open `/docs` for the interactive API view.

## Four-minute demo

1. **Scan.** Load a sample and compare with its punctuation variant. Explain the model score and cue chips.
2. **Replay v1.** Select Evaluation batch and point to labeled misses and false flags.
3. **Show feedback.** Open Review queue and save a label. Prepare a v2 with all 30 reference labels ahead of the viva if time is limited.
4. **Compare.** Return to Evaluation batch, select v2 and show both models on the same 24 rows.
5. **Release and recover.** Promote v2, scan to show the active version and roll back to v1. Show the deployment history and MLflow run ID.

Sample inputs to paste individually:

- Suspicious English — “URGENT! You've won a £500 cash prize. Reply CLAIM now to collect your reward.”
- Suspicious Hinglish — “Account verify karna hai. KYC ke liye PIN aur OTP bhej do.”
- Genuine lookalike — “Yaar PIN ya OTP share nahi karte.”
- Obfuscated — “Aapka account block hoga. K.Y.C ke liye O.T.P share kro abhi.”

Use the built-in Review queue for training. Some scanner examples are intentionally reserved for evaluation.

## Industry-style viva questions and model answers

### Data and modeling

1. **What is your exact prediction target?** The fitted model outputs a positive-class score. UCI positives are spam, and Hinglish feedback positives are scams. Report results separately because the labels have different meanings.
2. **Why Hinglish?** Code-switching and informal transliteration stress the historical SMS model. The authored batch makes the demo reproducible.
3. **Why deduplicate before splitting?** Exact normalized duplicates across splits can inflate measured performance. Hashing provides a deterministic identity.
4. **How do you prevent feedback leakage?** Review submission and training block normalized hashes of evaluation rows. Near-duplicate templates can still cross cohorts. A stronger evaluation would split by campaign group and time.
5. **How do you handle uncertain labels?** Unsure rows are excluded. Reviewers can revise a label. A production workflow would keep reviewer identity, label history and adjudication.
6. **Why combine word and character n-grams?** Word features capture phrases and character features capture some spelling variation. Unseen scripts and heavy spacing can still break patterns.
7. **Why logistic regression?** It is a lightweight fit for sparse TF-IDF. Compare alternatives on a common test set using latency, per-language metrics and error cost.
8. **What is C=4?** C is inverse regularization strength. A larger C gives weaker regularization. C=4 is the demo configuration.
9. **Why upweight the 30 feedback examples?** They are a small fraction of training data. Weighting makes adaptation observable. Tune the weight on independent validation and inspect its effect on false alarms.
10. **How did you address imbalance?** Stratification preserves class proportions. Logistic regression uses balanced class weights, with additional sample weights for feedback.
11. **Why 0.50?** It is a fixed demo operating point. For deployment, choose the threshold on validation data against recall, review capacity or misclassification cost, then freeze it for final testing.
12. **Does 90% on the scanner mean 90% fraud probability?** It is a positive-class output. Calibrate and verify it on representative labeled traffic before giving it a probability interpretation.
13. **Why focus on recall and false alarms?** The system needs to measure both missed scams and genuine messages flagged. Accuracy can hide missed positives when scams are rare.
14. **What improved and regressed?** Synthetic catches rose from 0/12 to 12/12, synthetic false flags from 0/12 to 3/12, and UCI recall fell from 95.03% to 93.79%.
15. **Is 25% false alarm rate acceptable?** It is the demo gate and v2 meets it exactly. A live threshold should reflect review capacity, message volumes and cost on representative data.
16. **What does 100% recall prove?** All 12 positive examples in that batch crossed the threshold. The fixed authored sample gives limited evidence about real traffic.
17. **Why can AP be positive with zero recall?** AP summarizes ranking across thresholds. Zero recall at 0.50 means all true positives scored below that operating point.
18. **Are cue chips model explanations?** They are regex matches. A model-specific explanation would need feature contribution analysis and checks that users interpret it correctly.
19. **Can the app read WhatsApp or discover campaigns?** It accepts pasted text and groups authored replay rows by supplied campaign labels. Messaging connectors and automatic clustering are future work.
20. **How would you handle adversarial messages?** Evaluate punctuation, spacing, Unicode lookalikes and paraphrases on a separate robustness suite. Assess false alarms before adding normalization or augmentation.

### MLOps and reliability

21. **Where is MLflow actually called?** In `store.py`, `log_run()` creates the run and logs parameters, metrics, the report and the fitted pipeline artifact.
22. **Is the dashboard using MLflow Model Registry?** The registry and active pointer are implemented in SQLite. Each version stores its MLflow Tracking run ID.
23. **What is the difference between a run and a model version?** A run records one execution. A model version is the stored candidate or deployable artifact associated with it.
24. **Does retraining immediately update serving?** It registers a candidate. The active pointer changes only through activation.
25. **What happens if two people click Train?** A process-local lock guards submission. A second request gets a conflict while training is in progress. Multiple replicas would need shared coordination.
26. **Can inference run while training?** Yes. The active version remains available and training uses a background worker. Throughput and resource contention have not been load tested.
27. **Why use HTTP 202 and polling?** Training can outlast a browser request. Accept the job quickly and check its progress separately.
28. **What makes promotion atomic?** The pointer update and deployment event are written in a single SQLite transaction. In-flight scans can finish with the version they already loaded.
29. **What makes rollback fast?** The older artifact is already stored. Change the pointer and record the event. No refit is required.
30. **What if MLflow fails?** A logging failure prevents candidate registration and leaves the active version available. If tracking completes and the database write fails, an unmatched run can remain. Production needs idempotent reconciliation.
31. **How reproducible is a run?** The project pins direct dependencies, fixes the split seed, stores the artifact and fingerprints base data and feedback. Exact reruns would benefit from a full dependency lock, code commit and immutable feedback/evaluation snapshots.
32. **What is CI versus CD versus continuous training?** CI is the test workflow. Model release and retraining are manual actions in this local demo. A production delivery pipeline would add deployment and rollout automation.
33. **What happens after campaign recall reaches 100%?** Strict improvement becomes impossible on the same fixed batch. A better gate would allow statistical non-inferiority on existing cohorts and require improvement on fresh failure cases.
34. **Why be careful with pickle?** Loading untrusted pickle can execute code. Validate artifact provenance and integrity, and evaluate a safer serving format. [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
35. **How would you make jobs durable?** Put training in a separate worker backed by a persistent queue, assign run IDs and idempotency keys, and persist status with retry rules. The current job lives in process memory.

### Monitoring, security and production design

36. **Shadow versus canary?** Shadow scores the same traffic without changing user decisions. Canary routes a controlled fraction of decisions to the new model. ScamShift demonstrates shadow-style comparison through offline replay.
37. **Do you detect live drift?** Replay measures performance on a fixed labeled batch. Live monitoring would compare changing language, input and score distributions over time and join delayed labels for performance measurement.
38. **Data drift versus concept drift?** Data drift changes P(X), such as more Hinglish inputs. Concept drift changes P(Y|X), the label relationship. A changed score distribution by itself does not identify the cause.
39. **What would you monitor before labels arrive?** Volume, p95 latency, errors, flag rates, score distributions and language mix by model version. Use delayed labels to estimate recall, precision and false alarm rates.
40. **How would you assess statistical confidence?** Report denominators and confidence intervals. Split by campaign and time, and reserve an untouched evaluation of real messages.
41. **How does the app handle privacy?** Scans persist the model version and flag. Reviews persist raw text locally. A real service needs redaction, access controls, protected storage and a retention policy.
42. **What if feedback is poisoned?** Authenticate reviewers, keep audit trails, review disputed labels and cap new data's influence. The current demo is designed for trusted local users.
43. **What security controls exist?** The API validates length and allowed labels, and the UI renders message text safely. Public deployment would add authentication, authorization, rate limits and monitoring.
44. **How would you scale serving?** Separate serving from training, cache validated model versions per replica, use shared metadata and object storage, and measure p95 latency and error rates under load.
45. **What would you do before production rollout?** Collect representative, consented data, evaluate on unseen campaigns, set operating targets, run measured shadow traffic and define rollback triggers before changing user-facing decisions.

## References

- Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection, CC BY 4.0. [UCI dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Tracking metadata and artifact storage. [MLflow backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- Word and character feature configuration. [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
