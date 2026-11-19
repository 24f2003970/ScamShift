# ScamShift

ScamShift is an MLOps app that flags suspicious messages and compares active and retrained models against emerging Hinglish scam patterns.

## Run locally

Use Python 3.12–3.14. On Windows, open the project folder and run `start-windows.cmd`. The first launch installs dependencies and trains the baseline. Open [the dashboard](http://127.0.0.1:8765). API documentation is available at [localhost:8765/docs](http://127.0.0.1:8765/docs).

To start it manually from the project folder:

```bash
python -m pip install -r requirements.txt
python -m uvicorn scamshift.app:app --host 127.0.0.1 --port 8765
```

## Demo

1. Open **Message scan**. Load the sample and analyze it. Try the obfuscated examples and compare the scores. Cue chips show rule matches such as a request for a credential or payment.
2. Open **Campaign replay**, select **Review queue** and run the replay. Save the displayed reference label for each of the 30 messages. The scanner can also save your own labeled examples. Unsure labels are skipped during training.
3. Open **Model registry** and train a candidate. Return to **Campaign replay**, choose **Evaluation batch** and run the active model against the candidate. The two models score the same messages. Compare misses and false flags at the message level.
4. Return to the registry and promote the candidate if its quality gates pass. Run a scan to see the new active version. Roll back to the previously deployed version from the same page.

## Model and evaluation

- A logistic regression pipeline uses word and character TF-IDF features. The baseline uses a stratified training split of the normalized, deduplicated UCI SMS Spam Collection. Its target is the dataset's **spam** label.
- The demo adds 30 labeled Hinglish review examples and a separate, fixed 24-message synthetic evaluation batch with three campaign types. The UCI holdout and evaluation examples are blocked from review-based training by normalized text hash.
- Promotion requires at least 85% benchmark spam recall, at most 3% benchmark false alarms, at most 25% synthetic campaign false alarms, and improved synthetic campaign recall over the active model. The bootstrap baseline and previously deployed rollback versions follow separate activation paths.
- In the verified demo, the baseline caught 0/12 synthetic scams with 0/12 genuine messages flagged. After review and retraining, the candidate caught 12/12 scams with 3/12 genuine messages flagged. The same synthetic evaluation is used for promotion, and its small, authored sample limits what the result says about real-world performance. Scores are classifier outputs and have not been calibrated as probabilities of fraud.

## MLOps workflow

`Message scan → human labels → candidate training → MLflow tracking → side-by-side replay → gated promotion → rollback`

FastAPI serves the browser dashboard and API. SQLite persists review labels, versions and deployment history. MLflow logs model parameters, metrics, the evaluation report and serialized classifier for each run. By default it uses the local file store under `runtime/mlruns`. Set `MLFLOW_TRACKING_URI` to use an existing tracking server.

This is a single-process local demo. Review submissions store message text in SQLite and can influence the model, so use sample messages and keep the server bound to loopback. Scan-only submissions persist the flag and model version.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python smoke_test.py
```

The smoke test uses an isolated temporary directory. It covers inference, input validation, review labels, evaluation separation, retraining, MLflow artifacts, failed promotion, successful promotion, rollback and state persistence. GitHub Actions runs it on pushes and pull requests.

## Data attribution

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original archive documentation is in `data/readme`. The Hinglish review and evaluation batches were authored for this demo.
