# ScamShift team handoff

ScamShift is an MLOps app that flags suspicious messages and compares active and retrained models against emerging Hinglish scam patterns.

## Start the project

1. Extract **ScamShift.zip**. On Windows with Python 3.12–3.14 installed, double-click **start-windows.cmd** in the extracted project.
2. Keep the server window open. First launch installs dependencies and trains v1.
3. Open **http://127.0.0.1:8765/** on the same computer. Every teammate starts their own instance.
4. On macOS or Linux, create and activate a Python virtual environment and run these commands from the extracted project.

```bash
python -m pip install -r requirements.txt
python -m uvicorn scamshift.app:app --host 127.0.0.1 --port 8765
```

## Reproduce the demo

1. In **Message scan**, load a sample and analyze it. Compare with an obfuscated variant.
2. In **Campaign replay**, choose **Review queue**, run replay and save all 30 displayed reference labels using Scam or Genuine.
3. In **Model registry**, click **Train candidate**. A clean run creates v2.
4. In **Campaign replay**, choose **Evaluation batch**, select **Version 2** and run replay. The baseline caught 0/12 synthetic scams. The retrained model caught 12/12 and incorrectly flagged 3/12 genuine messages.
5. In **Model registry**, promote v2, scan again to show the active version, then roll back to v1.

If the browser refuses the connection, check that the server window is open and has finished starting. If the port is occupied, use a free port in the manual command and browser URL. The source ZIP starts with fresh state, so v2 appears after review and training. Rehearse once on the presentation laptop and prepare a candidate for the viva.

## Suggested speaking split

- **Problem and data** — Explain the problem, the UCI spam labels, and the synthetic Hinglish cohorts.
- **ML and evaluation** — Explain TF-IDF, logistic regression, the 0.50 threshold and the false-alarm trade-off.
- **MLOps and demo** — Show replay, human feedback, MLflow Tracking, the version registry, quality gates and rollback.
- **Engineering and limitations** — Explain FastAPI, background training, tests, security and production requirements.

The team pack includes the source ZIP, this handoff, the technical viva guide, and a screenshot of the verified model comparison.
