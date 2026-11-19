import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import httpx
import pytest
from fastapi.testclient import TestClient

from scamshift.model import ROOT, examples, probabilities

URL = "http://127.0.0.1"
MESSAGES = ["I'll bring the blue notebook to class tomorrow.", "Urgent: your account is blocked. Share OTP now."]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    folder = tmp_path_factory.mktemp("baseline")
    subprocess.run([sys.executable, "train.py", "--output-dir", str(folder)], cwd=ROOT, check=True)
    return folder


@pytest.fixture
def isolated(tmp_path, monkeypatch, baseline):
    monkeypatch.delenv("SCAMSHIFT_REGISTRY", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    state = tmp_path / "app"
    monkeypatch.setenv("SCAMSHIFT_STATE", str(state))
    monkeypatch.setenv("SCAMSHIFT_BOOTSTRAP_DIR", str(baseline))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "tracking").as_uri())

    from scamshift import store
    from main import app

    monkeypatch.setattr(store, "STATE", state)
    store.load_model.cache_clear()
    yield app, store
    store.load_model.cache_clear()


def post(client, path, body=None, status=200):
    response = client.post(path, json=body)
    assert response.status_code == status, response.text
    return response.json()


def assert_parity(client, model, threshold, predict=probabilities):
    for text, score in zip(MESSAGES, predict(model, MESSAGES)):
        response = post(client, "/api/scan", {"text": text})
        assert response["score"] == pytest.approx(score, abs=1e-12)
        assert response["threshold"] == threshold
        assert response["flagged"] == bool(score >= threshold)


def test_trained_artifact_bootstrap_and_rerun(isolated, baseline):
    app, store = isolated
    raw = (baseline / "classifier.pkl").read_bytes()
    report = json.loads((baseline / "evaluation.json").read_text(encoding="utf-8"))
    with TestClient(app, base_url=URL) as client:
        assert client.get("/").status_code == 200
        assert client.get("/docs").status_code == 200
        state = client.get("/api/status").json()
        assert state["mlflow_registry"]["mode"] == "local"
        assert client.get("/api/readiness").json()["active"] == state["active"]
        with store.db() as conn:
            assert conn.execute("SELECT model FROM models WHERE id=?", (state["active"],)).fetchone()["model"] == raw
        assert_parity(client, pickle.loads(raw), report["threshold"])
        assert post(client, "/api/scan", {"text": " "}, status=422)["detail"] == "Request validation failed."
        subprocess.run([sys.executable, "train.py", "--output-dir", str(baseline)], cwd=ROOT, check=True)
        after = client.get("/api/status").json()
        assert (after["active"], after["events"], after["models"]) == (state["active"], state["events"], state["models"])
    with TestClient(app, base_url=URL) as client:
        assert client.get("/api/readiness").json()["active"] == state["active"]


def test_missing_bootstrap_reports_unavailable(isolated, monkeypatch, tmp_path):
    app, _ = isolated
    monkeypatch.setenv("SCAMSHIFT_BOOTSTRAP_DIR", str(tmp_path / "missing"))
    with TestClient(app, base_url=URL) as client:
        assert client.get("/api/readiness").status_code == 503
        post(client, "/api/scan", {"text": MESSAGES[0]}, status=503)


@pytest.fixture
def registry_server(tmp_path, monkeypatch):
    server_folder = tmp_path / "registry"
    server_folder.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    uri = f"{URL}:{port}"
    log_path = server_folder / "server.log"
    artifacts = server_folder / "artifacts"
    if os.name == "nt":
        artifact_options = ["--no-serve-artifacts", "--default-artifact-root", artifacts.as_uri()]
    else:
        artifact_options = ["--artifacts-destination", str(artifacts)]
    server_env = {**os.environ, "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "5"}
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(port),
             "--workers", "1", "--backend-store-uri", "sqlite:///tracking.db", *artifact_options],
            cwd=server_folder, env=server_env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = monotonic() + 75
            while monotonic() < deadline and proc.poll() is None:
                try:
                    if httpx.get(f"{uri}/health", timeout=1).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                sleep(.3)
            else:
                pytest.fail(f"MLflow server failed:\n{log_path.read_text(encoding='utf-8', errors='replace')}")
            monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
            monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
            yield uri
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def test_registry_backfill_promotion_failure_and_rollback(isolated, registry_server, monkeypatch):
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import get_model_info
    from scamshift import registry

    app, store = isolated
    with TestClient(app, base_url=URL) as client:
        original = client.get("/api/status").json()
        baseline = original["active"]
        post(client, "/api/reviews", {"text": "This synthetic review is undecided.", "label": "unsure"})
        with store.db() as conn:
            stored = tuple(conn.execute("SELECT * FROM models WHERE id=?", (baseline,)).fetchone())
            conn.execute("PRAGMA user_version=2")

    monkeypatch.setenv("SCAMSHIFT_REGISTRY", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_server)
    tracker = MlflowClient(tracking_uri=registry_server)

    def alias_version():
        return str(tracker.get_model_version_by_alias(registry.NAME, registry.ALIAS).version)

    def check_champion(client, local_id):
        state = client.get("/api/status").json()
        assert state["active"] == local_id
        assert state["mlflow_registry"]["committed_registry_version"] == alias_version()
        assert state["mlflow_registry"]["sync"] == "confirmed"
        uri = f"models:/{registry.NAME}@{registry.ALIAS}"
        model = mlflow.sklearn.load_model(uri)
        threshold = get_model_info(uri).metadata["threshold"]
        assert_parity(client, model, threshold, lambda model, texts: model.predict_proba(texts)[:, 1])

    with TestClient(app, base_url=URL) as client:
        if client.get("/api/status").json()["mlflow_registry"]["sync"] != "confirmed":
            store.reconcile_registry()
            pytest.fail("Startup did not synchronize champion, but explicit reconciliation succeeded.")
        check_champion(client, baseline)
        initial_alias = alias_version()
        post(client, "/api/registry/reconcile")
        assert alias_version() == initial_alias
        assert len(tracker.search_model_versions(f"name = '{registry.NAME}'")) == 1
        assert client.get("/api/status").json()["events"] == original["events"]
        with store.db() as conn:
            assert tuple(conn.execute("SELECT * FROM models WHERE id=?", (baseline,)).fetchone()) == stored
            assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1

        held_out = examples("holdout")[0]
        post(client, "/api/reviews", {"text": held_out["text"], "label": "scam"}, status=409)
        reviews = examples("review")
        for row in reviews:
            post(client, "/api/reviews", {"text": row["text"], "label": "scam" if row["label"] else "genuine"})
        job = post(client, "/api/train", status=202)["id"]
        deadline = monotonic() + 150
        while monotonic() < deadline:
            result = client.get(f"/api/train/{job}").json()
            if result["state"] == "finished":
                break
            assert result["state"] == "training", result
            sleep(.25)
        else:
            pytest.fail("Candidate training timed out.")
        candidate = result["version"]
        state = client.get("/api/status").json()
        candidate_data = next(row for row in state["models"] if row["id"] == candidate)
        assert candidate_data["metrics"]["reviewed_rows"] == len(reviews)
        assert candidate_data["mlflow"]["registry_version"] != initial_alias
        assert {"classifier.pkl", "evaluation.json", "model"} <= {
            artifact.path for artifact in tracker.list_artifacts(candidate_data["run_id"])
        }
        assert alias_version() == initial_alias
        assert len(post(client, "/api/replay", {"candidate": candidate})["comparison"]) == 2
        post(client, f"/api/models/{candidate}/rollback", status=409)
        post(client, f"/api/models/{baseline}/promote", status=409)
        assert alias_version() == initial_alias

        original_set_alias = registry.set_alias

        def ambiguous_write(client, version):
            original_set_alias(client, version)
            raise OSError("Synthetic connection loss after MLflow processed the write.")

        with monkeypatch.context() as fault:
            fault.setattr(registry, "set_alias", ambiguous_write)
            post(client, f"/api/models/{candidate}/promote", status=503)
        assert client.get("/api/status").json()["mlflow_registry"]["sync"] == "needs_sync"
        assert post(client, "/api/scan", {"text": MESSAGES[1]})["model"] == baseline
        post(client, "/api/registry/reconcile")
        assert alias_version() == initial_alias
        post(client, f"/api/models/{candidate}/promote")
        check_champion(client, candidate)
        post(client, f"/api/models/{baseline}/promote", status=409)
        check_champion(client, candidate)
        post(client, f"/api/models/{baseline}/rollback")
        check_champion(client, baseline)

    with TestClient(app, base_url=URL) as client:
        check_champion(client, baseline)
        assert alias_version() == initial_alias
