import os
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

with tempfile.TemporaryDirectory() as folder:
    os.environ["SCAMSHIFT_STATE"] = folder
    os.environ["MLFLOW_TRACKING_URI"] = (Path(folder) / "mlruns").as_uri()
    from scamshift.app import app

    with TestClient(app, base_url="http://127.0.0.1") as client:
        def post(url, payload, expected=200):
            response = client.post(url, json=payload)
            assert response.status_code == expected, (url, response.status_code, response.text)
            return response.json()

        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        baseline = client.get("/api/status").json()["active"]
        assert client.get("/api/health").json() == {"state": "ready", "active": baseline}
        assert client.get("/api/readiness").headers["X-Request-ID"]
        assert client.post("/api/train", headers={"Origin": "https://example.test"}).status_code == 403
        scan = post("/api/scan", {"text": "Share your OTP now"})
        assert scan["model"] == baseline and scan["cues"]
        assert 0 <= scan["score"] <= 1
        assert client.get("/api/status").json()["recent_scans"][0]["scans"] == 1
        post("/api/scan", {"text": " "}, 422)
        assert "private test content" not in client.post(
            "/api/reviews", json={"text": "private test content", "label": "invalid"}
        ).text
        post("/api/scan", {"text": "a" * 2501}, 422)
        post("/api/scan", {"text": "a" * 20_000}, 413)
        drift = post("/api/drift", {"days": 7, "minimum": 10})
        assert drift["current_count"] == 1
        assert all(item["status"] == "insufficient" for item in drift["features"])
        assert client.get("/api/drift").json()["reports"][0]["id"] == drift["id"]
        post("/api/train", {}, 409)
        audit = post("/api/replay", {"cohort": "audit"})
        assert len(audit["comparison"][0]["by_campaign"]) == 3
        post("/api/reviews", {"text": audit["rows"][0]["text"], "label": "scam"}, 409)
        from scamshift.model import examples
        from scamshift.store import quality, start_training
        holdout = examples("holdout")
        assert {row["campaign"] for row in holdout}.isdisjoint(row["campaign"] for row in audit["rows"])
        assert {row["id"] for row in holdout}.isdisjoint(row["id"] for row in audit["rows"] + examples("review"))
        post("/api/reviews", {"text": holdout[0]["text"], "label": "scam"}, 409)
        post("/api/replay", {"candidate": 9999}, 404)

        queue = post("/api/replay", {"cohort": "review"})
        post("/api/reviews", {"text": queue["rows"][0]["text"], "label": "scam"})
        post("/api/train", {}, 409)
        for row in queue["rows"]:
            post("/api/reviews", {"text": row["text"], "label": "scam" if row["label"] else "genuine"})
        post("/api/reviews", {"text": queue["rows"][0]["text"], "label": "scam"})
        post("/api/reviews", {"text": "Example awaiting manual judgment.", "label": "unsure"})
        assert client.get("/api/status").json()["reviewed"] == len(queue["rows"])
        first_review = post("/api/replay", {"cohort": "review"})["rows"][0]
        assert len(first_review["review_history"]) == 3
        assert first_review["review_label"] == 1

        job_id = post("/api/train", {}, 202)["id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            response = client.get(f"/api/train/{job_id}")
            assert response.status_code == 200, response.text
            training = response.json()
            if training["state"] == "finished":
                break
            time.sleep(.25)
        else:
            raise AssertionError("Candidate training timed out.")
        candidate = training["version"]
        assert client.get("/api/status").json()["active"] == baseline
        result = post("/api/replay", {"cohort": "audit", "candidate": candidate})["comparison"]
        before, after = [row["metrics"] for row in result]
        assert after["caught"] > before["caught"], result
        post(f"/api/models/{candidate}/rollback", {}, 409)
        post(f"/api/models/{candidate}/promote", {})
        assert post("/api/scan", {"text": "Share your OTP now"})["model"] == candidate
        post(f"/api/models/{baseline}/promote", {}, 409)
        post(f"/api/models/{baseline}/rollback", {})
        assert post("/api/scan", {"text": "Share your OTP now"})["model"] == baseline
        assert len(list(Path(folder).glob("mlruns/*/*/artifacts/classifier.pkl"))) == 2
        previous = {
            "benchmark": {"recall": .95, "false_alarm": .01, "caught": 19, "missed": 1,
                          "false_flags": 1, "genuine": 99},
            "campaign_audit": {"recall": 1, "false_alarm": .25, "caught": 12, "missed": 0,
                               "false_flags": 3, "genuine": 9},
        }
        improved = {**previous, "campaign_audit": {**previous["campaign_audit"], "false_alarm": 0,
                                                   "false_flags": 0, "genuine": 12}}
        assert all(gate["pass"] for gate in quality(improved, previous))
        assert not all(gate["pass"] for gate in quality(previous, previous))
        interrupted_id, _ = start_training()
        post("/api/train", {}, 409)
        print(f"Campaign scams caught {before['caught']} -> {after['caught']} of {after['caught'] + after['missed']}")
        print(f"Campaign false flags {before['false_flags']} -> {after['false_flags']}")
        print(client.get("/api/status").json()["models"][0]["metrics"]["benchmark"])
        print("End-to-end checks passed.")

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/status").json()["active"] == baseline
        assert len(client.get("/api/status").json()["models"]) == 2
        assert client.get(f"/api/train/{job_id}").json()["state"] == "finished"
        assert client.get(f"/api/train/{interrupted_id}").json()["state"] == "interrupted"
        from scamshift import store
        with store.db() as conn:
            conn.execute("UPDATE models SET model=? WHERE id=?", (b"invalid", candidate))
        assert client.post(f"/api/models/{candidate}/rollback").status_code == 409
        with store.db() as conn:
            conn.execute("UPDATE models SET model=NULL WHERE id=?", (candidate,))
        missing = client.post(f"/api/models/{candidate}/rollback")
        assert missing.status_code == 409
        assert missing.json()["detail"] == "Stored model artifact is missing."
        assert client.post("/api/models/9999/rollback").status_code == 404
        assert client.get("/api/health").json()["active"] == baseline
