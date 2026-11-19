import os
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

with tempfile.TemporaryDirectory() as folder:
    os.environ["SCAMSHIFT_STATE"] = folder
    from scamshift.app import app

    with TestClient(app) as client:
        def post(url, payload, expected=200):
            response = client.post(url, json=payload)
            assert response.status_code == expected, (url, response.status_code, response.text)
            return response.json()

        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        baseline = client.get("/api/status").json()["active"]
        scan = post("/api/scan", {"text": "Share your OTP now"})
        assert scan["model"] == baseline and scan["cues"]
        assert 0 <= scan["score"] <= 1
        post("/api/scan", {"text": " "}, 422)
        post("/api/train", {}, 409)
        audit = post("/api/replay", {"cohort": "audit"})
        post("/api/reviews", {"text": audit["rows"][0]["text"], "label": "scam"}, 409)
        post("/api/replay", {"candidate": 9999}, 404)

        queue = post("/api/replay", {"cohort": "review"})
        for row in queue["rows"]:
            post("/api/reviews", {"text": row["text"], "label": "scam" if row["label"] else "genuine"})
        post("/api/reviews", {"text": queue["rows"][0]["text"], "label": "scam"})
        post("/api/reviews", {"text": "Example awaiting manual judgment.", "label": "unsure"})
        assert client.get("/api/status").json()["reviewed"] == len(queue["rows"])

        post("/api/train", {}, 202)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            response = client.get("/api/train")
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
        print(f"Campaign scams caught {before['caught']} -> {after['caught']} of {after['caught'] + after['missed']}")
        print(f"Campaign false flags {before['false_flags']} -> {after['false_flags']}")
        print(client.get("/api/status").json()["models"][0]["metrics"]["benchmark"])
        print("End-to-end checks passed.")

    with TestClient(app) as client:
        assert client.get("/api/status").json()["active"] == baseline
        assert len(client.get("/api/status").json()["models"]) == 2
