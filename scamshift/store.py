"""Local model registry, review labels and MLflow experiment tracking."""
import json
import os
import pickle
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from .model import ROOT, key, reserved, train

STATE = Path(os.environ.get("SCAMSHIFT_STATE", ROOT / "runtime")).resolve()
LOCK = RLock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    STATE.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(STATE / "scamshift.db", timeout=30)) as conn, conn:
        conn.row_factory = sqlite3.Row
        yield conn


def initialize():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS models(
                id INTEGER PRIMARY KEY, created TEXT, run_id TEXT, metrics TEXT, model BLOB);
            CREATE TABLE IF NOT EXISTS active(singleton INTEGER PRIMARY KEY CHECK(singleton=1), model_id INTEGER);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created TEXT, kind TEXT, model_id INTEGER);
            CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, text TEXT, label INTEGER, updated TEXT);
            CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, model_id INTEGER, flagged INTEGER);
        """)
    with LOCK:
        if current() is None:
            with db() as conn:
                existing = conn.execute("SELECT id FROM models ORDER BY id LIMIT 1").fetchone()
            activate(existing["id"] if existing else register(baseline=True), bootstrap=True)


def log_run(meta, model_bytes):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or (STATE / "mlruns").as_uri())
    experiment = client.get_experiment_by_name("ScamShift")
    exp_id = experiment.experiment_id if experiment else client.create_experiment("ScamShift")
    run_id = client.create_run(exp_id, run_name=f"reviewed-{meta['reviewed_rows']}").info.run_id
    try:
        client.log_param(run_id, "classifier", "logistic_regression_word_char_tfidf")
        for field in ("threshold", "training_rows", "reviewed_rows", "dataset_sha256", "feedback_sha256"):
            client.log_param(run_id, field, meta[field])
        for cohort in ("benchmark", "campaign_audit"):
            for field, value in meta[cohort].items():
                client.log_metric(run_id, f"{cohort}_{field}", value)
        client.log_dict(run_id, meta, "evaluation.json")
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            artifact = Path(folder) / "classifier.pkl"
            artifact.write_bytes(model_bytes)
            client.log_artifact(run_id, str(artifact))
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id)
    return run_id


def current(conn=None):
    if conn is None:
        with db() as connection:
            return current(connection)
    row = conn.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
    return row["model_id"] if row else None


def register(baseline=False):
    with LOCK:
        with db() as conn:
            reviewed = [(row["text"], row["label"]) for row in conn.execute(
                "SELECT text,label FROM reviews WHERE label IN (0,1) ORDER BY id")]
        if not baseline and not reviewed:
            raise ValueError("Review at least one message before training.")
        model, meta = train([] if baseline else reviewed)
        raw = pickle.dumps(model)
        run_id = log_run(meta, raw)
        with db() as conn:
            version = conn.execute("INSERT INTO models(created,run_id,metrics,model) VALUES(?,?,?,?)",
                                   (now(), run_id, json.dumps(meta), raw)).lastrowid
            conn.execute("INSERT INTO events(created,kind,model_id) VALUES(?,?,?)", (now(), "Trained", version))
        return version


def quality(candidate, active):
    benchmark = candidate["benchmark"]
    audit = candidate["campaign_audit"]
    return [
        {"label": "Benchmark recall at least 85%", "pass": benchmark["recall"] >= .85},
        {"label": "Benchmark false alarms at most 3%", "pass": benchmark["false_alarm"] <= .03},
        {"label": "Campaign false alarms at most 25%", "pass": audit["false_alarm"] <= .25},
        {"label": "Campaign recall improves on active model", "pass": audit["recall"] > active["campaign_audit"]["recall"]},
    ]


def activate(version, rollback=False, bootstrap=False):
    with LOCK, db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = current(conn)
        row = conn.execute("SELECT metrics FROM models WHERE id=?", (version,)).fetchone()
        if not row:
            raise LookupError("Unknown model version.")
        if previous == version:
            raise ValueError("This model is already active.")
        if rollback:
            if not conn.execute("SELECT 1 FROM events WHERE model_id=? AND kind IN ('Promoted','Rolled back')",
                                (version,)).fetchone():
                raise ValueError("Rollback requires a previously deployed version.")
        elif not bootstrap:
            active = conn.execute("SELECT metrics FROM models WHERE id=?", (previous,)).fetchone()
            failed = [gate["label"] for gate in quality(json.loads(row["metrics"]), json.loads(active["metrics"]))
                      if not gate["pass"]]
            if failed:
                raise ValueError("Quality gates failed — " + ", ".join(failed))
        conn.execute("INSERT OR REPLACE INTO active(singleton,model_id) VALUES(1,?)", (version,))
        conn.execute("INSERT INTO events(created,kind,model_id) VALUES(?,?,?)",
                     (now(), "Rolled back" if rollback else "Promoted", version))


def load(version=None):
    with db() as conn:
        version = version if version is not None else current(conn)
        row = conn.execute("SELECT model,metrics FROM models WHERE id=?", (version,)).fetchone()
    if not row:
        raise LookupError("Unknown model version.")
    return version, pickle.loads(row["model"]), json.loads(row["metrics"])


def review(text, label):
    identity = key(text)
    if identity in reserved():
        raise ValueError("This message is reserved for evaluation.")
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO reviews(id,text,label,updated) VALUES(?,?,?,?)",
                     (identity, text, label, now()))


def status():
    with db() as conn:
        active = current(conn)
        versions = [
            {"id": row["id"], "run_id": row["run_id"], "created": row["created"],
             "metrics": json.loads(row["metrics"]), "deployed": bool(row["deployed"])}
            for row in conn.execute("""SELECT models.id, models.run_id, models.created, models.metrics,
                EXISTS(SELECT 1 FROM events WHERE model_id=models.id AND kind IN ('Promoted','Rolled back')) deployed
                FROM models ORDER BY id DESC""")
        ]
        events = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 6")]
        scans = conn.execute("SELECT COUNT(*) total, COALESCE(SUM(flagged),0) flagged FROM scans").fetchone()
        reviews = conn.execute("SELECT COUNT(*) FROM reviews WHERE label IS NOT NULL").fetchone()[0]
    active_meta = next(item["metrics"] for item in versions if item["id"] == active)
    for item in versions:
        item["gates"] = [] if item["id"] == active else quality(item["metrics"], active_meta)
    return {"active": active, "models": versions, "events": events, "scans": dict(scans), "reviewed": reviews}
