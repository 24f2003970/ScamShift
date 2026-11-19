"""Local model registry, review labels and MLflow experiment tracking."""
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from threading import RLock
from uuid import uuid4

from . import registry
from .drift import bucket, compare, input_features
from .evaluation import quality
from .model import ROOT, THRESHOLD, FEEDBACK_WEIGHT, eligible_feedback, key, reserved, train, training_keys

STATE = Path(os.environ.get("SCAMSHIFT_STATE", ROOT / "runtime")).resolve()
LOCK = RLock()
LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 3


class ArtifactError(ValueError):
    pass


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
        old_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if old_version > SCHEMA_VERSION:
            raise RuntimeError("This database was created by a newer ScamShift version.")
        if old_version < SCHEMA_VERSION and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='models'"
        ).fetchone():
            backup = STATE / f"scamshift-pre-v{SCHEMA_VERSION}.db"
            if not backup.exists():
                with closing(sqlite3.connect(backup)) as target:
                    conn.backup(target)
        conn.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS models(
                id INTEGER PRIMARY KEY, created TEXT, run_id TEXT, metrics TEXT, model BLOB);
            CREATE TABLE IF NOT EXISTS active(singleton INTEGER PRIMARY KEY CHECK(singleton=1), model_id INTEGER);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created TEXT, kind TEXT, model_id INTEGER);
            CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, text TEXT, label INTEGER, updated TEXT);
            CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, model_id INTEGER, flagged INTEGER);
            CREATE TABLE IF NOT EXISTS training_jobs(
                id INTEGER PRIMARY KEY, state TEXT NOT NULL, created TEXT NOT NULL,
                finished TEXT, version INTEGER, detail TEXT);
            CREATE TABLE IF NOT EXISTS review_events(
                id INTEGER PRIMARY KEY, review_id TEXT NOT NULL, label INTEGER, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scan_features(
                id INTEGER PRIMARY KEY, created TEXT NOT NULL, model_id INTEGER NOT NULL,
                features TEXT NOT NULL, score TEXT NOT NULL, flagged TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS drift_reports(
                id INTEGER PRIMARY KEY, created TEXT NOT NULL, report TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS registry_instance(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS mlflow_versions(
                scope TEXT NOT NULL, model_id INTEGER NOT NULL, registry_run TEXT NOT NULL,
                registry_version TEXT NOT NULL, PRIMARY KEY(scope,model_id));
            CREATE TABLE IF NOT EXISTS mlflow_sync(
                scope TEXT PRIMARY KEY, detail TEXT);
            CREATE INDEX IF NOT EXISTS scan_features_created ON scan_features(created);
        """)
        model_columns = {row["name"] for row in conn.execute("PRAGMA table_info(models)")}
        if "digest" not in model_columns:
            conn.execute("ALTER TABLE models ADD COLUMN digest TEXT")
        scan_columns = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
        if "created" not in scan_columns:
            conn.execute("ALTER TABLE scans ADD COLUMN created TEXT")
        if "latency_ms" not in scan_columns:
            conn.execute("ALTER TABLE scans ADD COLUMN latency_ms REAL")
        job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(training_jobs)")}
        if "phase" not in job_columns:
            conn.execute("ALTER TABLE training_jobs ADD COLUMN phase TEXT")
        if "retry_of" not in job_columns:
            conn.execute("ALTER TABLE training_jobs ADD COLUMN retry_of INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS scans_created ON scans(created)")
        conn.execute("CREATE INDEX IF NOT EXISTS review_events_review_id ON review_events(review_id)")
        conn.execute("INSERT OR IGNORE INTO registry_instance VALUES(1,?)", (uuid4().hex,))
        conn.execute(
            "UPDATE training_jobs SET state='interrupted', phase='interrupted', finished=?, detail=? WHERE state='training'",
            (now(), "Server restarted before training completed. Start a new run to retry."),
        )
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    load_model.cache_clear()
    with LOCK:
        try:
            if current() is None:
                with db() as conn:
                    existing = conn.execute("SELECT id FROM models ORDER BY id LIMIT 1").fetchone()
                activate(existing["id"] if existing else register(baseline=True), bootstrap=True)
            elif registry.enabled():
                reconcile_registry()
        except (registry.RegistryError, ArtifactError):
            LOGGER.error("MLflow champion synchronization failed. Check registry status and retry.")
        try:
            load()
        except (LookupError, ArtifactError):
            LOGGER.error("Active artifact unavailable. Restore a previously deployed model in the registry.")


def record_scan(model_id, text, score, flagged, band, latency_ms):
    created = now()
    features = json.dumps({**input_features(text), "score_band": band})
    with db() as conn:
        conn.execute("INSERT INTO scans(model_id,flagged,created,latency_ms) VALUES(?,?,?,?)",
                     (model_id, flagged, created, latency_ms))
        conn.execute("INSERT INTO scan_features(created,model_id,features,score,flagged) VALUES(?,?,?,?,?)",
                     (created, model_id, features, bucket(score, (.2, .4, .6, .8)), "Yes" if flagged else "No"))


def save_drift_report(days, minimum):
    end = datetime.now(timezone.utc).replace(microsecond=0)
    midpoint = end - timedelta(days=days)
    start = midpoint - timedelta(days=days)
    with db() as conn:
        conn.execute("BEGIN")
        active = current(conn)
        rows = [dict(row) for row in conn.execute(
            "SELECT created,model_id,features,score,flagged FROM scan_features WHERE created >= ? AND created <= ?",
            (start.isoformat(), end.isoformat()),
        )]
    reference, recent = [], []
    for row in rows:
        row["features"] = json.loads(row["features"])
        (reference if row["created"] < midpoint.isoformat() else recent).append(row)
    report = {
        "created": end.isoformat(), "reference_start": start.isoformat(), "current_start": midpoint.isoformat(),
        "active": active, "minimum": minimum, "reference_count": len(reference), "current_count": len(recent),
        "features": compare(reference, recent, active=active, minimum=minimum),
    }
    with db() as conn:
        report["id"] = conn.execute("INSERT INTO drift_reports(created,report) VALUES(?,?)",
                                    (report["created"], json.dumps(report))).lastrowid
    return report


def drift_history():
    with db() as conn:
        return [{**json.loads(row["report"]), "id": row["id"]} for row in conn.execute(
            "SELECT id,report FROM drift_reports ORDER BY id DESC LIMIT 15"
        )]


def tracking_uri():
    return os.environ.get("MLFLOW_TRACKING_URI") or (STATE / "mlruns").as_uri()


def registry_scope():
    uri = tracking_uri()
    return json.dumps([uri, os.environ.get("MLFLOW_REGISTRY_URI") or uri, registry.NAME])


def log_run(meta, model_bytes, job_id=None, original_run_id=None):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    client = registry.client_for(tracking_uri())
    experiment = client.get_experiment_by_name("ScamShift")
    exp_id = experiment.experiment_id if experiment else client.create_experiment("ScamShift")
    tags = {"scamshift.registry_state": "pending"}
    if job_id is not None:
        tags["scamshift.job_id"] = str(job_id)
    if original_run_id:
        tags["scamshift.original_run_id"] = original_run_id
    run_id = client.create_run(exp_id, run_name=f"reviewed-{meta['reviewed_rows']}", tags=tags).info.run_id
    try:
        client.log_param(run_id, "classifier", "logistic_regression_word_char_tfidf")
        for field in ("threshold", "feedback_weight", "training_rows", "reviewed_rows", "validation_rows",
                      "dataset_sha256", "feedback_sha256", "campaign_sha256", "validation_sha256",
                      "code_sha256", "artifact_sha256"):
            if field in meta:
                client.log_param(run_id, field, meta[field])
        for field, value in meta.get("hyperparameters", {}).items():
            client.log_param(run_id, f"classifier_{field}", value)
        if "training_seconds" in meta:
            client.log_metric(run_id, "training_seconds", meta["training_seconds"])
        for cohort in ("benchmark", "campaign_audit", "validation"):
            for field, value in meta.get(cohort, {}).items():
                if isinstance(value, (int, float)):
                    client.log_metric(run_id, f"{cohort}_{field}", value)
        client.log_dict(run_id, meta, "evaluation.json")
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            artifact = Path(folder) / "classifier.pkl"
            artifact.write_bytes(model_bytes)
            client.log_artifact(run_id, str(artifact))
        if registry.enabled() or has_flavor_dependencies():
            registry.log_flavor(client, run_id, pickle.loads(model_bytes), meta)
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id)
    return client, run_id


def has_flavor_dependencies():
    from importlib.util import find_spec

    return find_spec("pandas") is not None


def mark_registry_state(client, run_id, state, version=None):
    try:
        client.set_tag(run_id, "scamshift.registry_state", state)
        if version is not None:
            client.set_tag(run_id, "scamshift.model_version", str(version))
        if state == "unregistered":
            client.set_terminated(run_id, "FAILED")
    except Exception as exc:
        LOGGER.warning("Could not update registry state for MLflow run %s (%s)", run_id, type(exc).__name__)


def current(conn=None):
    if conn is None:
        with db() as connection:
            return current(connection)
    row = conn.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
    return row["model_id"] if row else None


def review_snapshot(conn):
    return [(row["text"], row["label"]) for row in conn.execute(
        "SELECT text,label FROM reviews WHERE label IN (0,1,2) ORDER BY id")
        if eligible_feedback(row["text"])]


def register(baseline=False, reviewed=None, job_id=None):
    if baseline:
        reviewed = []
    elif reviewed is None:
        with db() as conn:
            reviewed = review_snapshot(conn)
    if not baseline and not {0, 1}.issubset(label for _, label in reviewed):
        raise ValueError("Save at least one Scam and one Genuine review before training.")
    threshold = THRESHOLD if baseline else float(os.environ.get("SCAMSHIFT_THRESHOLD", THRESHOLD))
    weight = FEEDBACK_WEIGHT if baseline else float(os.environ.get("SCAMSHIFT_FEEDBACK_WEIGHT", FEEDBACK_WEIGHT))
    if not 0 < threshold < 1 or not 0 < weight <= 100:
        raise ValueError("Training configuration is outside the supported range.")
    bootstrap_path = os.environ.get("SCAMSHIFT_BOOTSTRAP_DIR") if baseline else None
    raw, meta = (read_baseline(Path(bootstrap_path)) if bootstrap_path
                 else fit_artifact(reviewed, threshold=threshold, weight=weight))
    if job_id is not None:
        with db() as conn:
            conn.execute("UPDATE training_jobs SET phase='tracking' WHERE id=?", (job_id,))
    try:
        client, run_id = log_run(meta, raw, job_id)
    except Exception as exc:
        if not registry.enabled():
            raise
        if baseline:
            update_registry_sync("Could not log the baseline to the MLflow tracking server.")
        raise registry.RegistryError("Could not log the model to the MLflow tracking server.") from exc
    try:
        with LOCK, db() as conn:
            version = conn.execute(
                "INSERT INTO models(created,run_id,metrics,model,digest) VALUES(?,?,?,?,?)",
                (now(), run_id, json.dumps(meta, allow_nan=False), raw, meta["artifact_sha256"]),
            ).lastrowid
            if registry.enabled():
                ensure_registry_version(conn, version, client, run_id)
            conn.execute("INSERT INTO events(created,kind,model_id) VALUES(?,?,?)", (now(), "Trained", version))
            if job_id is not None:
                conn.execute("UPDATE training_jobs SET state='finished', phase='completed', finished=?, version=? WHERE id=?",
                             (now(), version, job_id))
    except Exception as exc:
        mark_registry_state(client, run_id, "unregistered")
        if registry.enabled():
            if baseline:
                update_registry_sync("Could not register the baseline in MLflow.")
            raise registry.RegistryError("Could not register the model in MLflow.") from exc
        raise
    mark_registry_state(client, run_id, "registered", version)
    return version


def fit_artifact(reviewed, threshold=THRESHOLD, weight=FEEDBACK_WEIGHT):
    model, meta = train(reviewed, threshold=threshold, feedback_weight=weight)
    raw = pickle.dumps(model)
    digest = hashlib.sha256(raw).hexdigest()
    code = hashlib.sha256()
    for source in sorted((ROOT / "scamshift").glob("*.py")):
        code.update(source.name.encode() + b"\0" + source.read_bytes())
    meta.update(
        artifact_sha256=digest, code_sha256=code.hexdigest(),
        environment={
            "python": sys.version.split()[0],
            **{name: package_version(name) for name in ("scikit-learn", "numpy", "scipy", "mlflow-skinny")},
        },
    )
    return raw, meta


def read_baseline(folder):
    try:
        raw = (folder / "classifier.pkl").read_bytes()
        meta = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        if (hashlib.sha256(raw).hexdigest() != meta["artifact_sha256"]
                or meta["reviewed_rows"] != 0 or meta["threshold"] != THRESHOLD):
            raise ValueError("Baseline artifact or metadata failed validation.")
        model = pickle.loads(raw)
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("Baseline artifact is not a fitted classifier.")
    except (OSError, ValueError, TypeError, KeyError, pickle.UnpicklingError, EOFError, ImportError,
            AttributeError) as exc:
        raise ArtifactError("Could not load the trusted baseline artifact and its evaluation report.") from exc
    return raw, meta


def start_training():
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM training_jobs WHERE state='training'").fetchone():
            raise ValueError("Training is already in progress.")
        reviewed = review_snapshot(conn)
        if not {0, 1}.issubset(label for _, label in reviewed):
            raise ValueError("Save at least one Scam and one Genuine review before training.")
        previous = conn.execute("SELECT id,state FROM training_jobs ORDER BY id DESC LIMIT 1").fetchone()
        retry_of = previous["id"] if previous and previous["state"] in {"failed", "interrupted"} else None
        job_id = conn.execute(
            "INSERT INTO training_jobs(state,phase,created,retry_of) VALUES('training',?,?,?)",
            ("retrying" if retry_of else "queued", now(), retry_of),
        ).lastrowid
    return job_id, reviewed


def fail_training(job_id):
    with db() as conn:
        conn.execute("UPDATE training_jobs SET state='failed', phase='failed', finished=?, detail=? WHERE id=? AND state='training'",
                     (now(), "Training failed. Check the server logs and retry.", job_id))


def run_training(job_id, reviewed):
    try:
        with db() as conn:
            conn.execute("UPDATE training_jobs SET phase='running' WHERE id=?", (job_id,))
        register(reviewed=reviewed, job_id=job_id)
    except Exception as exc:
        LOGGER.error("Training job %s failed (%s)", job_id, type(exc).__name__)
        fail_training(job_id)


def training_status(job_id=None):
    query = "SELECT * FROM training_jobs"
    with db() as conn:
        row = (conn.execute(query + " ORDER BY id DESC LIMIT 1").fetchone() if job_id is None
               else conn.execute(query + " WHERE id=?", (job_id,)).fetchone())
    return dict(row) if row else None


def activate(version, rollback=False, bootstrap=False):
    with LOCK:
        # Verify the target before changing the active pointer, including on a cache hit.
        model, metadata = read_artifact(version)
        try:
            return activate_verified(version, rollback, bootstrap, model, metadata)
        except registry.RegistryError:
            update_registry_sync("Champion update failed. SQLite still serves the committed model. Retry or reconcile.")
            raise


def activate_verified(version, rollback, bootstrap, model, metadata):
    with db() as conn:
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
        if not registry.enabled():
            commit_deployment(conn, version, rollback)
            return
        conn.execute("INSERT OR REPLACE INTO mlflow_sync VALUES(?,?)",
                     (registry_scope(), "Champion deployment in progress."))
    try:
        with db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if current(conn) != previous:
                raise registry.RegistryError("Active model changed during deployment. Retry.")
            mirror_alias(conn, version, model, metadata)
            commit_deployment(conn, version, rollback)
    except sqlite3.Error as exc:
        raise registry.RegistryError("Could not commit the champion deployment to SQLite.") from exc


def commit_deployment(conn, version, rollback):
    conn.execute("INSERT OR REPLACE INTO active(singleton,model_id) VALUES(1,?)", (version,))
    conn.execute("INSERT INTO events(created,kind,model_id) VALUES(?,?,?)",
                 (now(), "Rolled back" if rollback else "Promoted", version))


def ensure_registry_version(conn, version, client, run_id=None):
    scope = registry_scope()
    mapping = conn.execute("SELECT * FROM mlflow_versions WHERE scope=? AND model_id=?",
                           (scope, version)).fetchone()
    if mapping:
        return mapping
    row = conn.execute("SELECT run_id,model,metrics FROM models WHERE id=?", (version,)).fetchone()
    state_id = conn.execute("SELECT id FROM registry_instance WHERE singleton=1").fetchone()["id"]
    digest = hashlib.sha256(row["model"]).hexdigest()
    identity = registry.identity(state_id, version, digest)
    if run_id is None:
        existing = registry.find_version(client, identity)
        if existing is not None:
            registered = existing
        else:
            metadata = {**json.loads(row["metrics"]), "artifact_sha256": digest}
            _, run_id = log_run(metadata, row["model"], original_run_id=row["run_id"])
            registered = registry.register(client, run_id, identity)
    else:
        registered = registry.register(client, run_id, identity)
    mapping = {"registry_run": registered.run_id, "registry_version": str(registered.version)}
    conn.execute("INSERT INTO mlflow_versions VALUES(?,?,?,?)",
                 (scope, version, mapping["registry_run"], mapping["registry_version"]))
    return mapping


def mirror_alias(conn, version, model, metadata):
    try:
        client = registry.client_for(tracking_uri())
        mapping = ensure_registry_version(conn, version, client)
        row = conn.execute("SELECT model FROM models WHERE id=?", (version,)).fetchone()
        metadata = {**metadata, "artifact_sha256": hashlib.sha256(row["model"]).hexdigest()}
        registry.verify(client, mapping, model, metadata)
        registry.set_alias(client, mapping["registry_version"])
        conn.execute("DELETE FROM mlflow_sync WHERE scope=?", (registry_scope(),))
    except Exception as exc:
        raise registry.RegistryError("MLflow champion could not be synchronized.") from exc


def update_registry_sync(detail):
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO mlflow_sync VALUES(?,?)", (registry_scope(), detail))


def reconcile_registry():
    if not registry.enabled():
        raise ValueError("MLflow registry mode is disabled.")
    with LOCK:
        try:
            version = current()
            if version is None:
                with db() as conn:
                    existing = conn.execute("SELECT id FROM models ORDER BY id LIMIT 1").fetchone()
                version = existing["id"] if existing else register(baseline=True)
                activate(version, bootstrap=True)
                return version
            model, metadata = read_artifact(version)
            update_registry_sync("Champion reconciliation in progress.")
            with db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if current(conn) != version:
                    raise registry.RegistryError("Active model changed during reconciliation. Retry.")
                mirror_alias(conn, version, model, metadata)
            return version
        except (registry.RegistryError, ArtifactError, LookupError, sqlite3.Error) as exc:
            update_registry_sync("Champion reconciliation failed. Check the MLflow server and retry.")
            raise registry.RegistryError("MLflow champion could not be reconciled.") from exc


def load(version=None):
    version = current() if version is None else version
    model, meta = load_model(version)
    return version, model, meta


@lru_cache(maxsize=4)
def load_model(version):
    return read_artifact(version)


def read_artifact(version):
    with db() as conn:
        row = conn.execute("SELECT model,metrics,digest FROM models WHERE id=?", (version,)).fetchone()
    if not row:
        raise LookupError("Unknown model version.")
    raw = row["model"]
    if not isinstance(raw, bytes) or not raw:
        raise ArtifactError("Stored model artifact is missing.")
    if row["digest"] and hashlib.sha256(raw).hexdigest() != row["digest"]:
        raise ArtifactError("Stored model failed the integrity check.")
    try:
        return pickle.loads(raw), json.loads(row["metrics"])
    except (pickle.UnpicklingError, EOFError, ImportError, ValueError, AttributeError) as exc:
        raise ArtifactError("Stored model could not be loaded.") from exc


def review(text, label):
    identity = key(text)
    if identity in reserved():
        raise ValueError("This message is reserved for evaluation.")
    if identity in training_keys():
        raise ValueError("This message is already in the base training corpus.")
    if not eligible_feedback(text):
        raise ValueError("This message is too similar to an evaluation example.")
    updated = now()
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO reviews(id,text,label,updated) VALUES(?,?,?,?)",
                     (identity, text, label, updated))
        conn.execute("INSERT INTO review_events(review_id,label,created) VALUES(?,?,?)", (identity, label, updated))
    return updated


def review_status(ids):
    placeholders = ",".join("?" for _ in ids)
    with db() as conn:
        conn.execute("BEGIN")
        statuses = {row["id"]: {**dict(row), "history": []} for row in conn.execute(
            f"SELECT id,label,updated FROM reviews WHERE id IN ({placeholders})", ids)}
        changes = conn.execute(
            "SELECT review_id,label,created FROM (SELECT review_id,label,created,"
            " ROW_NUMBER() OVER (PARTITION BY review_id ORDER BY id DESC) AS rank"
            f" FROM review_events WHERE review_id IN ({placeholders})) WHERE rank <= 3"
            " ORDER BY review_id, rank", ids)
        for row in changes:
            if row["review_id"] in statuses:
                statuses[row["review_id"]]["history"].append({"label": row["label"], "created": row["created"]})
    return statuses


def status():
    with db() as conn:
        conn.execute("BEGIN")
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
        labels = [label for _, label in review_snapshot(conn)]
        recent = [dict(row) for row in conn.execute(
            "SELECT model_id, COUNT(*) scans, SUM(flagged) flagged, AVG(latency_ms) latency_ms "
            "FROM scans WHERE created >= strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-7 days') "
            "GROUP BY model_id ORDER BY model_id DESC"
        )]
        registry_status = {"mode": "local"}
        if registry.enabled():
            scope = registry_scope()
            mappings = {row["model_id"]: dict(row) for row in conn.execute(
                "SELECT model_id,registry_run,registry_version FROM mlflow_versions WHERE scope=?", (scope,))}
            problem = conn.execute("SELECT detail FROM mlflow_sync WHERE scope=?", (scope,)).fetchone()
            active_mapping = mappings.get(active)
            registry_status = {
                "mode": "mlflow", "name": registry.NAME, "alias": registry.ALIAS,
                "sync": "needs_sync" if problem or not active_mapping else "confirmed",
                "detail": problem["detail"] if problem else None,
                "committed_registry_version": active_mapping["registry_version"] if active_mapping else None,
            }
            for item in versions:
                item["mlflow"] = mappings.get(item["id"])
    active_meta = next((item["metrics"] for item in versions if item["id"] == active), {})
    for item in versions:
        item["gates"] = [] if item["id"] == active else quality(item["metrics"], active_meta)
    return {"active": active, "models": versions, "events": events, "scans": dict(scans),
            "reviewed": len(labels), "review_counts": {"scam": labels.count(1), "genuine": labels.count(0),
                                                       "promotional_spam": labels.count(2)},
            "recent_scans": recent, "mlflow_registry": registry_status}
