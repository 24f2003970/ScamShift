"""Assessment workflow: validate labelled CSV, monitor, retrain and record a gate.

Only load pickle artifacts from your own trusted ScamShift training runs.
"""
import argparse
import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from uuid import uuid4

from scamshift.datasets import examples
from scamshift.evaluation import quality
from scamshift.model import ROOT, corpus, eligible_feedback, evaluate, key, normalize, robust_normalize
from scamshift.store import fit_artifact, load, log_run


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate(path, *, batch_bytes=None):
    if batch_bytes is None:
        batch_bytes = path.read_bytes()
    with StringIO(batch_bytes.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {"id", "text", "label"}.issubset(fields):
            raise ValueError("Required CSV columns: id,text,label")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV contains duplicate column names")
        rows = list(reader)
    if not rows:
        raise ValueError("Incoming batch is empty")
    ids, texts = set(), set()
    for number, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f"Row {number}: too many CSV fields")
        if any(not (row.get(field) or "").strip() for field in ("id", "text", "label")):
            raise ValueError(f"Row {number}: missing required value")
        row.update({field: row[field].strip() for field in ("id", "text", "label")})
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Row {number}: label must be 0 (genuine) or 1 (spam/scam)")
        identity = key(row["text"])
        if row["id"] in ids or identity in texts:
            raise ValueError(f"Row {number}: duplicate ID or normalized message")
        if not eligible_feedback(row["text"]):
            raise ValueError(f"Row {number}: overlaps original training or protected evaluation data")
        ids.add(row["id"])
        texts.add(identity)
        row["label"] = int(row["label"])
    if {r["label"] for r in rows} != {0, 1}:
        raise ValueError("Batch must contain both classes for quality monitoring")
    return rows


def score(model, meta, pairs):
    preprocess = robust_normalize if meta.get("preprocessing") == "robust" else normalize
    return evaluate(model, pairs, meta["threshold"], preprocess=preprocess)


def gate(candidate, champion):
    checks = list(quality(candidate, champion))
    previous_f1 = champion["benchmark"]["f1"]
    candidate_f1 = candidate["benchmark"]["f1"]
    checks.append({"label": "Benchmark F1 does not decrease",
                   "pass": previous_f1 is not None and candidate_f1 is not None and candidate_f1 >= previous_f1,
                   "actual": [previous_f1, candidate_f1]})
    return {"decision": "eligible_for_promotion" if all(c["pass"] for c in checks) else "retain_champion",
            "checks": checks, "champion_overwritten": False,
            "justification": "Scam recall is primary because missed scams expose users to fraud. False-alarm limits protect genuine messages; benchmark F1 balances precision and recall on the imbalanced SMS corpus. Both models use identical untouched evaluation cohorts."}


def run(args):
    output = args.output_dir.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    try:
        batch_bytes = args.batch.read_bytes()
        rows = validate(args.batch, batch_bytes=batch_bytes)
    except (OSError, ValueError) as exc:
        save(output / "validation.json", {"valid": False, "error": str(exc)})
        raise
    validation = {"valid": True, "rows": len(rows), "columns": ["id", "text", "label"],
                  "missing_values": 0, "duplicate_records_or_ids": 0,
                  "target_distribution": {str(label): sum(r["label"] == label for r in rows) for label in (0, 1)},
                  "batch_sha256": hashlib.sha256(batch_bytes).hexdigest()}
    save(output / "validation.json", validation)
    print("Task A:", json.dumps(validation))
    if args.champion_dir:
        raw = (args.champion_dir / "classifier.pkl").read_bytes()
        champion_meta = json.loads((args.champion_dir / "evaluation.json").read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != champion_meta["artifact_sha256"]:
            raise ValueError("Champion checksum mismatch")
        champion = pickle.loads(raw)
        champion_id = str(args.champion_dir)
        champion_hash = champion_meta["artifact_sha256"]
    else:
        champion_id, champion, champion_meta = load()
        champion_hash = champion_meta.get("artifact_sha256")
    pairs = [(r["text"], r["label"]) for r in rows]
    incoming = score(champion, champion_meta, pairs)
    save(output / "existing_model_evaluation.json", incoming)
    print("Task B:", json.dumps(incoming))
    if champion_meta.get("preprocessing", "baseline") != "baseline":
        raise ValueError("This CLI supports baseline preprocessing champions; use the dashboard for robust models")
    candidate_raw, candidate_meta = fit_artifact(pairs, threshold=champion_meta["threshold"],
                                                weight=champion_meta["feedback_weight"])
    (output / "candidate.pkl").write_bytes(candidate_raw)
    candidate = pickle.loads(candidate_raw)
    # Re-evaluate BOTH artifacts on the SAME current untouched cohorts (never batch training scores).
    benchmark = corpus()[1]
    audit = [(r["text"], r["label"]) for r in examples("audit")]
    champion_current = dict(champion_meta, benchmark=score(champion, champion_meta, benchmark),
                            campaign_audit=score(champion, champion_meta, audit))
    decision = gate(candidate_meta, champion_current)
    save(output / "candidate_evaluation.json", candidate_meta)
    save(output / "quality_gate.json", decision)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "champion": champion_id,
              "champion_sha256": champion_hash, "validation": validation,
              "existing_model_new_batch": incoming, "champion_comparison": champion_current,
              "candidate_comparison": candidate_meta, "quality_gate": decision,
              "combined_training_rows": candidate_meta["training_rows"] + candidate_meta["reviewed_rows"],
              "candidate_batch_training_metrics": score(candidate, candidate_meta, pairs),
              "note": "Batch metrics for the candidate are training diagnostics, never promotion evidence. Demo batches are authored fixtures, not production observations."}
    save(output / "report.json", report)
    client, run_id = log_run(candidate_meta, candidate_raw)
    report["mlflow_run_id"] = run_id
    try:
        client.log_dict(run_id, report, "monitoring_report.json")
        client.set_tag(run_id, "assessment.quality_gate", decision["decision"])
        client.set_tag(run_id, "scamshift.registry_state", "assessment_only")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    save(output / "report.json", report)
    lines = ["# ScamShift monitoring and retraining execution", "", f"MLflow run: `{run_id}`",
             f"Validated {len(rows)} incoming labelled rows; combined training rows: {report['combined_training_rows']}.",
             "", "| Metric | Existing model on new batch | Champion benchmark | Candidate benchmark |",
             "|---|---:|---:|---:|"]
    for metric in ("accuracy", "precision", "recall", "f1"):
        values = [incoming[metric], champion_current['benchmark'][metric], candidate_meta['benchmark'][metric]]
        lines.append("| " + metric + " | " + " | ".join("undefined" if v is None else f"{v:.4f}" for v in values) + " |")
    lines.extend(["", f"Decision: **{decision['decision']}**. Champion unchanged.", "", decision["justification"], ""])
    lines.extend(f"- {'PASS' if c['pass'] else 'FAIL'}: {c['label']} ({c['actual']})" for c in decision["checks"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Task C: candidate saved:", output / "candidate.pkl", "MLflow run:", run_id)
    print("Task D:", decision["decision"], "(champion unchanged)")
    print("Task E: evidence:", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--champion-dir", type=Path, help="Trusted train.py output; omit to read deployed SQLite champion")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "monitoring")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
