"""Report final synthetic holdout metrics without using the set for promotion."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scamshift import store
from scamshift.datasets import DATA, examples
from scamshift.evaluation import metrics, sliced_metrics
from scamshift.model import probabilities


def format_markdown(report):
    lines = [f"# ScamShift {report['cohort']} evaluation", "", f"Evaluated at {report['evaluated_at']}.",
             f"Dataset SHA-256 `{report['dataset_sha256']}`.",
             f"Baseline version v{report['baseline']}."]
    for version, model in report["models"].items():
        lines.extend(["", f"## Version {version}", "", f"Decision threshold {model['threshold']:.2f}.", "",
                      "| Slice | N | TP | FP | FN | TN | Precision | Recall | False alarms | F1 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        cohorts = {"Overall": model["overall"], **model["by_campaign"], **model["by_script"]}
        for name, result in cohorts.items():
            counts = [result["total"], *(result["confusion"][key] for key in ("tp", "fp", "fn", "tn"))]
            rates = ["—" if result[key] is None else f"{result[key]:.1%}"
                     for key in ("precision", "recall", "false_alarm", "f1")]
            lines.append("| " + " | ".join(map(str, (name, *counts, *rates))) + " |")
        delta = model["vs_baseline"]
        overall = model["overall"]
        rank_metrics = ", ".join(f"{label} {overall[key]:.3f}" for key, label in
                                 (("accuracy", "accuracy"), ("average_precision", "average precision"),
                                  ("roc_auc", "ROC AUC")) if overall[key] is not None)
        lines.extend(["", f"Overall {rank_metrics}." if rank_metrics else "Overall ranking metrics are unavailable.",
                      f"Compared with baseline, caught {delta['caught']:+d} scams and produced "
                      f"{delta['false_flags']:+d} false flags."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("versions", nargs="+", type=int, help="Model version IDs")
    parser.add_argument("--cohort", choices=("holdout", "robustness"), default="holdout")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    rows = examples(args.cohort)
    texts, labels = [row["text"] for row in rows], [row["label"] for row in rows]
    models = {}
    for version in args.versions:
        _, model, meta = store.load(version)
        threshold = meta["threshold"]
        scores = probabilities(model, texts)
        models[version] = {
            "threshold": threshold,
            "overall": metrics(labels, scores, threshold),
            "by_campaign": sliced_metrics(rows, scores, threshold, "campaign"),
            "by_script": sliced_metrics(rows, scores, threshold, "script"),
        }
    baseline = args.versions[0]
    for model in models.values():
        model["vs_baseline"] = {name: model["overall"][name] - models[baseline]["overall"][name]
                                for name in ("caught", "missed", "false_flags")}
    digest = hashlib.sha256((DATA / f"campaign_{args.cohort}.json").read_bytes()).hexdigest()
    report = {"dataset_sha256": digest, "cohort": args.cohort, "baseline": baseline, "models": models,
              "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json_out:
        with args.json_out.open("x", encoding="utf-8") as target:
            target.write(output)
    if args.markdown_out:
        with args.markdown_out.open("x", encoding="utf-8") as target:
            target.write(format_markdown(report))
    print(output, end="")


if __name__ == "__main__":
    main()
