"""Train a standalone ScamShift baseline without touching deployment state."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scamshift.model import ROOT
from scamshift.store import fit_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "baseline")
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, report = fit_artifact([])
    with TemporaryDirectory(dir=output) as folder:
        artifact = Path(folder) / "classifier.pkl"
        evaluation = Path(folder) / "evaluation.json"
        artifact.write_bytes(model)
        evaluation.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifact.replace(output / artifact.name)
        evaluation.replace(output / evaluation.name)
    print(f"Model: {output / artifact.name}")
    print(f"Evaluation: {output / evaluation.name}")
    for name in ("benchmark", "campaign_audit", "validation"):
        metrics = report[name]
        print(f"{name}: recall={metrics['recall']:.1%}, false alarms={metrics['false_alarm']:.1%}")
    print(f"Threshold: {report['threshold']}")


if __name__ == "__main__":
    main()
