"""Compare operating points using the separate validation campaign families."""
import json

from scamshift.datasets import DATA, examples, fingerprint
from scamshift.evaluation import metrics, sliced_metrics
from scamshift.model import probabilities, train
from scamshift.text_features import normalize, robust_normalize


def main():
    feedback = [(row["text"], row["label"]) for row in examples("review")]
    validation = examples("validation")
    messages = [row["text"] for row in validation]
    labels = [row["label"] for row in validation]
    results = []
    for robust, preprocess in ((False, normalize), (True, robust_normalize)):
        for weight in (1, 6, 12, 18):
            model, meta = train(feedback, feedback_weight=weight, robust=robust, promotion_metrics=False)
            scores = probabilities(model, messages, preprocess=preprocess)
            for threshold in (.4, .5, .6):
                results.append({
                    "feedback_weight": weight, "threshold": threshold, "preprocessing": meta["preprocessing"],
                    "training_seconds": meta["training_seconds"], "validation": metrics(labels, scores, threshold),
                    "by_campaign": sliced_metrics(validation, scores, threshold, "campaign"),
                })
    # Only the existing preprocessing is deployable via the current model registry.
    eligible = [row for row in results if row["preprocessing"] == "baseline" and row["validation"]["false_alarm"] <= .25]
    best = max(eligible, key=lambda row: (row["validation"]["f1"], -row["validation"]["false_alarm"],
                                        row["validation"]["recall"], -abs(row["threshold"] - .5)), default=None)
    print(json.dumps({"validation_sha256": fingerprint((DATA / "campaign_validation.json").read_bytes()),
                      "suggested_operating_point": best, "configurations": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
