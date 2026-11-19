"""Binary classification metrics, with undefined rates represented by None."""
from math import isfinite, sqrt


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, count):
    if not count:
        return None
    p = successes / count
    z = 1.96
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    radius = z * sqrt((p * (1 - p) + z * z / (4 * count)) / count) / divisor
    return [max(0, center - radius), min(1, center + radius)]


def metrics(labels, scores, threshold):
    samples = [(int(label), float(score)) for label, score in zip(labels, scores, strict=True)]
    tp = sum(label == 1 and score >= threshold for label, score in samples)
    fp = sum(label == 0 and score >= threshold for label, score in samples)
    positives = sum(label for label, _ in samples)
    negatives = len(samples) - positives
    fn, tn = positives - tp, negatives - fp
    # Group tied scores before calculating threshold-independent ranking metrics.
    score_groups = {}
    for label, score in samples:
        group = score_groups.setdefault(score, [0, 0])
        group[label] += 1
    seen_tp = seen_fp = average_precision = auc_pairs = 0
    for score in sorted(score_groups, reverse=True):
        group_neg, group_pos = score_groups[score]
        auc_pairs += (seen_tp + group_pos / 2) * group_neg
        seen_tp += group_pos
        seen_fp += group_neg
        average_precision += group_pos * seen_tp / (seen_tp + seen_fp)
    return {
        "accuracy": ratio(tp + tn, len(samples)),
        "recall": ratio(tp, positives), "recall_ci95": wilson(tp, positives),
        "false_alarm": ratio(fp, negatives), "false_alarm_ci95": wilson(fp, negatives),
        "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "average_precision": ratio(average_precision, positives) if negatives else None,
        "roc_auc": ratio(auc_pairs, positives * negatives),
        "caught": tp, "missed": fn, "false_flags": fp, "genuine": tn,
        "positives": positives, "negatives": negatives, "total": len(samples),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def grouped_indices(rows, field):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row[field], []).append(index)
    return groups


def sliced_metrics(rows, scores, threshold, field):
    return {
        name: metrics([rows[i]["label"] for i in indices], [scores[i] for i in indices], threshold)
        for name, indices in grouped_indices(rows, field).items()
    }


def quality(candidate, active):
    cohorts = [meta.get(name, {}) for meta in (candidate, active) for name in ("benchmark", "campaign_audit")]
    valid = all(
        all(isinstance(cohort.get(rate), (int, float)) and isfinite(cohort[rate])
            and 0 <= cohort[rate] <= 1 for rate in ("recall", "false_alarm"))
        and all(isinstance(cohort.get(count), int) and cohort[count] >= 0
                for count in ("caught", "missed", "false_flags", "genuine"))
        and cohort["caught"] + cohort["missed"] >= 10
        and cohort["false_flags"] + cohort["genuine"] >= 10
        for cohort in cohorts
    )
    if not valid:
        return [{"label": "Both models have valid metrics and at least 10 examples per class per benchmark",
                 "pass": False, "actual": "Metrics unavailable, invalid, or sample minimum unmet"}]
    benchmark, audit, previous_benchmark, previous_audit = cohorts
    recall_loss = previous_benchmark["recall"] - benchmark["recall"]
    alarm_increase = benchmark["false_alarm"] - previous_benchmark["false_alarm"]
    meets_campaign_tradeoff = (
        audit["recall"] > previous_audit["recall"]
        or (audit["recall"] == previous_audit["recall"]
            and audit["false_alarm"] < previous_audit["false_alarm"])
    )
    return [
        {"label": "Benchmark and campaign datasets match the active version",
         "pass": all(not (candidate.get(field) and active.get(field)) or candidate[field] == active[field]
                     for field in ("dataset_sha256", "campaign_sha256")),
         "actual": "Compared available fingerprints"},
        {"label": "Benchmark recall at least 85%", "pass": benchmark["recall"] >= .85,
         "actual": f"{benchmark['recall']:.1%}"},
        {"label": "Benchmark false alarms at most 3%", "pass": benchmark["false_alarm"] <= .03,
         "actual": f"{benchmark['false_alarm']:.1%}"},
        {"label": "Benchmark recall loss at most 2 percentage points", "pass": recall_loss <= .02 + 1e-9,
         "actual": f"{recall_loss * 100:+.2f} pp"},
        {"label": "Benchmark false-alarm increase at most 1 percentage point", "pass": alarm_increase <= .01 + 1e-9,
         "actual": f"{alarm_increase * 100:+.2f} pp"},
        {"label": "Campaign false alarms at most 25%", "pass": audit["false_alarm"] <= .25,
         "actual": f"{audit['false_alarm']:.1%}"},
        {"label": "Campaign recall improves, or recall holds and false alarms fall", "pass": meets_campaign_tradeoff,
         "actual": f"Recall {previous_audit['recall']:.1%} to {audit['recall']:.1%}, false alarms "
                   f"{previous_audit['false_alarm']:.1%} to {audit['false_alarm']:.1%}"},
    ]
