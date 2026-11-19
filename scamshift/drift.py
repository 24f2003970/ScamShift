"""Compare adjacent traffic windows using Jensen–Shannon divergence (base 2)."""
from collections import Counter
from math import log2, sqrt
import re
import unicodedata

from .text_features import SPACED_CREDENTIAL, URL, normalize, script

FEATURES = {
    "length": "Message length", "words": "Word count", "script": "Writing script",
    "url": "URL present", "digits": "Digit share", "punctuation": "Punctuation and symbol share",
    "unicode_format": "Unicode formatting", "unusual_spacing": "Credential spacing",
    "repeated_characters": "Repeated characters",
}
PREDICTIONS = {"score": "Score distribution", "flagged": "Flagged classification"}


def bucket(value, bounds):
    lower = None
    for bound in bounds:
        if value < bound:
            return f"< {bound:g}" if lower is None else f"{lower:g} to < {bound:g}"
        lower = bound
    return f"≥ {bounds[-1]:g}"


def priority_band(score, threshold):
    if score >= threshold:
        return "Flagged"
    if score >= max(0, threshold - .10):
        return "Near threshold"
    return "Below threshold"


def input_features(text):
    categories = [unicodedata.category(char) for char in text]
    length = max(len(text), 1)
    return {
        "length": bucket(len(text), (40, 80, 160, 320)),
        "words": bucket(len(text.split()), (5, 10, 20, 40)),
        "script": script(text),
        "url": "Yes" if URL.search(text) else "No",
        "digits": bucket(sum(char.isdigit() for char in text) / length, (.01, .05, .15)),
        "punctuation": bucket(sum(cat.startswith(("P", "S")) for cat in categories) / length, (.01, .05, .15)),
        "unicode_format": "Yes" if "Cf" in categories or unicodedata.normalize("NFKC", text) != text else "No",
        "unusual_spacing": "Yes" if SPACED_CREDENTIAL.search(normalize(text)) else "No",
        "repeated_characters": "Yes" if re.search(r"(.)\1{3,}", text) else "No",
    }


def distance(left, right):
    left_count, right_count = sum(left.values()), sum(right.values())
    if not left_count or not right_count:
        return None
    divergence = 0.0
    for category in left.keys() | right.keys():
        p, q = left[category] / left_count, right[category] / right_count
        mean = (p + q) / 2
        divergence += ((p * log2(p / mean) if p else 0.0) + (q * log2(q / mean) if q else 0.0)) / 2
    return sqrt(max(0.0, divergence))


def compare(reference, current, *, active, minimum):
    reference_active = [row for row in reference if row["model_id"] == active]
    current_active = [row for row in current if row["model_id"] == active]
    result = []

    def add(name, kind, before, after):
        left, right = Counter(before), Counter(after)
        enough = min(sum(left.values()), sum(right.values())) >= minimum
        score = distance(left, right) if enough else None
        state = ("insufficient" if score is None else "significant" if score >= .35
                 else "warning" if score >= .20 else "normal")
        result.append({
            "feature": name, "kind": kind, "distance": score, "status": state,
            "reference_count": sum(left.values()), "current_count": sum(right.values()),
            "reference_bins": dict(left), "current_bins": dict(right),
        })

    for feature, name in FEATURES.items():
        add(name, "input", (row["features"][feature] for row in reference),
            (row["features"][feature] for row in current))
    for feature, name in PREDICTIONS.items():
        add(name, "prediction", (row[feature] for row in reference_active),
            (row[feature] for row in current_active))
    add("Review priority band", "prediction",
        (row["features"]["score_band"] for row in reference_active if "score_band" in row["features"]),
        (row["features"]["score_band"] for row in current_active if "score_band" in row["features"]))
    add("Model version mix", "operational", (str(row["model_id"]) for row in reference),
        (str(row["model_id"]) for row in current))
    return result
