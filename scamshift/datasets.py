"""Versioned campaign fixtures and evaluation boundaries."""
import hashlib
import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from .text_features import key, script, template

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLIT_SEED = 42
REVIEW_LABELS = {"genuine": 0, "scam": 1, "promotional_spam": 2, "unsure": None}
EVALUATION_SPLITS = ("validation", "audit", "holdout", "robustness")


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=5)
def _examples(split):
    if split in {"review", "audit"}:
        groups = json.loads((DATA / "campaigns.json").read_text(encoding="utf-8-sig"))[split]
    else:
        groups = json.loads((DATA / f"campaign_{split}.json").read_text(encoding="utf-8"))
    return tuple(
        {"id": key(text), "campaign": campaign, "text": text, "label": REVIEW_LABELS[label],
         "script": script(text)}
        for campaign, categories in groups.items()
        for label, messages in categories.items()
        for text in messages
    )


def examples(split):
    return [row.copy() for row in _examples(split)]


def binary_review(label):
    return 1 if label == REVIEW_LABELS["promotional_spam"] else label


def near_duplicate(left, right):
    if not left or not right:
        return False
    if left == right:
        return True
    shorter = min(len(left), len(right))
    return (shorter >= 20 and 2 * shorter / (len(left) + len(right)) >= .88
            and SequenceMatcher(None, left, right, autojunk=False).ratio() >= .88)


@lru_cache(maxsize=1)
def protected_templates():
    return tuple(template(row["text"]) for split in EVALUATION_SPLITS for row in examples(split))


def overlaps_evaluation(text):
    normalized = template(text)
    return any(near_duplicate(normalized, protected) for protected in protected_templates())


def audit_splits():
    """Count exact and near-duplicate templates across the authored cohorts."""
    splits = ("review", *EVALUATION_SPLITS)
    result = []
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            left_rows, right_rows = examples(left), examples(right)
            overlaps = [
                (a["id"], b["id"])
                for a in left_rows for b in right_rows
                if near_duplicate(template(a["text"]), template(b["text"]))
            ]
            result.append({"left": left, "right": right, "template_matches": overlaps,
                           "shared_families": sorted({row["campaign"] for row in left_rows}
                                                     & {row["campaign"] for row in right_rows})})
    return result
