import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PATTERNS = {
    "Account access": r"\b(?:otp|pin|kyc|password|o\.t\.p|k\.y\.c)\b|ओटीपी|केवाईसी",
    "Urgency": r"\b(?:abhi|urgent|expire|suspend|freeze|block|warna)\b|अभी",
    "Money request": r"\b(?:fee|deposit|upi|payment|refund|pay)\b",
    "Link": r"(?:https?://|www\.)[^\s]+|\b[\w.-]+\.example\b",
}


def normalize(message):
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def key(message):
    return hashlib.sha256(normalize(message).encode()).hexdigest()


@lru_cache(maxsize=1)
def corpus():
    pairs = {}
    for line in (DATA / "SMSSpamCollection").read_text(encoding="utf-8").splitlines():
        label, sep, message = line.partition("\t")
        if sep and label in ("ham", "spam"):
            pairs.setdefault(key(message), (message, int(label == "spam")))
    return train_test_split(list(pairs.values()), test_size=.25, random_state=42,
                            stratify=[label for _, label in pairs.values()])


@lru_cache(maxsize=1)
def campaigns():
    return json.loads((DATA / "campaigns.json").read_text(encoding="utf-8-sig"))


def examples(split):
    return [
        {"id": key(message), "campaign": campaign, "text": message, "label": int(label == "scam")}
        for campaign, groups in campaigns()[split].items()
        for label, messages in groups.items()
        for message in messages
    ]


@lru_cache(maxsize=1)
def reserved():
    return {key(message) for message, _ in corpus()[1]} | {row["id"] for row in examples("audit")}


def metrics(labels, scores, threshold):
    tn, fp, fn, tp = map(int, confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel())
    return {
        "recall": tp / (tp + fn), "false_alarm": fp / (fp + tn),
        "average_precision": float(average_precision_score(labels, scores)),
        "caught": tp, "missed": fn, "false_flags": fp, "genuine": tn, "total": len(labels),
    }


def probabilities(model, messages):
    return model.predict_proba([normalize(message) for message in messages])[:, 1]


def evaluate(model, pairs, threshold):
    messages, labels = zip(*pairs)
    return metrics(labels, probabilities(model, messages), threshold)


def train(reviewed):
    base, holdout = corpus()
    feedback = [(text, label) for text, label in reviewed if key(text) not in reserved()]
    pairs = base + feedback
    messages, labels = zip(*pairs)
    model = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(C=4, max_iter=500, random_state=42, class_weight="balanced")),
    ])
    weights = [1] * len(base) + [18] * len(feedback)
    model.fit([normalize(message) for message in messages], labels, classifier__sample_weight=weights)
    threshold = .5
    audit = [(row["text"], row["label"]) for row in examples("audit")]
    meta = {
        "threshold": threshold, "training_rows": len(base), "reviewed_rows": len(feedback),
        "dataset_sha256": hashlib.sha256((DATA / "SMSSpamCollection").read_bytes()).hexdigest(),
        "feedback_sha256": hashlib.sha256(json.dumps(sorted((key(text), label) for text, label in feedback)).encode()).hexdigest(),
        "benchmark": evaluate(model, holdout, threshold), "campaign_audit": evaluate(model, audit, threshold),
    }
    return model, meta


def cues(message):
    return [
        {"kind": name, "text": match.group()}
        for name, pattern in PATTERNS.items()
        for match in list(re.finditer(pattern, message, re.IGNORECASE))[:2]
    ]
