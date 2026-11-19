import hashlib
import json
import re
from functools import lru_cache
from time import perf_counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .datasets import DATA, ROOT, EVALUATION_SPLITS, SPLIT_SEED, binary_review, examples, overlaps_evaluation
from .evaluation import metrics
from .text_features import key, normalize, robust_normalize

FEEDBACK_WEIGHT = 18
THRESHOLD = .5
PATTERNS = {
    "Credential mention": r"\b(?:o[.\s]?t[.\s]?p|pin|k[.\s]?y[.\s]?c|password)\b|ओटीपी|केवाईसी",
    "Urgency": r"\b(?:abhi|urgent|expire|suspend|freeze|block|warna)\b|अभी",
    "Money request": r"\b(?:fee|deposit|upi|payment|refund|pay)\b",
    "Link": r"(?:https?://|www\.)[^\s]+|\b[\w.-]+\.example\b",
}


@lru_cache(maxsize=1)
def corpus():
    pairs = {}
    for line in (DATA / "SMSSpamCollection").read_text(encoding="utf-8").splitlines():
        label, sep, message = line.partition("\t")
        if sep and label in ("ham", "spam"):
            pairs.setdefault(key(message), (message, int(label == "spam")))
    return train_test_split(list(pairs.values()), test_size=.25, random_state=SPLIT_SEED,
                            stratify=[label for _, label in pairs.values()])


@lru_cache(maxsize=1)
def reserved():
    return {key(message) for message, _ in corpus()[1]} | {
        row["id"] for split in EVALUATION_SPLITS for row in examples(split)
    }


@lru_cache(maxsize=1)
def training_keys():
    return {key(message) for message, _ in corpus()[0]}


@lru_cache(maxsize=1)
def excluded_reviews():
    return reserved() | training_keys()


def eligible_feedback(text):
    return key(text) not in excluded_reviews() and not overlaps_evaluation(text)


def probabilities(model, messages, *, preprocess=normalize):
    return model.predict_proba([preprocess(message) for message in messages])[:, 1]


def evaluate(model, pairs, threshold, *, preprocess=normalize):
    messages, labels = zip(*pairs)
    return metrics(labels, probabilities(model, messages, preprocess=preprocess), threshold)


def train(reviewed, *, feedback_weight=FEEDBACK_WEIGHT, threshold=THRESHOLD, robust=False, promotion_metrics=True):
    started = perf_counter()
    preprocess = robust_normalize if robust else normalize
    base, holdout = corpus()
    feedback = [(text, label) for text, label in reviewed if label is not None and eligible_feedback(text)]
    pairs = base + [(text, binary_review(label)) for text, label in feedback]
    messages, labels = zip(*pairs)
    model = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(C=4, max_iter=500, random_state=SPLIT_SEED, class_weight="balanced")),
    ])
    weights = [1] * len(base) + [feedback_weight] * len(feedback)
    model.fit([preprocess(message) for message in messages], labels, classifier__sample_weight=weights)
    validation = [(row["text"], row["label"]) for row in examples("validation")]
    meta = {
        "threshold": threshold, "feedback_weight": feedback_weight,
        "training_rows": len(base), "reviewed_rows": len(feedback),
        "dataset_sha256": hashlib.sha256((DATA / "SMSSpamCollection").read_bytes()).hexdigest(),
        "campaign_sha256": hashlib.sha256((DATA / "campaigns.json").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256((DATA / "campaign_validation.json").read_bytes()).hexdigest(),
        "validation_rows": len(validation), "training_seconds": perf_counter() - started,
        "model_type": "Logistic regression with word and character TF-IDF",
        "hyperparameters": model.named_steps["classifier"].get_params(),
        "preprocessing": "robust" if robust else "baseline",
        "feedback_sha256": hashlib.sha256(json.dumps(sorted((key(text), label) for text, label in feedback)).encode()).hexdigest(),
        "validation": evaluate(model, validation, threshold, preprocess=preprocess),
    }
    if promotion_metrics:
        audit = [(row["text"], row["label"]) for row in examples("audit")]
        meta["benchmark"] = evaluate(model, holdout, threshold, preprocess=preprocess)
        meta["campaign_audit"] = evaluate(model, audit, threshold, preprocess=preprocess)
    return model, meta


def cues(message):
    return [
        {"kind": name, "text": match.group()}
        for name, pattern in PATTERNS.items()
        for match in list(re.finditer(pattern, message, re.IGNORECASE))[:2]
    ]
