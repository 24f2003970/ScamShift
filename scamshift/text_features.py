"""Shared normalization and privacy-preserving text characteristics."""
import hashlib
import re
import unicodedata


URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.example\b", re.IGNORECASE)
SPACED_CREDENTIAL = re.compile(r"\b(?:o[.\s]+t[.\s]+p|k[.\s]+y[.\s]+c)\b")


def normalize(message):
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def key(message):
    return hashlib.sha256(normalize(message).encode()).hexdigest()


def robust_normalize(message):
    """An opt-in experiment. The existing fitted models retain original preprocessing."""
    message = normalize(message)
    message = "".join(char for char in message if unicodedata.category(char) != "Cf")
    message = SPACED_CREDENTIAL.sub(lambda match: re.sub(r"[.\s]", "", match.group()), message)
    return URL.sub(" urltoken ", message)


def script(message):
    latin = bool(re.search(r"[a-z]", normalize(message)))
    devanagari = bool(re.search(r"[\u0900-\u097f]", message))
    if devanagari:
        return "Mixed Latin and Devanagari" if latin else "Devanagari"
    return "Latin script" if latin else "Other"


def template(message):
    message = URL.sub(" urltoken ", robust_normalize(message))
    message = re.sub(r"\d+", " numbertoken ", message)
    return " ".join(re.findall(r"[\w\u0900-\u097f]+", message))
