"""Linguistic cue analysis + shared helpers."""
import re

SENSATIONAL = {"shocking", "unbelievable", "amazing", "insane", "secret", "miracle", "exposed",
               "destroys", "slams", "epic", "outrage", "bombshell", "scandal", "terrifying",
               "hoax", "banned", "cover-up", "truth", "crazy", "horrifying"}
CURIOSITY = ["you won't believe", "this is why", "what happened next", "you need to", "will blow your mind",
             "here's why", "here is why", "the reason", "number", "what they don't", "doctors hate"]
CUE_NAMES = ["Starts with number", "Question mark", "Exclamation mark", "ALL-CAPS words",
             "Sensational words", "Reader address (you/your)", "Curiosity-gap phrase", "Headline length"]


def cues(h: str) -> list:
    words = re.findall(r"[A-Za-z']+", h)
    low = h.lower()
    caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    return [
        float(bool(re.match(r"^\s*\d", h))),
        float("?" in h),
        min(h.count("!"), 3) / 3,
        min(caps, 4) / 4,
        min(sum(w.lower() in SENSATIONAL for w in words), 3) / 3,
        float(bool(re.search(r"\b(you|your)\b", low))),
        float(any(p in low for p in CURIOSITY)),
        min(len(words), 25) / 25,
    ]


def cue_matrix(texts):
    import numpy as np
    return np.array([cues(t) for t in texts])
