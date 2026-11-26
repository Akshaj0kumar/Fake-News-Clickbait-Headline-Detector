"""Trains the two heads: (1) clickbait, (2) misinformation. Run: python train.py"""
import os
import gzip
import json
import random
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nlp_utils import cue_matrix

FAKE_LABEL = 1

def build():
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("cues", FunctionTransformer(cue_matrix)),
        ])),
        ("clf", LogisticRegression(max_iter=1000, C=3)),
    ])

# Fallback samples if CSV datasets are missing
CB = ["You won't believe what this dog did next", "10 secrets doctors don't want you to know",
      "This one trick will change your life forever!", "Which celebrity are you? Take the quiz",
      "17 photos that will make you cry", "What happened next will blow your mind",
      "She opened the door and the truth was shocking", "Here's why everyone is talking about this",
      "9 insane facts about sleep you need to see", "You'll never guess who just got exposed!"]
NCB = ["Parliament passes annual budget bill", "Central bank raises interest rates by 25 basis points",
       "Scientists publish study on coastal erosion", "City council approves new bus routes",
       "Court upholds ruling on trade dispute", "Company reports quarterly earnings growth",
       "Government announces vaccination schedule", "Flood warning issued for river districts",
       "University opens new research laboratory", "Opposition and ruling party agree on education reform"]
FAKE = ["BREAKING: Secret cure banned by government EXPOSED", "Celebrity dead hoax shocks the world",
        "Miracle pill melts fat overnight, media hiding truth", "Elections rigged, insiders reveal bombshell cover-up",
        "Scientists SHOCKED: vaccines contain mind control chips", "Aliens confirmed living in the White House",
        "Banned video proves moon landing was fake", "Government secretly poisoning water, whistleblower says"]
REAL = ["Senate votes on infrastructure funding", "Health ministry reports drop in seasonal flu cases",
        "Researchers develop more efficient solar cell", "Local school wins national science award",
        "Treasury releases monthly employment figures", "Stock markets close slightly higher",
        "Election commission announces polling schedule", "Hospital expands emergency department"]

def load_misinfo():
    """Load from fake.csv + true.csv, or WELFake_Dataset.csv, or fallbacks."""
    fake_path = "data/fake.csv"
    true_path = "data/true.csv"
    sample_fake_path = "data/sample_fake.csv"
    sample_true_path = "data/sample_true.csv"
    welfake_path = "data/WELFake_Dataset.csv"

    if os.path.exists(fake_path) and os.path.exists(true_path):
        print(f"[misinfo] Loading full datasets: {fake_path} and {true_path}...")
        df_fake = pd.read_csv(fake_path).dropna(subset=["title"])
        df_true = pd.read_csv(true_path).dropna(subset=["title"])
        X = df_fake["title"].astype(str).tolist() + df_true["title"].astype(str).tolist()
        y = [1] * len(df_fake) + [0] * len(df_true)
        print(f"[misinfo] Loaded {len(df_fake)} fake and {len(df_true)} real news titles (Total: {len(X)})")
        return X, y
    elif os.path.exists(sample_fake_path) and os.path.exists(sample_true_path):
        print(f"[misinfo] Loading sample datasets: {sample_fake_path} and {sample_true_path}...")
        df_fake = pd.read_csv(sample_fake_path).dropna(subset=["title"])
        df_true = pd.read_csv(sample_true_path).dropna(subset=["title"])
        X = df_fake["title"].astype(str).tolist() + df_true["title"].astype(str).tolist()
        y = [1] * len(df_fake) + [0] * len(df_true)
        print(f"[misinfo] Loaded {len(df_fake)} fake and {len(df_true)} real news titles (Total: {len(X)})")
        return X, y
    elif os.path.exists(welfake_path):
        print(f"[misinfo] Loading {welfake_path}...")
        df = pd.read_csv(welfake_path).dropna(subset=["title", "label"])
        X = df["title"].astype(str).tolist()
        y = (df["label"] == FAKE_LABEL).astype(int).tolist()
        print(f"[misinfo] Loaded {len(X)} rows from {welfake_path}")
        return X, y
    else:
        print("[misinfo] Real datasets not found -> using fallback samples")
        random.seed(1)
        X = (FAKE + REAL) * 6
        y = ([1] * len(FAKE) + [0] * len(REAL)) * 6
        return X, y

def load_clickbait():
    cb_path = "data/clickbait_data.csv"
    if os.path.exists(cb_path):
        print(f"[clickbait] Loading {cb_path}...")
        df = pd.read_csv(cb_path).dropna(subset=["headline", "clickbait"])
        X = df["headline"].astype(str).tolist()
        y = (df["clickbait"] == 1).astype(int).tolist()
        print(f"[clickbait] Loaded {len(X)} rows from {cb_path}")
        return X, y
    else:
        print("[clickbait] Dataset not found -> using fallback samples")
        random.seed(1)
        X = (CB + NCB) * 6
        y = ([1] * len(CB) + [0] * len(NCB)) * 6
        return X, y

def export_lightweight(pipeline, name):
    """Export TF-IDF + LogisticRegression weights to compact .json.gz for dependency-free inference."""
    tfidf = pipeline.named_steps["features"].transformer_list[0][1]
    clf = pipeline.named_steps["clf"]
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_
    coef = clf.coef_[0]
    n_vocab = len(vocab)
    weights = {
        tok: [round(float(idf[idx]), 8), round(float(coef[idx]), 8)]
        for tok, idx in vocab.items()
    }
    payload = {
        "weights": weights,
        "cue_coefs": [float(x) for x in coef[n_vocab:]],
        "intercept": float(clf.intercept_[0]),
    }
    out_path = f"models/{name}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Exported lightweight inference model {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB).")


def fit(X, y, name):
    os.makedirs("models", exist_ok=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = build().fit(Xtr, ytr)
    target_names = ["Legit", "Clickbait"] if name == "clickbait" else ["Real", "Fake/Misinfo"]
    print(f"\n=== Classification Report: {name} ===\n", classification_report(yte, m.predict(Xte), target_names=target_names))
    model_path = f"models/{name}.joblib"
    joblib.dump(m, model_path)
    print(f"Saved {model_path} successfully.")
    export_lightweight(m, name)
    print()

if __name__ == "__main__":
    X_cb, y_cb = load_clickbait()
    fit(X_cb, y_cb, "clickbait")

    X_mi, y_mi = load_misinfo()
    fit(X_mi, y_mi, "misinfo")
