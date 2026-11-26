"""Flask backend for Fake News & Clickbait Headline Verifier.
Combines:
1. Dual-head Machine Learning models (Clickbait & Misinformation)
2. Explainability through word-level occlusion and linguistic cues
3. LLM Model / API Integration (Gemini, Groq, OpenAI, Anthropic, Ollama, and Built-in Neural Engine)
Run: python app.py -> open http://127.0.0.1:5000
"""
import gzip
import json
import math
import os
import re
from flask import Flask, request, jsonify, render_template
from nlp_utils import cues, CUE_NAMES
from llm_service import get_llm_analysis, PROVIDER_DEFAULTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.url_map.strict_slashes = False


class FastHeadlineModel:
    """Dependency-free TF-IDF (1-2 grams, sublinear TF, L2 norm) + LogisticRegression inference."""

    def __init__(self, gz_path: str):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        self.weights = payload["weights"]
        self.cue_coefs = payload["cue_coefs"]
        self.intercept = payload["intercept"]

    def _predict_one(self, text: str) -> float:
        tokens = TOKEN_RE.findall(text.lower())
        counts = {}
        for tok in tokens:
            if tok in self.weights:
                counts[tok] = counts.get(tok, 0) + 1
        for i in range(len(tokens) - 1):
            bg = tokens[i] + " " + tokens[i + 1]
            if bg in self.weights:
                counts[bg] = counts.get(bg, 0) + 1

        sq_sum = 0.0
        dot = 0.0
        for gram, tf in counts.items():
            idf_val, weight = self.weights[gram]
            val = (1.0 + math.log(tf)) * idf_val
            sq_sum += val * val
            dot += val * weight

        if sq_sum > 0.0:
            dot /= math.sqrt(sq_sum)

        c_vals = cues(text)
        z = self.intercept + dot + sum(cv * cw for cv, cw in zip(c_vals, self.cue_coefs))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_proba(self, texts):
        return [[1.0 - p, p] for p in (self._predict_one(t) for t in texts)]


def load_model(name: str):
    """Load compact .json.gz model (pure Python) or fallback to .joblib if available."""
    gz_path = os.path.join(BASE_DIR, "models", f"{name}.json.gz")
    if os.path.exists(gz_path):
        return FastHeadlineModel(gz_path)

    joblib_path = os.path.join(BASE_DIR, "models", f"{name}.joblib")
    if os.path.exists(joblib_path):
        import joblib
        return joblib.load(joblib_path)

    raise FileNotFoundError(f"Model '{name}' not found in models/. Please run train.py first.")


CB = load_model("clickbait")
MI = load_model("misinfo")
print("Loaded ML models successfully from models/")


def influence(model, headline):
    """Explainability by occlusion: remove each word, measure the drop in probability."""
    words = headline.split()
    base = float(model.predict_proba([headline])[0][1])
    texts = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = [float(p[1]) for p in model.predict_proba(texts)] if texts else []
    return [round(base - p, 4) for p in probs]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/providers", methods=["GET"])
def get_providers():
    return jsonify(providers=PROVIDER_DEFAULTS)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    h = (data.get("headline") or "").strip()

    if len(re.findall(r"\w+", h)) < 3:
        return jsonify(error="Enter a headline with at least three words."), 400

    # ML Classifier Predictions
    cb_prob = float(CB.predict_proba([h])[0][1])
    mi_prob = float(MI.predict_proba([h])[0][1])
    cb_pct = round(cb_prob * 100)
    mi_pct = round(mi_prob * 100)
    credibility_pct = round((1 - mi_prob) * 100)

    # Explainability by occlusion
    icb = influence(CB, h)
    imi = influence(MI, h)

    # Linguistic cues
    raw_cues = cues(h)
    cue_dict = {n: round(float(v), 2) for n, v in zip(CUE_NAMES, raw_cues)}
    cue_list = [{"name": n, "value": v} for n, v in cue_dict.items()]

    # LLM Analysis
    provider = data.get("provider") or "offline"
    api_key = (data.get("apiKey") or "").strip()
    model = (data.get("model") or "").strip()
    endpoint = (data.get("endpoint") or "").strip()

    llm_result = get_llm_analysis(
        headline=h,
        provider=provider,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        cb_score=cb_pct,
        mi_score=mi_pct,
        cues_dict=cue_dict
    )

    return jsonify(
        clickbait=cb_pct,
        misinfo=mi_pct,
        credibility=credibility_pct,
        words=[{"w": w, "cb": a, "mi": b} for w, a, b in zip(h.split(), icb, imi)],
        cues=cue_list,
        llm=llm_result
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
