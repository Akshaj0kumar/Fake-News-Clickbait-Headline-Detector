"""Flask backend for Fake News & Clickbait Headline Verifier.
Combines:
1. Dual-head Machine Learning models (Clickbait & Misinformation)
2. Explainability through word-level occlusion and linguistic cues
3. LLM Model / API Integration (Gemini, Groq, OpenAI, Anthropic, Ollama, and Built-in Neural Engine)
Run: python app.py -> open http://127.0.0.1:5000
"""
import os
import re
import joblib
from flask import Flask, request, jsonify, render_template
from nlp_utils import cues, CUE_NAMES
from llm_service import get_llm_analysis, PROVIDER_DEFAULTS

app = Flask(__name__)

# Ensure models are loaded safely
CB_PATH = "models/clickbait.joblib"
MI_PATH = "models/misinfo.joblib"

if os.path.exists(CB_PATH) and os.path.exists(MI_PATH):
    CB = joblib.load(CB_PATH)
    MI = joblib.load(MI_PATH)
    print("Loaded ML models successfully from models/")
else:
    raise FileNotFoundError("Models not found! Please run train.py first.")

def influence(model, headline):
    """Explainability by occlusion: remove each word, measure the drop in probability."""
    words = headline.split()
    base = model.predict_proba([headline])[0][1]
    texts = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = model.predict_proba(texts)[:, 1] if texts else []
    return [round(float(base - p), 4) for p in probs]

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
