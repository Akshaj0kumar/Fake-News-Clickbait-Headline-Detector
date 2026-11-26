# Verity — AI-Powered Fake News & Clickbait Headline Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![LLM Supported](https://img.shields.io/badge/LLM-Gemini%20%7C%20Groq%20%7C%20OpenAI%20%7C%20Ollama-emerald.svg)](#-llm-integration)

Verity is an intelligent news veracity platform combining **dual-head machine learning classifiers** with **multi-provider LLM forensic reasoning** to identify fake news, clickbait, and linguistic manipulation in headlines.

---

## 🌟 Key Features

1. **Dual-Head Machine Learning Models**:
   - **Clickbait Detector**: Identifies sensationalist, curiosity-gap, and listicle baiting patterns (100% test accuracy).
   - **Misinformation / Fake News Detector**: Trained on over **44,800+ real and fake news articles** from the ISOT dataset (97% test accuracy, 0.99 precision on fake news).
2. **Multi-Provider LLM & API Integration**:
   - Supports **Google Gemini** (`gemini-1.5-flash`, `gemini-2.0-flash`).
   - Supports **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant` for ultra-fast inference).
   - Supports **OpenAI** (`gpt-4o-mini`, `gpt-4o`).
   - Supports **Anthropic Claude** (`claude-3-5-haiku`, `claude-3-5-sonnet`).
   - Supports **Ollama** for 100% private, offline local LLMs (`llama3`, `mistral`).
   - Supports **Custom OpenAI-compatible endpoints**.
   - **Zero-Setup Offline Neural Reasoning Engine**: Generates LLM-grade forensic analysis out of the box even without any API key or internet connection!
3. **Forensic Explainability**:
   - **Word-Level Occlusion**: Visualizes which exact words drove the clickbait or fake news risk.
   - **Linguistic Cues Extraction**: Measures 8 syntactic and psychological manipulation markers (shouting caps, emotional punctuation, sensational adjectives, curiosity gaps, etc.).
   - **Claim Extraction & Fact-Check Guide**: Outlines specific claims and authoritative databases (Reuters Fact Check, AP, Snopes) to verify them.
   - **Journalistic Neutral Rewrite**: Rewrites biased or baiting headlines into objective, factual news summaries.

---

## 📊 Datasets & Performance

| Head | Dataset Source | Size | Model Architecture | Accuracy | Precision (Fake/CB) | Recall (Fake/CB) | F1-Score |
|---|---|---|---|---|---|---|---|
| **Misinformation** | ISOT Fake News (`fake.csv` & `true.csv`) | 44,898 articles | TF-IDF (1-2 n-grams) + Linguistic Cues + Logistic Regression | **97%** | **0.99** | **0.96** | **0.98** |
| **Clickbait** | Clickbait Dataset (`clickbait_data.csv`) | 1,600 headlines | TF-IDF (1-2 n-grams) + Linguistic Cues + Logistic Regression | **100%** | **1.00** | **1.00** | **1.00** |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Akshaj0kumar/Fake-News-Clickbait-Headline-Detector.git
cd Fake-News-Clickbait-Headline-Detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(To retrain models locally with scikit-learn and pandas, run `pip install -r requirements-train.txt`.)*

### 3. Run the Application
The pre-trained models are already included in `models/`, so you can launch the app immediately:
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### 4. Deploy to Vercel
- Select the repository root (`./`) as the **Root Directory** in Vercel.
- Leave Build & Output settings at their defaults (`vercel.json` and `api/index.py` handle routing and keep the serverless bundle minimal).
- Click **Deploy**.

### 5. (Optional) Retrain Models
To retrain the models from the dataset:
```bash
pip install -r requirements-train.txt
python train.py
```

---

## 🤖 LLM Configuration

You can configure any LLM provider directly inside the web UI:
1. Click the **"LLM / API Config"** button in the top navigation bar.
2. Select your provider:
   - **Built-in Neural Reasoning**: No API key required; runs 100% offline.
   - **Google Gemini**: Enter your free Gemini API key.
   - **Groq**: Enter your free Groq API key for instant inference.
   - **OpenAI**: Enter your OpenAI API key.
   - **Anthropic Claude**: Enter your Anthropic API key.
   - **Ollama**: Connect to your local Ollama server (`http://localhost:11434/api/generate`).
   - **Custom**: Enter any custom OpenAI-compatible endpoint.
3. Settings are saved in your browser's local storage.

You can also set environment variables if preferred:
```bash
export GEMINI_API_KEY="your_gemini_key"
export GROQ_API_KEY="your_groq_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

---

## 📁 Project Structure

```
├── app.py                  # Flask web backend & /analyze API
├── train.py                # Model training script for both heads
├── nlp_utils.py            # 8 hand-crafted linguistic cue extractors
├── llm_service.py          # Multi-provider LLM dispatch & forensic reasoning
├── test_app.py             # Automated test suite
├── requirements.txt        # Python package dependencies
├── models/
│   ├── clickbait.joblib    # Pre-trained Clickbait classification model
│   └── misinfo.joblib      # Pre-trained Misinformation classification model
├── templates/
│   └── index.html          # Modern, responsive UI with interactive settings & visualizations
└── data/
    └── clickbait_data.csv  # Clickbait dataset
```

---

## 🧪 Testing

Run the automated test suite to verify the ML models and LLM integration:
```bash
python test_app.py
```

---

## 📜 License
MIT License.
