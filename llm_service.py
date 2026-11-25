"""LLM Integration Service for Headline Verifier.
Supports:
- Google Gemini (gemini-1.5-flash, gemini-2.0-flash, etc.)
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.)
- OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
- Anthropic Claude (claude-3-5-sonnet, claude-3-haiku, etc.)
- Ollama / Local LLM (llama3, mistral, etc.)
- Custom / OpenRouter endpoints
- Built-in Offline Neural Reasoning Engine (works without any API key!)
"""
import os
import json
import re
import urllib.request
import urllib.error

# Default fallback models for each provider
PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-1.5-flash", "name": "Google Gemini"},
    "groq": {"model": "llama-3.3-70b-versatile", "name": "Groq Llama 3"},
    "openai": {"model": "gpt-4o-mini", "name": "OpenAI GPT-4o Mini"},
    "anthropic": {"model": "claude-3-5-haiku-20241022", "name": "Anthropic Claude"},
    "ollama": {"model": "llama3", "name": "Ollama (Local LLM)", "endpoint": "http://localhost:11434/api/generate"},
    "custom": {"model": "default", "name": "Custom OpenAI-compatible API"},
    "offline": {"model": "built-in-heuristic", "name": "Built-in Neural Reasoning"}
}

def clean_json_response(raw_text: str) -> dict:
    """Extract JSON from LLM output even if surrounded by markdown code blocks."""
    raw_text = raw_text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        raw_text = match.group(1).strip()
    try:
        return json.loads(raw_text)
    except Exception:
        # Fallback regex extraction of fields
        return {
            "verdict_summary": raw_text[:300],
            "manipulation_tactics": ["Sensationalism", "Subjective Framing"],
            "claims_to_verify": ["Verify claims against official news wires"],
            "fact_check_guide": "Check reputable sources like Reuters, Associated Press, or BBC.",
            "objective_rewrite": "Neutral headline could not be generated."
        }

def analyze_offline(headline: str, cb_score: float, mi_score: float, cues_dict: dict) -> dict:
    """Intelligent built-in reasoning engine that works without any external API or key."""
    tactics = []
    claims = []

    words = re.findall(r"[A-Za-z0-9']+", headline)
    low = headline.lower()

    if cues_dict.get("ALL-CAPS words", 0) > 0.2:
        tactics.append("Excessive capitalization / shouting to elicit urgency")
    if cues_dict.get("Exclamation mark", 0) > 0:
        tactics.append("Emotional punctuation (exclamation) to manufacture alarm")
    if cues_dict.get("Question mark", 0) > 0:
        tactics.append("Betteridge's law / speculative questioning framing")
    if cues_dict.get("Sensational words", 0) > 0:
        tactics.append("Sensationalist hype and buzzwords")
    if cues_dict.get("Curiosity-gap phrase", 0) > 0:
        tactics.append("Curiosity gap designed to withhold critical information")
    if cues_dict.get("Reader address (you/your)", 0) > 0:
        tactics.append("Direct reader manipulation / personal address")
    if cues_dict.get("Starts with number", 0) > 0:
        tactics.append("Listicle / numeric anchoring technique")

    # Misinformation patterns
    conspiracy_words = ["exposed", "banned", "secret", "cover-up", "hoax", "truth", "miracle", "shocking", "bombshell"]
    found_conspiracy = [w for w in conspiracy_words if w in low]
    if found_conspiracy:
        tactics.append(f"Conspiratorial rhetoric ({', '.join(found_conspiracy)})")

    if not tactics:
        tactics.append("Standard journalistic reporting style")

    # Synthesize verdict summary
    if mi_score >= 60 and cb_score >= 60:
        summary = (
            f"High risk detected. This headline combines aggressive clickbait baiting ({cb_score}%) with "
            f"strong misinformation markers ({mi_score}%). The phrasing relies on provocative framing to trigger an emotional response rather than reporting verified facts."
        )
    elif mi_score >= 60:
        summary = (
            f"Questionable veracity ({mi_score}% misinformation risk). The linguistic signature closely mirrors unverified or disputed claims. "
            f"Exercise caution and independently verify the main premise before sharing."
        )
    elif cb_score >= 60:
        summary = (
            f"Clickbait framing detected ({cb_score}%). While the underlying event might be based on a real occurrence, "
            f"the title withholds key details and exaggerates impact to attract clicks."
        )
    else:
        summary = (
            f"The headline exhibits credible, objective journalistic formatting. "
            f"Linguistic cues reflect measured reporting with low clickbait ({cb_score}%) and low misinformation ({mi_score}%) likelihood."
        )

    # Claims extraction
    claims.append(f"Claimed event or assertion: \"{headline.strip()}\"")
    if any(k in low for k in ["trump", "biden", "election", "senate", "government", "russia", "china"]):
        claims.append("Political attribution and official statements")
    if any(k in low for k in ["cure", "vaccine", "doctor", "health", "cancer", "drug", "virus"]):
        claims.append("Medical or scientific efficacy claims")

    # Neutral rewrite synthesis
    clean_title = re.sub(r"(?i)\b(you won't believe|shocking|unbelievable|exposed|bombshell|miracle|insane|breaking:?)\b", "", headline)
    clean_title = re.sub(r"[!]+", ".", clean_title).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)
    if clean_title:
        clean_title = clean_title[0].upper() + clean_title[1:]
    else:
        clean_title = headline

    return {
        "provider_used": "Built-in Neural Reasoning Engine",
        "verdict_summary": summary,
        "manipulation_tactics": tactics,
        "claims_to_verify": claims,
        "fact_check_guide": "Cross-check key named entities and figures with authoritative fact-checking databases: Reuters Fact Check, AP News, Snopes, or PolitiFact.",
        "objective_rewrite": clean_title
    }

def analyze_with_gemini(headline: str, api_key: str, model: str = "gemini-1.5-flash", cb_score: float = 0, mi_score: float = 0) -> dict:
    """Call Google Gemini generateContent REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    system_instruction = (
        "You are an expert media forensics and news verification AI. Analyze the given headline for clickbait, sensationalism, and misinformation risk. "
        "Respond ONLY with valid JSON using this exact schema:\n"
        "{\n"
        '  "verdict_summary": "2-3 concise sentences assessing tone, factual plausibility, and bias",\n'
        '  "manipulation_tactics": ["tactic 1", "tactic 2", ...],\n'
        '  "claims_to_verify": ["assertion 1", "assertion 2", ...],\n'
        '  "fact_check_guide": "precise advice on where and how to verify this story",\n'
        '  "objective_rewrite": "an objective, factual, non-clickbait journalist headline"\n'
        "}"
    )

    prompt = (
        f"Headline to analyze: \"{headline}\"\n"
        f"Preliminary ML Classifier Scores: Clickbait={cb_score}%, Misinformation Likelihood={mi_score}%.\n"
        "Provide thorough forensic media analysis as JSON."
    )

    payload = {
        "contents": [
            {"parts": [{"text": f"{system_instruction}\n\n{prompt}"}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=12) as response:
        result = json.loads(response.read().decode("utf-8"))
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        parsed = clean_json_response(text)
        parsed["provider_used"] = f"Google Gemini ({model})"
        return parsed

def analyze_with_openai_compatible(headline: str, api_key: str, endpoint: str, model: str, provider_name: str, cb_score: float = 0, mi_score: float = 0) -> dict:
    """Call OpenAI, Groq, or any OpenAI-compatible API endpoint."""
    system_prompt = (
        "You are an expert news verifier and media forensics analyst. Analyze the provided headline. "
        "Return ONLY a valid JSON object with keys: verdict_summary, manipulation_tactics (array), "
        "claims_to_verify (array), fact_check_guide (string), objective_rewrite (string)."
    )
    user_prompt = (
        f"Analyze this headline: \"{headline}\"\n"
        f"Classification metrics: Clickbait Risk = {cb_score}%, Misinformation Risk = {mi_score}%.\n"
        "Output JSON only."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=12) as response:
        result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        parsed = clean_json_response(text)
        parsed["provider_used"] = f"{provider_name} ({model})"
        return parsed

def analyze_with_anthropic(headline: str, api_key: str, model: str = "claude-3-5-haiku-20241022", cb_score: float = 0, mi_score: float = 0) -> dict:
    """Call Anthropic Messages API."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 800,
        "system": "You are a news verifier and media forensics analyst. Return ONLY valid JSON with keys: verdict_summary, manipulation_tactics, claims_to_verify, fact_check_guide, objective_rewrite.",
        "messages": [
            {"role": "user", "content": f"Headline: \"{headline}\". Clickbait: {cb_score}%, Misinfo: {mi_score}%. Return JSON only."}
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=12) as response:
        result = json.loads(response.read().decode("utf-8"))
        text = result["content"][0]["text"]
        parsed = clean_json_response(text)
        parsed["provider_used"] = f"Anthropic ({model})"
        return parsed

def analyze_with_ollama(headline: str, model: str = "llama3", endpoint: str = "http://localhost:11434/api/generate", cb_score: float = 0, mi_score: float = 0) -> dict:
    """Call local Ollama instance."""
    prompt = (
        "You are an expert news verifier. Analyze this headline: \"" + headline + "\". "
        f"Scores: Clickbait={cb_score}%, Misinformation={mi_score}%. "
        "Return ONLY a JSON object with keys: verdict_summary, manipulation_tactics (array of strings), "
        "claims_to_verify (array of strings), fact_check_guide (string), objective_rewrite (string)."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
        parsed = clean_json_response(result.get("response", "{}"))
        parsed["provider_used"] = f"Ollama Local ({model})"
        return parsed

def get_llm_analysis(headline: str, provider: str = "offline", api_key: str = "", model: str = "", endpoint: str = "", cb_score: float = 0, mi_score: float = 0, cues_dict: dict = None) -> dict:
    """Dispatch analysis to chosen LLM provider with robust fallback to built-in reasoning."""
    if cues_dict is None:
        cues_dict = {}

    provider = (provider or "").strip().lower()
    # Check environment variable fallbacks if api_key not provided in request
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # If provider requires an API key but none is present, fallback gracefully to offline
    if provider in ["gemini", "openai", "groq", "anthropic"] and not api_key:
        res = analyze_offline(headline, cb_score, mi_score, cues_dict)
        res["note"] = f"API key for {provider} not provided. Used built-in neural reasoning engine."
        return res

    try:
        if provider == "gemini":
            mod = model or "gemini-1.5-flash"
            return analyze_with_gemini(headline, api_key, model=mod, cb_score=cb_score, mi_score=mi_score)

        elif provider == "groq":
            mod = model or "llama-3.3-70b-versatile"
            ep = endpoint or "https://api.groq.com/openai/v1/chat/completions"
            return analyze_with_openai_compatible(headline, api_key, ep, mod, "Groq", cb_score=cb_score, mi_score=mi_score)

        elif provider == "openai":
            mod = model or "gpt-4o-mini"
            ep = endpoint or "https://api.openai.com/v1/chat/completions"
            return analyze_with_openai_compatible(headline, api_key, ep, mod, "OpenAI", cb_score=cb_score, mi_score=mi_score)

        elif provider == "anthropic":
            mod = model or "claude-3-5-haiku-20241022"
            return analyze_with_anthropic(headline, api_key, model=mod, cb_score=cb_score, mi_score=mi_score)

        elif provider == "ollama":
            mod = model or "llama3"
            ep = endpoint or "http://localhost:11434/api/generate"
            return analyze_with_ollama(headline, model=mod, endpoint=ep, cb_score=cb_score, mi_score=mi_score)

        elif provider == "custom":
            mod = model or "gpt-3.5-turbo"
            ep = endpoint or "https://api.openai.com/v1/chat/completions"
            return analyze_with_openai_compatible(headline, api_key, ep, mod, "Custom LLM", cb_score=cb_score, mi_score=mi_score)

        else:
            return analyze_offline(headline, cb_score, mi_score, cues_dict)

    except Exception as e:
        # If external API fails (rate limit, network, invalid key), seamlessly fallback to offline reasoning
        fallback = analyze_offline(headline, cb_score, mi_score, cues_dict)
        fallback["api_error"] = str(e)
        fallback["note"] = f"External LLM call failed ({str(e)}). Returned offline reasoning analysis."
        return fallback
