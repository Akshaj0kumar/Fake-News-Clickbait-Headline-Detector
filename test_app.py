import json
from app import app

client = app.test_client()

samples = [
    "BREAKING: Secret cure banned by big pharma EXPOSED to public!",
    "10 shocking secrets doctors don't want you to know - number 4 will blow your mind!",
    "Federal Reserve maintains benchmark interest rate after policy meeting"
]

print("=== Starting Flask & LLM Integration Verification ===")
for s in samples:
    print("-" * 50)
    print(f"Testing Headline: {s}")
    res = client.post("/analyze", data=json.dumps({"headline": s, "provider": "offline"}), content_type="application/json")
    assert res.status_code == 200, f"Failed with status: {res.status_code}"
    data = res.get_json()
    print(f"Credibility: {data['credibility']} | Clickbait: {data['clickbait']}% | Misinfo: {data['misinfo']}%")
    print("LLM Engine:", data["llm"]["provider_used"])
    print("LLM Verdict:", data["llm"]["verdict_summary"])
    print("LLM Tactics:", data["llm"]["manipulation_tactics"])
    print("LLM Rewrite:", data["llm"]["objective_rewrite"])

print("\n=== ALL FLASK & LLM BACKEND TESTS PASSED SUCCESSFULLY! ===")
