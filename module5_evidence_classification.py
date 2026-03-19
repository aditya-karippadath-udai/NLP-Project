import os
import requests
from google import genai

# ======================================================
# GEMINI SETUP (PRIMARY CLASSIFIER)
# ======================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# ======================================================
# GEMINI STANCE CLASSIFIER
# ======================================================
def _gemini_classify_stance(claim: str, evidence: str) -> str | None:
    """
    Returns:
        "pro", "against", "neutral", or None if failed
    """

    if not client:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
You are an expert debate analyst.

TASK:
Classify the relationship between the CLAIM and the EVIDENCE.

LABELS:
- PRO → Evidence supports the claim
- AGAINST → Evidence contradicts or challenges the claim
- NEUTRAL → Evidence is unrelated or unclear

RULES:
- Focus only on logical meaning
- Ignore writing style
- If evidence weakly supports → PRO
- If evidence raises criticism or risks → AGAINST
- If unsure → NEUTRAL

Respond with ONLY one word:
pro
against
neutral

CLAIM:
"{claim}"

EVIDENCE:
"{evidence}"
"""
        )

        if not response.text:
            return None

        answer = response.text.strip().lower()

        if "pro" in answer:
            return "pro"
        if "against" in answer:
            return "against"
        if "neutral" in answer:
            return "neutral"

        return None

    except Exception as e:
        print("❌ Gemini stance error:", e)
        return None


# ======================================================
# HF ZERO-SHOT FALLBACK
# ======================================================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
} if HF_API_TOKEN else {}


def _hf_classify_stance(claim: str, evidence: str) -> str:
    """
    Returns: pro / against / neutral
    """

    text = f"Claim: {claim} Evidence: {evidence}"

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": [
                "supports the claim",
                "opposes the claim",
                "neutral or unrelated"
            ]
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            return "neutral"

        result = response.json()
        labels = result.get("labels", [])

        if not labels:
            return "neutral"

        top_label = labels[0].lower()

        if "support" in top_label:
            return "pro"
        elif "oppose" in top_label:
            return "against"
        else:
            return "neutral"

    except Exception:
        return "neutral"


# ======================================================
# MAIN MODULE FUNCTION
# ======================================================
def classify_evidence_stance(retrieved_results: list[dict]) -> list[dict]:
    """
    Input:
        Output from Module 4

    Output:
        [
            {
                "claim_id": int,
                "claim": str,
                "pro_evidence": [...],
                "against_evidence": [...],
                "neutral_evidence": [...]
            }
        ]
    """

    final_results = []

    for item in retrieved_results:

        claim_id = item.get("claim_id")
        claim_text = item.get("claim")
        label = item.get("label")

        pro_list = []
        against_list = []
        neutral_list = []

        # Only process debatable claims
        if label == "debatable":

            for chunk in item.get("evidence_chunks", []):

                content = chunk.get("content", "")
                source = chunk.get("source")
                url = chunk.get("url")

                if not content or len(content) < 50:
                    continue

                # ============================
                # Step 1: Gemini Classification
                # ============================
                stance = _gemini_classify_stance(claim_text, content)

                # ============================
                # Step 2: Fallback
                # ============================
                if not stance:
                    stance = _hf_classify_stance(claim_text, content)

                evidence_obj = {
                    "source": source,
                    "url": url,
                    "content": content
                }

                # ============================
                # Step 3: Store by category
                # ============================
                if stance == "pro":
                    pro_list.append(evidence_obj)

                elif stance == "against":
                    against_list.append(evidence_obj)

                else:
                    neutral_list.append(evidence_obj)

        final_results.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "pro_evidence": pro_list,
            "against_evidence": against_list,
            "neutral_evidence": neutral_list
        })

    return final_results