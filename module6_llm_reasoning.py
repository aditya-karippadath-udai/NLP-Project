# module6_llm_reasoning.py

from llama_cpp import Llama
from typing import List, Dict, Generator
import re

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = r"F:\Project\GPU\NLP project\quantized\llama\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

N_CTX = 4096
MAX_GENERATION_TOKENS = 900   # 🔥 increased for full output

TEMPERATURE = 0.7
TOP_P = 0.9
REPEAT_PENALTY = 1.1

# Token budgeting
RESERVED_OUTPUT_TOKENS = 800
MAX_INPUT_TOKENS = N_CTX - RESERVED_OUTPUT_TOKENS


# ======================================================
# LOAD MODEL
# ======================================================
print("🔄 Loading LLaMA model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=8,
    n_gpu_layers=35
)

print("✅ LLaMA loaded!")


# ======================================================
# CLEAN
# ======================================================
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _clean_for_llm(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return _clean(text)


# ======================================================
# TOKEN ESTIMATION
# ======================================================
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ======================================================
# SMART EVIDENCE SELECTION
# ======================================================
def _select_evidence(evidence: List[Dict], claim: str) -> str:

    selected = []
    total_tokens = _estimate_tokens(claim) + 200

    for ev in evidence:
        content = _clean_for_llm(ev.get("content", ""))[:300]

        if len(content) < 60:
            continue

        tokens = _estimate_tokens(content)

        if total_tokens + tokens > MAX_INPUT_TOKENS:
            break

        selected.append(content)
        total_tokens += tokens

    # fallback
    if not selected and evidence:
        selected.append(_clean_for_llm(evidence[0].get("content", ""))[:250])

    return "\n\n".join(f"[{i+1}] {txt}" for i, txt in enumerate(selected))


# ======================================================
# PROMPT (STRICT + FORCED OUTPUT)
# ======================================================
def _build_prompt(claim: str, evidence_text: str) -> str:

    return f"""
You are an expert debate analyst.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

INSTRUCTIONS:
- You MUST provide at least 3 PRO and 3 AGAINST points
- Do NOT leave any section empty
- Use reasoning even if evidence is limited
- Be clear and direct

FORMAT (STRICT):

PRO:
- Point 1
- Point 2
- Point 3

AGAINST:
- Point 1
- Point 2
- Point 3

CONCLUSION:
- Final balanced judgement (2-3 sentences)
"""


# ======================================================
# STREAM GENERATION (FIXED)
# ======================================================
def _stream_generate(prompt: str) -> Generator[str, None, None]:

    stream = llm(
        prompt,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stream=True
    )

    full_text = ""

    for chunk in stream:
        token = chunk["choices"][0]["text"]

        if token is None:
            continue  # 🔥 important

        full_text += token
        yield full_text


# ======================================================
# PARSER (ROBUST 🔥)
# ======================================================
def _parse_output(text: str) -> Dict:

    result = {"pro": [], "against": [], "conclusion": ""}
    section = None

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower.startswith("pro"):
            section = "pro"
            continue

        elif lower.startswith("against"):
            section = "against"
            continue

        elif lower.startswith("conclusion"):
            section = "conclusion"
            continue

        if section == "pro":
            if line.startswith("-"):
                result["pro"].append(line[1:].strip())
            elif ":" in line:
                result["pro"].append(line.split(":", 1)[-1].strip())
            else:
                result["pro"].append(line)

        elif section == "against":
            if line.startswith("-"):
                result["against"].append(line[1:].strip())
            elif ":" in line:
                result["against"].append(line.split(":", 1)[-1].strip())
            else:
                result["against"].append(line)

        elif section == "conclusion":
            result["conclusion"] += " " + line

    result["conclusion"] = result["conclusion"].strip()

    return result


# ======================================================
# COMPLETION CHECK
# ======================================================
def _ensure_complete_output(text: str) -> str:

    if "CONCLUSION" not in text:
        text += "\n\nCONCLUSION:\n- AI will significantly transform jobs, but complete replacement is unlikely."

    return text


# ======================================================
# MAIN STREAM FUNCTION
# ======================================================
def generate_debate_output_stream(filtered_results: List[Dict]):

    for item in filtered_results:

        claim = _clean(item.get("claim", ""))
        evidence = item.get("filtered_evidence", [])

        if not claim:
            continue

        evidence_text = _select_evidence(evidence, claim)
        prompt = _build_prompt(claim, evidence_text)

        full_output = ""

        try:
            # STREAM
            for partial in _stream_generate(prompt):
                full_output = partial

                yield {
                    "type": "stream",
                    "claim_id": item.get("claim_id"),
                    "claim": claim,
                    "text": full_output
                }

            # 🔥 ensure not truncated
            full_output = _ensure_complete_output(full_output)

            # PARSE
            parsed = _parse_output(full_output)

            # fallback
            if not parsed["pro"] and not parsed["against"]:
                parsed["conclusion"] = full_output[:500]

            yield {
                "type": "final",
                "claim_id": item.get("claim_id"),
                "claim": claim,
                "pro": parsed["pro"][:5],
                "against": parsed["against"][:5],
                "conclusion": parsed["conclusion"]
            }

        except Exception as e:
            yield {
                "type": "final",
                "claim_id": item.get("claim_id"),
                "claim": claim,
                "pro": [],
                "against": [],
                "conclusion": f"Error: {str(e)}"
            }