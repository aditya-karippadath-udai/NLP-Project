import re
import hashlib
from typing import List, Dict
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# LOAD MINI LM (PYTORCH ONLY 🔥)
# ======================================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()


# ======================================================
# CONFIG
# ======================================================
MIN_TEXT_LENGTH = 80
MAX_TEXT_LENGTH = 500
TOP_K = 10
MAX_PER_SOURCE = 2

MIN_RELEVANCE = 2
MIN_ARGUMENT_SCORE = 1

SEMANTIC_WEIGHT = 3.0


# ======================================================
# CLEAN
# ======================================================
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


# ======================================================
# GENERIC FILTER
# ======================================================
GENERIC_PATTERNS = [
    "this article", "we explore", "this study",
    "in this article", "this chapter",
    "discussion", "learn more", "click here",
    "sign up", "newsletter", "advertisement",
    "the question of", "we examine"
]


def _is_generic(text: str) -> bool:
    return any(p in text.lower() for p in GENERIC_PATTERNS)


# ======================================================
# WEAK FILTER
# ======================================================
def _is_weak(text: str) -> bool:
    t = text.lower()

    weak_patterns = [
        "experts say", "studies show",
        "it is believed", "it is expected",
        "many believe"
    ]

    if any(p in t for p in weak_patterns):
        if not re.search(r"\d+", t):
            return True

    return False


# ======================================================
# TOKENIZE
# ======================================================
def _tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))


# ======================================================
# RELEVANCE
# ======================================================
def _relevance(claim: str, text: str) -> int:
    return len(_tokenize(claim).intersection(_tokenize(text)))


# ======================================================
# ARG SCORE
# ======================================================
ARG_KEYWORDS = [
    "job", "employment", "replace", "automation",
    "workers", "labor", "economy", "impact",
    "increase", "decrease", "growth", "loss"
]


def _arg_score(text: str) -> int:
    return sum(1 for w in ARG_KEYWORDS if w in text.lower())


# ======================================================
# EMBEDDING (MEAN POOLING)
# ======================================================
def _embed(texts: List[str]):

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()


# ======================================================
# COSINE SIMILARITY
# ======================================================
def _cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# ======================================================
# MAIN FUNCTION
# ======================================================
def filter_and_rank_evidence(retrieved_results: List[Dict]) -> List[Dict]:

    final_results = []

    for item in retrieved_results:

        if item.get("label") != "debatable":
            continue

        claim = _clean(item.get("claim", ""))
        chunks = item.get("evidence_chunks", [])

        candidates = []
        seen = set()
        source_count = defaultdict(int)

        texts = []
        meta = []

        # =========================
        # FILTER
        # =========================
        for ch in chunks:

            text = _clean(ch.get("content", ""))

            if len(text) < MIN_TEXT_LENGTH:
                continue

            if _is_generic(text) or _is_weak(text):
                continue

            rel = _relevance(claim, text)
            arg = _arg_score(text)

            if rel < MIN_RELEVANCE or arg < MIN_ARGUMENT_SCORE:
                continue

            text = text[:MAX_TEXT_LENGTH]

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue

            source = ch.get("source", "")
            if source_count[source] >= MAX_PER_SOURCE:
                continue

            seen.add(h)
            source_count[source] += 1

            texts.append(text)

            meta.append({
                "text": text,
                "source": source,
                "url": ch.get("url", ""),
                "rel": rel,
                "arg": arg
            })

        # =========================
        # SEMANTIC SIMILARITY 🔥
        # =========================
        if texts:
            claim_emb = _embed([claim])[0]
            text_embs = _embed(texts)

            for i, m in enumerate(meta):

                sim = _cosine(claim_emb, text_embs[i])

                score = (
                    (1.5 * m["rel"]) +
                    (2.0 * m["arg"]) +
                    (SEMANTIC_WEIGHT * sim)
                )

                candidates.append({
                    "source": m["source"],
                    "url": m["url"],
                    "content": m["text"],
                    "score": round(score, 4),
                    "semantic": round(float(sim), 4)
                })

        # =========================
        # SORT
        # =========================
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # =========================
        # FALLBACK
        # =========================
        if not candidates:
            fallback = []
            for ch in chunks[:TOP_K]:
                text = _clean(ch.get("content", ""))
                if len(text) > 50:
                    fallback.append({
                        "source": ch.get("source", ""),
                        "url": ch.get("url", ""),
                        "content": text[:MAX_TEXT_LENGTH],
                        "score": 0.1,
                        "semantic": 0.0
                    })
            candidates = fallback

        final_results.append({
            "claim_id": item.get("claim_id"),
            "claim": claim,
            "filtered_evidence": candidates[:TOP_K]
        })

    return final_results