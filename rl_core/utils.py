"""
utils.py
────────
Shared utilities.  Contains:
  • build_judge_llm()         — constructs a dedicated judge LLM instance
  • JudgeChain                — wraps judge prompt + LLM into a callable chain
  • approx_token_count        — lightweight token estimator for retrieved docs
  • compute_routing_metrics   — confusion-matrix routing metrics (PDF p.9-10)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import CONFIG

logger = logging.getLogger("disaster_chatbot")


# ──────────────────────────────────────────────────────────────────────────────
# Judge LLM
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict judge evaluating a disaster-response assistant \
(domain: floods and landslides only).

Assess the response against ALL four criteria below, then output ONE final overall score (1-10).

1. **Relevance** — Did the response directly address what the user asked?
   Consider the conversation summary for context on the user's intent.
   Deduct points if the response is off-topic or answers a different question.

2. **Correctness / Faithfulness** — Are the facts consistent with the retrieved documents (if any) and general domain knowledge?
   Penalise heavily for hallucinations, dangerous advice, or contradictions with the retrieved context.

3. **Completeness** — Did the response answer enough of the user's query?
   Deduct points for important omissions (e.g. missing evacuation steps, no emergency contact given when relevant).

4. **Appropriateness of grounding** — action_taken={action_taken}  ("retrieve" = retrieval was used, "skip" = retrieval was skipped)
   - If action_taken=retrieve and the response correctly draws on the retrieved documents → positive contribution.
   - If action_taken=retrieve but the response ignores the retrieved context → deduct points.
   - If action_taken=skip and the response is still accurate and sufficient → positive contribution.
   - If action_taken=skip but the response contains uncertain or hallucinated domain facts → deduct points.

Combine your assessment of all four criteria into a single overall score.

Inputs
------
User query:
{query}

Conversation summary (prior context):
{conversation_summary}

Retrieved documents (empty if retrieval was skipped):
{retrieved_docs}

Generated response:
{response}

Output ONLY valid JSON on a single line — no markdown, no preamble:
{{"score": <int 1-10>, "reason": "<brief explanation under 40 words>"}}
"""

JUDGE_PROMPT = ChatPromptTemplate.from_template(_JUDGE_PROMPT_TEMPLATE)


def build_judge_llm():
    """Create a low-temperature LLM instance dedicated to judging."""
    import importlib

    provider = CONFIG.llm_provider
    model    = CONFIG.llm_model
    temp     = CONFIG.judge_temperature  # defaults to 0.0 for deterministic scoring

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    if provider == "openai":
        openai_module = importlib.import_module("langchain_openai")
        cls = getattr(openai_module, "ChatOpenAI")
        return cls(
            model=model,
            temperature=temp,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


class JudgeChain:
    """
    Wraps JUDGE_PROMPT + judge LLM into a simple callable.

    Usage:
        judge = JudgeChain()
        result = judge.invoke({
            "query": "...",
            "conversation_summary": "...",   # prior turn summary or empty string
            "action_taken": "retrieve",       # "retrieve" or "skip"
            "retrieved_docs": "...",           # formatted doc text or empty string
            "response": "...",
        })
        # result → {"score": 8, "reason": "..."}
    """

    def __init__(self):
        llm = build_judge_llm()
        self._chain = JUDGE_PROMPT | llm | StrOutputParser()

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        """Return parsed JSON dict with 'score' and 'reason'."""
        raw = self._chain.invoke(inputs)
        return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> dict[str, Any]:
    """Robustly parse the judge's JSON output."""
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data  = json.loads(text)
        score = max(1, min(10, int(data.get("score", 5))))
        return {"score": score, "reason": data.get("reason", "")}
    except Exception:
        pass

    numbers = re.findall(r"\b([1-9]|10)\b", text)
    score   = int(numbers[0]) if numbers else 5
    logger.warning("Judge output parse failed; defaulted score=%d. Raw: %r", score, raw)
    return {"score": score, "reason": raw[:120]}


# ──────────────────────────────────────────────────────────────────────────────
# Token cost helper
# ──────────────────────────────────────────────────────────────────────────────

def approx_token_count(docs: list[Document]) -> int:
    """
    Rough token estimate for retrieved docs (4 chars ≈ 1 token).
    Used to compute the retrieval cost term β * tokens.
    """
    total_chars = sum(len(d.page_content) for d in docs)
    return total_chars // 4


# ──────────────────────────────────────────────────────────────────────────────
# Routing metrics  (PDF p.9-10)
# ──────────────────────────────────────────────────────────────────────────────

def compute_routing_metrics(
    actions: list[int],
    retrieval_required: list[bool],
) -> dict[str, float]:
    """
    Compute confusion-matrix routing metrics from parallel action / label lists.

    PDF p.9-10 definitions:
      TP = retrieval chosen  AND retrieval required
      FP = retrieval chosen  BUT skip sufficient
      FN = skip chosen       BUT retrieval required
      TN = skip chosen       AND skip sufficient

    Returns:
      TP, FP, FN, TN                 — raw counts
      accuracy                       — (TP + TN) / total
      precision                      — TP / (TP + FP)
      recall                         — TP / (TP + FN)
      f1                             — 2PR / (P + R)
      unsafe_skip_rate               — FN / total   (most dangerous failure mode)
      wasteful_retrieve_rate         — FP / total   (efficiency waste)
    """
    if len(actions) != len(retrieval_required):
        raise ValueError("actions and retrieval_required must have the same length")

    tp = fp = fn = tn = 0
    for act, req in zip(actions, retrieval_required):
        if act == 1 and req:
            tp += 1
        elif act == 1 and not req:
            fp += 1
        elif act == 0 and req:
            fn += 1
        else:
            tn += 1

    total     = len(actions)
    precision = tp / (tp + fp)       if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total    if total > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy":               accuracy,
        "precision":              precision,
        "recall":                 recall,
        "f1":                     f1,
        "unsafe_skip_rate":       fn / total if total > 0 else 0.0,
        "wasteful_retrieve_rate": fp / total if total > 0 else 0.0,
    }