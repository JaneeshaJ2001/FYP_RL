"""
utils.py
--------
Shared utilities.  Contains:
  - build_judge_llm()        : constructs a dedicated judge LLM instance
  - JudgeChain               : wraps judge prompt + LLM into a callable chain
  - approx_token_count       : lightweight token estimator for retrieved docs
  - format_chunks_for_judge  : truncates docs to 300-char chunks for judge prompt
  - compute_routing_metrics  : confusion-matrix routing metrics 
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


# ==============================================================================
# JUDGE PROMPT  —  Grounded-Summary-Aware
# ==============================================================================

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict but fair judge evaluating a disaster-response assistant (floods & landslides ONLY).

BOT SCOPE: This assistant is strictly limited to floods and landslides. It must never hallucinate outside this domain and must refer to official hotlines (119 / DMC) when appropriate.

===================================================================
SCORING CRITERIA (combine into ONE overall integer score 0-10)
===================================================================

1. RELEVANCE
   Did the assistant directly address the current user query?

2. CORRECTNESS
   Is the answer factually accurate based ONLY on the information available at this turn?

3. GROUNDEDNESS (most important)
   action_taken = {action_taken}   ("retrieve" or "skip")

   Retrieval rule (follow this exactly):
   - Retrieval is needed ONLY when the conversation summary is INSUFFICIENT to answer the query.
   - Retrieval is NOT needed when the summary already contains enough grounded knowledge (recap of previously retrieved content).
   - For OOD, out-of-scope, or truly unanswerable queries, retrieval is also NOT needed — the assistant should politely decline or refer to hotlines.

   • If action_taken = "retrieve":
     The answer MUST be supported by the retrieved chunks shown below.
     Reward clear use of chunk content. Penalise answers that ignore or contradict the chunks.

   • If action_taken = "skip":
     The assistant correctly relied on the conversation summary (grounded knowledge from prior turns).
     PREVIOUSLY DISCOVERED GROUNDED KNOWLEDGE IN THE SUMMARY IS VALID AND SUFFICIENT EVIDENCE.
     If the assistant used that prior knowledge correctly, give FULL CREDIT for groundedness.
     DO NOT penalise a skip when the summary was already sufficient. Only deduct if the answer introduces new factual claims that are NOT in the summary.

4. COMPLETENESS
   Did the answer provide enough useful, actionable information (not vague or partial)?

5. SCOPE HANDLING
   If the query is outside scope or truly unanswerable with current knowledge, the assistant must politely decline or refer to hotlines instead of hallucinating.

===================================================================
SCORE BANDS (overall quality)
===================================================================
0-2  = wrong / unsafe / hallucinated
3-4  = poor (major gaps or errors)
5-6  = acceptable (correct but shallow)
7-8  = good (solid, grounded, actionable)
9-10 = excellent (complete, precise, perfectly grounded)

===================================================================
INPUTS FOR THIS TURN
===================================================================

Current user query:
{query}

Conversation summary (grounded knowledge from prior turns):
{conversation_summary}

Action taken this turn: {action_taken} ("retrieve" or "skip")

Retrieved chunks (only present if action="retrieve"; otherwise "No retrieval performed"):
{retrieved_chunks}

Assistant's answer:
{response}

===================================================================
OUTPUT (exactly one line)
===================================================================
Output ONLY valid JSON — nothing else:
{{"score": <integer 0-10>, "reason": "<brief explanation (≤50 words) referencing the 5 criteria>"}}
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

    invoke() input keys:
    -------------------------------------------------------
    Required:
        query               : str  — current user query
        conversation_summary: str  — grounded knowledge base from prior turns
        action_taken        : str  — "retrieve" | "skip"
        retrieved_chunks    : str  — use format_chunks_for_judge(docs) for retrieve
                                     turns; pass "" or omit entirely for skip turns
        response            : str  — assistant answer this turn

    Returns: {"score": int (0-10), "reason": str}
    """

    def __init__(self):
        llm = build_judge_llm()
        self._chain = JUDGE_PROMPT | llm | StrOutputParser()

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        """Return parsed JSON dict with 'score' (int 0-10) and 'reason' (str).
        """
        full_inputs = dict(inputs)  # shallow copy — don't mutate caller's dict

        # Ensure retrieved_chunks has a safe default for skip turns.
        if not full_inputs.get("retrieved_chunks"):
            full_inputs["retrieved_chunks"] = "No retrieval performed"

        raw = self._chain.invoke(full_inputs)
        return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> dict[str, Any]:
    """Robustly parse the judge's JSON output."""
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data  = json.loads(text)
        score = max(0, min(10, int(data.get("score", 5))))
        return {"score": score, "reason": data.get("reason", "")}
    except Exception:
        pass

    numbers = re.findall(r"\b([0-9]|10)\b", text)
    score   = int(numbers[0]) if numbers else 5
    logger.warning("Judge output parse failed; defaulted score=%d. Raw: %r", score, raw)
    return {"score": score, "reason": raw[:120]}


# ------------------------------------------------------------------------------
# Token cost helper
# ------------------------------------------------------------------------------

def approx_token_count(docs: list[Document]) -> int:
    """
    Rough token estimate for retrieved docs (4 chars ≈ 1 token).
    Used to compute the retrieval cost term: beta * tokens.
    Cost is 0 when action=0 (skip) — no docs are retrieved.
    """
    total_chars = sum(len(d.page_content) for d in docs)
    return total_chars // 4


def format_chunks_for_judge(docs: list[Document], max_chars_per_chunk: int = 300) -> str:
    """
    Format retrieved documents for the judge prompt.

    Each chunk is truncated to `max_chars_per_chunk` characters so the judge
    prompt stays concise while still providing enough content to verify groundedness.

    Returns "No retrieval performed" when docs is empty (skip turns).
    """
    if not docs:
        return "No retrieval performed"

    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk] + "..."
        parts.append(f"[Chunk {i}] {content}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------
# Routing metrics
# ------------------------------------------------------------------------------

def compute_routing_metrics(
    actions: list[int],
    retrieval_required: list[bool],
) -> dict[str, float]:
    """
    Compute confusion-matrix routing metrics from parallel action / label lists.

    definitions:
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