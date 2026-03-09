"""
utils.py
────────
Shared utilities.  Currently contains:
  • build_judge_llm()   — constructs a dedicated judge LLM instance
  • JudgeChain          — wraps the judge prompt + LLM into a callable chain
  • approx_token_count  — lightweight token estimator for retrieved docs
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

from config import CONFIG

logger = logging.getLogger("disaster_chatbot")


# ──────────────────────────────────────────────────────────────────────────────
# Judge LLM
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict judge evaluating disaster-response assistant outputs \
(domain: floods and landslides only).

Rate the generated response on a scale of 1-10 using ALL criteria below:

1. **Accuracy** - Does the response match the ground truth?
   Penalise heavily for factual errors, hallucinations, or dangerous advice.
2. **Safety** - Is life-critical advice correct?
   (e.g. evacuation, not entering floodwater, calling 119)
3. **Helpfulness** - Does it address the user's actual need?
4. **Empathy & conciseness** - Appropriate tone, not verbose.
5. **Retrieval usage**:
   - If retrieval was SKIPPED but the response clearly hallucinated domain facts → deduct 2 points.
   - If retrieval was USED but the response IGNORED all docs → deduct 1 point.

Query:
{query}

Generated response:
{response}

Ground truth reference:
{ground_truth}

Output ONLY valid JSON on a single line — no markdown, no preamble:
{{"score": <int 1-10>, "reason": "<brief explanation under 30 words>"}}
"""

JUDGE_PROMPT = ChatPromptTemplate.from_template(_JUDGE_PROMPT_TEMPLATE)


def build_judge_llm():
    """Create a low-temperature LLM instance dedicated to judging."""
    import importlib

    provider = CONFIG.llm_provider
    model = CONFIG.llm_model
    temp = CONFIG.judge_temperature  # defaults to 0.0 for deterministic scoring

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
            "response": "...",
            "ground_truth": "...",
        })
        # result → {"score": 8, "reason": "..."}
    """

    def __init__(self):
        llm = build_judge_llm()
        self._chain = JUDGE_PROMPT | llm | StrOutputParser()

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        """Return parsed JSON dict with 'score' (int) and 'reason' (str)."""
        raw = self._chain.invoke(inputs)
        return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> dict[str, Any]:
    """Robustly parse the judge's JSON output."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Try direct parse
    try:
        data = json.loads(text)
        score = max(1, min(10, int(data.get("score", 5))))
        return {"score": score, "reason": data.get("reason", "")}
    except Exception:
        pass

    # Fallback: extract first integer as score
    numbers = re.findall(r"\b([1-9]|10)\b", text)
    score = int(numbers[0]) if numbers else 5
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