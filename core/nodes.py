"""State and node functions for the chatbot."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

from core.config import CONFIG
from core.ingestion import load_or_create_vectorstore
from core.observability import extract_invoke_config
from core.prompts import ANSWER_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger("disaster_chatbot")


class DisasterState(TypedDict):
    # Full message history for display; add_messages handles append semantics
    messages: Annotated[list[AnyMessage], add_messages]
    # Running conversation summary (kept small deliberately)
    summary: str
    # Cleaned user input for this turn
    query: str
    # Documents retrieved this turn
    retrieved_docs: list[Document]
    # Final assistant answer
    answer: str
    # Retrieval decision: 1 = retrieve, 0 = skip  (set by decide_retrieve node)
    action: int
    # Graph operation mode: "baseline" forces action=1; "policy" uses the RL policy
    mode: Literal["baseline", "policy"]


# ──────────────────────────────────────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────────────────────────────────────
_llm: Any = None
_vectorstore: Any = None
# Lazy-loaded RL policy (only needed in "policy" mode)
_rl_policy: Any = None


def build_llm():
    """Factory - swap LLM provider via CONFIG without touching node code."""
    provider = CONFIG.llm_provider
    model = CONFIG.llm_model
    temp = CONFIG.llm_temperature

    if provider == "groq":
        return ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    if provider == "openai":
        try:
            openai_module = importlib.import_module("langchain_openai")
            chat_openai_cls = getattr(openai_module, "ChatOpenAI")
        except Exception as exc:
            raise ImportError(
                "OpenAI provider selected but 'langchain_openai' is not installed."
            ) from exc

        return chat_openai_cls(
            model=model,
            temperature=temp,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_or_create_vectorstore()
    return _vectorstore


def _load_rl_policy():
    """Lazy-load the saved PPO policy.  Returns None if not available."""
    global _rl_policy
    if _rl_policy is not None:
        return _rl_policy
    try:
        from stable_baselines3 import PPO
        path = CONFIG.policy_model_path
        if os.path.exists(path):
            _rl_policy = PPO.load(path)
            logger.info("RL policy loaded from %s", path)
        else:
            logger.warning("Policy file %s not found; falling back to baseline.", path)
    except Exception as exc:
        logger.warning("Could not load RL policy: %s", exc)
    return _rl_policy


# ──────────────────────────────────────────────────────────────────────────────
# decide_retrieve node
# ──────────────────────────────────────────────────────────────────────────────

def decide_retrieve(state: DisasterState, config: RunnableConfig) -> dict:
    """
    Decide whether to retrieve (action=1) or skip retrieval (action=0).

    Priority order:
      1. config["configurable"]["forced_action"] - injected per-invocation by RL env
      2. mode="baseline" - always retrieve
      3. mode="policy"   - ask the RL policy
    """
    forced = (config or {}).get("configurable", {}).get("forced_action")
    if forced is not None:
        logger.info("[decide_retrieve] forced_action=%d (from config)", forced)
        return {"action": int(forced)}

    mode = state.get("mode", "baseline")
    if mode == "baseline":
        logger.info("[decide_retrieve] baseline mode → retrieve")
        return {"action": 1}

    # policy mode
    policy = _load_rl_policy()
    if policy is None:
        logger.warning("[decide_retrieve] policy unavailable → fallback retrieve")
        return {"action": 1}

    from rl_core.state_encoder import encode_state  # local import to avoid circular deps
    obs = encode_state(state.get("summary", ""), state["query"])
    action_arr, _ = policy.predict(obs, deterministic=True)
    action = int(action_arr)
    logger.info("[decide_retrieve] policy action=%d", action)
    return {"action": action}


def retrieval_router(state: DisasterState) -> str:
    """Conditional edge function: route to 'retrieve' or 'generate_answer'."""
    return "retrieve" if state.get("action", 1) == 1 else "generate_answer"


# ──────────────────────────────────────────────────────────────────────────────
# retrieve node
# ──────────────────────────────────────────────────────────────────────────────

def retrieve(state: DisasterState, config: RunnableConfig) -> dict:
    """Always retrieve top-K documents from Chroma for the current query."""
    query = state["query"]
    logger.info("[retrieve] query=%r", query)

    retriever = _get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG.top_k},
    )
    invoke_config = extract_invoke_config(config)
    docs = (
        retriever.invoke(query, config=invoke_config)
        if invoke_config
        else retriever.invoke(query)
    )
    logger.info("[retrieve] retrieved %d docs", len(docs))
    return {"retrieved_docs": docs}


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No retrieved context available for this turn."
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_tag = " | ".join(
            f"{key}: {value}" for key, value in meta.items() if value and value != ""
        )
        if source_tag:
            parts.append(f"[{i}] {doc.page_content}\n    Source -> {source_tag}")
        else:
            parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# generate_answer node
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(state: DisasterState, config: RunnableConfig) -> dict:
    """Generate a safety-focused answer using retrieved context and summary."""
    llm = _get_llm()
    # retrieved_docs may be empty if retrieval was skipped
    docs = state.get("retrieved_docs", [])
    context = _format_docs(docs)
    summary = state.get("summary", "")
    query = state["query"]

    logger.info(
        "[generate_answer] building prompt (context docs=%d, retrieval_skipped=%s)",
        len(docs),
        len(docs) == 0,
    )

    chain = ANSWER_PROMPT | llm
    invoke_payload = {
        "summary": summary,
        "context": context,
        "question": query,
    }
    invoke_config = extract_invoke_config(config)
    result = (
        chain.invoke(invoke_payload, config=invoke_config)
        if invoke_config
        else chain.invoke(invoke_payload)
    )

    answer = result.content if hasattr(result, "content") else str(result)
    logger.info("[generate_answer] answer length=%d chars", len(answer))
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ──────────────────────────────────────────────────────────────────────────────
# summarize_conversation node
# ──────────────────────────────────────────────────────────────────────────────

def summarize_conversation(
    state: DisasterState,
    config: RunnableConfig,
) -> dict:
    """Maintain a running summary to keep the context window small."""
    messages = state.get("messages", [])

    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]

    if not human_msgs or not ai_msgs:
        return {"summary": state.get("summary", "")}

    last_user = human_msgs[-1].content
    last_assistant = ai_msgs[-1].content
    current_summary = state.get("summary", "")

    logger.info("[summarize] updating summary")

    llm = _get_llm()
    chain = SUMMARY_PROMPT | llm
    invoke_payload = {
        "max_tokens": CONFIG.summary_max_tokens,
        "summary": current_summary or "(no prior summary)",
        "user_msg": last_user,
        "assistant_msg": last_assistant,
    }
    invoke_config = extract_invoke_config(config)
    result = (
        chain.invoke(invoke_payload, config=invoke_config)
        if invoke_config
        else chain.invoke(invoke_payload)
    )

    new_summary = result.content if hasattr(result, "content") else str(result)
    logger.info("[summarize] new summary length=%d chars", len(new_summary))
    return {"summary": new_summary}