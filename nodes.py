"""State, node functions, and runtime helpers for the Disaster RAG chatbot."""

from __future__ import annotations

import ast
import importlib
import json
import logging
import os
from typing import Annotated, Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import TypedDict

from config import CONFIG
from prompts import ANSWER_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger("disaster_chatbot")

# Langfuse tracing (optional, lazy-imported)
get_langfuse_client = None
LangfuseCallbackHandler = None


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


# Singletons (initialized once at module level)
_llm: Any = None
_vectorstore: Any = None
_langfuse_handler: Any = None
_langfuse_initialized = False


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


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=CONFIG.embed_model)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_chunks_file(file_path: str) -> "pd.DataFrame":
    """Load chunk data from a JSON array file."""
    with open(file_path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    if raw.startswith("["):
        try:
            records = json.loads(raw)
            df = pd.DataFrame(records)
            logger.info("Loaded %d records from JSON array", len(df))
            return df
        except Exception as exc:
            logger.warning("JSON array parse failed: %s", exc)

    raise ValueError(
        f"Cannot parse {file_path}.\n"
        "Expected: JSON array [ {...}, ... ] with columns 'chunk' and 'metadata'."
    )


def load_or_create_vectorstore(
    json_path: str | None = None,
    force_recreate: bool = False,
) -> Chroma:
    """Load existing Chroma collection from disk, or create it from a JSON file."""
    global _vectorstore

    embeddings = build_embeddings()
    chroma_dir = CONFIG.chroma_dir
    collection = CONFIG.chroma_collection

    db_exists = os.path.isdir(chroma_dir) and any(
        file_name.endswith(".sqlite3") for file_name in os.listdir(chroma_dir)
    ) if os.path.isdir(chroma_dir) else False

    if db_exists and not force_recreate:
        logger.info("Loading existing Chroma DB from %s", chroma_dir)
        vectorstore = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        logger.info("Collection '%s' loaded (count=%s)", collection, vectorstore._collection.count())
        _vectorstore = vectorstore
        return vectorstore

    if json_path is None:
        json_path = CONFIG.chunks_path

    logger.info("Creating new Chroma DB from %s", json_path)
    df = _load_chunks_file(json_path)

    documents: list[Document] = []
    for _, row in df.iterrows():
        chunk = str(row["chunk"])

        raw_meta = row.get("metadata", "{}")
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta)
            except json.JSONDecodeError:
                try:
                    meta = ast.literal_eval(raw_meta)
                except Exception:
                    meta = {}
        elif isinstance(raw_meta, dict):
            meta = raw_meta
        else:
            meta = {}

        clean_meta = {
            key: (str(value) if value is not None else "")
            for key, value in meta.items()
        }
        documents.append(Document(page_content=chunk, metadata=clean_meta))

    if not documents:
        raise ValueError("No documents loaded from JSON - check your file path and schema.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=chroma_dir,
    )
    logger.info("Chroma DB created with %d documents", len(documents))
    _vectorstore = vectorstore
    return vectorstore


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


def _get_langfuse_handler():
    """Create a singleton Langfuse callback handler when credentials exist."""
    global _langfuse_handler, _langfuse_initialized
    global get_langfuse_client, LangfuseCallbackHandler

    if _langfuse_initialized:
        return _langfuse_handler

    _langfuse_initialized = True

    tracing_enabled = CONFIG.langfuse_enabled and _env_flag(
        "LANGFUSE_TRACING_ENABLED", default=True
    )
    if not tracing_enabled:
        logger.info("Langfuse tracing disabled via configuration.")
        return None

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Langfuse keys not found in environment; tracing disabled.")
        return None

    if get_langfuse_client is None or LangfuseCallbackHandler is None:
        try:
            langfuse_module = importlib.import_module("langfuse")
            langfuse_lc_module = importlib.import_module("langfuse.langchain")
            get_langfuse_client = getattr(langfuse_module, "get_client")
            LangfuseCallbackHandler = getattr(langfuse_lc_module, "CallbackHandler")
        except Exception:
            logger.warning("Langfuse SDK not installed; tracing disabled.")
            return None

    try:
        get_langfuse_client()
        _langfuse_handler = LangfuseCallbackHandler()
        logger.info("Langfuse tracing enabled.")
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse tracing: %s", exc)
        _langfuse_handler = None

    return _langfuse_handler


def _extract_invoke_config(config: RunnableConfig | None) -> dict[str, Any]:
    """Forward callbacks/metadata from graph runtime config to LangChain calls."""
    if not config:
        return {}

    invoke_config: dict[str, Any] = {}
    callbacks = config.get("callbacks")
    metadata = config.get("metadata")

    if callbacks:
        invoke_config["callbacks"] = callbacks
    if metadata:
        invoke_config["metadata"] = metadata

    return invoke_config


def build_run_config(thread_id: str) -> dict[str, Any]:
    """Build LangGraph run config and attach Langfuse callbacks when available."""
    run_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    handler = _get_langfuse_handler()

    if handler is not None:
        run_config["callbacks"] = [handler]
        run_config["metadata"] = {
            "langfuse_session_id": thread_id,
            "langfuse_tags": list(CONFIG.langfuse_tags),
        }

    return run_config


def flush_langfuse() -> None:
    """Flush telemetry so short-lived runs do not lose traces on shutdown."""
    if _langfuse_handler is None or get_langfuse_client is None:
        return

    try:
        get_langfuse_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush skipped: %s", exc)


def retrieve(state: DisasterState, config: RunnableConfig | None = None) -> dict:
    """Always retrieve top-K documents from Chroma for the current query."""
    query = state["query"]
    logger.info("[retrieve] query=%r", query)

    retriever = _get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG.top_k},
    )
    invoke_config = _extract_invoke_config(config)
    docs = (
        retriever.invoke(query, config=invoke_config)
        if invoke_config
        else retriever.invoke(query)
    )
    logger.info("[retrieve] retrieved %d docs", len(docs))
    return {"retrieved_docs": docs}


def _format_docs(docs: list[Document]) -> str:
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
    return "\n\n".join(parts) if parts else "No relevant context found."


def generate_answer(state: DisasterState, config: RunnableConfig | None = None) -> dict:
    """Generate a safety-focused answer using retrieved context and summary."""
    llm = _get_llm()
    docs = state.get("retrieved_docs", [])
    context = _format_docs(docs)
    summary = state.get("summary", "")
    query = state["query"]

    logger.info("[generate_answer] building prompt (context docs=%d)", len(docs))

    chain = ANSWER_PROMPT | llm
    invoke_payload = {
        "summary": summary,
        "context": context,
        "question": query,
    }
    invoke_config = _extract_invoke_config(config)
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


def summarize_conversation(
    state: DisasterState,
    config: RunnableConfig | None = None,
) -> dict:
    """Maintain a running summary to keep the context window small."""
    messages = state.get("messages", [])

    human_msgs = [message for message in messages if isinstance(message, HumanMessage)]
    ai_msgs = [message for message in messages if isinstance(message, AIMessage)]

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
    invoke_config = _extract_invoke_config(config)
    result = (
        chain.invoke(invoke_payload, config=invoke_config)
        if invoke_config
        else chain.invoke(invoke_payload)
    )

    new_summary = result.content if hasattr(result, "content") else str(result)
    logger.info("[summarize] new summary length=%d chars", len(new_summary))
    return {"summary": new_summary}
