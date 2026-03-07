"""
============================================================
  Disaster-Domain RAG Chatbot  (Floods & Landslides only)
  Phase 1 – Always-retrieve, summary memory, LangGraph
============================================================
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import json
import logging
import ast
import importlib
from typing import Annotated, Any

import pandas as pd
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_chroma import Chroma

# LLM providers
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI

# Langfuse tracing (optional, lazy-imported)
get_langfuse_client = None
LangfuseCallbackHandler = None

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from config import CONFIG
from prompts import ANSWER_PROMPT, SUMMARY_PROMPT

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("disaster_chatbot")


def build_llm():
    """Factory – swap LLM provider via CONFIG without touching node code."""
    provider = CONFIG.llm_provider
    model = CONFIG.llm_model
    temp = CONFIG.llm_temperature

    if provider == "groq":
        return ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
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
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=CONFIG.embed_model)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ──────────────────────────────────────────────────────────────
# 3. INGESTION FUNCTION  (run once to populate Chroma)
# ──────────────────────────────────────────────────────────────


def _load_chunks_file(file_path: str) -> "pd.DataFrame":
    """
    Load chunk data from:
      1. JSON array:
             [ {"metadata": {...}, "chunk": "..."}, ... ]

    Returns a DataFrame with at minimum columns: 'chunk', 'metadata'.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    # ── 1. JSON array ──────────────────────────────────────────
    if raw.startswith("["):
        try:
            records = json.loads(raw)
            df = pd.DataFrame(records)
            logger.info("Loaded %d records from JSON array", len(df))
            return df
        except Exception as e:
            logger.warning("JSON array parse failed: %s", e)

    raise ValueError(
        f"Cannot parse {file_path}.\n"
        "Expected: JSON array [ {{...}}, ... ], JSON-lines, or a CSV with "
        "columns 'chunk' and 'metadata'."
    )


def load_or_create_vectorstore(
    json_path: str | None = None,
    force_recreate: bool = False,
) -> Chroma:
    """
    Load an existing Chroma collection from disk, or create it from a JSON file.

    File format — JSON array: [ {"metadata": {...}, "chunk": "..."}, ... ]

    Parameters
    ----------
    json_path       : path to your chunks JSON file (only used on first creation)
    force_recreate : if True, wipe the existing DB and rebuild from JSON
    """
    embeddings = build_embeddings()
    chroma_dir = CONFIG.chroma_dir
    collection = CONFIG.chroma_collection

    db_exists = os.path.isdir(chroma_dir) and any(
        f.endswith(".sqlite3") for f in os.listdir(chroma_dir)
    ) if os.path.isdir(chroma_dir) else False

    if db_exists and not force_recreate:
        logger.info("Loading existing Chroma DB from %s", chroma_dir)
        vectorstore = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        logger.info("Collection '%s' loaded (count=%s)",
                    collection, vectorstore._collection.count())
        return vectorstore

    # ── Build from JSON ──────────────────────────────────────────
    if json_path is None:
        json_path = CONFIG.chunks_path

    logger.info("Creating new Chroma DB from %s", json_path)
    df = _load_chunks_file(json_path)

    documents: list[Document] = []
    for _, row in df.iterrows():
        chunk = str(row["chunk"])

        # Safely parse metadata column (JSON string or Python dict string)
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

        # Chroma metadata values must be str / int / float / bool – sanitise
        clean_meta = {
            k: (str(v) if v is not None else "")
            for k, v in meta.items()
        }
        documents.append(Document(page_content=chunk, metadata=clean_meta))

    if not documents:
        raise ValueError("No documents loaded from CSV – check your file path and column names.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=chroma_dir,
    )
    logger.info("Chroma DB created with %d documents", len(documents))
    return vectorstore


# ──────────────────────────────────────────────────────────────
# 4. STATE DEFINITION
# ──────────────────────────────────────────────────────────────

class DisasterState(TypedDict):
    # Full message history for display; add_messages handles append semantics
    messages:       Annotated[list[AnyMessage], add_messages]
    # Running conversation summary (kept small deliberately)
    summary:        str
    # Cleaned user input for this turn
    query:          str
    # Documents retrieved this turn
    retrieved_docs: list[Document]
    # Final assistant answer
    answer:         str


# ──────────────────────────────────────────────────────────────
# 5. NODE FUNCTIONS
# ──────────────────────────────────────────────────────────────

# ── Singletons (initialised once at module level) ──────────────
_llm:         Any = None
_vectorstore: Any = None
_langfuse_handler: Any = None
_langfuse_initialized = False


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
        # Ensure client singleton is initialised before creating callbacks.
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


def _build_run_config(thread_id: str) -> dict[str, Any]:
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


def _flush_langfuse() -> None:
    """Flush telemetry so short-lived runs do not lose traces on shutdown."""
    if _langfuse_handler is None or get_langfuse_client is None:
        return

    try:
        get_langfuse_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush skipped: %s", exc)


# ── Node 1: retrieve ───────────────────────────────────────────
# NOTE ▼▼▼  Phase-2 insertion point ▼▼▼
# To add a "should_retrieve" decision node later:
#   1. Create a new node `decide_retrieval(state)` that returns {"skip_retrieval": True/False}
#   2. Add a conditional edge: START → decide_retrieval → retrieve | generate_answer
#   3. The retrieve node itself stays unchanged.
# ──────────────────────────────────────────────────────────────

def retrieve(state: DisasterState, config: RunnableConfig | None = None) -> dict:
    """
    Always retrieve top-K documents from Chroma for the current query.
    Phase 2: wrap with a conditional router to make retrieval optional.
    """
    query = state["query"]
    logger.info("[retrieve] query=%r", query)

    retriever = _get_vectorstore().as_retriever(
        search_type="similarity",               # swap to "mmr" for diversity
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


# ── Node 2: generate_answer ────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents with their source metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_tag = " | ".join(
            f"{k}: {v}" for k, v in meta.items() if v and v != ""
        )
        parts.append(f"[{i}] {doc.page_content}\n    Source → {source_tag}" if source_tag
                     else f"[{i}] {doc.page_content}")
    return "\n\n".join(parts) if parts else "No relevant context found."


def generate_answer(state: DisasterState, config: RunnableConfig | None = None) -> dict:
    """Generate a safety-focused answer using retrieved context + conversation summary."""
    llm     = _get_llm()
    docs    = state.get("retrieved_docs", [])
    context = _format_docs(docs)
    summary = state.get("summary", "")
    query   = state["query"]

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

    # Append the assistant turn to the message list
    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }


# ── Node 3: summarize_conversation ────────────────────────────


def summarize_conversation(state: DisasterState, config: RunnableConfig | None = None) -> dict:
    """
    Maintain a running summary instead of full history to keep the context window small.
    First turn: skip (no previous exchange to summarise yet).
    """
    messages = state.get("messages", [])

    # Need at least one user + one assistant message to summarise
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    ai_msgs    = [m for m in messages if isinstance(m, AIMessage)]

    if not human_msgs or not ai_msgs:
        return {"summary": state.get("summary", "")}

    last_user      = human_msgs[-1].content
    last_assistant = ai_msgs[-1].content
    current_summary = state.get("summary", "")

    logger.info("[summarize] updating summary")

    llm    = _get_llm()
    chain = SUMMARY_PROMPT | llm
    invoke_payload = {
        "max_tokens":    CONFIG.summary_max_tokens,
        "summary":       current_summary or "(no prior summary)",
        "user_msg":      last_user,
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


# ──────────────────────────────────────────────────────────────
# 6. EDGE / ROUTER LOGIC
# ──────────────────────────────────────────────────────────────
# Current flow (Phase 1):
#   START → retrieve → generate_answer → summarize_conversation → END
#
# Phase 2 – insert before retrieve:
#   START → [decide_retrieval] → retrieve (or skip) → generate_answer → ...
#
# The conditional edge would look like:
#   graph.add_conditional_edges(
#       "decide_retrieval",
#       lambda state: "retrieve" if state["should_retrieve"] else "generate_answer",
#       {"retrieve": "retrieve", "generate_answer": "generate_answer"},
#   )


# ──────────────────────────────────────────────────────────────
# 7. GRAPH CONSTRUCTION & COMPILATION
# ──────────────────────────────────────────────────────────────

def build_graph():
    """Construct, compile, and return the LangGraph StateGraph."""
    graph = StateGraph(DisasterState)

    # ── Register nodes ─────────────────────────────────────────
    graph.add_node("retrieve",              retrieve)
    graph.add_node("generate_answer",       generate_answer)
    graph.add_node("summarize_conversation", summarize_conversation)

    # ── Edges ──────────────────────────────────────────────────
    graph.add_edge(START,                   "retrieve")
    graph.add_edge("retrieve",              "generate_answer")
    graph.add_edge("generate_answer",       "summarize_conversation")
    graph.add_edge("summarize_conversation", END)

    # ── Checkpointing (thread-level persistence) ───────────────
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Graph compiled successfully")
    return compiled


# ──────────────────────────────────────────────────────────────
# 8. CHAT LOOP / EXAMPLE RUN
# ──────────────────────────────────────────────────────────────

def run_chat_loop(compiled_graph, thread_id: str = "default-session"):
    """
    Simple console chat loop.
    Each user session should use a unique thread_id so MemorySaver
    maintains separate checkpoints (summaries) per conversation.
    """
    print("\n" + "="*60)
    print("  Disaster Response Assistant  (floods & landslides)")
    print("  Type 'exit' or 'quit' to end | thread:", thread_id)
    print("="*60 + "\n")

    run_config = _build_run_config(thread_id)

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Assistant: Stay safe. Goodbye.")
                break

            # Build initial state for this turn
            input_state: DisasterState = {
                "messages":       [HumanMessage(content=user_input)],
                "query":          user_input,
                "summary":        "",        # MemorySaver will restore previous value
                "retrieved_docs": [],
                "answer":         "",
            }

            try:
                # stream() yields intermediate state dicts; we take the final answer
                final_state = None
                for event in compiled_graph.stream(input_state, config=run_config):
                    # Each event is {node_name: partial_state}
                    for node, partial in event.items():
                        if "answer" in partial and partial["answer"]:
                            final_state = partial

                if final_state and final_state.get("answer"):
                    print(f"\nAssistant: {final_state['answer']}\n")
                else:
                    print("\nAssistant: [No answer generated – please try again]\n")

            except Exception as exc:
                logger.error("Graph execution error: %s", exc, exc_info=True)
                print(f"\n[Error] {exc}\n")
    finally:
        _flush_langfuse()


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disaster RAG Chatbot")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Force re-ingest chunks.json into Chroma (wipes existing DB).",
    )
    parser.add_argument(
        "--json", default=CONFIG.chunks_path,
        help="Path to chunks file – JSON array (default: chunks.json).",
    )
    parser.add_argument(
        "--thread", default="session-001",
        help="Thread/session ID for memory checkpointing.",
    )
    args = parser.parse_args()

    # ── Optional one-time ingestion ────────────────────────────
    if args.ingest:
        logger.info("Ingestion requested – rebuilding vector store from %s", args.json)
        _vectorstore = load_or_create_vectorstore(json_path=args.json, force_recreate=True)
    else:
        # Lazy-load on first retrieval call
        logger.info("Vector store will be loaded lazily on first query.")

    # ── Build & run ────────────────────────────────────────────
    app = build_graph()
    run_chat_loop(app, thread_id=args.thread)


# ──────────────────────────────────────────────────────────────
# OPTIONAL: Programmatic / Streamlit integration snippet
# ──────────────────────────────────────────────────────────────
"""
# ── Streamlit integration example ──────────────────────────────
import streamlit as st

@st.cache_resource
def get_app():
    return build_graph()

def streamlit_chat():
    st.title("Disaster Response Assistant")
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about floods or landslides...")
    if user_input:
        st.session_state.display_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        app    = get_app()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "query": user_input, "summary": "", "retrieved_docs": [], "answer": "",
        }
        answer = ""
        for event in app.stream(input_state, config=config):
            for _, partial in event.items():
                if partial.get("answer"):
                    answer = partial["answer"]

        st.session_state.display_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__streamlit__":
    streamlit_chat()
"""