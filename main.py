"""
============================================================
  Disaster-Domain RAG Chatbot  (Floods & Landslides only)
  Phase 1 – Always-retrieve, summary memory, LangGraph
============================================================

Directory layout expected:
  disaster_chatbot.py   ← this file
  chunks.csv            ← your pre-chunked data (columns: chunk + metadata JSON)
  .env                  ← GROQ_API_KEY=... (or OPENAI_API_KEY=...)
  chroma_db_disaster/   ← auto-created on first run
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import json
import logging
import ast
from typing import Annotated, Any

import pandas as pd
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_chroma import Chroma

# LLM providers (swap via CFG below)
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("disaster_chatbot")


# ──────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────
CFG = {
    # ── Vector store ──
    "chroma_dir":        "./chroma_db_disaster",
    "chroma_collection": "disaster_rag",
    "top_k":             3,

    # ── CSV ingestion ──
    # Expected columns: "chunk" (str) and "metadata" (JSON string or dict)
    "chunks_path":       "chunks.json",   # JSON array, JSONL, or CSV

    # ── Embeddings ──
    "embed_model":       "sentence-transformers/all-MiniLM-L6-v2",

    # ── LLM ──
    # Switch provider by changing "provider" + "model"
    "llm_provider":      "groq",          # "groq" | "openai" | "ollama"
    "llm_model":         "llama-3.3-70b-versatile",
    "llm_temperature":   0.1,

    # ── Memory ──
    "summary_max_tokens": 300,
}


def build_llm():
    """Factory – swap LLM provider via CFG without touching node code."""
    provider = CFG["llm_provider"]
    model    = CFG["llm_model"]
    temp     = CFG["llm_temperature"]

    if provider == "groq":
        return ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        return ChatOpenAI(          # noqa: F821  (import at top when needed)
            model=model,
            temperature=temp,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=CFG["embed_model"])


# ──────────────────────────────────────────────────────────────
# 3. INGESTION FUNCTION  (run once to populate Chroma)
# ──────────────────────────────────────────────────────────────


def _load_chunks_file(file_path: str) -> "pd.DataFrame":
    """
    Load chunk data from:
      1. JSON array  (your current format):
             [ {"metadata": {...}, "chunk": "..."}, ... ]
      2. Bare JSON objects, one per line (JSONL):
             {"metadata": {...}, "chunk": "..."}
             {"metadata": {...}, "chunk": "..."}
      3. Bare JSON objects with no separator (auto-repaired):
             {"metadata": {...}, "chunk": "..."}{"metadata": {...}, "chunk": "..."}
      4. CSV with columns 'chunk' and 'metadata'

    Returns a DataFrame with at minimum columns: 'chunk', 'metadata'.
    """
    import re as _re

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

    # ── 2. JSON-lines (one valid JSON object per line) ─────────
    if raw.startswith("{"):
        try:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            df = pd.DataFrame(records)
            logger.info("Loaded %d records from JSON-lines", len(df))
            return df
        except Exception:
            pass

        # ── 3. Bare objects with no newline / comma separator ──
        try:
            fixed = "[" + _re.sub(r"}\s*{", "},{", raw) + "]"
            records = json.loads(fixed)
            df = pd.DataFrame(records)
            logger.info("Loaded %d records after JSON repair", len(df))
            return df
        except Exception as e:
            logger.warning("JSON repair failed: %s", e)

    # ── 4. CSV fallback ────────────────────────────────────────
    for kwargs in [
        {"quotechar": '"', "doublequote": True, "engine": "c"},
        {"engine": "python"},
        {"engine": "python", "on_bad_lines": "skip"},
    ]:
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info("Loaded %d rows from CSV (%s)", len(df), kwargs)
            return df
        except Exception as e:
            logger.debug("CSV attempt failed (%s): %s", kwargs, e)

    raise ValueError(
        f"Cannot parse {file_path}.\n"
        "Expected: JSON array [ {{...}}, ... ], JSON-lines, or a CSV with "
        "columns 'chunk' and 'metadata'."
    )


def load_or_create_vectorstore(
    csv_path: str | None = None,
    force_recreate: bool = False,
) -> Chroma:
    """
    Load an existing Chroma collection from disk, or create it from a CSV.

    File format — accepts any of:
      1. JSON array (your current format):
           [ {"metadata": {...}, "chunk": "..."}, ... ]
      2. JSON-lines: one JSON object per line
      3. CSV with columns "chunk" and "metadata" (save with QUOTE_ALL)

    Parameters
    ----------
    csv_path       : path to your chunks CSV (only used on first creation)
    force_recreate : if True, wipe the existing DB and rebuild from CSV
    """
    embeddings = build_embeddings()
    chroma_dir = CFG["chroma_dir"]
    collection = CFG["chroma_collection"]

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

    # ── Build from CSV ──────────────────────────────────────────
    if csv_path is None:
        csv_path = CFG["chunks_path"]

    logger.info("Creating new Chroma DB from %s", csv_path)
    df = _load_chunks_file(csv_path)

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


def _truncate_text(text: str, max_chars: int = 220) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _serialize_message(msg: Any) -> dict[str, str]:
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or "message")
    else:
        role = getattr(msg, "type", msg.__class__.__name__)

    if isinstance(msg, dict):
        raw_content = msg.get("content", "")
    else:
        raw_content = getattr(msg, "content", "")

    content = raw_content if isinstance(raw_content, str) else str(raw_content)
    return {
        "role": role,
        "content": _truncate_text(content, max_chars=300),
    }


def _state_snapshot_for_print(state_values: dict[str, Any]) -> dict[str, Any]:
    docs = state_values.get("retrieved_docs", []) or []
    messages = state_values.get("messages", []) or []

    docs_preview = []
    for i, doc in enumerate(docs, 1):
        if isinstance(doc, Document):
            docs_preview.append({
                "rank": i,
                "metadata": doc.metadata,
                "content_preview": _truncate_text(doc.page_content),
            })
        else:
            docs_preview.append({
                "rank": i,
                "value": _truncate_text(str(doc)),
            })

    return {
        "query": state_values.get("query", ""),
        "summary": state_values.get("summary", ""),
        "answer": _truncate_text(state_values.get("answer", ""), max_chars=500),
        "message_count": len(messages),
        "messages_last_6": [_serialize_message(m) for m in messages[-6:]],
        "retrieved_docs": docs_preview,
    }


# ──────────────────────────────────────────────────────────────
# 5. NODE FUNCTIONS
# ──────────────────────────────────────────────────────────────

# ── Singletons (initialised once at module level) ──────────────
_llm:         Any = None
_vectorstore: Any = None


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


# ── Node 1: retrieve ───────────────────────────────────────────
# NOTE ▼▼▼  Phase-2 insertion point ▼▼▼
# To add a "should_retrieve" decision node later:
#   1. Create a new node `decide_retrieval(state)` that returns {"skip_retrieval": True/False}
#   2. Add a conditional edge: START → decide_retrieval → retrieve | generate_answer
#   3. The retrieve node itself stays unchanged.
# ──────────────────────────────────────────────────────────────

def retrieve(state: DisasterState) -> dict:
    """
    Always retrieve top-K documents from Chroma for the current query.
    Phase 2: wrap with a conditional router to make retrieval optional.
    """
    query = state["query"]
    logger.info("[retrieve] query=%r", query)

    retriever = _get_vectorstore().as_retriever(
        search_type="similarity",               # swap to "mmr" for diversity
        search_kwargs={"k": CFG["top_k"]},
    )
    docs = retriever.invoke(query)
    logger.info("[retrieve] retrieved %d docs", len(docs))
    return {"retrieved_docs": docs}


# ── Node 2: generate_answer ────────────────────────────────────

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful, calm, authoritative disaster response assistant specialised \
ONLY in floods and landslides.

STRICT RULES:
• NEVER give medical, legal, or financial advice beyond official source citations.
• ALWAYS prioritise life safety: evacuate to higher ground, never enter floodwater, etc.
• Use ONLY the retrieved context below. If information is missing or uncertain, say so \
clearly and direct the user to call official emergency lines (119 in Sri Lanka, \
Disaster Management Centre – DMC, or local police).
• Cite sources briefly where possible, e.g. [UN OCHA], [NDMA guideline].
• For instructions, use numbered steps.
• Respond empathetically but concisely.

─────────────────────────────────────────
Conversation summary (use as background context):
{summary}

─────────────────────────────────────────
Retrieved context (most relevant first):
{context}

─────────────────────────────────────────
Current user question:
{question}

─────────────────────────────────────────
Answer:"""
)


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


def generate_answer(state: DisasterState) -> dict:
    """Generate a safety-focused answer using retrieved context + conversation summary."""
    llm     = _get_llm()
    docs    = state.get("retrieved_docs", [])
    context = _format_docs(docs)
    summary = state.get("summary", "")
    query   = state["query"]

    logger.info("[generate_answer] building prompt (context docs=%d)", len(docs))

    chain  = _ANSWER_PROMPT | llm
    result = chain.invoke({
        "summary":  summary,
        "context":  context,
        "question": query,
    })

    answer = result.content if hasattr(result, "content") else str(result)
    logger.info("[generate_answer] answer length=%d chars", len(answer))

    # Append the assistant turn to the message list
    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }


# ── Node 3: summarize_conversation ────────────────────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Given the current conversation summary and the latest exchange, create an updated \
concise summary (max {max_tokens} tokens).

Focus ONLY on facts relevant to floods, landslides, the user's situation, risks, \
actions taken, location, family status, etc.
Do NOT include chit-chat, greetings, or generic advice already widely known.

Current summary:
{summary}

Latest exchange:
User: {user_msg}
Assistant: {assistant_msg}

Updated summary (plain text, no bullet points):"""
)


def summarize_conversation(state: DisasterState) -> dict:
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
    chain  = _SUMMARY_PROMPT | llm
    result = chain.invoke({
        "max_tokens":    CFG["summary_max_tokens"],
        "summary":       current_summary or "(no prior summary)",
        "user_msg":      last_user,
        "assistant_msg": last_assistant,
    })

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

    config = {"configurable": {"thread_id": thread_id}}

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
            for event in compiled_graph.stream(input_state, config=config):
                # Each event is {node_name: partial_state}
                for node, partial in event.items():
                    if "answer" in partial and partial["answer"]:
                        final_state = partial

            if final_state and final_state.get("answer"):
                print(f"\nAssistant: {final_state['answer']}\n")
            else:
                print("\nAssistant: [No answer generated – please try again]\n")

            # Show end-of-turn state snapshot (this becomes next-turn input via checkpointing)
            state_snapshot = compiled_graph.get_state(config)
            state_values = state_snapshot.values if state_snapshot else {}
            printable = _state_snapshot_for_print(state_values)
            print("State snapshot (end of turn / next-turn input):")
            print(json.dumps(printable, indent=2, ensure_ascii=False))
            print()

        except Exception as exc:
            logger.error("Graph execution error: %s", exc, exc_info=True)
            print(f"\n[Error] {exc}\n")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disaster RAG Chatbot")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Force re-ingest chunks.csv into Chroma (wipes existing DB).",
    )
    parser.add_argument(
        "--csv", default=CFG["chunks_path"],
        help="Path to chunks file – JSON array, JSONL, or CSV (default: chunks.json).",
    )
    parser.add_argument(
        "--thread", default="session-001",
        help="Thread/session ID for memory checkpointing.",
    )
    args = parser.parse_args()

    # ── Optional one-time ingestion ────────────────────────────
    if args.ingest:
        logger.info("Ingestion requested – rebuilding vector store from %s", args.csv)
        _vectorstore = load_or_create_vectorstore(csv_path=args.csv, force_recreate=True)
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