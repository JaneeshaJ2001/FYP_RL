"""
============================================================
  Disaster-Domain RAG Chatbot  (Floods & Landslides only)
  Phase 1 - Always-retrieve, summary memory, LangGraph
============================================================
"""

import logging

from langchain_core.messages import HumanMessage

from config import CONFIG
from graph import build_graph
from ingestion import load_or_create_vectorstore
from nodes import (
    DisasterState,
    build_run_config,
    flush_langfuse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("disaster_chatbot")


def run_chat_loop(compiled_graph, thread_id: str = "default-session"):
    """
    Simple console chat loop.
    Each user session should use a unique thread_id so MemorySaver
    maintains separate checkpoints (summaries) per conversation.
    """
    print("\n" + "=" * 60)
    print("  Disaster Response Assistant  (floods & landslides)")
    print("  Type 'exit' or 'quit' to end | thread:", thread_id)
    print("=" * 60 + "\n")

    run_config = build_run_config(thread_id)

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Assistant: Stay safe. Goodbye.")
                break

            input_state: DisasterState = {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "summary": "",  # MemorySaver restores previous summary
                "retrieved_docs": [],
                "answer": "",
            }

            try:
                final_state = None
                for event in compiled_graph.stream(input_state, config=run_config):
                    for _, partial in event.items():
                        if "answer" in partial and partial["answer"]:
                            final_state = partial

                if final_state and final_state.get("answer"):
                    print(f"\nAssistant: {final_state['answer']}\n")
                else:
                    print("\nAssistant: [No answer generated - please try again]\n")

            except Exception as exc:
                logger.error("Graph execution error: %s", exc, exc_info=True)
                print(f"\n[Error] {exc}\n")
    finally:
        flush_langfuse()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disaster RAG Chatbot")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Force re-ingest chunks.json into Chroma (wipes existing DB).",
    )
    parser.add_argument(
        "--json",
        default=CONFIG.chunks_path,
        help="Path to chunks file - JSON array (default: chunks.json).",
    )
    parser.add_argument(
        "--thread",
        default="session-001",
        help="Thread/session ID for memory checkpointing.",
    )
    args = parser.parse_args()

    if args.ingest:
        logger.info("Ingestion requested - rebuilding vector store from %s", args.json)
        load_or_create_vectorstore(json_path=args.json, force_recreate=True)
    else:
        logger.info("Vector store will be loaded lazily on first query.")

    app = build_graph()
    run_chat_loop(app, thread_id=args.thread)


"""
# Streamlit integration example
import streamlit as st
from langchain_core.messages import HumanMessage

from graph import build_graph
from nodes import build_run_config

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

        app = get_app()
        config = build_run_config(st.session_state.thread_id)
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "query": user_input,
            "summary": "",
            "retrieved_docs": [],
            "answer": "",
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
