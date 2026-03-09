"""
============================================================
  Disaster-Domain RAG Chatbot  (Floods & Landslides only)
  Phase 2 - Conditional retrieval via RL policy
============================================================
"""

import logging
import os

from langchain_core.messages import HumanMessage

from config import CONFIG
from graph import build_graph
from ingestion import load_or_create_vectorstore
from nodes import DisasterState
from observability import build_run_config, flush_langfuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("disaster_chatbot")


def run_chat_loop(
    compiled_graph,
    thread_id: str = "default-session",
    mode: str = "baseline",   # "baseline" | "policy"
):
    """
    Simple console chat loop.

    mode="baseline"  → always retrieve (original behaviour, action forced=1)
    mode="policy"    → use the RL-trained policy to decide per turn
    """
    print("\n" + "=" * 60)
    print("  Disaster Response Assistant  (floods & landslides)")
    print(f"  Mode: {mode.upper()}  |  Thread: {thread_id}")
    print("  Type 'exit' or 'quit' to end")
    print("=" * 60 + "\n")

    run_config = build_run_config(thread_id)
    summary = ""

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
                "summary": summary,
                "retrieved_docs": [],
                "answer": "",
                "action": 1,
                "mode": mode,          # "baseline" forces retrieve; "policy" uses RL
                "forced_action": None, # None = let decide_retrieve handle it
            }

            try:
                final_state = None
                for event in compiled_graph.stream(input_state, config=run_config):
                    for _, partial in event.items():
                        if "answer" in partial and partial["answer"]:
                            final_state = partial

                if final_state:
                    summary = final_state.get("summary", summary)
                    action_taken = final_state.get("action", 1)
                    retrieval_label = "🔍 retrieved" if action_taken == 1 else "⚡ skipped"
                    print(f"\nAssistant [{retrieval_label}]: {final_state['answer']}\n")
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
        help="Path to chunks file (default: chunks.json).",
    )
    parser.add_argument(
        "--thread",
        default="session-001",
        help="Thread/session ID for memory checkpointing.",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "policy"],
        help=(
            "baseline = always retrieve (original behaviour); "
            "policy = use RL-trained retrieval policy."
        ),
    )
    args = parser.parse_args()

    if args.ingest:
        logger.info("Ingestion requested - rebuilding vector store from %s", args.json)
        load_or_create_vectorstore(json_path=args.json, force_recreate=True)
    else:
        logger.info("Vector store will be loaded lazily on first query.")

    if args.mode == "policy" and not os.path.exists(CONFIG.policy_model_path):
        logger.warning(
            "Policy file '%s' not found. Run rl_train.py first, or use --mode baseline.",
            CONFIG.policy_model_path,
        )

    app = build_graph()
    run_chat_loop(app, thread_id=args.thread, mode=args.mode)