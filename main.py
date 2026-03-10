"""
============================================================
  Disaster-Domain RAG Chatbot  (Floods & Landslides only)
  Phase 2 - Conditional retrieval via RL policy
============================================================
"""

import logging
import os
import pprint
from typing import Any

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

_NODE_SEP = "─" * 56

# All 8 fields of DisasterState, in display order
_STATE_FIELDS = (
    "messages", "summary", "query", "retrieved_docs",
    "answer", "action", "mode", "forced_action",
)


def _print_node_state(node_name: str, state: dict[str, Any]) -> None:
    """Pretty-print all 8 DisasterState fields after a graph node completes."""
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage

    print(f"\n┌── NODE: {node_name} " + "─" * max(0, 46 - len(node_name)) + "┐")
    for key in _STATE_FIELDS:
        val = state.get(key, "<not set>")
        if isinstance(val, list) and val and isinstance(val[0], Document):
            print(f"│  {key}: [{len(val)} doc(s)]")
            for i, doc in enumerate(val, 1):
                snippet = doc.page_content[:160].replace("\n", " ")
                print(f"│    [{i}] {snippet!r}")
                if doc.metadata:
                    print(f"│         meta={doc.metadata}")
        elif isinstance(val, list) and val and isinstance(val[0], BaseMessage):
            print(f"│  {key}: [{len(val)} message(s)]")
            for msg in val:
                preview = str(msg.content)[:160].replace("\n", " ")
                print(f"│    {type(msg).__name__}: {preview!r}")
        elif isinstance(val, str) and len(val) > 200:
            print(f"│  {key}: {val[:200]!r}  … ({len(val)} chars total)")
        else:
            print(f"│  {key}: {pprint.pformat(val, width=80, depth=3)}")
    print("└" + _NODE_SEP + "┘")


def run_chat_loop(
    compiled_graph,
    thread_id: str = "default-session",
    mode: str = "baseline",   # "baseline" | "policy"
    debug: bool = False,
):
    """
    Simple console chat loop.

    mode="baseline"  → always retrieve (original behaviour, action forced=1)
    mode="policy"    → use the RL-trained policy to decide per turn
    debug=True       → print each node's state updates as the query flows through
    """
    print("\n" + "=" * 60)
    print("  Disaster Response Assistant  (floods & landslides)")
    print(f"  Mode: {mode.upper()}  |  Thread: {thread_id}  |  Debug: {'ON' if debug else 'OFF'}")
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
                if debug:
                    # stream_mode=["updates","values"] yields (tag, data) pairs:
                    #   "updates" → {node_name: partial}  (tells us which node ran)
                    #   "values"  → full DisasterState snapshot after that node
                    pending_node: str | None = None
                    for mode_tag, data in compiled_graph.stream(
                        input_state, config=run_config,
                        stream_mode=["updates", "values"],
                    ):
                        if mode_tag == "updates":
                            for node_name in data:
                                pending_node = node_name
                        elif mode_tag == "values" and pending_node is not None:
                            _print_node_state(pending_node, data)
                            final_state = data   # keep last; all fields present
                            pending_node = None
                else:
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the full state updates of each graph node for every query.",
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
    run_chat_loop(app, thread_id=args.thread, mode=args.mode, debug=args.debug)