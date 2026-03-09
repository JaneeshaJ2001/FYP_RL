"""Graph construction for the Disaster RAG chatbot."""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    DisasterState,
    decide_retrieve,
    generate_answer,
    retrieve,
    retrieval_router,
    summarize_conversation,
)

logger = logging.getLogger("disaster_chatbot")


def build_graph():
    """Construct, compile, and return the LangGraph StateGraph.

    Graph topology
    ──────────────
    START
      └─► decide_retrieve
              ├─ action=1 ──► retrieve ──► generate_answer
              └─ action=0 ──────────────►  generate_answer
                                               └─► summarize_conversation ──► END
    """
    graph = StateGraph(DisasterState)

    graph.add_node("decide_retrieve", decide_retrieve)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("summarize_conversation", summarize_conversation)

    # Entry point
    graph.add_edge(START, "decide_retrieve")

    # Conditional branch: retrieve or skip straight to generation
    graph.add_conditional_edges(
        "decide_retrieve",
        retrieval_router,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "summarize_conversation")
    graph.add_edge("summarize_conversation", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Graph compiled successfully (conditional retrieval enabled)")
    return compiled