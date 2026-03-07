"""Graph construction for the Disaster RAG chatbot."""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import DisasterState, generate_answer, retrieve, summarize_conversation

logger = logging.getLogger("disaster_chatbot")


def build_graph():
    """Construct, compile, and return the LangGraph StateGraph."""
    graph = StateGraph(DisasterState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("summarize_conversation", summarize_conversation)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "summarize_conversation")
    graph.add_edge("summarize_conversation", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Graph compiled successfully")
    return compiled
