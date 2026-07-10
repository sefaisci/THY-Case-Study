"""LangGraph builders for the agentic RAG workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .adapters import RagAdapters, create_fake_adapters
from .nodes import (
    RagNodeSet,
    route_after_answer,
    route_after_query_understanding,
    route_after_retrieval,
)
from .schemas import RagState
from .settings import RagSettings


def build_retrieval_subgraph(nodes: RagNodeSet):
    """Build the retrieval planning, retrieval, and reranking subgraph."""

    builder = StateGraph(RagState)
    builder.add_node("retrieval_planner", nodes.retrieval_planner)
    builder.add_node("hybrid_retrieval", nodes.hybrid_retrieval)
    builder.add_node("reranking", nodes.reranking)
    builder.add_edge(START, "retrieval_planner")
    builder.add_edge("retrieval_planner", "hybrid_retrieval")
    builder.add_edge("hybrid_retrieval", "reranking")
    builder.add_edge("reranking", END)
    return builder.compile(name="retrieval_subgraph")


def build_answer_subgraph(nodes: RagNodeSet):
    """Build the answer, citation validation, and reflection subgraph."""

    builder = StateGraph(RagState)
    builder.add_node("answer_generation", nodes.answer_generation)
    builder.add_node("citation_validation", nodes.citation_validation)
    builder.add_node("claim_evidence_reflection", nodes.claim_evidence_reflection)
    builder.add_edge(START, "answer_generation")
    builder.add_edge("answer_generation", "citation_validation")
    builder.add_edge("citation_validation", "claim_evidence_reflection")
    builder.add_edge("claim_evidence_reflection", END)
    return builder.compile(name="answer_subgraph")


def build_rag_graph(
    adapters: RagAdapters | None = None,
    settings: RagSettings | None = None,
    *,
    checkpointer=None,
    use_memory_saver: bool = False,
):
    """Build the main Agentic RAG graph with modular subgraphs."""

    settings = settings or RagSettings.from_env()
    adapters = adapters or create_fake_adapters(settings)
    nodes = RagNodeSet(adapters=adapters, settings=settings)
    retrieval_subgraph = build_retrieval_subgraph(nodes)
    answer_subgraph = build_answer_subgraph(nodes)

    builder = StateGraph(RagState)
    builder.add_node("query_understanding", nodes.query_understanding)
    builder.add_node("retrieval_subgraph", retrieval_subgraph)
    builder.add_node("answer_subgraph", answer_subgraph)
    builder.add_node("conversation_generation", nodes.conversation_generation)
    builder.add_node("final_response", nodes.final_response)

    builder.add_edge(START, "query_understanding")
    builder.add_conditional_edges(
        "query_understanding",
        route_after_query_understanding,
        {
            "retrieval_subgraph": "retrieval_subgraph",
            "conversation_generation": "conversation_generation",
        },
    )
    builder.add_conditional_edges(
        "retrieval_subgraph",
        route_after_retrieval,
        {
            "answer_subgraph": "answer_subgraph",
            "conversation_generation": "conversation_generation",
        },
    )
    builder.add_conditional_edges(
        "answer_subgraph",
        route_after_answer,
        {
            "final_response": "final_response",
            "conversation_generation": "conversation_generation",
        },
    )
    builder.add_edge("conversation_generation", "final_response")
    builder.add_edge("final_response", END)

    active_checkpointer = checkpointer
    if use_memory_saver and active_checkpointer is None:
        active_checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=active_checkpointer,
        name="thy_agentic_rag_graph",
    )
