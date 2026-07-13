"""LangGraph builders for the agentic RAG workflow."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .adapters import RagAdapters, create_fake_adapters
from .nodes import (
    RagNodeSet,
    route_after_answer,
    route_after_general_generation,
    route_after_grounding_reflection,
    route_after_query_understanding,
    route_after_retrieval,
)
from .schemas import RagState
from .settings import RagSettings


def _as_async(function: Callable[[RagState], Any]):
    """Run a short, provider-free callable inline on the async graph loop.

    The installed LangGraph runtime delegates synchronous graph callables to a
    thread executor during ``ainvoke``. That executor path can remain pending in
    constrained ASGI/test runtimes. These functions only perform bounded state
    transformations, so running them inline is safe and keeps the graph fully
    asynchronous end to end.
    """

    @wraps(function)
    async def invoke(state: RagState):
        return function(state)

    return invoke


def build_retrieval_subgraph(nodes: RagNodeSet):
    """Build adaptive one-or-two-query mapping, fusion, and reranking."""

    builder = StateGraph(RagState)
    builder.add_node("retrieval_planner", _as_async(nodes.retrieval_planner))
    builder.add_node("retrieve_variant", nodes.retrieve_variant)
    builder.add_node(
        "fuse_variant_results",
        _as_async(nodes.fuse_variant_results),
    )
    builder.add_node("reranking", nodes.reranking)
    builder.add_node(
        "retrieval_outcome_classification",
        _as_async(nodes.retrieval_outcome_classification),
    )
    builder.add_edge(START, "retrieval_planner")
    builder.add_conditional_edges(
        "retrieval_planner",
        _as_async(nodes.dispatch_query_variants),
        ["retrieve_variant"],
    )
    builder.add_edge("retrieve_variant", "fuse_variant_results")
    builder.add_edge("fuse_variant_results", "reranking")
    builder.add_edge("reranking", "retrieval_outcome_classification")
    builder.add_edge("retrieval_outcome_classification", END)
    return builder.compile(name="retrieval_subgraph")


def build_answer_subgraph(nodes: RagNodeSet):
    """Build the answer, citation validation, and reflection subgraph."""

    builder = StateGraph(RagState)
    builder.add_node("answer_generation", nodes.answer_generation)
    builder.add_node("citation_validation", _as_async(nodes.citation_validation))
    builder.add_node("claim_evidence_reflection", nodes.claim_evidence_reflection)
    builder.add_node("grounded_repair", nodes.grounded_repair)
    builder.add_edge(START, "answer_generation")
    builder.add_edge("answer_generation", "citation_validation")
    builder.add_edge("citation_validation", "claim_evidence_reflection")
    builder.add_conditional_edges(
        "claim_evidence_reflection",
        _as_async(route_after_grounding_reflection),
        {
            "grounded_repair": "grounded_repair",
            "end": END,
        },
    )
    builder.add_edge("grounded_repair", "citation_validation")
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
    builder.add_node(
        "general_knowledge_generation",
        nodes.general_knowledge_generation,
    )
    builder.add_node(
        "compose_hybrid_response",
        _as_async(nodes.compose_hybrid_response),
    )
    builder.add_node("explicit_no_answer", _as_async(nodes.explicit_no_answer))
    builder.add_node("final_response", _as_async(nodes.final_response))

    builder.add_edge(START, "query_understanding")
    builder.add_conditional_edges(
        "query_understanding",
        _as_async(route_after_query_understanding),
        {
            "retrieval_subgraph": "retrieval_subgraph",
            "conversation_generation": "conversation_generation",
            "explicit_no_answer": "explicit_no_answer",
        },
    )
    builder.add_conditional_edges(
        "retrieval_subgraph",
        _as_async(route_after_retrieval),
        {
            "answer_subgraph": "answer_subgraph",
            "general_knowledge_generation": "general_knowledge_generation",
            "explicit_no_answer": "explicit_no_answer",
        },
    )
    builder.add_conditional_edges(
        "answer_subgraph",
        _as_async(route_after_answer),
        {
            "final_response": "final_response",
            "general_knowledge_generation": "general_knowledge_generation",
            "explicit_no_answer": "explicit_no_answer",
        },
    )
    builder.add_conditional_edges(
        "general_knowledge_generation",
        _as_async(route_after_general_generation),
        {
            "compose_hybrid_response": "compose_hybrid_response",
            "final_response": "final_response",
        },
    )
    builder.add_edge("compose_hybrid_response", "final_response")
    builder.add_edge("conversation_generation", "final_response")
    builder.add_edge("explicit_no_answer", "final_response")
    builder.add_edge("final_response", END)

    active_checkpointer = checkpointer
    if use_memory_saver and active_checkpointer is None:
        active_checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=active_checkpointer,
        name="thy_agentic_rag_graph",
    )
