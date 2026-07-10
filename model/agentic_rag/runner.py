"""Notebook-friendly runner for the agentic RAG graph."""

from __future__ import annotations

from .adapters import RagAdapters, create_fake_adapters
from .graphs import build_rag_graph
from .schemas import CollectionScope, ConversationTurn, RagRequest, RagResponse, RagState
from .settings import RagSettings


def run_rag_question(
    question: str,
    user_id: str = "local-demo-user",
    *,
    thread_id: str | None = "notebook-thread",
    adapters: RagAdapters | None = None,
    settings: RagSettings | None = None,
    use_memory_saver: bool = False,
    collection_scope: CollectionScope = "both",
    conversation_history: list[ConversationTurn] | None = None,
) -> RagResponse:
    """Run one RAG question through the LangGraph workflow."""

    settings = settings or RagSettings.from_env()
    adapters = adapters or create_fake_adapters(settings)
    request = RagRequest(
        question=question,
        user_id=user_id,
        thread_id=thread_id,
        collection_scope=collection_scope,
        conversation_history=conversation_history or [],
    )
    graph = build_rag_graph(
        adapters=adapters,
        settings=settings,
        use_memory_saver=use_memory_saver,
    )
    initial_state: RagState = {
        "question": request.question,
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "collection_scope": request.collection_scope,
        "conversation_history": request.conversation_history,
        "errors": [],
    }
    config = {"configurable": {"thread_id": thread_id or "notebook-thread", "user_id": user_id}}
    final_state = graph.invoke(initial_state, config=config)
    response = final_state.get("response")
    if response is not None:
        return response
    return RagResponse(
        answer=final_state.get("final_answer", ""),
        citations=final_state.get("citations", []),
        no_answer=final_state.get("no_answer", True),
        checked_collections=final_state.get("checked_collections", []),
        citation_validation=final_state.get("citation_validation"),
        reflection=final_state.get("reflection"),
        errors=final_state.get("errors", []),
    )
