"""Notebook-friendly runner for the agentic RAG graph."""

from __future__ import annotations

import asyncio
import hashlib
import json

from .adapters import RagAdapters, create_fake_adapters
from .graphs import build_rag_graph
from .schemas import CollectionScope, ConversationTurn, RagRequest, RagResponse, RagState
from .settings import RagSettings


def _checkpoint_thread_namespace(user_id: str, thread_id: str | None) -> str:
    """Return an opaque checkpoint key scoped to one user and raw thread ID."""

    raw_thread_id = thread_id or "notebook-thread"
    namespace_input = json.dumps(
        [user_id, raw_thread_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(namespace_input.encode("utf-8")).hexdigest()


async def run_rag_question(
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
    """Run one RAG question asynchronously through the LangGraph workflow."""

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
    config = {
        "configurable": {
            "thread_id": _checkpoint_thread_namespace(
                request.user_id,
                request.thread_id,
            ),
            "user_id": request.user_id,
        }
    }
    final_state = await graph.ainvoke(initial_state, config=config)
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


def run_rag_question_sync(*args, **kwargs) -> RagResponse:
    """Notebook-only synchronous wrapper around :func:`run_rag_question`.

    Application and backend code must await ``run_rag_question`` directly. This
    wrapper intentionally refuses to run inside an active event loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_rag_question(*args, **kwargs))
    raise RuntimeError(
        "run_rag_question_sync is notebook-only and cannot run inside an active "
        "event loop; await run_rag_question instead."
    )
