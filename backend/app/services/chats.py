"""Session-scoped chat memory and existing LangGraph RAG integration."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
import weakref
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from model.agentic_rag import (
    ConversationTurn,
    RagSettings,
    create_openai_qdrant_adapters,
    run_rag_question,
)
from model.usage import ModelUsage

from ..config import Settings
from ..exceptions import NotFoundError, ProviderError
from ..models import ChatSession, utc_now
from ..repositories import ChatRepository, DocumentRepository, UsageRepository
from ..schemas.chats import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    ChatTurnResponse,
)
from .model_catalog import ModelCatalogService
from .pricing import PricingRegistry
from .usage import UsageService


logger = logging.getLogger(__name__)

_USAGE_STAGE_ORDER = {
    "retrieval_query_rewrite": 0,
    "retrieval_embedding": 1,
    "answer_generation": 2,
    "retrieval_grounding": 3,
}


# Every HTTP request receives its own ``ChatService`` and ``AsyncSession``, so
# turn serialization must live outside a service instance.  Locks are scoped to
# their event loop (``asyncio`` primitives must never be shared across loops)
# and weakly held so completed sessions do not create an unbounded registry.
_turn_lock_registry_guard = threading.Lock()
_turn_locks_by_loop: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()


def _session_turn_lock(user_id: str, session_id: str) -> asyncio.Lock:
    """Return the process-wide lock for one owned chat session on this loop."""

    loop = asyncio.get_running_loop()
    key = f"{user_id}:{session_id}"
    with _turn_lock_registry_guard:
        loop_locks = _turn_locks_by_loop.get(loop)
        if loop_locks is None:
            loop_locks = weakref.WeakValueDictionary()
            _turn_locks_by_loop[loop] = loop_locks
        lock = loop_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[key] = lock
        return lock


def _ordered_usage_events(events: list[ModelUsage]) -> list[ModelUsage]:
    """Stabilize parallel provider telemetry without changing token totals."""

    return sorted(
        events,
        key=lambda event: (
            _USAGE_STAGE_ORDER.get(event.stage, len(_USAGE_STAGE_ORDER)),
            event.stage,
            str(event.metadata.get("query_variant_id", "")),
            str(event.metadata.get("page_number", "")),
            event.model,
            event.request_id or "",
        ),
    )


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        model_catalog: ModelCatalogService,
        *,
        adapter_factory: Callable = create_openai_qdrant_adapters,
        runner: Callable = run_rag_question,
    ) -> None:
        self.session = session
        self.settings = settings
        self.model_catalog = model_catalog
        self.repository = ChatRepository(session)
        self.document_repository = DocumentRepository(session)
        self.adapter_factory = adapter_factory
        self.runner = runner

    async def create_session(self, user_id: str, title: str | None = None) -> ChatSession:
        chat = await self.repository.create_session(
            user_id=user_id,
            title=(title or "New chat").strip() or "New chat",
            identifier=f"chat-{uuid.uuid4()}",
        )
        await self.session.commit()
        return chat

    async def list_sessions(self, user_id: str) -> list[ChatSessionResponse]:
        return [
            ChatSessionResponse.model_validate(item)
            for item in await self.repository.list_sessions(user_id)
        ]

    async def list_messages(self, user_id: str, session_id: str) -> list[ChatMessageResponse]:
        chat = await self._owned_session(user_id, session_id)
        return [
            ChatMessageResponse.model_validate(item)
            for item in await self.repository.list_messages(chat.id)
        ]

    async def send_message(
        self,
        *,
        user_id: str,
        session_id: str,
        request: ChatMessageRequest,
    ) -> ChatTurnResponse:
        # Preserve turn order for one conversation from history read through
        # commit. Different sessions receive different locks and can continue
        # through model and retrieval work concurrently.
        async with _session_turn_lock(user_id, session_id):
            return await self._send_message_serialized(
                user_id=user_id,
                session_id=session_id,
                request=request,
            )

    async def _send_message_serialized(
        self,
        *,
        user_id: str,
        session_id: str,
        request: ChatMessageRequest,
    ) -> ChatTurnResponse:
        await self.model_catalog.validate(request.chat_model, request.chat_reasoning_effort)
        chat = await self._owned_session(user_id, session_id)
        previous = await self.repository.list_messages(
            chat.id,
            limit=self.settings.max_session_history_messages,
        )
        history = [
            ConversationTurn(role=item.role, content=item.content)
            for item in previous
            if item.role in {"user", "assistant"}
        ]
        chat_id = chat.id
        chat_identifier = chat.session_identifier
        initial_title = chat.title
        rag_settings = await self._rag_settings(
            request.chat_model,
            request.chat_reasoning_effort,
            user_id=user_id,
        )
        # Release the read transaction before potentially long provider calls.
        # A read-only commit works with ``expire_on_commit=False`` and avoids
        # rollback-expiring owner/session objects held by the request.
        await self.session.commit()
        events: list[ModelUsage] = []
        adapters = None
        try:
            adapters = self.adapter_factory(
                rag_settings,
                usage_callback=events.append,
            )
            response = await self.runner(
                request.question.strip(),
                user_id=user_id,
                thread_id=chat_identifier,
                adapters=adapters,
                settings=rag_settings,
                collection_scope=request.collection_scope,
                conversation_history=history,
            )
            if response.errors:
                failure_summary = "; ".join(
                    f"{item.node}: {item.message}" for item in response.errors
                )[:1000]
                if response.no_answer or not response.answer.strip():
                    raise ProviderError(
                        f"The chat request could not be completed. {failure_summary}",
                        code="rag_provider_failure",
                    )
                logger.warning(
                    "A RAG stage failed, but the graph returned a successful fallback response.",
                    extra={
                        "event": "rag_stage_recovered",
                        "user_id": user_id,
                        "chat_session_id": chat_id,
                        "failed_nodes": [item.node for item in response.errors],
                    },
                )
            citations = [
                {
                    "filename": item.document_name,
                    "document_id": item.document_id,
                    "page_number": item.page_number,
                    "slide_number": item.slide_number,
                    "chunk_id": item.chunk_id,
                    "source_excerpt": item.source_excerpt,
                    "retrieval_score": item.retrieval_score,
                    "ingestion_method": item.ingestion_method,
                    "source_collection": item.collection_name,
                    "source_pipeline": item.source_pipeline,
                }
                for item in response.citations
            ]
            actual_model = next(
                (event.model for event in events if event.stage == "answer_generation"),
                request.chat_model,
            )
            chat = await self._owned_session(user_id, session_id)
            user_message = await self.repository.add_message(
                session_id=chat.id,
                role="user",
                content=request.question.strip(),
            )
            assistant_message = await self.repository.add_message(
                session_id=chat.id,
                role="assistant",
                content=response.answer,
                citations=citations,
                model=actual_model,
                reasoning_effort=request.chat_reasoning_effort,
            )
            if initial_title == "New chat" and chat.title == "New chat":
                chat.title = request.question.strip()[:80]
            chat.last_activity_at = utc_now()
            usage_service = UsageService(
                self.session,
                PricingRegistry(self.settings.pricing_registry_path),
            )
            request_records = await usage_service.persist_events(
                _ordered_usage_events(events),
                user_id=user_id,
                operation="chat",
                reasoning_effort=request.chat_reasoning_effort,
                chat_session_id=chat.id,
                chat_message_id=assistant_message.id,
            )
            recorded_stages = {item.stage for item in request_records}
            for stage in (
                "retrieval_query_rewrite",
                "retrieval_embedding",
                "answer_generation",
                "retrieval_grounding",
            ):
                if stage not in recorded_stages:
                    request_records.append(
                        await usage_service.record_not_applicable(
                            user_id=user_id,
                            operation="chat",
                            stage=stage,
                            chat_session_id=chat.id,
                            chat_message_id=assistant_message.id,
                        )
                    )
            usage_repository = UsageRepository(self.session)
            session_records = await usage_repository.list_for_user(user_id, session_id=chat.id)
            total_records = await usage_repository.list_for_user(user_id, limit=10_000)
            request_totals = usage_service.totals(request_records)
            session_totals = usage_service.totals(session_records)
            total_totals = usage_service.totals(total_records)
            await self.session.commit()
            return ChatTurnResponse(
                user_message=ChatMessageResponse.model_validate(user_message),
                assistant_message=ChatMessageResponse.model_validate(assistant_message),
                no_answer=response.no_answer,
                checked_collections=list(response.checked_collections),
                request_usage=request_totals,
                session_usage=session_totals,
                total_usage=total_totals,
            )
        except Exception:
            if self.session.in_transaction():
                await self.session.rollback()
            raise
        finally:
            close = getattr(adapters, "aclose", None) if adapters is not None else None
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.warning(
                        "Async RAG provider clients could not be closed cleanly.",
                        extra={
                            "event": "rag_clients_close_failed",
                            "user_id": user_id,
                            "chat_session_id": session_id,
                        },
                        exc_info=True,
                    )

    async def _owned_session(self, user_id: str, session_id: str) -> ChatSession:
        chat = await self.repository.get_owned(user_id, session_id)
        if chat is None:
            raise NotFoundError("Chat session not found.", code="chat_session_not_found")
        return chat

    async def _rag_settings(self, model: str, effort: str, *, user_id: str) -> RagSettings:
        return RagSettings(
            app_env=self.settings.app_env,
            runtime_mode="hybrid",
            llm_provider="openai",
            embedding_provider="openai",
            self_service_llm_model=model,
            self_service_reasoning_effort=effort,
            embedding_model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
            openai_base_url=self.settings.openai_base_url,
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
            semantic_collection=self.settings.qdrant_collection_semantic,
            docling_collection=self.settings.qdrant_collection_docling,
            dense_vector_name=self.settings.qdrant_dense_vector_name,
            sparse_vector_name=self.settings.qdrant_sparse_vector_name,
            retrieval_top_k=self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            max_context_chunks=self.settings.max_context_chunks,
            hybrid_dense_weight=self.settings.hybrid_dense_weight,
            hybrid_sparse_weight=self.settings.hybrid_sparse_weight,
            no_answer_min_score=self.settings.no_answer_min_score,
            citation_min_score=self.settings.citation_min_score,
            llm_request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            enable_reranker=self.settings.enable_reranker,
            reranker_provider=self.settings.reranker_provider,
            allowed_document_ids=tuple(
                await self.document_repository.list_retrievable_ids(user_id)
            ),
        )
