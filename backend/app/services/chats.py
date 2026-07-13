"""Session-scoped chat memory and existing LangGraph RAG integration."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
from model.agentic_rag.schemas import RagResponse
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

_CHAT_USAGE_STAGES = (
    "retrieval_query_rewrite",
    "retrieval_embedding",
    "retrieval_reranking",
    "answer_generation",
    "retrieval_grounding",
    "grounded_repair",
    "general_knowledge_generation",
)
_USAGE_STAGE_ORDER = {
    stage: index for index, stage in enumerate(_CHAT_USAGE_STAGES)
}
_DOCUMENT_NO_ANSWER = (
    "The uploaded documents do not contain enough evidence to answer this question."
)
_PARTIAL_RETRIEVAL_ERROR_NODES = (
    "hybrid_retrieval",
)
_GENERIC_RAG_PROVIDER_ERROR = "The chat request could not be completed."


def _is_partial_retrieval_error_node(node: str) -> bool:
    return (
        any(node.startswith(f"{base}:") for base in _PARTIAL_RETRIEVAL_ERROR_NODES)
        and ":payload" not in node
    )


def _is_recoverable_rag_error_node(node: str) -> bool:
    """Return whether a single Qdrant collection degraded safely."""

    return _is_partial_retrieval_error_node(node)


def _is_safe_document_no_answer(response: RagResponse) -> bool:
    return (
        response.no_answer
        and response.answer.strip() == _DOCUMENT_NO_ANSWER
        and not response.citations
    )


def _is_grounded_accepted_response(response: RagResponse) -> bool:
    validation = response.citation_validation
    reflection = response.reflection
    return (
        not response.no_answer
        and bool(response.answer.strip())
        and bool(response.citations)
        and validation is not None
        and validation.is_valid
        and reflection is not None
        and reflection.is_grounded
        and reflection.decision == "accept"
        and reflection.hallucination_risk != "high"
        and not reflection.unsupported_claims
        and not reflection.missing_citations
    )


def _is_general_knowledge_response(response: RagResponse) -> bool:
    """Recognize only the server-labeled, citation-free fallback contract."""

    return (
        not response.no_answer
        and not response.citations
        and response.citation_validation is None
        and response.reflection is None
        and response.answer.lstrip().startswith(
            "> **Genel model bilgisi:** Aşağıdaki bölüm yüklediğiniz "
            "belgelerde doğrulanmamıştır."
        )
    )


def _validate_rag_response_error_policy(response: RagResponse) -> None:
    """Reject fatal or inconsistently recovered graph errors with a generic API error."""

    if not response.errors:
        return
    all_recoverable = all(
        _is_recoverable_rag_error_node(error.node)
        for error in response.errors
    )
    safe_no_answer = _is_safe_document_no_answer(response)
    recovered_with_only_partial_retrieval_errors = (
        (
            _is_grounded_accepted_response(response)
            or _is_general_knowledge_response(response)
        )
        and all(
            _is_partial_retrieval_error_node(error.node)
            for error in response.errors
        )
    )
    if not all_recoverable or not (
        safe_no_answer or recovered_with_only_partial_retrieval_errors
    ):
        raise ProviderError(
            _GENERIC_RAG_PROVIDER_ERROR,
            code="rag_provider_failure",
        )


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


def _missing_usage_stages(recorded_stages: set[str]) -> tuple[str, ...]:
    """Return chat stages that need explicit not-applicable records."""

    return tuple(
        stage for stage in _CHAT_USAGE_STAGES if stage not in recorded_stages
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
            rag_started_at = time.perf_counter()
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
            latency_ms = max(
                0,
                round((time.perf_counter() - rag_started_at) * 1_000),
            )
            if response.errors:
                _validate_rag_response_error_policy(response)
                logger.warning(
                    "A RAG stage failed, but the graph returned a safe terminal response.",
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
                (
                    event.model
                    for event in events
                    if event.stage
                    in {
                        "answer_generation",
                        "grounded_repair",
                        "general_knowledge_generation",
                    }
                ),
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
                latency_ms=latency_ms,
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
            for stage in _missing_usage_stages(recorded_stages):
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
            logger.info(
                "Chat answer generated",
                extra={
                    "event": "chat_answer_generated",
                    "user_id": user_id,
                    "chat_session_id": chat.id,
                    "chat_message_id": assistant_message.id,
                    "latency_ms": latency_ms,
                },
            )
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
                except Exception as exc:
                    logger.warning(
                        "Async RAG provider clients could not be closed cleanly.",
                        extra={
                            "event": "rag_clients_close_failed",
                            "exception_type": type(exc).__name__,
                            "user_id": user_id,
                            "chat_session_id": session_id,
                        },
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
            sparse_encoder_provider=self.settings.sparse_encoder_provider,
            sparse_encoder_model=self.settings.sparse_encoder_model,
            sparse_encoder_cache_dir=self.settings.sparse_encoder_cache_dir,
            retrieval_prefetch_k=self.settings.retrieval_prefetch_k,
            retrieval_collection_k=self.settings.retrieval_collection_k,
            rerank_candidate_k=self.settings.rerank_candidate_k,
            rerank_top_k=self.settings.rerank_top_k,
            max_context_chunks=self.settings.max_context_chunks,
            hybrid_dense_weight=self.settings.hybrid_dense_weight,
            hybrid_sparse_weight=self.settings.hybrid_sparse_weight,
            no_answer_min_score=self.settings.no_answer_min_score,
            citation_min_score=self.settings.citation_min_score,
            llm_request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            enable_reranker=self.settings.enable_reranker,
            reranker_provider=self.settings.reranker_provider,
            reranker_model=self.settings.reranker_model,
            reranker_reasoning_effort=self.settings.reranker_reasoning_effort,
            reranker_max_candidates=self.settings.reranker_max_candidates,
            reranker_text_max_chars=self.settings.reranker_text_max_chars,
            rerank_min_score=self.settings.rerank_min_score,
            reranker_allow_partial_support=(
                self.settings.reranker_allow_partial_support
            ),
            grounding_reasoning_effort=self.settings.grounding_reasoning_effort,
            grounding_max_retries=self.settings.grounding_max_retries,
            allowed_document_ids=tuple(
                await self.document_repository.list_retrievable_ids(user_id)
            ),
        )
