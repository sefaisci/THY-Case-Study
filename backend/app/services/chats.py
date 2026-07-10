"""Session-scoped chat memory and existing LangGraph RAG integration."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

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


class ChatService:
    def __init__(
        self,
        session: Session,
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

    def create_session(self, user_id: str, title: str | None = None) -> ChatSession:
        chat = self.repository.create_session(
            user_id=user_id,
            title=(title or "New chat").strip() or "New chat",
            identifier=f"chat-{uuid.uuid4()}",
        )
        self.session.commit()
        return chat

    def list_sessions(self, user_id: str) -> list[ChatSessionResponse]:
        return [ChatSessionResponse.model_validate(item) for item in self.repository.list_sessions(user_id)]

    def list_messages(self, user_id: str, session_id: str) -> list[ChatMessageResponse]:
        chat = self._owned_session(user_id, session_id)
        return [
            ChatMessageResponse.model_validate(item)
            for item in self.repository.list_messages(chat.id)
        ]

    def send_message(
        self,
        *,
        user_id: str,
        session_id: str,
        request: ChatMessageRequest,
    ) -> ChatTurnResponse:
        chat = self._owned_session(user_id, session_id)
        self.model_catalog.validate(request.chat_model, request.chat_reasoning_effort)
        previous = self.repository.list_messages(
            chat.id,
            limit=self.settings.max_session_history_messages,
        )
        history = [
            ConversationTurn(role=item.role, content=item.content)
            for item in previous
            if item.role in {"user", "assistant"}
        ]
        user_message = self.repository.add_message(
            session_id=chat.id,
            role="user",
            content=request.question.strip(),
        )
        events: list[ModelUsage] = []
        try:
            rag_settings = self._rag_settings(
                request.chat_model,
                request.chat_reasoning_effort,
                user_id=user_id,
            )
            adapters = self.adapter_factory(
                rag_settings,
                usage_callback=events.append,
            )
            response = self.runner(
                request.question.strip(),
                user_id=user_id,
                thread_id=chat.session_identifier,
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
                        "chat_session_id": chat.id,
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
            assistant_message = self.repository.add_message(
                session_id=chat.id,
                role="assistant",
                content=response.answer,
                citations=citations,
                model=actual_model,
                reasoning_effort=request.chat_reasoning_effort,
            )
            if chat.title == "New chat":
                chat.title = request.question.strip()[:80]
            chat.last_activity_at = utc_now()
            usage_service = UsageService(
                self.session,
                PricingRegistry(self.settings.pricing_registry_path),
            )
            request_records = usage_service.persist_events(
                events,
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
                        usage_service.record_not_applicable(
                            user_id=user_id,
                            operation="chat",
                            stage=stage,
                            chat_session_id=chat.id,
                            chat_message_id=assistant_message.id,
                        )
                    )
            usage_repository = UsageRepository(self.session)
            session_records = usage_repository.list_for_user(user_id, session_id=chat.id)
            total_records = usage_repository.list_for_user(user_id, limit=10_000)
            request_totals = usage_service.totals(request_records)
            session_totals = usage_service.totals(session_records)
            total_totals = usage_service.totals(total_records)
            self.session.commit()
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
            self.session.rollback()
            raise

    def _owned_session(self, user_id: str, session_id: str) -> ChatSession:
        chat = self.repository.get_owned(user_id, session_id)
        if chat is None:
            raise NotFoundError("Chat session not found.", code="chat_session_not_found")
        return chat

    def _rag_settings(self, model: str, effort: str, *, user_id: str) -> RagSettings:
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
                self.document_repository.list_retrievable_ids(user_id)
            ),
        )
