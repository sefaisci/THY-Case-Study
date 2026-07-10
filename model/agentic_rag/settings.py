"""Environment-backed settings for the agentic RAG graph."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class RagSettings:
    """Runtime configuration for notebook-first RAG experiments."""

    app_env: str = "local"
    runtime_mode: str = "hybrid"
    llm_provider: str = "openai"
    embedding_provider: str = "openai"
    self_service_llm_model: str = "gpt-5.5"
    self_service_reasoning_effort: str = "low"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    semantic_collection: str = "semantic_chunks"
    docling_collection: str = "docling_fixed_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    retrieval_top_k: int = 12
    rerank_top_k: int = 6
    max_context_chunks: int = 8
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    no_answer_min_score: float = 0.20
    citation_min_score: float = 0.20
    llm_request_timeout_seconds: int = 120
    enable_reranker: bool = False
    reranker_provider: str = "noop"
    allowed_document_ids: tuple[str, ...] | None = None

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "RagSettings":
        """Load settings from environment variables and an optional dotenv file."""

        load_dotenv(dotenv_path=dotenv_path, override=False)
        return cls(
            app_env=os.getenv("APP_ENV", "local"),
            runtime_mode=os.getenv("RAG_RUNTIME_MODE", "hybrid"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            self_service_llm_model=os.getenv("SELF_SERVICE_LLM_MODEL", "gpt-5.5"),
            self_service_reasoning_effort=os.getenv(
                "SELF_SERVICE_REASONING_EFFORT", "low"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            semantic_collection=os.getenv("QDRANT_COLLECTION_SEMANTIC", "semantic_chunks"),
            docling_collection=os.getenv("QDRANT_COLLECTION_DOCLING", "docling_fixed_chunks"),
            dense_vector_name=os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense"),
            sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
            retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 12),
            rerank_top_k=_get_int("RERANK_TOP_K", 6),
            max_context_chunks=_get_int("MAX_CONTEXT_CHUNKS", 8),
            hybrid_dense_weight=_get_float("HYBRID_DENSE_WEIGHT", 0.65),
            hybrid_sparse_weight=_get_float("HYBRID_SPARSE_WEIGHT", 0.35),
            no_answer_min_score=_get_float("NO_ANSWER_MIN_SCORE", 0.20),
            citation_min_score=_get_float("CITATION_MIN_SCORE", 0.20),
            llm_request_timeout_seconds=_get_int("LLM_REQUEST_TIMEOUT_SECONDS", 120),
            enable_reranker=_get_bool("ENABLE_RERANKER", False),
            reranker_provider=os.getenv("RERANKER_PROVIDER", "noop"),
        )
