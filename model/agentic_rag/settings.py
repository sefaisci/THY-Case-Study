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


def _get_optional_text(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or None


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
    sparse_encoder_provider: str = "stable_hash"
    sparse_encoder_model: str = "Qdrant/bm25"
    sparse_encoder_cache_dir: str | None = None
    retrieval_prefetch_k: int = 20
    retrieval_collection_k: int = 15
    rerank_candidate_k: int = 30
    rerank_top_k: int = 6
    max_context_chunks: int = 8
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    no_answer_min_score: float = 0.20
    citation_min_score: float = 0.20
    llm_request_timeout_seconds: int = 120
    enable_reranker: bool = True
    reranker_provider: str = "openai"
    reranker_model: str | None = None
    reranker_reasoning_effort: str | None = "low"
    reranker_max_candidates: int = 30
    reranker_text_max_chars: int = 1600
    rerank_min_score: float = 0.50
    reranker_allow_partial_support: bool = False
    grounding_reasoning_effort: str = "low"
    grounding_max_retries: int = 1
    allowed_document_ids: tuple[str, ...] | None = None

    @property
    def effective_reranker_model(self) -> str:
        """Return the dedicated reranker model or the answer model fallback."""

        return self.reranker_model or self.self_service_llm_model

    @property
    def effective_reranker_provider(self) -> str:
        """Return the single normalized reranker mode used by all runtime layers."""

        if not self.enable_reranker:
            return "noop"
        return self.reranker_provider.strip().casefold()

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
            sparse_encoder_provider=os.getenv(
                "SPARSE_ENCODER_PROVIDER",
                "stable_hash",
            ),
            sparse_encoder_model=os.getenv("SPARSE_ENCODER_MODEL", "Qdrant/bm25"),
            sparse_encoder_cache_dir=os.getenv("SPARSE_ENCODER_CACHE_DIR") or None,
            retrieval_prefetch_k=_get_int("RETRIEVAL_PREFETCH_K", 20),
            retrieval_collection_k=_get_int("RETRIEVAL_COLLECTION_K", 15),
            rerank_candidate_k=_get_int("RERANK_CANDIDATE_K", 30),
            rerank_top_k=_get_int("RERANK_TOP_K", 6),
            max_context_chunks=_get_int("MAX_CONTEXT_CHUNKS", 8),
            hybrid_dense_weight=_get_float("HYBRID_DENSE_WEIGHT", 0.65),
            hybrid_sparse_weight=_get_float("HYBRID_SPARSE_WEIGHT", 0.35),
            no_answer_min_score=_get_float("NO_ANSWER_MIN_SCORE", 0.20),
            citation_min_score=_get_float("CITATION_MIN_SCORE", 0.20),
            llm_request_timeout_seconds=_get_int("LLM_REQUEST_TIMEOUT_SECONDS", 120),
            enable_reranker=_get_bool("ENABLE_RERANKER", True),
            reranker_provider=os.getenv("RERANKER_PROVIDER", "openai"),
            reranker_model=os.getenv("RERANKER_MODEL") or None,
            reranker_reasoning_effort=_get_optional_text(
                "RERANKER_REASONING_EFFORT",
                "low",
            ),
            reranker_max_candidates=_get_int("RERANKER_MAX_CANDIDATES", 30),
            reranker_text_max_chars=_get_int("RERANKER_TEXT_MAX_CHARS", 1600),
            rerank_min_score=_get_float("RERANK_MIN_SCORE", 0.50),
            reranker_allow_partial_support=_get_bool(
                "RERANKER_ALLOW_PARTIAL_SUPPORT",
                False,
            ),
            grounding_reasoning_effort=os.getenv(
                "GROUNDING_REASONING_EFFORT",
                "low",
            ),
            grounding_max_retries=_get_int("GROUNDING_MAX_RETRIES", 1),
        )
