"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration shared by API, services, and migrations."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "THY Agentic RAG"
    app_env: str = "local"
    app_debug: bool = False
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8501,http://127.0.0.1:8501"
    )

    database_url: str = "postgresql+psycopg://thy_app:change-me@localhost:5432/thy_case_study"
    upload_dir: Path = Path("data/uploads")
    page_image_dir: Path = Path("data/page_images")
    processing_dir: Path = Path("data/processing")
    max_upload_size_mb: int = Field(default=100, gt=0, le=500)
    allowed_file_extensions: str = "pdf,docx,pptx"

    openai_api_key: str | None = None
    openai_base_url: str | None = "https://api.openai.com/v1"
    openai_model_cache_seconds: int = Field(default=300, ge=0)
    model_capabilities_path: Path = Path("config/model-capabilities.v1.json")
    pricing_registry_path: Path = Path("config/pricing/openai-pricing.v1.json")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_semantic: str = "semantic_chunks"
    qdrant_collection_docling: str = "docling_fixed_chunks"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_dense_vector_size: int = 1536

    embedding_model: str = "text-embedding-3-small"
    doc_conversion_dpi: int = 200
    semantic_flush_batch_size: int = Field(default=32, gt=0, le=1000)
    fixed_chunk_size_tokens: int = 800
    fixed_chunk_overlap_tokens: int = 120
    embedding_batch_size: int = 32
    ingestion_job_concurrency: int = Field(default=4, gt=0, le=16)
    semantic_page_max_concurrency: int = Field(default=3, gt=0, le=32)
    document_max_concurrency: int = Field(default=2, gt=0, le=16)
    llm_request_timeout_seconds: int = 120
    sparse_encoder_provider: str = "stable_hash"
    sparse_encoder_model: str = "Qdrant/bm25"
    sparse_encoder_cache_dir: str | None = None
    retrieval_prefetch_k: int = Field(default=20, gt=0, le=200)
    retrieval_collection_k: int = Field(default=15, gt=0, le=100)
    rerank_candidate_k: int = Field(default=30, gt=0, le=100)
    rerank_top_k: int = Field(default=6, gt=0, le=30)
    max_context_chunks: int = Field(default=8, gt=0, le=30)
    hybrid_dense_weight: float = Field(default=0.65, ge=0.0)
    hybrid_sparse_weight: float = Field(default=0.35, ge=0.0)
    enable_reranker: bool = True
    reranker_provider: str = "openai"
    reranker_model: str | None = None
    reranker_reasoning_effort: str | None = "low"
    reranker_max_candidates: int = Field(default=30, gt=0, le=100)
    reranker_text_max_chars: int = Field(default=1600, gt=0, le=8000)
    rerank_min_score: float = Field(default=0.50, ge=0.0, le=1.0)
    reranker_allow_partial_support: bool = False
    grounding_reasoning_effort: str = "low"
    grounding_max_retries: int = Field(default=1, ge=0, le=3)
    no_answer_min_score: float = Field(default=0.20, ge=0.0, le=1.0)
    citation_min_score: float = Field(default=0.20, ge=0.0, le=1.0)
    max_session_history_messages: int = Field(default=20, ge=0, le=100)

    @field_validator("reranker_reasoning_effort", mode="before")
    @classmethod
    def validate_reranker_reasoning_effort(cls, value: object) -> str | None:
        """Normalize an optional supported OpenAI reasoning effort."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Reranker reasoning effort must be text or null.")
        normalized = value.strip().casefold()
        if not normalized:
            return None
        if normalized not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported reranker reasoning effort.")
        return normalized

    @field_validator("grounding_reasoning_effort", mode="before")
    @classmethod
    def validate_grounding_reasoning_effort(cls, value: object) -> str:
        """Normalize the independent structured grounding reasoning effort."""

        if not isinstance(value, str):
            raise ValueError("Grounding reasoning effort must be text.")
        normalized = value.strip().casefold()
        if normalized not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported grounding reasoning effort.")
        return normalized

    @field_validator(
        "upload_dir",
        "page_image_dir",
        "processing_dir",
        "model_capabilities_path",
        "pricing_registry_path",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not process cwd."""

        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            item.strip().lower().lstrip(".")
            for item in self.allowed_file_extensions.split(",")
            if item.strip()
        }

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allowed_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""

    return Settings()
