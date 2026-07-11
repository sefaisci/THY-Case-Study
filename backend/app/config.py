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
    retrieval_top_k: int = 12
    rerank_top_k: int = 6
    max_context_chunks: int = 8
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    enable_reranker: bool = True
    reranker_provider: str = "heuristic"
    no_answer_min_score: float = 0.20
    citation_min_score: float = 0.20
    max_session_history_messages: int = Field(default=20, ge=0, le=100)

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
