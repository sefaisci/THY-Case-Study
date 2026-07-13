"""Environment-backed settings for document ingestion pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _integer(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)


@dataclass(frozen=True)
class IngestionSettings:
    """Connected ingestion configuration shared by notebooks and model code."""

    project_root: Path
    page_image_dir: Path
    processing_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    qdrant_url: str
    qdrant_api_key: str | None
    semantic_collection: str
    docling_collection: str
    dense_vector_name: str
    sparse_vector_name: str
    dense_vector_size: int
    embedding_model: str
    semantic_model: str
    semantic_reasoning_effort: str
    conversion_dpi: int
    fixed_chunk_size_tokens: int
    fixed_chunk_overlap_tokens: int
    embedding_batch_size: int
    llm_request_timeout_seconds: int
    semantic_flush_batch_size: int = 32
    semantic_page_max_concurrency: int = 3
    document_max_concurrency: int = 2
    sparse_encoder_provider: str = "stable_hash"
    sparse_encoder_model: str = "Qdrant/bm25"
    sparse_encoder_cache_dir: str | None = None

    @classmethod
    def from_env(
        cls,
        dotenv_path: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> "IngestionSettings":
        """Load local settings without overriding already-set process values."""

        root = Path(project_root or Path.cwd()).resolve()
        if root.name == "notebook":
            root = root.parent
        env_path = Path(dotenv_path).resolve() if dotenv_path else root / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

        def rooted(name: str, default: str) -> Path:
            value = Path(os.getenv(name, default)).expanduser()
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        return cls(
            project_root=root,
            page_image_dir=rooted("PAGE_IMAGE_DIR", "data/page_images"),
            processing_dir=rooted("PROCESSING_DIR", "data/processing"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            semantic_collection=os.getenv("QDRANT_COLLECTION_SEMANTIC", "semantic_chunks"),
            docling_collection=os.getenv("QDRANT_COLLECTION_DOCLING", "docling_fixed_chunks"),
            dense_vector_name=os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense"),
            sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
            dense_vector_size=_integer("QDRANT_DENSE_VECTOR_SIZE", 1536),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            semantic_model=os.getenv("SEMANTIC_CHUNKING_LLM_MODEL", "gpt-5.5"),
            semantic_reasoning_effort=os.getenv("SEMANTIC_CHUNKING_REASONING_EFFORT", "low"),
            conversion_dpi=_integer("DOC_CONVERSION_DPI", 200),
            fixed_chunk_size_tokens=_integer("FIXED_CHUNK_SIZE_TOKENS", 800),
            fixed_chunk_overlap_tokens=_integer("FIXED_CHUNK_OVERLAP_TOKENS", 120),
            embedding_batch_size=_integer("EMBEDDING_BATCH_SIZE", 32),
            llm_request_timeout_seconds=_integer("LLM_REQUEST_TIMEOUT_SECONDS", 120),
            semantic_flush_batch_size=_integer("SEMANTIC_FLUSH_BATCH_SIZE", 32),
            semantic_page_max_concurrency=_integer(
                "SEMANTIC_PAGE_MAX_CONCURRENCY",
                3,
            ),
            document_max_concurrency=_integer("DOCUMENT_MAX_CONCURRENCY", 2),
            sparse_encoder_provider=os.getenv(
                "SPARSE_ENCODER_PROVIDER",
                "stable_hash",
            ),
            sparse_encoder_model=os.getenv(
                "SPARSE_ENCODER_MODEL",
                "Qdrant/bm25",
            ),
            sparse_encoder_cache_dir=(
                os.getenv("SPARSE_ENCODER_CACHE_DIR") or None
            ),
        )
