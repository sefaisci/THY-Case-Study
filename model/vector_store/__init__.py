"""Shared dense, sparse, and Qdrant utilities for both ingestion paths."""

from .qdrant_store import (
    OpenAIEmbedder,
    QdrantChunkStore,
    build_user_document_filter,
    build_user_documents_filter,
    build_user_filter,
)
from .schemas import ChunkRecord
from .sparse import StableHashSparseEncoder

__all__ = [
    "ChunkRecord",
    "OpenAIEmbedder",
    "QdrantChunkStore",
    "StableHashSparseEncoder",
    "build_user_document_filter",
    "build_user_documents_filter",
    "build_user_filter",
]
