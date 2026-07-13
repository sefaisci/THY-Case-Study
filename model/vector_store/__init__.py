"""Shared dense, sparse, and Qdrant utilities for both ingestion paths."""

from .qdrant_store import (
    OpenAIEmbedder,
    QdrantChunkStore,
    build_user_document_filter,
    build_user_documents_filter,
    build_user_filter,
)
from .schemas import ChunkRecord
from .sparse import (
    FastEmbedBM25SparseEncoder,
    SparseEncoder,
    StableHashSparseEncoder,
    create_sparse_encoder,
)

__all__ = [
    "ChunkRecord",
    "FastEmbedBM25SparseEncoder",
    "OpenAIEmbedder",
    "QdrantChunkStore",
    "SparseEncoder",
    "StableHashSparseEncoder",
    "build_user_document_filter",
    "build_user_documents_filter",
    "build_user_filter",
    "create_sparse_encoder",
]
