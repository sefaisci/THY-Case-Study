"""Selectable sparse vector encoders used by ingestion and retrieval."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from threading import Lock
from typing import Any, Protocol

from qdrant_client import models

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class SparseEncoder(Protocol):
    """Shared sparse-vector contract for indexing and querying."""

    version: str
    requires_idf_modifier: bool

    def encode_document(self, text: str) -> models.SparseVector:
        """Encode document content for Qdrant indexing."""

    def encode_query(self, text: str) -> models.SparseVector:
        """Encode a search query for Qdrant retrieval."""


class StableHashSparseEncoder:
    """Legacy deterministic token-frequency encoder."""

    version = "blake2b-frequency-v1"
    requires_idf_modifier = False
    hash_space = 2_000_000_000

    def encode_document(self, text: str) -> models.SparseVector:
        """Encode document text through the legacy deterministic path."""

        return self._encode(text)

    def encode_query(self, text: str) -> models.SparseVector:
        """Encode query text through the legacy deterministic path."""

        return self._encode(text)

    def _encode(self, text: str) -> models.SparseVector:
        frequencies: dict[int, float] = {}
        for token in TOKEN_PATTERN.findall(text.lower()):
            index = self._index(token)
            frequencies[index] = frequencies.get(index, 0.0) + 1.0
        if not frequencies:
            frequencies[self._index("__empty__")] = 1.0
        ordered = sorted(frequencies.items())
        return models.SparseVector(
            indices=[index for index, _ in ordered],
            values=[value for _, value in ordered],
        )

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self.hash_space


class FastEmbedBM25SparseEncoder:
    """FastEmbed BM25 adapter with distinct passage and query paths."""

    version = "fastembed-qdrant-bm25-v1"
    requires_idf_modifier = True

    def __init__(
        self,
        *,
        model_name: str = "Qdrant/bm25",
        cache_dir: str | None = None,
        backend: Any | None = None,
    ) -> None:
        if model_name != "Qdrant/bm25":
            raise ValueError("FastEmbed BM25 requires model_name='Qdrant/bm25'.")
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._backend = backend
        self._backend_lock = Lock()

    def _get_backend(self) -> Any:
        """Construct and retain the FastEmbed backend only on first encoding."""

        backend = self._backend
        if backend is not None:
            return backend
        with self._backend_lock:
            backend = self._backend
            if backend is None:
                from fastembed import SparseTextEmbedding

                backend = SparseTextEmbedding(
                    model_name=self._model_name,
                    cache_dir=self._cache_dir,
                    lazy_load=True,
                )
                self._backend = backend
        return backend

    def encode_document(self, text: str) -> models.SparseVector:
        """Encode index content with FastEmbed's passage-specific path."""

        embedding = next(iter(self._get_backend().passage_embed([text])))
        return self._to_qdrant(embedding)

    def encode_query(self, text: str) -> models.SparseVector:
        """Encode search text with FastEmbed's query-specific path."""

        embedding = next(iter(self._get_backend().query_embed([text])))
        return self._to_qdrant(embedding)

    @staticmethod
    def _to_qdrant(embedding: Any) -> models.SparseVector:
        return models.SparseVector(
            indices=[int(value) for value in embedding.indices],
            values=[float(value) for value in embedding.values],
        )


@lru_cache(maxsize=8)
def create_sparse_encoder(
    provider: str,
    model_name: str = "Qdrant/bm25",
    cache_dir: str | None = None,
) -> SparseEncoder:
    """Return one process-cached sparse encoder for a provider configuration."""

    normalized = provider.strip().casefold()
    if normalized == "stable_hash":
        return StableHashSparseEncoder()
    if normalized == "fastembed_bm25":
        return FastEmbedBM25SparseEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported sparse encoder provider: {provider!r}.")
