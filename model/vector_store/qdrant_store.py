"""OpenAI embedding and Qdrant storage adapters for normalized chunks."""

from __future__ import annotations

import asyncio
import inspect
import uuid
import weakref
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from model.usage import UsageCallback, emit_usage, usage_from_response

from .schemas import ChunkRecord
from .sparse import StableHashSparseEncoder


_COLLECTION_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()


def _collection_lock(collection_name: str) -> asyncio.Lock:
    """Return one process-wide collection setup lock for the active event loop."""

    loop = asyncio.get_running_loop()
    by_collection = _COLLECTION_LOCKS.setdefault(loop, {})
    return by_collection.setdefault(collection_name, asyncio.Lock())


def build_user_filter(user_id: str) -> models.Filter:
    """Build the mandatory server-side owner filter."""

    if not user_id.strip():
        raise ValueError("user_id must not be empty.")
    return models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
    )


def build_user_document_filter(user_id: str, document_id: str) -> models.Filter:
    """Build the mandatory owner-and-document filter used for deletion."""

    if not document_id.strip():
        raise ValueError("document_id must not be empty.")
    user_filter = build_user_filter(user_id)
    return models.Filter(
        must=[
            *user_filter.must,
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ]
    )


def build_user_documents_filter(
    user_id: str,
    document_ids: Sequence[str],
) -> models.Filter:
    """Build an owner filter restricted to an explicit active-document allowlist."""

    allowed = [document_id.strip() for document_id in document_ids if document_id.strip()]
    if not allowed:
        raise ValueError("document_ids must contain at least one identifier.")
    user_filter = build_user_filter(user_id)
    return models.Filter(
        must=[
            *user_filter.must,
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=allowed),
            ),
        ]
    )


class OpenAIEmbedder:
    """Batched OpenAI document and query embedding adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        vector_size: int,
        batch_size: int = 32,
        base_url: str | None = None,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
        usage_stage: str = "embeddings",
    ) -> None:
        if not api_key and client is None:
            raise ValueError("OPENAI_API_KEY is required for connected embeddings.")
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self.model = model
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.usage_callback = usage_callback
        self.usage_stage = usage_stage

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text batches and retain the input order."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = await self._client.embeddings.create(model=self.model, input=batch)
            emit_usage(
                self.usage_callback,
                usage_from_response(
                    response,
                    stage=self.usage_stage,
                    fallback_model=self.model,
                    metadata={"batch_size": len(batch)},
                ),
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        if len(vectors) != len(texts):
            raise ValueError(f"Embedding count mismatch: expected {len(texts)}, received {len(vectors)}.")
        dimensions = {len(vector) for vector in vectors}
        if dimensions != {self.vector_size}:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.vector_size}, received {sorted(dimensions)}."
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query using the same configured model."""

        return (await self.embed_documents([text]))[0]


class QdrantChunkStore:
    """Create compatible collections and idempotently upsert named vectors."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        dense_vector_name: str,
        sparse_vector_name: str,
        dense_vector_size: int,
        sparse_encoder: StableHashSparseEncoder | None = None,
        client: Any | None = None,
    ) -> None:
        self._owns_client = client is None
        self.client = client or AsyncQdrantClient(url=url, api_key=api_key)
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.dense_vector_size = dense_vector_size
        self.sparse_encoder = sparse_encoder or StableHashSparseEncoder()
        self._ready_collections: set[str] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def aclose(self) -> None:
        """Idempotently close only a Qdrant client created by this store."""

        if not self._owns_client or self._closed:
            return
        async with self._close_lock:
            if self._closed:
                return
            close = getattr(self.client, "aclose", None) or getattr(
                self.client,
                "close",
                None,
            )
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
            self._closed = True

    async def ensure_compatible_collection(self, collection_name: str) -> None:
        """Create a missing collection or reject incompatible vector settings."""

        if collection_name in self._ready_collections:
            return
        lock = _collection_lock(collection_name)
        async with lock:
            if collection_name in self._ready_collections:
                return
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                try:
                    await self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            self.dense_vector_name: models.VectorParams(
                                size=self.dense_vector_size,
                                distance=models.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={
                            self.sparse_vector_name: models.SparseVectorParams(
                                modifier=models.Modifier.IDF
                            )
                        },
                    )
                except Exception as exc:
                    message = str(exc).casefold()
                    if "already exists" not in message:
                        raise
            info = await self.client.get_collection(collection_name)
            vectors = info.config.params.vectors
            sparse_vectors = info.config.params.sparse_vectors or {}
            dense_config = vectors.get(self.dense_vector_name) if isinstance(vectors, dict) else None
            if dense_config is None or dense_config.size != self.dense_vector_size:
                actual = getattr(dense_config, "size", None)
                raise ValueError(
                    f"Collection {collection_name!r} has incompatible dense vector {self.dense_vector_name!r}: "
                    f"expected {self.dense_vector_size}, received {actual}."
                )
            if self.sparse_vector_name not in sparse_vectors:
                raise ValueError(
                    f"Collection {collection_name!r} is missing sparse vector {self.sparse_vector_name!r}."
                )
            await self._ensure_payload_indexes(collection_name)
            self._ready_collections.add(collection_name)

    async def upsert_chunks(
        self,
        *,
        collection_name: str,
        chunks: Sequence[ChunkRecord],
        dense_vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert normalized chunks with named dense and sparse vectors."""

        if len(chunks) != len(dense_vectors):
            raise ValueError("Chunk and dense-vector counts must match.")
        if not chunks:
            return 0
        await self.ensure_compatible_collection(collection_name)
        points = await asyncio.to_thread(
            self._build_points,
            collection_name,
            chunks,
            dense_vectors,
        )
        await self.client.upsert(collection_name=collection_name, points=points, wait=True)
        return len(points)

    def _build_points(
        self,
        collection_name: str,
        chunks: Sequence[ChunkRecord],
        dense_vectors: Sequence[Sequence[float]],
    ) -> list[models.PointStruct]:
        """Build CPU-bound sparse vectors and point payloads in a worker thread."""

        points: list[models.PointStruct] = []
        for chunk, dense in zip(chunks, dense_vectors, strict=True):
            if chunk.collection_name != collection_name:
                raise ValueError(
                    f"Chunk {chunk.chunk_id!r} targets {chunk.collection_name!r}, not {collection_name!r}."
                )
            if len(dense) != self.dense_vector_size:
                raise ValueError(
                    f"Dense vector for {chunk.chunk_id!r} has {len(dense)} dimensions; "
                    f"expected {self.dense_vector_size}."
                )
            points.append(
                models.PointStruct(
                    id=self.point_id(collection_name, chunk),
                    vector={
                        self.dense_vector_name: list(dense),
                        self.sparse_vector_name: self.sparse_encoder.encode(chunk.text),
                    },
                    payload=chunk.payload(),
                )
            )
        return points

    async def count_user_document_points(
        self,
        *,
        collection_name: str,
        user_id: str,
        document_id: str,
    ) -> int:
        """Count exact owner/document points through server-side filters."""

        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        )
        response = await self.client.count(
            collection_name=collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(response.count)

    async def delete_user_document_points(
        self,
        *,
        collection_name: str,
        user_id: str,
        document_id: str,
    ) -> int:
        """Idempotently clear one owner's document points from one collection.

        This method is safe for both first ingestion and retry preparation: a
        missing collection or an owner/document pair with no points is a no-op.
        """

        if not await self.client.collection_exists(collection_name):
            return 0
        point_filter = build_user_document_filter(user_id, document_id)
        existing = await self.client.count(
            collection_name=collection_name,
            count_filter=point_filter,
            exact=True,
        )
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=point_filter),
            wait=True,
        )
        return int(existing.count)

    @staticmethod
    def point_id(collection_name: str, chunk: ChunkRecord) -> str:
        """Return a stable UUIDv5 for an owner-scoped chunk."""

        stable_key = f"{collection_name}:{chunk.user_id}:{chunk.document_id}:{chunk.chunk_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        for field_name in ("user_id", "document_id", "document_type", "source_pipeline"):
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:  # Qdrant versions differ on existing-index responses.
                message = str(exc).lower()
                if "already exists" not in message and "already indexed" not in message:
                    raise
