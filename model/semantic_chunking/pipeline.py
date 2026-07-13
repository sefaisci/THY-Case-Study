"""End-to-end semantic page-image chunking and Qdrant ingestion."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from model.document_processing.rendering import render_document
from model.document_processing.schemas import DocumentSource
from model.ingestion.progress import ProgressCallback, report_progress
from model.ingestion.schemas import DocumentIngestionResult, LocationFailure
from model.ingestion.settings import IngestionSettings
from model.vector_store import (
    ChunkRecord,
    OpenAIEmbedder,
    QdrantChunkStore,
    SparseEncoder,
)

from .openai_adapter import OpenAISemanticChunker
from .schemas import SemanticPageResult

_MAX_CITATION_EXCERPT_CHARS = 2_000


class SemanticChunkingPipeline:
    """Render, understand, embed, and store every page of one document."""

    def __init__(
        self,
        *,
        settings: IngestionSettings,
        chunker: OpenAISemanticChunker,
        embedder: OpenAIEmbedder,
        store: QdrantChunkStore,
        sparse_encoder: SparseEncoder | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.sparse_encoder = (
            sparse_encoder if sparse_encoder is not None else store.sparse_encoder
        )
        self.progress_callback = progress_callback
        if self.settings.semantic_flush_batch_size <= 0:
            raise ValueError("semantic_flush_batch_size must be greater than zero.")
        if self.settings.semantic_page_max_concurrency <= 0:
            raise ValueError("semantic_page_max_concurrency must be greater than zero.")

    async def process_document(
        self,
        source: DocumentSource,
        user_id: str,
    ) -> DocumentIngestionResult:
        """Analyze independent pages and flush bounded vector batches."""

        failures: list[LocationFailure] = []
        try:
            await self.store.delete_user_document_points(
                collection_name=self.settings.semantic_collection,
                user_id=user_id,
                document_id=source.document_id,
            )
        except Exception as exc:
            return _failed_document(
                source,
                f"Semantic vector cleanup failed before ingestion: {str(exc)[:900]}",
            )
        try:
            pages = await render_document(
                source,
                page_image_dir=self.settings.page_image_dir,
                processing_dir=self.settings.processing_dir,
                dpi=self.settings.conversion_dpi,
            )
        except Exception as exc:
            return _failed_document(source, str(exc))

        total_locations = len(pages)
        await report_progress(
            self.progress_callback,
            total_pages=total_locations,
            processed_pages=0,
        )

        records: list[ChunkRecord] = []
        processed_locations = 0
        reported_locations = 0
        page_concurrency = self.settings.semantic_page_max_concurrency
        for start in range(0, len(pages), page_concurrency):
            page_batch = pages[start : start + page_concurrency]
            page_results = await asyncio.gather(
                *(self.chunker.chunk_page(page) for page in page_batch),
                return_exceptions=True,
            )
            for page, result in zip(page_batch, page_results, strict=True):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    failures.append(
                        LocationFailure(
                            location_number=page.location_number,
                            message=str(result)[:1000],
                        )
                    )
                    continue
                page_records = self._map_records(source, user_id, result)
                records.extend(page_records)
                processed_locations += 1
                # Reserve the final location until every embedding and Qdrant
                # write has succeeded. This prevents the UI from presenting
                # 100% while a buffered vector flush is still in flight.
                visible_locations = min(
                    processed_locations,
                    max(0, total_locations - 1),
                )
                if visible_locations > reported_locations:
                    await report_progress(
                        self.progress_callback,
                        total_pages=total_locations,
                        processed_pages=visible_locations,
                    )
                    reported_locations = visible_locations
        chunk_count = len(records)
        point_count = 0
        if chunk_count == 0:
            failures.append(
                LocationFailure(message="Semantic chunking produced no chunks for this document.")
            )

        optimistic_point_count = 0
        if not failures:
            for start in range(0, chunk_count, self.settings.semantic_flush_batch_size):
                batch = records[start : start + self.settings.semantic_flush_batch_size]
                try:
                    optimistic_point_count += await self._flush_records(batch)
                except Exception as exc:
                    failures.append(
                        LocationFailure(message=f"Vector storage failed: {str(exc)[:900]}")
                    )
                    break

        if not failures and optimistic_point_count != chunk_count:
            failures.append(
                LocationFailure(
                    message=(
                        "Semantic vector persistence is incomplete: "
                        f"expected {chunk_count} point(s), stored {optimistic_point_count}."
                    )
                )
            )

        if not failures:
            try:
                point_count = await self.store.verify_chunks_persisted(
                    collection_name=self.settings.semantic_collection,
                    chunks=records,
                )
            except Exception as exc:
                failures.append(
                    LocationFailure(
                        message=f"Semantic vector verification failed: {str(exc)[:900]}"
                    )
                )

        if not failures and point_count != chunk_count:
            failures.append(
                LocationFailure(
                    message=(
                        "Semantic vector verification is incomplete: "
                        f"expected {chunk_count} point(s), verified {point_count}."
                    )
                )
            )

        if failures:
            try:
                await self.store.delete_user_document_points(
                    collection_name=self.settings.semantic_collection,
                    user_id=user_id,
                    document_id=source.document_id,
                )
                point_count = 0
            except Exception as exc:
                failures.append(
                    LocationFailure(
                        message=f"Failed semantic-point cleanup was incomplete: {str(exc)[:900]}"
                    )
                )

        status = "completed" if not failures else "failed"
        if status == "completed":
            await report_progress(
                self.progress_callback,
                total_pages=total_locations,
                processed_pages=total_locations,
            )
        return DocumentIngestionResult(
            document_id=source.document_id,
            document_name=source.document_name,
            document_type=source.document_type,
            method="semantic",
            status=status,
            total_locations=total_locations,
            processed_locations=processed_locations,
            chunk_count=chunk_count,
            point_count=point_count,
            failures=failures,
        )

    async def _flush_records(self, records: list[ChunkRecord]) -> int:
        """Embed and upsert one bounded record batch."""

        vectors = await self.embedder.embed_documents([record.text for record in records])
        return await self.store.upsert_chunks(
            collection_name=self.settings.semantic_collection,
            chunks=records,
            dense_vectors=vectors,
        )

    def _map_records(
        self,
        source: DocumentSource,
        user_id: str,
        page_result: SemanticPageResult,
    ) -> list[ChunkRecord]:
        created_at = datetime.now(timezone.utc).isoformat()
        location = (
            {"page_number": None, "slide_number": page_result.page_number}
            if source.document_type == "pptx"
            else {"page_number": page_result.page_number, "slide_number": None}
        )
        records: list[ChunkRecord] = []
        for index, chunk in enumerate(page_result.chunks):
            safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", chunk.chunk_key).strip("-").lower()
            location_label = "slide" if source.document_type == "pptx" else "page"
            chunk_id = f"{source.document_id}-{location_label}-{page_result.page_number:04d}-{safe_key}"
            records.append(
                ChunkRecord(
                    collection_name=self.settings.semantic_collection,
                    collection_type="semantic",
                    user_id=user_id,
                    document_id=source.document_id,
                    document_name=source.document_name,
                    document_type=source.document_type,
                    chunk_id=chunk_id,
                    chunk_index=index,
                    source_pipeline="semantic_image_chunking",
                    source_excerpt=_citation_excerpt(chunk.text),
                    text=chunk.text,
                    created_at=created_at,
                    source_sha256=source.source_sha256,
                    embedding_model=self.settings.embedding_model,
                    sparse_encoder_version=self.sparse_encoder.version,
                    metadata={
                        "title": chunk.title,
                        "chunk_key": chunk.chunk_key,
                        "keywords": chunk.keywords,
                        "relationships": chunk.relationships,
                        "confidence": chunk.confidence,
                        "page_summary": page_result.page_summary,
                        "page_classification": page_result.page_classification,
                        "model_source_excerpt": chunk.source_excerpt,
                        "semantic_model": self.settings.semantic_model,
                        "semantic_reasoning_effort": self.settings.semantic_reasoning_effort,
                    },
                    **location,
                )
            )
        return records


def _failed_document(source: DocumentSource, message: str) -> DocumentIngestionResult:
    return DocumentIngestionResult(
        document_id=source.document_id,
        document_name=source.document_name,
        document_type=source.document_type,
        method="semantic",
        status="failed",
        failures=[LocationFailure(message=message[:1000])],
    )


def _citation_excerpt(text: str) -> str:
    """Return the bounded evidence text actually exposed to grounded generation."""

    normalized = text.strip()
    if len(normalized) <= _MAX_CITATION_EXCERPT_CHARS:
        return normalized

    candidate = normalized[: _MAX_CITATION_EXCERPT_CHARS - 1]
    minimum_boundary = int(_MAX_CITATION_EXCERPT_CHARS * 0.8)
    boundary = max(
        candidate.rfind("\n\n", minimum_boundary),
        candidate.rfind("\n", minimum_boundary),
        candidate.rfind(" ", minimum_boundary),
    )
    if boundary >= minimum_boundary:
        candidate = candidate[:boundary]
    return f"{candidate.rstrip()}…"
