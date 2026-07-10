"""End-to-end semantic page-image chunking and Qdrant ingestion."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from model.document_processing.rendering import render_document
from model.document_processing.schemas import DocumentSource
from model.ingestion.schemas import DocumentIngestionResult, LocationFailure
from model.ingestion.settings import IngestionSettings
from model.vector_store import (
    ChunkRecord,
    OpenAIEmbedder,
    QdrantChunkStore,
    StableHashSparseEncoder,
)

from .openai_adapter import OpenAISemanticChunker
from .schemas import SemanticPageResult


class SemanticChunkingPipeline:
    """Render, understand, embed, and store every page of one document."""

    def __init__(
        self,
        *,
        settings: IngestionSettings,
        chunker: OpenAISemanticChunker,
        embedder: OpenAIEmbedder,
        store: QdrantChunkStore,
        sparse_encoder: StableHashSparseEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.sparse_encoder = sparse_encoder or store.sparse_encoder
        if self.settings.semantic_flush_batch_size <= 0:
            raise ValueError("semantic_flush_batch_size must be greater than zero.")

    def process_document(self, source: DocumentSource, user_id: str) -> DocumentIngestionResult:
        """Analyze independent pages and flush bounded vector batches."""

        failures: list[LocationFailure] = []
        try:
            self.store.delete_user_document_points(
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
            pages = render_document(
                source,
                page_image_dir=self.settings.page_image_dir,
                processing_dir=self.settings.processing_dir,
                dpi=self.settings.conversion_dpi,
            )
        except Exception as exc:
            return _failed_document(source, str(exc))

        buffered_records: list[ChunkRecord] = []
        processed_locations = 0
        chunk_count = 0
        point_count = 0
        storage_failed = False
        for page in pages:
            try:
                result = self.chunker.chunk_page(page)
                page_records = self._map_records(source, user_id, result)
            except Exception as exc:
                failures.append(
                    LocationFailure(
                        location_number=page.location_number,
                        message=str(exc)[:1000],
                    )
                )
                continue

            buffered_records.extend(page_records)
            chunk_count += len(page_records)
            processed_locations += 1
            while len(buffered_records) >= self.settings.semantic_flush_batch_size:
                batch = buffered_records[: self.settings.semantic_flush_batch_size]
                del buffered_records[: self.settings.semantic_flush_batch_size]
                try:
                    point_count += self._flush_records(batch)
                except Exception as exc:
                    failures.append(
                        LocationFailure(
                            location_number=page.location_number,
                            message=f"Vector storage failed: {str(exc)[:900]}",
                        )
                    )
                    storage_failed = True
                    break
            if storage_failed:
                break

        if buffered_records and not storage_failed:
            try:
                point_count += self._flush_records(list(buffered_records))
                buffered_records.clear()
            except Exception as exc:
                failures.append(LocationFailure(message=f"Vector storage failed: {str(exc)[:900]}"))

        if chunk_count == 0:
            failures.append(
                LocationFailure(message="Semantic chunking produced no chunks for this document.")
            )
        if point_count != chunk_count:
            failures.append(
                LocationFailure(
                    message=(
                        "Semantic vector persistence is incomplete: "
                        f"expected {chunk_count} point(s), stored {point_count}."
                    )
                )
            )
        status = "completed" if not failures else "failed"
        return DocumentIngestionResult(
            document_id=source.document_id,
            document_name=source.document_name,
            document_type=source.document_type,
            method="semantic",
            status=status,
            processed_locations=processed_locations,
            chunk_count=chunk_count,
            point_count=point_count,
            failures=failures,
        )

    def _flush_records(self, records: list[ChunkRecord]) -> int:
        """Embed and upsert one bounded record batch."""

        vectors = self.embedder.embed_documents([record.text for record in records])
        return self.store.upsert_chunks(
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
                    source_excerpt=chunk.source_excerpt,
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
