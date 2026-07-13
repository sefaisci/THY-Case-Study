"""Docling extraction with page-scoped token-aware fixed chunking."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import tiktoken

from model.ingestion.progress import ProgressCallback, report_progress
from model.ingestion.schemas import DocumentIngestionResult, LocationFailure
from model.ingestion.settings import IngestionSettings
from model.vector_store import (
    ChunkRecord,
    OpenAIEmbedder,
    QdrantChunkStore,
    SparseEncoder,
)

from .schemas import DocumentSource


def normalize_markdown(text: str) -> str:
    """Normalize incidental whitespace while retaining Markdown structure."""

    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(normalized_lines)).strip()


def split_token_windows(
    text: str,
    *,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split one page without allowing a window to cross its boundary."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")
    tokens = tokenizer.encode(text)
    if not tokens:
        return []
    step = chunk_size - overlap
    windows: list[str] = []
    for start in range(0, len(tokens), step):
        token_window = tokens[start : start + chunk_size]
        if not token_window:
            break
        decoded = tokenizer.decode(token_window).strip()
        if decoded:
            windows.append(decoded)
        if start + chunk_size >= len(tokens):
            break
    return windows


class DoclingFixedChunkingPipeline:
    """Parse original documents with Docling and write fixed chunks to Qdrant."""

    def __init__(
        self,
        *,
        settings: IngestionSettings,
        embedder: OpenAIEmbedder,
        store: QdrantChunkStore,
        sparse_encoder: SparseEncoder | None = None,
        converter: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.sparse_encoder = (
            sparse_encoder if sparse_encoder is not None else store.sparse_encoder
        )
        self.converter = converter
        self.progress_callback = progress_callback
        self._conversion_lock = asyncio.Lock()
        try:
            self.tokenizer = tiktoken.encoding_for_model(settings.embedding_model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def process_document(
        self,
        source: DocumentSource,
        user_id: str,
    ) -> DocumentIngestionResult:
        """Convert one original source, chunk each location, embed, and upsert."""

        try:
            await self.store.delete_user_document_points(
                collection_name=self.settings.docling_collection,
                user_id=user_id,
                document_id=source.document_id,
            )
        except Exception as exc:
            return DocumentIngestionResult(
                document_id=source.document_id,
                document_name=source.document_name,
                document_type=source.document_type,
                method="docling",
                status="failed",
                failures=[
                    LocationFailure(
                        message=f"Docling vector cleanup failed before ingestion: {str(exc)[:900]}"
                    )
                ],
            )

        try:
            async with self._conversion_lock:
                page_numbers, records = await asyncio.to_thread(
                    self._convert_and_extract_records,
                    source,
                    user_id,
                )
        except Exception as exc:
            return DocumentIngestionResult(
                document_id=source.document_id,
                document_name=source.document_name,
                document_type=source.document_type,
                method="docling",
                status="failed",
                failures=[LocationFailure(message=str(exc)[:1000])],
            )

        total_locations = len(page_numbers)
        await report_progress(
            self.progress_callback,
            total_pages=total_locations,
            processed_pages=0,
        )
        if not records:
            await report_progress(
                self.progress_callback,
                total_pages=total_locations,
                processed_pages=total_locations,
            )
            return DocumentIngestionResult(
                document_id=source.document_id,
                document_name=source.document_name,
                document_type=source.document_type,
                method="docling",
                status="failed",
                total_locations=total_locations,
                processed_locations=total_locations,
                failures=[LocationFailure(message="Docling produced no non-empty text chunks.")],
            )

        try:
            vectors = await self.embedder.embed_documents([record.text for record in records])
        except Exception as exc:
            return DocumentIngestionResult(
                document_id=source.document_id,
                document_name=source.document_name,
                document_type=source.document_type,
                method="docling",
                status="failed",
                total_locations=total_locations,
                chunk_count=len(records),
                failures=[LocationFailure(message=f"Vector embedding failed: {str(exc)[:900]}")],
            )

        records_by_page: dict[int, list[ChunkRecord]] = {number: [] for number in page_numbers}
        vectors_by_page: dict[int, list[list[float]]] = {number: [] for number in page_numbers}
        for record, vector in zip(records, vectors, strict=True):
            page_number = record.slide_number or record.page_number
            if page_number is None:
                continue
            records_by_page.setdefault(page_number, []).append(record)
            vectors_by_page.setdefault(page_number, []).append(vector)

        processed_locations = 0
        point_count = 0
        failures: list[LocationFailure] = []
        for page_number in page_numbers:
            page_records = records_by_page.get(page_number, [])
            if page_records:
                try:
                    point_count += await self.store.upsert_chunks(
                        collection_name=self.settings.docling_collection,
                        chunks=page_records,
                        dense_vectors=vectors_by_page[page_number],
                    )
                except Exception as exc:
                    failures.append(
                        LocationFailure(
                            location_number=page_number,
                            message=f"Vector storage failed: {str(exc)[:900]}",
                        )
                    )
                    break
            processed_locations += 1
            await report_progress(
                self.progress_callback,
                total_pages=total_locations,
                processed_pages=processed_locations,
            )

        if point_count != len(records):
            failures.append(
                LocationFailure(
                    message=(
                        "Docling vector persistence is incomplete: "
                        f"expected {len(records)} point(s), stored {point_count}."
                    )
                )
            )
        return DocumentIngestionResult(
            document_id=source.document_id,
            document_name=source.document_name,
            document_type=source.document_type,
            method="docling",
            status="failed" if failures else "completed",
            total_locations=total_locations,
            processed_locations=processed_locations,
            chunk_count=len(records),
            point_count=point_count,
            failures=failures,
        )

    def _convert_and_extract_records(
        self,
        source: DocumentSource,
        user_id: str,
    ) -> tuple[list[int], list[ChunkRecord]]:
        """Run blocking Docling conversion, export, and tokenization in one worker."""

        if self.converter is None:
            self.converter = self._create_converter()
        conversion_result = self.converter.convert(source.path)
        document = conversion_result.document
        page_numbers = sorted(int(number) for number in document.pages)
        if not page_numbers:
            full_text = normalize_markdown(document.export_to_markdown())
            if not full_text:
                raise ValueError("Docling returned no pages, slides, or document text.")
            page_numbers = [1]
            records = self._extract_records(
                document,
                source,
                user_id,
                page_numbers,
                page_text_overrides={1: full_text},
            )
        else:
            records = self._extract_records(document, source, user_id, page_numbers)
        return page_numbers, records

    @staticmethod
    def _create_converter() -> Any:
        """Import and construct Docling inside the conversion worker thread."""

        from docling.document_converter import DocumentConverter

        return DocumentConverter()

    def _extract_records(
        self,
        document: Any,
        source: DocumentSource,
        user_id: str,
        page_numbers: list[int],
        page_text_overrides: dict[int, str] | None = None,
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        created_at = datetime.now(timezone.utc).isoformat()
        global_index = 0
        for page_number in page_numbers:
            page_text = (
                page_text_overrides[page_number]
                if page_text_overrides and page_number in page_text_overrides
                else normalize_markdown(document.export_to_markdown(page_no=page_number))
            )
            windows = split_token_windows(
                page_text,
                tokenizer=self.tokenizer,
                chunk_size=self.settings.fixed_chunk_size_tokens,
                overlap=self.settings.fixed_chunk_overlap_tokens,
            )
            for local_index, text in enumerate(windows):
                location_label = "slide" if source.document_type == "pptx" else "page"
                location = (
                    {"page_number": None, "slide_number": page_number}
                    if source.document_type == "pptx"
                    else {"page_number": page_number, "slide_number": None}
                )
                records.append(
                    ChunkRecord(
                        collection_name=self.settings.docling_collection,
                        collection_type="docling_fixed",
                        user_id=user_id,
                        document_id=source.document_id,
                        document_name=source.document_name,
                        document_type=source.document_type,
                        chunk_id=(
                            f"{source.document_id}-{location_label}-{page_number:04d}-"
                            f"fixed-{local_index:04d}"
                        ),
                        chunk_index=global_index,
                        source_pipeline="docling_fixed_chunking",
                        source_excerpt=re.sub(r"\s+", " ", text)[:1000],
                        text=text,
                        created_at=created_at,
                        source_sha256=source.source_sha256,
                        embedding_model=self.settings.embedding_model,
                        sparse_encoder_version=self.sparse_encoder.version,
                        metadata={
                            "chunk_size_tokens": self.settings.fixed_chunk_size_tokens,
                            "chunk_overlap_tokens": self.settings.fixed_chunk_overlap_tokens,
                            "tokenizer": self.tokenizer.name,
                            "docling_page_number": page_number,
                            "logical_page_fallback": bool(page_text_overrides),
                        },
                        **location,
                    )
                )
                global_index += 1
        return records
