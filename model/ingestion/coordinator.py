"""One entry point for optional multi-format semantic or Docling ingestion."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from model.document_processing.discovery import discover_documents
from model.document_processing.schemas import DocumentSource
from model.ingestion.schemas import (
    DocumentIngestionResult,
    IngestionRequest,
    IngestionRunResult,
)
from model.ingestion.progress import ProgressCallback
from model.ingestion.settings import IngestionSettings
from model.usage import UsageCallback


logger = logging.getLogger(__name__)


@dataclass
class IngestionCoordinator:
    """Discover all inputs and route every document through one selected method."""

    semantic_pipeline: object
    docling_pipeline: object
    document_max_concurrency: int = 2
    closeables: tuple[Any, ...] = ()
    _close_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    async def run(self, request: IngestionRequest) -> IngestionRunResult:
        """Process discovered documents independently and aggregate exact outcomes."""

        if self.document_max_concurrency <= 0:
            raise ValueError("document_max_concurrency must be greater than zero.")
        discovered = await asyncio.to_thread(
            discover_documents,
            pdf_path=request.pdf_path,
            docx_path=request.docx_path,
            pptx_path=request.pptx_path,
            filename_allowlist=request.filename_allowlist,
        )
        if not discovered.documents:
            return IngestionRunResult(
                method=request.method,
                user_id=request.user_id,
                discovered_count=0,
            )

        pipeline = self.semantic_pipeline if request.method == "semantic" else self.docling_pipeline
        results: list[DocumentIngestionResult] = []
        for start in range(0, len(discovered.documents), self.document_max_concurrency):
            document_batch = discovered.documents[
                start : start + self.document_max_concurrency
            ]
            batch_results = await asyncio.gather(
                *(
                    self._process_source(
                        pipeline,
                        source,
                        request,
                    )
                    for source in document_batch
                )
            )
            results.extend(batch_results)

        return IngestionRunResult(
            method=request.method,
            user_id=request.user_id,
            discovered_count=len(discovered.documents),
            completed_count=sum(result.status == "completed" for result in results),
            partial_count=sum(result.status == "partial" for result in results),
            failed_count=sum(result.status == "failed" for result in results),
            skipped_count=sum(result.status == "skipped" for result in results),
            chunk_count=sum(result.chunk_count for result in results),
            point_count=sum(result.point_count for result in results),
            documents=results,
        )

    async def _process_source(
        self,
        pipeline: object,
        source: DocumentSource,
        request: IngestionRequest,
    ) -> DocumentIngestionResult:
        try:
            return await pipeline.process_document(source, request.user_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return DocumentIngestionResult(
                document_id=source.document_id,
                document_name=source.document_name,
                document_type=source.document_type,
                method=request.method,
                status="failed",
                failures=[{"message": str(exc)[:1000]}],
            )

    async def aclose(self) -> None:
        """Idempotently attempt to close every coordinator-owned provider client."""

        if self._closed:
            return
        cancellation: asyncio.CancelledError | None = None
        async with self._close_lock:
            if self._closed:
                return
            for closeable in self.closeables:
                close = getattr(closeable, "aclose", None) or getattr(
                    closeable,
                    "close",
                    None,
                )
                if close is None:
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                except Exception:
                    logger.warning(
                        "Failed to close an ingestion provider client.",
                        exc_info=True,
                        extra={"event": "ingestion_client_close_failed"},
                    )
            self._closed = True
        if cancellation is not None:
            raise cancellation


async def create_connected_ingestion_coordinator(
    settings: IngestionSettings | None = None,
    *,
    usage_callback: UsageCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IngestionCoordinator:
    """Wire real OpenAI, Qdrant, semantic, and Docling ingestion adapters."""

    from model.document_processing.docling_pipeline import DoclingFixedChunkingPipeline
    from model.semantic_chunking import OpenAISemanticChunker, SemanticChunkingPipeline
    from model.vector_store import OpenAIEmbedder, QdrantChunkStore, StableHashSparseEncoder
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    settings = settings or IngestionSettings.from_env()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be configured for connected ingestion.")
    closeables: list[Any] = []
    try:
        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        closeables.append(openai_client)
        qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        closeables.append(qdrant_client)
        sparse_encoder = StableHashSparseEncoder()
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            vector_size=settings.dense_vector_size,
            batch_size=settings.embedding_batch_size,
            usage_callback=usage_callback,
            usage_stage="embeddings",
            client=openai_client,
        )
        store = QdrantChunkStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
            dense_vector_size=settings.dense_vector_size,
            sparse_encoder=sparse_encoder,
            client=qdrant_client,
        )
        semantic_pipeline = SemanticChunkingPipeline(
            settings=settings,
            chunker=OpenAISemanticChunker(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.semantic_model,
                reasoning_effort=settings.semantic_reasoning_effort,
                timeout_seconds=settings.llm_request_timeout_seconds,
                usage_callback=usage_callback,
                client=openai_client,
            ),
            embedder=embedder,
            store=store,
            sparse_encoder=sparse_encoder,
            progress_callback=progress_callback,
        )
        docling_pipeline = DoclingFixedChunkingPipeline(
            settings=settings,
            embedder=embedder,
            store=store,
            sparse_encoder=sparse_encoder,
            progress_callback=progress_callback,
        )
        return IngestionCoordinator(
            semantic_pipeline=semantic_pipeline,
            docling_pipeline=docling_pipeline,
            document_max_concurrency=settings.document_max_concurrency,
            closeables=tuple(closeables),
        )
    except BaseException:
        temporary = IngestionCoordinator(
            semantic_pipeline=None,
            docling_pipeline=None,
            closeables=tuple(closeables),
        )
        await temporary.aclose()
        raise
