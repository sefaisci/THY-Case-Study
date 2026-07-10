"""One entry point for optional multi-format semantic or Docling ingestion."""

from __future__ import annotations

from dataclasses import dataclass

from model.document_processing.discovery import discover_documents
from model.ingestion.schemas import (
    DocumentIngestionResult,
    IngestionRequest,
    IngestionRunResult,
)
from model.ingestion.settings import IngestionSettings
from model.usage import UsageCallback


@dataclass
class IngestionCoordinator:
    """Discover all inputs and route every document through one selected method."""

    semantic_pipeline: object
    docling_pipeline: object

    def run(self, request: IngestionRequest) -> IngestionRunResult:
        """Process discovered documents independently and aggregate exact outcomes."""

        discovered = discover_documents(
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
        for source in discovered.documents:
            try:
                result = pipeline.process_document(source, request.user_id)
            except Exception as exc:
                result = DocumentIngestionResult(
                    document_id=source.document_id,
                    document_name=source.document_name,
                    document_type=source.document_type,
                    method=request.method,
                    status="failed",
                    failures=[{"message": str(exc)[:1000]}],
                )
            results.append(result)

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


def create_connected_ingestion_coordinator(
    settings: IngestionSettings | None = None,
    *,
    usage_callback: UsageCallback | None = None,
) -> IngestionCoordinator:
    """Wire real OpenAI, Qdrant, semantic, and Docling ingestion adapters."""

    from model.document_processing.docling_pipeline import DoclingFixedChunkingPipeline
    from model.semantic_chunking import OpenAISemanticChunker, SemanticChunkingPipeline
    from model.vector_store import OpenAIEmbedder, QdrantChunkStore, StableHashSparseEncoder

    settings = settings or IngestionSettings.from_env()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be configured for connected ingestion.")
    sparse_encoder = StableHashSparseEncoder()
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        vector_size=settings.dense_vector_size,
        batch_size=settings.embedding_batch_size,
        usage_callback=usage_callback,
        usage_stage="embeddings",
    )
    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        dense_vector_size=settings.dense_vector_size,
        sparse_encoder=sparse_encoder,
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
        ),
        embedder=embedder,
        store=store,
        sparse_encoder=sparse_encoder,
    )
    docling_pipeline = DoclingFixedChunkingPipeline(
        settings=settings,
        embedder=embedder,
        store=store,
        sparse_encoder=sparse_encoder,
    )
    return IngestionCoordinator(
        semantic_pipeline=semantic_pipeline,
        docling_pipeline=docling_pipeline,
    )
