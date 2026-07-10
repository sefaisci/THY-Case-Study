# Agentic RAG Model Package

This package contains the connected, notebook-first LangGraph workflow for the THY self-service document chatbot. Reusable ingestion code lives beside it under `model/ingestion`, `model/document_processing`, `model/semantic_chunking`, and `model/vector_store`.

## Multi-Format Ingestion

One request can supply PDF, DOCX, and PPTX directories together. Each path can independently be `None`, a matching file, or an empty directory. Select exactly one ingestion method per run:

```python
from pathlib import Path

from model.ingestion import (
    IngestionRequest,
    IngestionSettings,
    create_connected_ingestion_coordinator,
)

settings = IngestionSettings.from_env(".env", project_root=Path.cwd())
coordinator = create_connected_ingestion_coordinator(settings)

result = coordinator.run(
    IngestionRequest(
        method="semantic",  # or "docling"
        user_id="local-demo-user",
        pdf_path=Path("files/pdf"),
        docx_path=Path("files/docx"),
        pptx_path=Path("files/pptx"),
    )
)
```

`semantic` renders every page or slide to an image and analyzes each location independently. It supplies no previous-page memory and returns one flat list of variable-length chunks; each chunk's authoritative `text` is embedded exactly as stored in `semantic_chunks`. `docling` parses the original source, creates page-scoped fixed token windows, and writes them to `docling_fixed_chunks`. Both paths use `text-embedding-3-small` and the same deterministic sparse encoder.

## Graph Shape

```text
START
-> query_understanding (standalone + bilingual retrieval rewrite)
-> conversation_generation, for explicit active-session history questions or users without completed documents
-> retrieval_subgraph, otherwise
   -> retrieval_planner
   -> hybrid_retrieval (active-document filter, dense + sparse, weighted Qdrant RRF)
   -> reranking
-> conversation_generation, when evidence is empty or below threshold
-> answer_subgraph, when evidence is sufficient
   -> answer_generation
   -> citation_validation
   -> claim_evidence_reflection
-> conversation_generation, when the grounded draft is rejected
-> final_response
-> END
```

The graph is intentionally modular. Retrieval, reranking, grounded answer generation, citation validation, reflection, and conversational generation can be replaced without changing the public runner. Conversational responses use only the active session's bounded PostgreSQL history, contain no document citations, and never cross into another chat session.

## Notebook Usage

```python
from model.agentic_rag import (
    RagSettings,
    create_openai_qdrant_adapters,
    run_rag_question,
)

settings = RagSettings.from_env(".env")
adapters = create_openai_qdrant_adapters(settings)
response = run_rag_question(
    "What three parts of differential equations study are listed on page 1?",
    user_id="local-demo-user",
    adapters=adapters,
    settings=settings,
)
response.model_dump()
```

The default runner still supports fake adapters for offline tests. Connected execution must explicitly pass `create_openai_qdrant_adapters(settings)`.

## Security Invariant

Every dense and sparse Qdrant prefetch applies `user_id == current_user_id`. Backend chat requests additionally restrict points to document IDs currently marked `completed` in PostgreSQL, preventing stale or deleted points from entering answer generation. Standalone questions do not inherit unrelated prior turns; referential follow-ups use only the immediately preceding user question before the Responses API produces an English retrieval variant.

Grounded answers must contain at least one exact internal chunk ID. Unknown, weak, or cross-user IDs are rejected before the structured claim-evidence evaluator runs against the same full chunk text used for answer generation. Missing per-line markers remain coverage diagnostics so one exact marker can support an adjacent coherent multiline list or project tree; the evaluator still rejects any factual item not supported by the cited chunks. Accepted IDs are rendered as compact numeric markers for the UI. If grounded evidence is absent or rejected, the graph continues in citation-free conversational mode instead of returning a fixed insufficient-evidence string.
