# Agentic RAG Model Package

This package contains the connected LangGraph document-QA workflow for the THY self-service chatbot. Reusable ingestion code lives under `model/ingestion`, `model/document_processing`, `model/semantic_chunking`, and `model/vector_store`.

## Multi-Format Ingestion

One request can supply PDF, DOCX, and PPTX paths together. Each path may be `None`, one matching file, or a directory. Select exactly one ingestion method per run:

```python
from pathlib import Path

from model.ingestion import (
    IngestionRequest,
    IngestionSettings,
    create_connected_ingestion_coordinator,
)

settings = IngestionSettings.from_env(".env", project_root=Path.cwd())
coordinator = await create_connected_ingestion_coordinator(settings)

try:
    result = await coordinator.run(
        IngestionRequest(
            method="semantic",  # or "docling"
            user_id="local-demo-user",
            pdf_path=Path("files/pdf"),
            docx_path=Path("files/docx"),
            pptx_path=Path("files/pptx"),
        )
    )
finally:
    await coordinator.aclose()
```

`semantic` renders every page or slide to an image and analyzes each location independently. It supplies no previous-page memory and returns a flat list of variable-length chunks; the authoritative chunk text is embedded exactly as stored in `semantic_chunks`. `docling` parses the original source, creates page-scoped fixed token windows, and writes them to `docling_fixed_chunks`. Both paths use the configured dense embedder and the same configured sparse encoder as retrieval.

The safe migration default is `SPARSE_ENCODER_PROVIDER=stable_hash`. `fastembed_bm25` uses `Qdrant/bm25` with distinct passage and query embedding paths. Do not activate it against legacy sparse vectors: completely reingest both `semantic_chunks` and `docling_fixed_chunks`, then change ingestion and retrieval together. Rollback requires restoring both legacy collection snapshots together with `stable_hash`; changing only the provider creates an invalid mixed representation.

At query time, the active encoder version is part of the Qdrant server-side
`must` filter shared by dense and sparse prefetches. The adapter also requires a
nonblank `sparse_encoder_version` payload and defensively drops any returned
chunk whose version differs from the active encoder. This is the runtime
mixed-index enforcement layer, not a migration mechanism: it never rewrites or
reindexes a point, and an incomplete matching index can make evidence invisible.

### Seven-step operator rollout

These are operator actions for a later connected rollout; none was executed by
this documentation and code-verification task:

1. Keep `SPARSE_ENCODER_PROVIDER=stable_hash`, evaluate the complete legacy-built index, and save the versioned, case-aligned prediction JSONL and metrics as the baseline.
2. In an isolated non-production Qdrant environment, select `fastembed_bm25` and fully reingest one non-production user's documents across both `semantic_chunks` and `docling_fixed_chunks`. Do not serve retrieval from a partially rebuilt collection.
3. Inspect every rebuilt point payload and require `sparse_encoder_version=fastembed-qdrant-bm25-v1`; any missing, legacy, or unexpected version rejects the candidate index.
4. Run and retain Recall@5, Recall@10, Recall@20, MRR, reranker recall at the final context cutoff, no-answer precision/recall, unauthorized chunk count, and unknown citation count. Compare the versioned result with explicit acceptance criteria rather than assuming an improvement.
5. After acceptance, use an isolated-index or maintenance-window cutover to reingest every completed document assigned to each collection, covering both collections, with `fastembed_bm25`.
6. Enable `SPARSE_ENCODER_PROVIDER=fastembed_bm25` globally only after the complete rebuilt index and evaluation results are accepted; ingestion and retrieval must both keep `SPARSE_ENCODER_MODEL=Qdrant/bm25`.
7. Roll back to `stable_hash` only against an index built entirely with the legacy encoder. Restore or select that complete legacy index atomically, and never query an index containing a mixture of legacy and BM25 sparse vectors.

## Implemented Graph

```text
normalized verbatim query
  -> optional standalone rewrite for a referential follow-up
  -> parallel owner-scoped dense + BM25 retrieval per selected collection
  -> collection-balanced candidate union
  -> optional two-query weighted RRF
  -> OpenAI structured evidence reranker
  -> retrieval outcome and fallback eligibility classification
  -> grounded generation
  -> exact citation validation
  -> claim-evidence reflection with question coverage
  -> at most one grounded repair
  -> grounded, hybrid, labeled general-model, or explicit document no-answer
```

In graph terms:

```text
START
-> query_understanding
   -> conversation_generation only for explicit active-session history intent
   -> explicit_no_answer when no completed accessible documents exist
   -> retrieval_subgraph otherwise
      -> retrieval_planner
      -> Send(retrieve_variant) x1 or x2
      -> fuse_variant_results
      -> reranking
      -> retrieval_outcome_classification
-> general_knowledge_generation when retrieval succeeds empty and policy permits
-> explicit_no_answer when the question is document/private-specific
-> fatal service error when retrieval or a provider stage fails
-> answer_subgraph only for sufficient validated evidence
   -> answer_generation
   -> citation_validation
   -> claim_evidence_reflection
      -> grounded_repair once after a semantic rejection with valid citations
      -> citation_validation
      -> claim_evidence_reflection
-> compose_hybrid_response after accepted partial coverage plus a general supplement
-> final_response
-> END
```

Normal document questions produce one exact normalized verbatim variant. A bounded, role-preserving history payload is sent to the query rewriter only when the current question contains an explicit reference that needs resolution; it may add one distinct standalone variant. Rewrite failure or duplication leaves the verbatim variant unchanged. There is no translation, keyword, source-style, or synthetic fan-out in this package.

Each query variant searches the selected Qdrant collections concurrently. Dense and sparse prefetches use the identical `user_id` and completed-document filter. Qdrant performs weighted RRF within each collection. Results are deduplicated by `(collection_name, chunk_id)` and balanced across collections before optional application-level weighted RRF for a genuine second query.

When a LangGraph checkpointer is enabled, the runner does not use the caller's
raw thread ID as the checkpoint partition. It hashes compact JSON containing
`[user_id, thread_id or "notebook-thread"]` with SHA-256 and supplies the lowercase 64-character
digest as `configurable.thread_id`. The same user/thread pair is stable, while a
different user or thread creates a separate opaque namespace. The raw thread ID
stays in request/state and `configurable.user_id` remains the validated user ID.
Backend chat currently does not enable `MemorySaver` or another graph
checkpointer; active session history is reconstructed from PostgreSQL.

## OpenAI Structured Reranker

OpenAI does not publish ChatGPT's internal reranker implementation. This package does not reproduce or claim parity with it. Instead, it preserves Qdrant and emulates OpenAI's publicly documented query-rewrite, keyword-plus-semantic search, rerank, and threshold pattern. Hosted `ranker: auto` is an OpenAI Vector Store/File Search capability; it is not called against arbitrary Qdrant candidates.

Official documentation:

- https://developers.openai.com/api/docs/guides/retrieval#ranking
- https://developers.openai.com/api/docs/assistants/tools/file-search#how-it-works

The adapter makes at most one Responses structured-output reranker request per document question. It serializes no more than `RERANKER_MAX_CANDIDATES`, truncates each candidate excerpt and text independently to `RERANKER_TEXT_MAX_CHARS`, and records usage under `retrieval_reranking`. `RERANKER_MODEL` falls back to `SELF_SERVICE_LLM_MODEL`. Leaving `RERANKER_REASONING_EFFORT` empty omits the OpenAI `reasoning` parameter; otherwise the normalized configured effort is sent and must be supported by the selected model.

Candidates are already authorized before reranking. The server maps every unique `(collection_name, chunk_id)` identity to an opaque, per-request evidence ID such as `c0001`. The model ranks only those aliases; source metadata always comes from the retained server-owned chunk. Unknown, duplicate, excessive, malformed, cross-user, or disallowed-document output fails closed. By default, only `direct` support at or above `RERANK_MIN_SCORE` enters context; `partial` support remains disabled.

`RERANK_MIN_SCORE=0.50` is a starting local judgment threshold, not a probability. The architecture alone does not establish an answer-accuracy or retrieval-recall improvement; calibrate limits and thresholds with representative labeled evaluation data.

## Stage Limits

| Setting | Default | Boundary |
| --- | ---: | --- |
| `RETRIEVAL_PREFETCH_K` | `20` | Each dense/sparse Qdrant branch per collection and query variant |
| `RETRIEVAL_COLLECTION_K` | `15` | Each collection after Qdrant RRF |
| `RERANK_CANDIDATE_K` | `30` | Balanced union before reranking |
| `RERANKER_MAX_CANDIDATES` | `30` | OpenAI request input |
| `RERANK_TOP_K` | `6` | Accepted structured reranker output |
| `MAX_CONTEXT_CHUNKS` | `8` | Answer prompt ceiling; currently bounded to six by `RERANK_TOP_K` |
| `RERANKER_TEXT_MAX_CHARS` | `1600` | Each candidate excerpt and chunk text |

## Notebook Usage

The canonical architecture walkthrough is
`notebook/08_langgraph_rag_flow_sandbox.ipynb`. Its default path imports the
production graph builders, renders the root/retrieval/answer graphs, validates
the complete `RagState` field inventory, and runs deterministic routing checks
without provider calls. The connected cell is explicitly gated and requires an
isolated owner plus completed document IDs.

```python
from dataclasses import replace

from model.agentic_rag import (
    RagSettings,
    create_openai_qdrant_adapters,
    run_rag_question,
)

settings = replace(
    RagSettings.from_env(".env"),
    allowed_document_ids=("completed-document-id",),
)
adapters = create_openai_qdrant_adapters(settings)
try:
    response = await run_rag_question(
        "What three parts of differential equations study are listed on page 1?",
        user_id="local-demo-user",
        adapters=adapters,
        settings=settings,
    )
finally:
    await adapters.aclose()
response.model_dump()
```

Fake adapters and explicit `noop` or `heuristic` rerankers remain available for deterministic offline development. They are not presented as production evidence ranking. Connected execution must explicitly pass `create_openai_qdrant_adapters(settings)`.

## Security and No-Answer Invariants

- Backend chat resolves the user and restricts retrieval to that user's document IDs currently marked `completed` in PostgreSQL.
- Every dense and sparse Qdrant prefetch repeats the owner/document filter; payload ownership is checked again before reranking and after model output.
- Retrieved text and source fields are labeled untrusted data. They cannot change reranker or answer instructions.
- Grounded answers require sufficient validated reranker evidence, at least one exact accepted evidence marker, and accepted claim-evidence reflection.
- A semantic grounding rejection with valid cited evidence may trigger exactly one evidence-only repair; the repaired draft passes through the same citation and grounding gates.
- General-model fallback requires at least one successfully completed selected collection and a conservative policy decision that the question is general domain knowledge. Accepted partial evidence for a document-specific question returns only the cited grounded portion; it never appends a general section that can contradict document availability. Document-, page-, upload-, person-, project-, assessment-content, and private-attribute questions remain strict no-answer without evidence.
- General generation receives only the current question and bounded role-preserving chat history. It receives no chunks, filenames, excerpts, citation IDs, ACL metadata, or provider errors.
- The application prepends `> **Genel model bilgisi:** Aşağıdaki bölüm yüklediğiniz belgelerde doğrulanmamıştır.` General text is stripped of document citation markers before release.
- Complete retrieval, payload schema, embedding, reranking, answer, grounded-repair, grounding, conversation, and pure-general generation failures are fatal. Before declaring grounding terminal, the structured evaluator uses independent low reasoning and one bounded retry for an incomplete or transiently unparsable successful Responses call. The backend raises the generic `rag_provider_failure` error and rolls back the turn only if no validated grounding result is recovered. Only a collection-isolated Qdrant failure may coexist with a safe answer from another successful collection.
- Citation-free conversation is available only for an explicit question about prior messages in the active session. It is never a recovery path for failed document evidence.

This change requires no PostgreSQL migration, Qdrant collection recreation, or document re-ingestion.

See [`../no-answer-policy.md`](../no-answer-policy.md) for the complete decision table and the root README for the offline evaluation CLI and safe activation procedure.
