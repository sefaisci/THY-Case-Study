<p align="center">
  <img src="frontend/public/favicon-192x192.png" alt="Turkish Airlines emblem" width="112" />
</p>

<h1 align="center">Cabin Knowledge Assistant</h1>

<p align="center">
  <strong>A THY-branded, multi-user Agentic RAG proof of concept for private document intelligence.</strong>
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-14213D?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Pydantic_v2-009688?logo=fastapi&logoColor=white" />
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agentic_RAG-C8102E" />
  <img alt="React 19" src="https://img.shields.io/badge/React-19-14213D?logo=react&logoColor=61DAFB" />
  <img alt="PostgreSQL 16" src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" />
  <img alt="Docker Compose" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-C8102E" />
  <img alt="Project status" src="https://img.shields.io/badge/status-demo--ready-C8102E" />
</p>

---

## Overview

Cabin Knowledge Assistant turns private PDF, DOCX, and PPTX files into a user-isolated knowledge base. It combines a professional React application, an asynchronous typed FastAPI boundary, async SQLAlchemy persistence, an external Qdrant cluster, OpenAI Responses and Embeddings APIs, Docling, and an asynchronous modular LangGraph workflow.

The system supports two ingestion strategies, bounded concurrent multi-document processing, post-write Qdrant verification, adaptive one-or-two-query retrieval, collection-balanced hybrid search, structured OpenAI evidence reranking, one bounded grounded-answer repair, citation-safe hybrid/general fallback, strict document-specific no-answer behavior, sanitized rich source citations, polished KaTeX mathematics, session-scoped conversational memory, provider token accounting, versioned USD pricing, and retryable document deletion. React is the only frontend and runs as a focused three-panel document chat workspace.

> [!IMPORTANT]
> This repository is a proof of concept. Username-based identity demonstrates tenant scoping but is not authentication. Add enterprise identity, authorization, rate limiting, malware scanning, and durable background workers before production use.

## What the system delivers

| Capability | Implementation |
| --- | --- |
| Multi-format ingestion | Multiple PDF, DOCX, and PPTX uploads with backend validation and bounded asynchronous document/page processing |
| Two chunking paths | Independently validated full-page semantic chunks or Docling fixed token windows |
| Private retrieval | Mandatory backend-resolved `user_id` and completed-document filters on every Qdrant query |
| Agentic RAG | One verbatim query plus at most one standalone rewrite, LangGraph `Send` map branches, owner-scoped dense plus sparse Qdrant retrieval, balanced candidate reduction, structured reranking, grounding, citation validation, and reflection |
| Conversational continuity | Citation-free active-session history answers plus a deterministic, labeled general-model fallback for eligible general questions after successful empty or partial retrieval; document/private questions remain strict no-answer |
| Traceable answers | Inline citation previews on hover, focus, or tap plus expandable evidence cards with actual provenance and sanitized Markdown, safe HTML, code, tables, lists, and KaTeX rendering |
| Usage observability | Input, cached input, output, reasoning tokens, known USD cost, model snapshot, and pricing registry version |
| Safe deletion | Retryable PostgreSQL state, owner-filtered deletion from both Qdrant collections, then physical file removal |
| Async application stack | AsyncOpenAI, AsyncQdrantClient, SQLAlchemy AsyncSession, async FastAPI services, and non-blocking React request orchestration |
| Portable operation | Docker Compose for PostgreSQL, FastAPI, and React; external Qdrant remains environment-configured |

## Complete system architecture

<p align="center">
  <img src="docs/architecture/complete-system-architecture.svg" alt="Complete system architecture showing frontend, FastAPI, model orchestration, persistence, Qdrant, and OpenAI" width="100%" />
</p>

The browser never supplies an internal user identifier. FastAPI resolves the logical username, establishes the tenant boundary, and passes the internal UUID into owner-scoped repositories and Qdrant filters. Source files remain in a persistent volume, relational state remains in PostgreSQL, and document chunks remain in the configured external Qdrant instance.

The editable Graphviz source and rendered PNG are stored beside the SVG in [`docs/architecture`](docs/architecture).

## LangGraph Agentic RAG architecture

<p align="center">
  <img src="docs/architecture/langgraph-agentic-rag.png" alt="Expanded executable LangGraph Agentic RAG workflow with retrieval and answer subgraphs" width="100%" />
</p>

The editable [`langgraph-agentic-rag.mmd`](docs/architecture/langgraph-agentic-rag.mmd) source and rendered PNG are generated from the executable graph. The generator asserts the complete expanded node/edge topology so documentation drift fails visibly.

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
  -> one bounded grounded repair when needed
  -> grounded, hybrid, labeled general-model, or explicit no-answer response
```

In this implementation, **agent** means the parent `thy_agentic_rag_graph`; **subagent** means one of its two compiled, typed subgraphs. The remaining named components are executable LangGraph nodes or conditional routing functions, not hidden autonomous agents.

| Architecture level | Component | Responsibility |
| --- | --- | --- |
| Parent agent | `thy_agentic_rag_graph` | Owns the turn lifecycle, conditional routing, strict no-answer path, final typed response, and optional checkpoint integration. |
| Retrieval subagent | `retrieval_subgraph` | Plans collection scope, maps one verbatim query and at most one genuine standalone rewrite, balances cross-collection candidates, reranks evidence, and classifies successful/partial/fatal retrieval outcomes. |
| Answer subagent | `answer_subgraph` | Produces an evidence-only draft, deterministically validates exact chunk citations, performs structured claim-level grounding reflection, and permits one bounded evidence-only repair. |

### Complete node and router catalog

| Scope | Executable component | What it does |
| --- | --- | --- |
| Parent | `START` | LangGraph entry boundary for a typed `RagState`. |
| Parent | `query_understanding` | Normalizes the exact verbatim question, classifies intent, and requests at most one standalone rewrite only when bounded active-session history is required to resolve a reference. Duplicate or failed rewrites leave one variant. |
| Parent router | `route_after_query_understanding` | Sends only explicit active-session history questions to conversation generation, sends document questions with no completed accessible documents to `explicit_no_answer`, and otherwise enters retrieval. |
| Retrieval | `retrieval_planner` | Selects semantic chunks, Docling fixed chunks, or both and records branch prefetch, per-collection retention, candidate-union, rerank, and hybrid-weight settings in a typed `RetrievalPlan`. |
| Retrieval router | `dispatch_query_variants` | Emits one asynchronous LangGraph `Send` task for each genuine query variant: one task normally, at most two for a referential follow-up. |
| Retrieval | `retrieve_variant` | Embeds one variant and concurrently queries each selected Qdrant collection. Every dense and sparse prefetch has the same mandatory owner and completed-document filter. Qdrant performs weighted RRF within each collection. |
| Retrieval | `fuse_variant_results` | Deduplicates by `(collection_name, chunk_id)`, preserves both collection quotas, and applies deterministic weighted RRF only when two genuine variants exist. The reducer remains associative, commutative, and idempotent. |
| Retrieval | `reranking` | Sends the bounded authorized union to one OpenAI Responses structured-output relevance judgment. It accepts configured direct support above the local threshold and fails closed on error, refusal, timeout, malformed output, unknown ID, or insufficient evidence. |
| Retrieval | `retrieval_outcome_classification` | Distinguishes complete retrieval failure, partial collection degradation, successful empty retrieval, and safe fallback eligibility. Complete failure remains fatal. |
| Parent router | `route_after_retrieval` | Sends sufficient evidence into the answer subgraph, eligible successful-empty retrieval into isolated general generation, and document-specific or fatal paths into `explicit_no_answer`. |
| Answer | `answer_generation` | Gives the answer model only selected evidence and requires exact internal chunk-ID markers. Evidence insufficiency cannot reach this node. An answer-model provider failure is captured as a fatal typed node error; the backend returns a generic service/provider error instead of claiming that the documents lack evidence. Active-session history can clarify the question but is not document evidence. |
| Answer | `citation_validation` | Deterministically extracts exact opaque evidence markers and rejects missing valid citations, unknown or ambiguous IDs, cross-user IDs, and weak chunks. Only server-owned metadata for explicitly cited evidence becomes public citation data. |
| Answer | `claim_evidence_reflection` | Uses the structured grounding evaluator to check each claim against the cited full chunk text and independently classify full, partial, or absent question coverage. The evaluator uses independent low reasoning and retries one incomplete or transiently unparsable HTTP-200 result; terminal provider failures remain fatal. |
| Answer | `grounded_repair` | Receives only the rejected draft, sanitized evaluator feedback, and the same authorized evidence. It may repair once and then re-enters citation validation and reflection. |
| Parent router | `route_after_answer` | Returns accepted grounding directly for document-specific questions even when coverage is partial. Only fallback-eligible general-domain questions may receive an isolated general supplement; document/private and provider-failure paths remain fail-closed. |
| Parent | `conversation_generation` | Produces a citation-free response from bounded active-session history only for an explicit history intent. It is not a document-evidence fallback. |
| Parent | `general_knowledge_generation` | Receives only the question and bounded chat history. It receives no chunks, source metadata, citation IDs, or provider errors and cannot emit document citations. |
| Parent | `compose_hybrid_response` | Places the accepted cited document section first and appends a server-labeled, citation-free general-model section. |
| Parent | `explicit_no_answer` | Clears retrieved, generated, citation, and reflection artifacts before returning a grounded-mode document no-answer with no citations. |
| Parent | `final_response` | Builds grounded, hybrid, general-knowledge, conversational-history, or strict no-answer output. Fatal retrieval, reranking, answer, repair, grounding, and pure-general provider errors become one generic backend service error. |
| Parent | `END` | LangGraph terminal boundary after the typed response is available. |

### Grounding, ownership, and extension invariants

- **Ownership is enforced before generation.** The backend resolves the current user, passes only that user's completed document IDs into `RagSettings`, and the Qdrant adapter applies both `user_id` and allowed-document filters to every dense and sparse prefetch. Retrieved payloads are checked again before chunks can reach reranking or prompts.
- **Grounded generation is evidence-only.** Retrieved chunks are the only factual document context. Conversation history is role-preserving, session-scoped clarification context and is never treated or cited as source evidence.
- **Document/private insufficiency fails closed.** File-, upload-, page-, person-, and private-attribute questions without acceptable evidence return `no_answer=true` with `citations=[]`.
- **General fallback is explicit and citation-free.** After at least one selected collection completes successfully, an eligible general-domain question may use model knowledge. The application prepends `> **Genel model bilgisi:** Aşağıdaki bölüm yüklediğiniz belgelerde doğrulanmamıştır.` General claims never receive uploaded-document citations.
- **Provider outage is not evidence insufficiency.** Complete retrieval, payload schema, embedding, reranking, answer, repair, grounding, conversation, and pure-general generation failures are fatal at the FastAPI boundary. The grounding evaluator first absorbs one incomplete or transiently unparsable successful Responses call through a bounded retry; it never releases an unverified answer. Only a collection-isolated Qdrant failure may coexist with a safe response from another successful collection. A failed optional general supplement may degrade to an already accepted grounded answer.
- **Evidence markers are collision-safe.** Before the model sees candidates, the reranker maps server-owned `(collection_name, chunk_id)` identities to opaque, per-request IDs such as `c0001`. The answer and grounding stages use that mapping, so equal raw chunk IDs from different collections cannot alias one another. Provider output never supplies citation metadata.
- **Checkpoint partitions are opaque and user-scoped.** The runner derives `configurable.thread_id` as the lowercase SHA-256 digest of compact JSON containing `[user_id, caller_thread_id or "notebook-thread"]`. The same user/thread pair is stable, while changing either value creates a different checkpoint namespace. The raw caller thread ID remains only in request/state, and `configurable.user_id` remains the validated user ID. Backend chat does not currently enable `MemorySaver` or another LangGraph checkpointer; PostgreSQL remains the active session-memory source.
- **Citation validation, reflection, and repair are bounded first-class stages.** Deterministic chunk-ID and ownership checks run before the structured evaluator. A semantic rejection with valid cited evidence permits exactly one evidence-only repair; no unbounded loop exists.
- **Nodes are replaceable and extensible.** `RagNodeSet` binds graph nodes to protocol-based `RagAdapters`, so query rewriting, embeddings, retrieval, reranking, answer generation, and grounding evaluation can be replaced independently without changing the public runner or typed graph contract. New routing branches remain explicit in `graphs.py` and will force diagram regeneration checks to be updated.
- **Concurrency is bounded by graph semantics and stage limits.** LangGraph schedules one or two `Send` workers; each awaits one embedding and concurrently searches the selected collections. Typed reducers provide a deterministic fan-in before the capped reranker and context windows.

### Async design provenance

The implementation applies four LangGraph patterns that were analyzed from local reference exports during development without adding those HTML exports to the public repository:

- **Parallelization:** explicit fan-out/fan-in barriers and deterministic business ordering instead of relying on completion order.
- **Subgraphs:** typed retrieval and answer subgraphs with clear input/state/output contracts.
- **Map-reduce:** adaptive dynamic `Send` workers and an associative, commutative, idempotent result reducer.
- **Research assistant:** nested asynchronous orchestration that separates query generation, source retrieval, evidence synthesis, and final response construction.

These patterns keep provider I/O concurrent while preserving reproducible answers and owner-isolation invariants.

### What "OpenAI-style" means here

OpenAI does not publish ChatGPT's internal reranker implementation. This project therefore makes no ChatGPT-parity claim. It retains Qdrant and emulates the publicly documented File Search pattern: bounded query rewriting, keyword plus semantic retrieval, reranking, and a score threshold. OpenAI's hosted `ranker: auto` belongs to OpenAI Vector Store/File Search retrieval; it is not a public standalone ranker applied to this project's Qdrant candidates.

The implementation uses one OpenAI Responses structured-output call as a strict relevance judge over candidates that have already passed server-side Qdrant ownership and document filters. Its score is a model judgment and `RERANK_MIN_SCORE=0.50` is only a local starting value, not a calibrated probability. Any accuracy or recall change must be established with the versioned evaluation set; no gain is inferred from the architecture alone.

Official references:

- [OpenAI Retrieval: ranking options](https://developers.openai.com/api/docs/guides/retrieval#ranking)
- [OpenAI File Search: how it works](https://developers.openai.com/api/docs/assistants/tools/file-search#how-it-works)

## Repository layout

```text
.
├── backend/
│   ├── alembic/                 # PostgreSQL migrations
│   ├── app/                     # FastAPI routes, services, repositories, and schemas
│   └── Dockerfile
├── config/
│   ├── model-capabilities.v1.json
│   └── pricing/openai-pricing.v1.json
├── docs/architecture/           # Generated architecture images and editable sources
├── frontend/                    # React 19, Vite, TypeScript, and Nginx application
├── model/
│   ├── agentic_rag/             # LangGraph graph, nodes, adapters, runner, and contracts
│   ├── document_processing/     # PDF and Office rendering plus Docling processing
│   ├── ingestion/               # Connected ingestion coordinator
│   ├── semantic_chunking/       # Flat, page-independent semantic chunking
│   └── vector_store/            # Qdrant persistence, sparse vectors, and retrieval
├── notebook/
│   └── 08_langgraph_rag_flow_sandbox.ipynb  # Executed production-graph walkthrough
├── scripts/
│   └── generate_architecture_diagrams.py
├── .env.example                 # Safe public configuration template
├── alembic.ini
├── docker-compose.yml
└── requirements.txt
```

Local tests, source documents, scratch notebooks, generated page images, QA captures, private planning notes, credentials, dependency folders, and build output are intentionally outside the public Git surface. The executed `08_langgraph_rag_flow_sandbox.ipynb` architecture walkthrough is the intentional notebook exception.

## Core data flows

### Semantic ingestion

1. FastAPI validates the upload and stores its source file under internal user and document identifiers.
2. PDF pages render directly to images. DOCX and PPTX files convert through LibreOffice before every page or slide renders to an image.
3. Each full page or slide is analyzed independently with AsyncOpenAI and the selected Responses model/reasoning effort. Bounded page batches limit concurrent requests through `SEMANTIC_PAGE_MAX_CONCURRENCY`.
4. No previous-page summary, text, chunk, image, dictionary memory, continuation context, retrieval result, or document-level generated memory is added to the semantic prompt.
5. Strict Pydantic output classifies every location as `content` or `blank`. A content location must produce at least one flat, variable-length semantic chunk; only an explicitly blank location may produce zero chunks. Duplicate identifiers, nested chunks, and continuation fields are rejected.
6. The model is instructed to inspect the complete image in reading order and cover meaningful headings, prose, lists, tables, equations, code, charts, diagrams, captions, and visible labels without importing facts from another location.
7. The authoritative chunk text is embedded through the configured OpenAI embedding model and stored unchanged in the owner-scoped `semantic_chunks` collection. Semantic citation previews use a bounded copy of that same retrieved evidence text; the model-selected short verbatim excerpt remains metadata for audit rather than replacing the evidence shown to users.
8. Dense embeddings and Qdrant upserts are awaited in bounded batches controlled by `SEMANTIC_FLUSH_BATCH_SIZE`. Stable UUIDv5 point identities make retries idempotent.
9. Completion requires a Qdrant read-back of every expected stable point ID plus matching `user_id`, `document_id`, and `chunk_id` payload provenance. An optimistic upsert count is not treated as proof of persistence.
10. Any page-analysis, embedding, upsert, count, or read-back failure marks the job failed, keeps the document out of completed-document retrieval, and triggers owner/document-filtered cleanup of partial semantic points.
11. A retry first removes only the same backend-resolved `user_id` and `document_id` points from the target collection and fails closed if cleanup is unsuccessful.
12. Document-first row locks and a PostgreSQL partial unique index enforce one active ingestion job per document and exclude deletion while a job is pending or processing. Multiple selected documents run concurrently within `INGESTION_JOB_CONCURRENCY`; model-level discovery runs document batches bounded by `DOCUMENT_MAX_CONCURRENCY`.

### Docling fixed ingestion

Docling extracts supported documents through a thread-offloaded blocking boundary, applies the configured token window and overlap, asynchronously embeds the chunks, and writes only to `docling_fixed_chunks`. LibreOffice conversion uses a cancellable asynchronous subprocess. Semantic model and reasoning controls do not apply to this path.

### Chat and retrieval

1. A per-session async turn lock preserves message order within one chat while different chat sessions continue concurrently.
2. PostgreSQL loads only the selected session's bounded message history and releases its read transaction before provider I/O.
3. The graph preserves the normalized verbatim question and requests at most one standalone rewrite only for a referential follow-up that needs bounded session context.
4. LangGraph dispatches one or two `Send` workers. Each worker asynchronously embeds its query and concurrently searches the selected collections.
5. Every dense and sparse Qdrant prefetch applies the same mandatory backend-resolved owner and completed-document filters.
6. Qdrant performs weighted dense/sparse RRF per collection. The reducer deduplicates by collection plus chunk ID, preserves a quota for each selected collection, and applies a second weighted RRF only when two genuine query variants exist.
7. One structured OpenAI reranker call judges the bounded candidate union. Only direct support above the configured local threshold proceeds by default; invalid or insufficient results fail closed.
8. Grounded generation uses opaque, per-request evidence IDs. Deterministic validation maps accepted IDs back to server-owned source metadata before claim-evidence reflection.
9. Grounding runs with independent `GROUNDING_REASONING_EFFORT=low`, not the user's answer reasoning setting. One unusable structured result may be retried according to `GROUNDING_MAX_RETRIES`; an accepted grounded answer receives only its validated citations. A semantic grounding rejection with valid cited evidence receives one bounded repair attempt and then passes through the same validation gates.
10. Full grounded coverage returns a cited answer. Partial coverage can append a separately labeled, uncited general section only for fallback-eligible general-domain questions. A document-specific partial answer remains cited and grounded without a contradictory general supplement. Successful empty retrieval can produce the same labeled general-only response for eligible domain questions; document/private questions remain strict no-answer when evidence is absent.
11. Complete retrieval and provider outages become a generic backend service/provider error rather than a document-evidence claim. Citation-free session conversation remains limited to explicit active-session history questions.
12. Messages persist through async SQLAlchemy transactions; chat messages are never written to Qdrant.

## Persistence model

| Table | Responsibility |
| --- | --- |
| `users` | Unique normalized username and timestamps |
| `documents` | Owner, hash, source path, MIME type, ingestion configuration, collection, and lifecycle state |
| `ingestion_jobs` | Pending/processing/completed/failed status, counts, timing, failure details, usage, and known cost |
| `chat_sessions` | Owner, LangGraph-compatible session identifier, title, and activity timestamps |
| `chat_messages` | Session-scoped user and assistant content, citations, actual model, and reasoning effort |
| `usage_records` | Operation stage, provider, model, token categories, USD cost, pricing version, and pricing status |

## API surface

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | PostgreSQL, Qdrant, and OpenAI readiness |
| `POST` | `/api/v1/users/resolve` | Resolve or create a logical username |
| `GET` | `/api/v1/models` | Return runtime-accessible configured models |
| `GET` | `/api/v1/documents` | List the current user's documents |
| `POST` | `/api/v1/documents/upload` | Validate and store multiple source files |
| `DELETE` | `/api/v1/documents/{document_id}` | Run owner-scoped, retryable deletion |
| `POST` | `/api/v1/ingestion-jobs` | Start ingestion for uploaded documents |
| `POST` | `/api/v1/ingestion-jobs/status` | Poll multiple owner-scoped jobs in one request |
| `GET` | `/api/v1/ingestion-jobs/{job_id}` | Poll one owner-scoped job |
| `GET` | `/api/v1/chat/sessions` | List the current user's chats |
| `POST` | `/api/v1/chat/sessions` | Create a clean short-term memory context |
| `GET` | `/api/v1/chat/sessions/{session_id}/messages` | Load one session's messages |
| `POST` | `/api/v1/chat/sessions/{session_id}/messages` | Run chat with independent model settings |
| `GET` | `/api/v1/usage` | Return request, session, workspace, and stage totals |

Interactive OpenAPI documentation is available at `http://localhost:8000/docs` while the backend is running.

## Quick start with Docker Compose

### Prerequisites

- Docker Engine or Docker Desktop with Compose
- An OpenAI API key
- A reachable external Qdrant cluster and API key

### Start

```bash
git clone https://github.com/<account>/<repository>.git
cd <repository>
cp .env.example .env
```

Edit `.env` and replace the OpenAI and Qdrant placeholders. For any shared environment, also replace the local PostgreSQL password and keep the two database URLs consistent.

```bash
docker compose up --build
```

| Surface | URL |
| --- | --- |
| React application | `http://localhost:3000` |
| FastAPI | `http://localhost:8000` |
| OpenAPI | `http://localhost:8000/docs` |

Stop containers without deleting persistent data:

```bash
docker compose down
```

> [!CAUTION]
> `docker compose down --volumes` permanently removes the Compose-managed PostgreSQL and document-processing volumes.

## Local development on Linux and macOS

The repository pins Python 3.12 and Node.js 22 through `.python-version`, `.node-version`, and `.nvmrc`.

### System packages

Linux:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libreoffice-writer libreoffice-impress graphviz libgraphviz-dev fontconfig fonts-dejavu-core fonts-liberation2
```

macOS:

```bash
brew install graphviz
brew install --cask libreoffice
```

If LibreOffice is not on `PATH` on macOS, add this to `.env`:

```dotenv
SOFFICE_BINARY=/Applications/LibreOffice.app/Contents/MacOS/soffice
```

### Install application dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

```bash
cd frontend
nvm use
npm ci
cd ..
```

### Start PostgreSQL and migrate

Do not run local Uvicorn while the Compose backend still owns port `8000`. Preserve the database while switching modes with:

```bash
docker compose stop frontend backend
docker compose up -d postgres
docker compose exec -T postgres pg_isready -U thy_app -d thy_case_study
python -m alembic -c alembic.ini upgrade head
```

### Terminal 1 — FastAPI

```bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2 — React

```bash
cd frontend
nvm use
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api`, `/health`, and `/ready` to FastAPI.

## Environment configuration

`.env.example` is the only public environment template. `.env` and every other environment override are ignored.

The minimum connected configuration is:

```dotenv
OPENAI_API_KEY=replace-with-openai-api-key
QDRANT_URL=https://your-qdrant-cluster.example
QDRANT_API_KEY=replace-with-qdrant-api-key
POSTGRES_PASSWORD=local-dev-password
```

| Group | Important variables |
| --- | --- |
| Application | `APP_ENV`, `LOG_LEVEL`, `API_PORT`, `FRONTEND_PORT`, `CORS_ALLOWED_ORIGINS` |
| PostgreSQL | `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_URL`, `DATABASE_INTERNAL_URL` |
| File processing | `UPLOAD_DIR`, `PAGE_IMAGE_DIR`, `PROCESSING_DIR`, `MAX_UPLOAD_SIZE_MB`, `DOC_CONVERSION_DPI`, `SOFFICE_BINARY` |
| External Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, collection names, named vectors, and dense vector size |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, model cache duration, embedding model, and request timeout |
| Ingestion | `INGESTION_JOB_CONCURRENCY`, `DOCUMENT_MAX_CONCURRENCY`, `SEMANTIC_PAGE_MAX_CONCURRENCY`, semantic flush size, embedding batch size, fixed chunk size, and fixed overlap |
| Retrieval | Sparse provider/model/cache, stage-specific candidate limits, hybrid weights, reranker and grounding controls, evidence thresholds, and session history limit |
| Registries | `MODEL_CAPABILITIES_PATH`, `PRICING_REGISTRY_PATH` |

Qdrant is intentionally not created by Compose. `QDRANT_URL` must be reachable from the backend container. A cloud HTTPS endpoint works directly; a Qdrant process on the Docker host can use `http://host.docker.internal:<port>`.

### Retrieval stage limits and reranker controls

The committed values are initial operating limits, not benchmark-derived optima:

| Variable | Default | Applied boundary |
| --- | ---: | --- |
| `RETRIEVAL_PREFETCH_K` | `20` | Candidates requested from each dense and sparse branch, for each selected collection and genuine query variant |
| `RETRIEVAL_COLLECTION_K` | `15` | Results retained after Qdrant weighted RRF within each selected collection |
| `RERANK_CANDIDATE_K` | `30` | Collection-balanced, globally deduplicated candidate union |
| `RERANKER_MAX_CANDIDATES` | `30` | Independent maximum number of authorized candidates serialized into the OpenAI reranker request |
| `RERANK_TOP_K` | `6` | Maximum structured reranker results accepted for the evidence gate |
| `MAX_CONTEXT_CHUNKS` | `8` | Final answer-context ceiling; the current rerank limit makes the effective maximum six |
| `RERANKER_TEXT_MAX_CHARS` | `1600` | Per-candidate cap applied separately to source excerpt and chunk text before reranking |
| `GROUNDING_REASONING_EFFORT` | `low` | Independent structured evidence-judge effort; it is deliberately decoupled from the user's answer reasoning selection |
| `GROUNDING_MAX_RETRIES` | `1` | Retry count for incomplete or transiently unparsable structured grounding results; terminal failure remains fail-closed |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `120` | OpenAI request timeout, including the structured reranker call |

`RERANKER_MODEL` may be empty, in which case the runtime uses `SELF_SERVICE_LLM_MODEL`. `RERANKER_REASONING_EFFORT=low` explicitly requests low effort. Leaving it empty normalizes it to `None` and omits the `reasoning` parameter from the OpenAI request; `minimal`, `low`, `medium`, `high`, and `xhigh` are accepted configuration values, but the selected model must support the value. `RERANKER_ALLOW_PARTIAL_SUPPORT=false` keeps the accuracy-first default: only direct support at or above `RERANK_MIN_SCORE` can enter context.

### Safe BM25 activation and rollback

The public template deliberately keeps `SPARSE_ENCODER_PROVIDER=stable_hash`. Ingestion and retrieval share one provider, and payloads record the sparse encoder version. Activating `fastembed_bm25` changes the sparse-vector representation and requires a complete rebuild of both `semantic_chunks` and `docling_fixed_chunks`; legacy hash vectors and BM25 vectors must never be treated as one valid production index.

Retrieval actively enforces this boundary. The configured encoder's exact
version is added to the server-side Qdrant `must` filter used by both dense and
sparse prefetches, alongside owner and completed-document conditions. Returned
payloads are checked again and any mismatched version is dropped before fusion.
This prevents mixed-version points from entering answer context; it does not
convert, repair, or reindex existing points. Selecting a provider whose matching
index has not been completely built can therefore make otherwise owned evidence
invisible and produce a safe evidence-insufficiency result.

The following are explicit operator rollout steps. They were documented but were
not executed by this repository change:

1. Keep `SPARSE_ENCODER_PROVIDER=stable_hash`, run the versioned evaluation against the complete legacy-built index, and save the case-aligned prediction JSONL and metrics as the rollback baseline.
2. In an isolated non-production Qdrant environment, select `fastembed_bm25` and fully reingest one non-production user's documents across both `semantic_chunks` and `docling_fixed_chunks`; do not query a partially rebuilt collection.
3. Verify that every rebuilt point payload reports `sparse_encoder_version=fastembed-qdrant-bm25-v1`. Reject the candidate index if any legacy, missing, or unexpected version remains.
4. Run and retain Recall@5, Recall@10, Recall@20, MRR, reranker recall at the final context cutoff, no-answer precision/recall, unauthorized chunk count, and unknown citation count. Apply explicit acceptance criteria to the versioned results; do not infer a gain from the provider change alone.
5. After non-production acceptance, use an isolated-index or maintenance-window cutover to reingest every completed document assigned to each collection, covering both `semantic_chunks` and `docling_fixed_chunks`, with `fastembed_bm25`.
6. Enable `SPARSE_ENCODER_PROVIDER=fastembed_bm25` globally only after the complete rebuilt index and evaluation results are accepted. Keep `SPARSE_ENCODER_MODEL=Qdrant/bm25` identical in ingestion and retrieval.
7. Roll back to `stable_hash` only while restoring or selecting an index built entirely with the legacy encoder. Never query a mixed legacy/BM25 index, and never switch the provider without switching to its matching complete index.

Before cutover, retain a recoverable snapshot of both legacy collections. This repository update does not contact Qdrant, reingest documents, inspect connected payloads, or mutate connected data.

### Concurrency controls

The defaults favor predictable local-resource use and can be tuned independently:

| Variable | Default | Boundary |
| --- | ---: | --- |
| `INGESTION_JOB_CONCURRENCY` | `4` | Maximum backend ingestion jobs executing simultaneously in one process; at the default, four selected files can run concurrently |
| `DOCUMENT_MAX_CONCURRENCY` | `2` | Maximum discovered documents processed concurrently inside a model ingestion run |
| `SEMANTIC_PAGE_MAX_CONCURRENCY` | `3` | Maximum page-image analysis requests in flight for one semantic document |
| `SEMANTIC_FLUSH_BATCH_SIZE` | `32` | Maximum chunks embedded and upserted in one awaited semantic vector batch |

These are bounded application-level limits, not global distributed quotas. When multiple backend replicas are deployed, use a durable queue and shared rate-limit coordination.

## Model access and pricing

`config/model-capabilities.v1.json` contains configured chat and semantic model candidates plus supported reasoning efforts. FastAPI intersects that list with the models visible to the configured OpenAI project. Unavailable entries remain available through the catalog API for diagnostics but are excluded from selectors; the end-user inspector does not show a noisy preview-model warning card. A provider failure is never silently substituted with another model.

Semantic chunking, answer generation, reranking, and grounding use explicit independent reasoning settings. A user-selected high answer effort therefore cannot consume the structured grounding judge's bounded output budget.

`config/pricing/openai-pricing.v1.json` is the versioned pricing registry. Every calculated usage record retains the actual provider model and pricing version. Unknown prices remain explicitly unpriced; the application never invents token counts or cost.

## Regenerate the architecture images

The committed images are ready for GitHub display. To regenerate them after changing the graph or system design, install the optional local documentation dependency:

```bash
source .venv/bin/activate
python -m pip install -r requirements-architecture.txt
python scripts/generate_architecture_diagrams.py
```

Linux requires the compiler toolchain, `pkg-config`, and `libgraphviz-dev` shown above; macOS uses the Graphviz installation shown above. Generated assets:

- `docs/architecture/langgraph-agentic-rag.mmd`
- `docs/architecture/langgraph-agentic-rag.png`
- `docs/architecture/complete-system-architecture.dot`
- `docs/architecture/complete-system-architecture.svg`
- `docs/architecture/complete-system-architecture.png`

## Executable LangGraph architecture notebook

[`notebook/08_langgraph_rag_flow_sandbox.ipynb`](notebook/08_langgraph_rag_flow_sandbox.ipynb) is the reader-facing, executed walkthrough of the production graph. It imports the real root/retrieval/answer builders, renders all three graphs locally, inventories node/provider/failure boundaries, classifies every `RagState` field, verifies twelve routing invariants, and demonstrates optional checkpointing. Default execution uses fake adapters and makes no OpenAI or Qdrant call.

Execute it from a clean project kernel:

```bash
.venv/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  notebook/08_langgraph_rag_flow_sandbox.ipynb \
  --ExecutePreprocessor.timeout=180
```

The optional connected cell remains disabled unless `RUN_CONNECTED_RAG_NOTEBOOK=1` and the isolated `NOTEBOOK_USER_ID`, `NOTEBOOK_ALLOWED_DOCUMENT_IDS`, and `NOTEBOOK_QUESTION` values are supplied. A connected smoke result proves integration, not retrieval accuracy or recall.

## Public repository boundary

The ignore rules are deliberately conservative:

- no `.env`, API key, credential override, or editor secret;
- no test source code;
- no uploaded sample or personal document;
- no unrelated notebook output or credential-bearing experiment; the sanitized,
  offline-executed LangGraph architecture notebook is the intentional exception;
- no generated page image, processing artifact, database, coverage, dependency folder, or compiled frontend;
- no internal QA screenshot, delivery plan, or agent workspace metadata.

Before the first push, inspect the exact staged surface:

```bash
git init
git add .
git status --short
git diff --cached --check
git grep --cached -l -E 'BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|sk-[A-Za-z0-9_-]{20,}' || true
```

The last command reports filenames only. Investigate any result and remove the staged secret before committing.

Then publish to the repository you control:

```bash
git commit -m "Publish THY-branded Agentic RAG proof of concept"
git branch -M main
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

No GitHub remote is configured or pushed automatically by this project.

## Operational validation

The following checks do not require connected OpenAI or Qdrant requests:

```bash
python -m compileall -q backend model tests scripts
docker compose config --quiet
```

The offline evaluator accepts strict, versioned case and prediction JSONL files:

```bash
python -m model.evaluation.runner \
  --cases model/evaluation/fixtures/rag_eval.v1.jsonl \
  --predictions model/evaluation/fixtures/rag_eval_predictions.v1.jsonl \
  --ks 5 10 20 \
  --reranker-cutoff 6
```

It reports macro Recall@K over answerable cases with labeled relevant chunks, per-case reciprocal rank and MRR, reranker recall at the explicit final-context cutoff, no-answer precision/recall with support counts, and unauthorized/unknown-citation event counts. Leakage counts are unique `(case_id, chunk_id)` events: duplicates inside one case count once, while the same ID in another case counts again. A citation is unknown when it is absent from that case's reranked context at `--reranker-cutoff`. No-answer precision or recall is JSON `null` when its denominator is zero; `0.0` is reserved for a defined metric whose numerator is zero. The evaluator does not infer answer correctness or citation entailment from retrieval labels.

The synthetic fixture is a runner smoke test, not evidence of production answer quality. Measure candidate changes against a representative, human-labeled local set before describing them as accuracy or recall improvements.

> [!NOTE]
> The July 13, 2026 verification run passed all 319 Python tests outside the filesystem sandbox. The sandboxed runner can stall its `aiosqlite` worker during fixture setup, so SQLite-backed backend integration tests should run in a normal local/container process rather than treating that sandbox limitation as an application failure.

```bash
cd frontend
npm test
npm run lint
npm run build
```

Connected ingestion and chat operations can incur provider charges.

## Proof-of-concept limitations

- Username identity is not password authentication, SSO, or an authorization protocol.
- FastAPI schedules ingestion as awaited async background coroutines, but in-process background tasks are not a durable distributed queue and do not survive a worker restart.
- Chat execution is asynchronous internally and browser timeouts abort fetches, but HTTP responses are non-streaming; durable request idempotency and provider-side cancellation after a client disconnect are not yet guaranteed.
- PostgreSQL reconstructs session context instead of using a distributed LangGraph checkpointer.
- Concurrency limits are process-local; multiple backend replicas require shared queue and rate-limit coordination.
- Existing Qdrant points are not automatically migrated when a chunk schema changes; reingest those documents.
- Rendering artifacts require a production retention and cleanup policy.
- DOCX pagination can vary when host fonts differ; the backend image installs repeatable DejaVu and Liberation fonts.
- Clickable source previews, malware scanning, audit export, OCR prewarming, and production rate limiting remain outside this POC.

## Brand and licensing notice

This is an independent technical proof of concept and is not an official Turkish Airlines product. Turkish Airlines and THY names, marks, and logo remain the property of their respective owner. The original software in this repository is released under the [MIT License](LICENSE); that license does not grant rights to third-party names, marks, logos, or other brand assets.
