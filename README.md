<p align="center">
  <img src="frontend/react/public/thy-logo.png" alt="Turkish Airlines emblem" width="92" />
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
  <img alt="Project status" src="https://img.shields.io/badge/status-demo--ready-C8102E" />
</p>

---

## Overview

Cabin Knowledge Assistant turns private PDF, DOCX, and PPTX files into a user-isolated knowledge base. It combines a professional React application, a typed FastAPI boundary, PostgreSQL metadata and chat persistence, an external Qdrant cluster, OpenAI Responses and Embeddings APIs, Docling, and a modular LangGraph workflow.

The system supports two ingestion strategies, deterministic cross-collection retrieval, actual source citations, session-scoped conversational memory, provider token accounting, versioned USD pricing, and retryable document deletion. React is the primary interface; the earlier Streamlit application remains available through an optional Compose profile.

> [!IMPORTANT]
> This repository is a proof of concept. Username-based identity demonstrates tenant scoping but is not authentication. Add enterprise identity, authorization, rate limiting, malware scanning, and durable background workers before production use.

## What the system delivers

| Capability | Implementation |
| --- | --- |
| Multi-format ingestion | Multiple PDF, DOCX, and PPTX uploads with backend extension, MIME, size, and SHA-256 duplicate validation |
| Two chunking paths | Variable-length semantic page chunks or Docling fixed token windows |
| Private retrieval | Mandatory backend-resolved `user_id` and completed-document filters on every Qdrant query |
| Agentic RAG | Query understanding, planning, hybrid retrieval, reranking, grounded answer generation, citation validation, and reflection |
| Conversational continuity | Citation-free fallback using only the active PostgreSQL chat session when useful document evidence is unavailable |
| Traceable answers | Actual filename, location, excerpt, score, ingestion method, collection, and chunk metadata |
| Usage observability | Input, cached input, output, reasoning tokens, known USD cost, model snapshot, and pricing registry version |
| Safe deletion | Retryable PostgreSQL state, owner-filtered deletion from both Qdrant collections, then physical file removal |
| Portable operation | Docker Compose for PostgreSQL, FastAPI, and React; external Qdrant remains environment-configured |

## Complete system architecture

<p align="center">
  <img src="docs/architecture/complete-system-architecture.svg" alt="Complete system architecture showing frontend, FastAPI, model orchestration, persistence, Qdrant, and OpenAI" width="100%" />
</p>

The browser never supplies an internal user identifier. FastAPI resolves the logical username, establishes the tenant boundary, and passes the internal UUID into owner-scoped repositories and Qdrant filters. Source files remain in a persistent volume, relational state remains in PostgreSQL, and document chunks remain in the configured external Qdrant instance.

The editable Graphviz source and rendered PNG are stored beside the SVG in [`docs/architecture`](docs/architecture).

## LangGraph Agentic RAG architecture

<p align="center">
  <img src="docs/architecture/langgraph-agentic-rag.png" alt="Expanded LangGraph Agentic RAG workflow" width="100%" />
</p>

This image is generated from the maintained compiled graph, not from a manually duplicated flowchart. The generator calls `build_rag_graph()`, expands nested subgraphs with `get_graph(xray=True)`, saves LangGraph's Mermaid representation, and uses LangGraph's local PNG drawing path to save the branded image. No project architecture is sent to an external diagram-rendering service.

The graph has two terminal response paths:

1. Relevant evidence moves through grounded answer generation, exact chunk-ID citation validation, and claim-evidence reflection.
2. Conversational requests, empty retrieval, below-threshold evidence, or rejected grounding move through a citation-free fallback that sees only the active session history.

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
├── frontend/
│   ├── react/                   # Primary React 19 application and Nginx image
│   └── streamlit/               # Optional compatibility interface
├── model/
│   ├── agentic_rag/             # LangGraph graph, nodes, adapters, runner, and contracts
│   ├── document_processing/     # PDF and Office rendering plus Docling processing
│   ├── ingestion/               # Connected ingestion coordinator
│   ├── semantic_chunking/       # Flat, page-independent semantic chunking
│   └── vector_store/            # Qdrant persistence, sparse vectors, and retrieval
├── scripts/
│   └── generate_architecture_diagrams.py
├── .env.example                 # Safe public configuration template
├── alembic.ini
├── docker-compose.yml
└── requirements.txt
```

Local tests, source documents, notebooks, generated page images, QA captures, private planning notes, credentials, dependency folders, and build output are intentionally outside the public Git surface.

## Core data flows

### Semantic ingestion

1. FastAPI validates the upload and stores its source file under internal user and document identifiers.
2. PDF pages render directly to images. DOCX and PPTX files convert through LibreOffice before every page or slide renders to an image.
3. Each page is analyzed independently with the selected OpenAI Responses model and reasoning effort.
4. No previous-page summary, dictionary memory, continuation context, or earlier page image is added to the semantic prompt.
5. Strict Pydantic output produces a flat list of variable-length semantic chunks; recursive and nested chunk structures are rejected.
6. The authoritative chunk text is embedded and stored unchanged in `semantic_chunks`.
7. Embeddings and Qdrant upserts use bounded batches controlled by `SEMANTIC_FLUSH_BATCH_SIZE`.
8. A retry first removes only the same backend-resolved `user_id` and `document_id` points from the target collection and fails closed if cleanup is unsuccessful.

### Docling fixed ingestion

Docling extracts supported documents, applies the configured token window and overlap, embeds the chunks, and writes only to `docling_fixed_chunks`. Semantic model and reasoning controls do not apply to this path.

### Chat and retrieval

1. PostgreSQL loads only the selected session's bounded message history.
2. The graph normalizes the question and rewrites a retrieval query when appropriate.
3. Retrieval targets the selected collection scope and applies mandatory owner and completed-document filters.
4. Dense and sparse candidates are merged deterministically, then reranked.
5. Grounded answers receive citations only for chunks actually cited by exact chunk ID.
6. If useful evidence is unavailable, the same chat remains usable through normal conversational generation without fabricated citations.
7. Messages persist in PostgreSQL; chat messages are never written to Qdrant.

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

Start the optional Streamlit interface as well:

```bash
docker compose --profile streamlit up --build
```

Streamlit is then available at `http://localhost:8501`.

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
sudo apt-get install -y libreoffice-writer libreoffice-impress graphviz fontconfig fonts-dejavu-core fonts-liberation2
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
cd frontend/react
nvm use
npm ci
cd ../..
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
cd frontend/react
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
| Application | `APP_ENV`, `LOG_LEVEL`, `API_PORT`, `FRONTEND_PORT`, `STREAMLIT_PORT`, `CORS_ALLOWED_ORIGINS` |
| PostgreSQL | `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DATABASE_URL`, `DATABASE_INTERNAL_URL` |
| File processing | `UPLOAD_DIR`, `PAGE_IMAGE_DIR`, `PROCESSING_DIR`, `MAX_UPLOAD_SIZE_MB`, `DOC_CONVERSION_DPI`, `SOFFICE_BINARY` |
| External Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, collection names, named vectors, and dense vector size |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, model cache duration, embedding model, and request timeout |
| Ingestion | Semantic batch size, embedding batch size, fixed chunk size, and fixed overlap |
| Retrieval | Top-k values, hybrid weights, reranker, score thresholds, and session history limit |
| Registries | `MODEL_CAPABILITIES_PATH`, `PRICING_REGISTRY_PATH` |

Qdrant is intentionally not created by Compose. `QDRANT_URL` must be reachable from the backend container. A cloud HTTPS endpoint works directly; a Qdrant process on the Docker host can use `http://host.docker.internal:<port>`.

## Model access and pricing

`config/model-capabilities.v1.json` contains configured chat and semantic model candidates plus supported reasoning efforts. FastAPI intersects that list with the models visible to the configured OpenAI project. An unavailable model is displayed as unavailable and cannot be silently selected or substituted.

Semantic chunking and answer generation use independent model and reasoning settings.

`config/pricing/openai-pricing.v1.json` is the versioned pricing registry. Every calculated usage record retains the actual provider model and pricing version. Unknown prices remain explicitly unpriced; the application never invents token counts or cost.

## Regenerate the architecture images

The committed images are ready for GitHub display. To regenerate them after changing the graph or system design, install the optional local documentation dependency:

```bash
source .venv/bin/activate
python -m pip install -r requirements-architecture.txt
python scripts/generate_architecture_diagrams.py
```

Linux may require `libgraphviz-dev`; macOS uses the Graphviz installation shown above. Generated assets:

- `docs/architecture/langgraph-agentic-rag.mmd`
- `docs/architecture/langgraph-agentic-rag.png`
- `docs/architecture/complete-system-architecture.dot`
- `docs/architecture/complete-system-architecture.svg`
- `docs/architecture/complete-system-architecture.png`

## Public repository boundary

The ignore rules are deliberately conservative:

- no `.env`, API key, credential override, or editor secret;
- no test source code;
- no uploaded sample or personal document;
- no notebook output or experimental credential check;
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
python -m compileall -q backend model scripts
docker compose config --quiet
docker compose --profile streamlit config --quiet
```

```bash
cd frontend/react
npm run lint
npm run build
```

Connected ingestion and chat operations can incur provider charges.

## Proof-of-concept limitations

- Username identity is not password authentication, SSO, or an authorization protocol.
- FastAPI background tasks are not a durable distributed ingestion queue.
- Chat responses are synchronous; token streaming and cancellation are not implemented.
- PostgreSQL reconstructs session context instead of using a distributed LangGraph checkpointer.
- Existing Qdrant points are not automatically migrated when a chunk schema changes; reingest those documents.
- Rendering artifacts require a production retention and cleanup policy.
- DOCX pagination can vary when host fonts differ; the backend image installs repeatable DejaVu and Liberation fonts.
- Clickable source previews, malware scanning, audit export, OCR prewarming, and production rate limiting remain outside this POC.

## Brand and licensing notice

This is an independent technical proof of concept and is not an official Turkish Airlines product. Turkish Airlines and THY names, marks, and logo remain the property of their respective owner. No open-source license is included; select and add an appropriate license before granting reuse or contribution rights.
