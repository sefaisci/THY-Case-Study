# THY Document Intelligence React Frontend

This directory contains the production-style React 19 interface for the THY Agentic RAG API. It uses Vite, TypeScript, Tailwind CSS, TanStack Query, Zustand, React Markdown, React Dropzone, and Lucide icons.

## Prerequisites

- Node.js 22.12 or newer (`.nvmrc` is included)
- npm 10.9 or newer
- FastAPI running on `http://127.0.0.1:8000`

The same commands work on Linux and macOS.

## Local development

```bash
cd frontend/react
nvm use
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api`, `/health`, and `/ready` to the local
FastAPI process, so browser requests remain same-origin during development.

Use `VITE_API_BASE_URL` only when the API must be addressed directly:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1 npm run dev
```

If direct cross-origin requests are used, the FastAPI `CORS_ALLOWED_ORIGINS` setting must include
the React origin.

## Verification

```bash
npm run lint
npm run build
```

## Runtime architecture

- The default API base is same-origin `/api/v1`.
- All owner-scoped requests send `X-Username` and a generated `X-Request-ID`.
- The backend-resolved username is the only logical identity stored by the frontend.
- TanStack Query keys include the normalized username for cache isolation.
- Semantic chunking and chat model selections are independent.
- Active ingestion job identifiers are versioned and persisted in `sessionStorage` so polling can
  resume after a soft reload.
- Static Docker delivery uses Nginx and proxies API calls to the Compose service named `backend`.

## Interface architecture

- `src/features/navigation`: username resolution and chat-session navigation.
- `src/features/chat`: responsive chat workspace, grounded answer rendering, and the composer.
- `src/features/citations`: compact source summaries with expandable excerpts.
- `src/features/usage`: lazy stage-level request, session, and workspace usage.
- `src/features/documents`: pending uploads, independent model controls, ingestion jobs, document lifecycle, and deletion.
- `src/features/workspace`: application orchestration and user-switch isolation.
- `src/api`: typed FastAPI contracts and request helpers.
- `src/state`: versioned session-only workspace preferences and active jobs.

Desktop uses a stable three-region application shell. Conversation and document panels become accessible modal drawers at narrower viewports; hidden drawers are removed from the accessibility tree.
