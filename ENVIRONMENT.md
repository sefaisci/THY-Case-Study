# Environment and Portability Guide

`README.md` is the canonical startup guide. This document summarizes cross-platform runtime requirements and common environment concerns.

## Supported toolchain

- Python 3.12 (`.python-version`)
- Node.js 22 (`.node-version` and `.nvmrc`)
- npm 10.9 or newer
- PostgreSQL 16 through Docker Compose
- Docker Engine or Docker Desktop with Compose
- LibreOffice for DOCX and PPTX rendering
- External Qdrant; the default Compose stack does not create Qdrant

## Safe configuration

Create a machine-local environment file from the public template:

```bash
cp .env.example .env
```

Replace the OpenAI and Qdrant placeholders. Keep `.env` outside Git; `.gitignore` already excludes it and all noncanonical environment overrides.

## Linux

```bash
sudo apt-get update
sudo apt-get install -y libreoffice-writer libreoffice-impress fontconfig fonts-dejavu-core fonts-liberation2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## macOS

```bash
brew install --cask libreoffice
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When LibreOffice is not discoverable automatically, configure:

```dotenv
SOFFICE_BINARY=/Applications/LibreOffice.app/Contents/MacOS/soffice
```

## React

```bash
cd frontend/react
nvm use
npm ci
npm run dev
```

Vite serves `http://127.0.0.1:5173` and proxies API and operational routes to FastAPI at `http://127.0.0.1:8000`.

## Avoiding port conflicts

The full Compose stack publishes FastAPI on `8000` and React on `3000`. Stop those application containers before starting local Uvicorn or another frontend server:

```bash
docker compose stop frontend backend
docker compose up -d postgres
```

This preserves PostgreSQL and the named document volumes. Use `docker compose down --volumes` only when permanent data deletion is intentional.

## Architecture documentation tooling

The application does not require diagram-generation dependencies. To regenerate the committed architecture assets:

```bash
sudo apt-get install -y graphviz libgraphviz-dev  # Linux
source .venv/bin/activate
python -m pip install -r requirements-architecture.txt
python scripts/generate_architecture_diagrams.py
```

On macOS, install Graphviz with `brew install graphviz` before installing the optional Python dependency.
