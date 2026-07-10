"""FastAPI application factory and operational endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .api.router import api_router
from .config import get_settings
from .database import SessionLocal
from .exceptions import AppError
from .logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    for directory in (settings.upload_dir, settings.page_image_dir, settings.processing_dir):
        directory.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="THY Agentic RAG API",
    summary="Multi-user document ingestion and grounded Agentic RAG proof of concept.",
    description=(
        "FastAPI boundary for username-scoped PostgreSQL metadata, external Qdrant retrieval, "
        "semantic and Docling ingestion, citations, chat memory, and usage observability."
    ),
    version="1.0.0-poc",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Username", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "HTTP request completed",
            extra={"request_id": request_id, "event": "http_request"},
        )
        logger.debug("HTTP request latency_ms=%s", elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": getattr(request.state, "request_id", None),
                "details": exc.details,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": {
                    "code": "request_validation_error",
                    "message": "Request validation failed.",
                    "request_id": getattr(request.state, "request_id", None),
                    "details": {"errors": exc.errors()},
                }
            }
        ),
    )


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception(
        "PostgreSQL operation failed",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "event": "database_unavailable",
        },
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "database_unavailable",
                "message": (
                    "PostgreSQL is unavailable. Start the database service and apply "
                    "the latest Alembic migrations."
                ),
                "request_id": getattr(request.state, "request_id", None),
                "details": {},
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled application error",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "event": "unhandled_error",
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": "An unexpected server error occurred.",
                "request_id": getattr(request.state, "request_id", None),
                "details": {},
            }
        },
    )


@app.get("/health", tags=["operations"], summary="Liveness check")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready", tags=["operations"], summary="Dependency-aware readiness check")
async def ready() -> JSONResponse:
    checks: dict[str, dict[str, object]] = {}
    database_ready = False
    qdrant_ready = False
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        database_ready = True
        checks["postgresql"] = {"ready": True}
    except Exception as exc:
        checks["postgresql"] = {"ready": False, "error": str(exc)[:300]}
    try:
        qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        qdrant.get_collections()
        qdrant_ready = True
        checks["qdrant"] = {"ready": True}
    except Exception as exc:
        checks["qdrant"] = {"ready": False, "error": str(exc)[:300]}
    checks["openai"] = {"ready": bool(settings.openai_api_key), "mode": "key_configured"}
    ready_state = database_ready and qdrant_ready and bool(settings.openai_api_key)
    return JSONResponse(
        status_code=200 if ready_state else 503,
        content={"status": "ready" if ready_state else "not_ready", "checks": checks},
    )


app.include_router(api_router, prefix=settings.api_prefix)
