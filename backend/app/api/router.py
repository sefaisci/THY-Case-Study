"""Versioned API router composition."""

from fastapi import APIRouter

from .routes import chats, documents, ingestion, model_catalog, usage, users

api_router = APIRouter()
api_router.include_router(users.router)
api_router.include_router(model_catalog.router)
api_router.include_router(documents.router)
api_router.include_router(ingestion.router)
api_router.include_router(chats.router)
api_router.include_router(usage.router)
