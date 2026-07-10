"""FastAPI dependency providers and backend-resolved logical identity."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..database import get_db
from ..exceptions import AppError
from ..models import User
from ..services import ModelCatalogService, UserService

DatabaseSession = Annotated[Session, Depends(get_db)]


async def get_application_settings() -> Settings:
    return get_settings()


ApplicationSettings = Annotated[Settings, Depends(get_application_settings)]


@lru_cache(maxsize=1)
def _cached_model_catalog_service() -> ModelCatalogService:
    return ModelCatalogService(get_settings())


async def get_model_catalog_service() -> ModelCatalogService:
    return _cached_model_catalog_service()


async def get_current_user(
    session: DatabaseSession,
    username: Annotated[str | None, Header(alias="X-Username")] = None,
) -> User:
    """Resolve a username server-side and never accept a frontend user ID."""

    if username is None:
        raise AppError(
            "X-Username header is required.",
            code="username_required",
            status_code=401,
        )
    user, _ = UserService(session).resolve(username)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
ModelCatalog = Annotated[ModelCatalogService, Depends(get_model_catalog_service)]
