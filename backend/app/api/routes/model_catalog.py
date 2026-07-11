"""Account-aware OpenAI model availability endpoint."""

from fastapi import APIRouter, Query

from ...schemas.model_catalog import ModelCatalogResponse
from ..dependencies import ModelCatalog

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "",
    response_model=ModelCatalogResponse,
    summary="List selectable and account-unavailable configured OpenAI models",
)
async def list_models(
    catalog: ModelCatalog,
    refresh: bool = Query(default=False, description="Bypass the short model-list cache."),
) -> ModelCatalogResponse:
    return await catalog.get_catalog(force=refresh)
