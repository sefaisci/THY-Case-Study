"""Account-aware OpenAI model and reasoning capability validation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..exceptions import AppError, ProviderError
from ..schemas.model_catalog import (
    AvailableModel,
    ModelCatalogResponse,
    UnavailableModel,
)


class ModelCatalogService:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._cached: ModelCatalogResponse | None = None
        self._cached_monotonic = 0.0
        self._capabilities = self._load_capabilities(settings.model_capabilities_path)

    def get_catalog(self, *, force: bool = False) -> ModelCatalogResponse:
        age = time.monotonic() - self._cached_monotonic
        if (
            not force
            and self._cached is not None
            and age <= self.settings.openai_model_cache_seconds
        ):
            return self._cached
        refreshed_at = datetime.now(timezone.utc).isoformat()
        if not self.settings.openai_api_key and self._client is None:
            catalog = ModelCatalogResponse(
                provider_available=False,
                error="OPENAI_API_KEY is not configured.",
                refreshed_at=refreshed_at,
            )
        else:
            try:
                client = self._client or self._create_client()
                response = client.models.list()
                available_ids = {str(item.id) for item in response.data}
                models = [
                    AvailableModel(**self._catalog_fields(item))
                    for item in self._capabilities
                    if item["id"] in available_ids
                ]
                unavailable_models = [
                    UnavailableModel(
                        **self._catalog_fields(item),
                        unavailable_reason=self._unavailable_reason(item),
                    )
                    for item in self._capabilities
                    if item["id"] not in available_ids
                ]
                catalog = ModelCatalogResponse(
                    provider_available=True,
                    models=models,
                    unavailable_models=unavailable_models,
                    refreshed_at=refreshed_at,
                )
            except Exception as exc:
                catalog = ModelCatalogResponse(
                    provider_available=False,
                    error=f"OpenAI model availability check failed: {str(exc)[:500]}",
                    refreshed_at=refreshed_at,
                )
        self._cached = catalog
        self._cached_monotonic = time.monotonic()
        return catalog

    def validate(self, model: str, reasoning_effort: str) -> None:
        catalog = self.get_catalog()
        if not catalog.provider_available:
            raise ProviderError(catalog.error or "OpenAI model availability is unavailable.")
        selected = next((item for item in catalog.models if item.id == model), None)
        if selected is None:
            raise AppError(
                f"Model {model!r} is not available to the configured OpenAI account.",
                code="model_unavailable",
                status_code=422,
            )
        if reasoning_effort not in selected.reasoning_efforts:
            raise AppError(
                f"Reasoning effort {reasoning_effort!r} is not supported by model {model!r}.",
                code="reasoning_effort_unsupported",
                status_code=422,
            )

    def _create_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

    @staticmethod
    def _catalog_fields(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "display_name": item.get("display_name") or item["id"],
            "family": item.get("family"),
            "variant": item.get("variant"),
            "release_stage": item.get("release_stage", "general_availability"),
            "description": item.get("description"),
            "documentation_url": item.get("documentation_url"),
            "reasoning_efforts": item["reasoning_efforts"],
        }

    @staticmethod
    def _unavailable_reason(item: dict[str, Any]) -> str:
        if item.get("release_stage") == "preview":
            return (
                "This preview model is configured but is not returned by the "
                "configured OpenAI account's Models API."
            )
        return "This configured model is not available to the OpenAI account."

    @staticmethod
    def _load_capabilities(path: Path) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documentation_url = payload.get("documentation_url")
        models = list(payload.get("models", []))
        for item in models:
            item.setdefault("documentation_url", documentation_url)
        return models
