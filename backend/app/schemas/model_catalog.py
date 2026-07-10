"""Runtime OpenAI model availability and capability schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReasoningEffort = Literal["low", "medium", "high"]
ModelReleaseStage = Literal["general_availability", "preview"]


class CatalogModel(BaseModel):
    id: str
    display_name: str | None = None
    family: str | None = None
    variant: str | None = None
    release_stage: ModelReleaseStage = "general_availability"
    description: str | None = None
    documentation_url: str | None = None
    reasoning_efforts: list[ReasoningEffort]


class AvailableModel(CatalogModel):
    """Configured model that the current OpenAI account can invoke."""


class UnavailableModel(CatalogModel):
    """Configured model omitted from selectors because the account lacks access."""

    unavailable_reason: str


class ModelCatalogResponse(BaseModel):
    provider: str = "openai"
    provider_available: bool
    models: list[AvailableModel] = Field(default_factory=list)
    unavailable_models: list[UnavailableModel] = Field(default_factory=list)
    error: str | None = None
    refreshed_at: str
