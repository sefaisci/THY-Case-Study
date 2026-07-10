"""Selectable semantic and Docling ingestion entry points."""

from .coordinator import IngestionCoordinator, create_connected_ingestion_coordinator
from .schemas import (
    DocumentIngestionResult,
    IngestionRequest,
    IngestionRunResult,
    LocationFailure,
)
from .settings import IngestionSettings

__all__ = [
    "DocumentIngestionResult",
    "IngestionCoordinator",
    "IngestionRequest",
    "IngestionRunResult",
    "IngestionSettings",
    "LocationFailure",
    "create_connected_ingestion_coordinator",
]
