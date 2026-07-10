"""Application exceptions with stable API error codes."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Expected application failure rendered by the centralized handler."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "application_error",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class NotFoundError(AppError):
    def __init__(self, message: str, *, code: str = "not_found") -> None:
        super().__init__(message, code=code, status_code=404)


class ConflictError(AppError):
    def __init__(self, message: str, *, code: str = "conflict") -> None:
        super().__init__(message, code=code, status_code=409)


class ProviderError(AppError):
    def __init__(self, message: str, *, code: str = "provider_error") -> None:
        super().__init__(message, code=code, status_code=502)
