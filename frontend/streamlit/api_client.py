"""Typed-enough HTTP client used by the Streamlit application only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiError(RuntimeError):
    message: str
    code: str = "api_error"
    status_code: int | None = None
    request_id: str | None = None

    def __str__(self) -> str:
        suffix = f" (Request ID: {self.request_id})" if self.request_id else ""
        return f"{self.message}{suffix}"


class ApiClient:
    """Backend-only application client with username-scoped requests."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    def resolve_user(self, username: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/users/resolve", json={"username": username})

    def get_models(self, *, refresh: bool = False) -> dict[str, Any]:
        return self._request("GET", "/api/v1/models", params={"refresh": str(refresh).lower()})

    def list_documents(self, username: str) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/documents", username=username)

    def upload_documents(
        self,
        username: str,
        files: list[Any],
        *,
        ingestion_method: str,
        semantic_model: str | None,
        semantic_reasoning_effort: str | None,
    ) -> dict[str, Any]:
        multipart = [
            (
                "files",
                (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                ),
            )
            for uploaded in files
        ]
        data = {"ingestion_method": ingestion_method}
        if semantic_model:
            data["semantic_model"] = semantic_model
        if semantic_reasoning_effort:
            data["semantic_reasoning_effort"] = semantic_reasoning_effort
        return self._request(
            "POST",
            "/api/v1/documents/upload",
            username=username,
            files=multipart,
            data=data,
        )

    def start_ingestion(self, username: str, document_ids: list[str]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/ingestion-jobs",
            username=username,
            json={"document_ids": document_ids},
        )

    def get_ingestion_job(self, username: str, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/ingestion-jobs/{job_id}",
            username=username,
        )

    def get_ingestion_jobs(self, username: str, job_ids: list[str]) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            "/api/v1/ingestion-jobs/status",
            username=username,
            json={"job_ids": job_ids},
        )
        return list(response.get("jobs", []))

    def delete_document(self, username: str, document_id: str) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/api/v1/documents/{document_id}",
            username=username,
        )

    def create_chat(self, username: str, title: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/chat/sessions",
            username=username,
            json={"title": title},
        )

    def list_chats(self, username: str) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/chat/sessions", username=username)

    def list_messages(self, username: str, session_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/api/v1/chat/sessions/{session_id}/messages",
            username=username,
        )

    def send_message(
        self,
        username: str,
        session_id: str,
        *,
        question: str,
        chat_model: str,
        chat_reasoning_effort: str,
        collection_scope: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/chat/sessions/{session_id}/messages",
            username=username,
            json={
                "question": question,
                "chat_model": chat_model,
                "chat_reasoning_effort": chat_reasoning_effort,
                "collection_scope": collection_scope,
            },
        )

    def get_usage(
        self,
        username: str,
        *,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        params = {}
        if session_id:
            params["session_id"] = session_id
        if message_id:
            params["message_id"] = message_id
        return self._request("GET", "/api/v1/usage", username=username, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        username: str | None = None,
        **kwargs: Any,
    ) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if username:
            headers["X-Username"] = username
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise ApiError(
                "The FastAPI backend is unavailable. Verify that it is running and reachable.",
                code="backend_unavailable",
            ) from exc
        if response.is_success:
            return response.json()
        try:
            payload = response.json().get("error", {})
        except ValueError:
            payload = {}
        raise ApiError(
            message=payload.get("message") or f"Backend request failed with status {response.status_code}.",
            code=payload.get("code", "api_error"),
            status_code=response.status_code,
            request_id=payload.get("request_id") or response.headers.get("X-Request-ID"),
        )
