"""OpenAI Responses adapter for strict multimodal semantic chunking."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from model.document_processing.schemas import RenderedPage
from model.usage import UsageCallback, emit_usage, usage_from_response

from .prompts import SEMANTIC_SYSTEM_PROMPT, build_page_prompt
from .schemas import SemanticPageResult


class SemanticChunkingError(RuntimeError):
    """Raised when a page cannot produce a validated structured result."""


class SemanticChunkingRefusal(SemanticChunkingError):
    """Raised when the provider refuses a page request."""


def image_to_data_url(image_path: str | Path, mime_type: str = "image/png") -> str:
    """Encode a local image for transport without persisting the base64 body."""

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class OpenAISemanticChunker:
    """Analyze one current page image without cross-page memory."""

    _SUPPORTED_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.5",
        reasoning_effort: str = "low",
        base_url: str | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if reasoning_effort not in self._SUPPORTED_REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
        if not api_key and client is None:
            raise ValueError("OPENAI_API_KEY is required for semantic image chunking.")
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.usage_callback = usage_callback

    async def chunk_page(self, page: RenderedPage) -> SemanticPageResult:
        """Return a strict semantic result or one bounded terminal error."""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                image_url = await asyncio.to_thread(
                    image_to_data_url,
                    page.image_path,
                    page.mime_type,
                )
                response = await self._client.responses.parse(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    instructions=SEMANTIC_SYSTEM_PROMPT,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": build_page_prompt(page),
                                },
                                {
                                    "type": "input_image",
                                    "image_url": image_url,
                                },
                            ],
                        }
                    ],
                    text_format=SemanticPageResult,
                    timeout=self.timeout_seconds,
                )
                emit_usage(
                    self.usage_callback,
                    usage_from_response(
                        response,
                        stage="semantic_chunking",
                        fallback_model=self.model,
                        metadata={
                            "document_id": page.document_id,
                            "location_number": page.location_number,
                        },
                    ),
                )
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    refusal = _extract_refusal(response)
                    if refusal:
                        raise SemanticChunkingRefusal(f"Semantic chunking request was refused: {refusal}")
                    if getattr(response, "status", None) == "incomplete":
                        details = getattr(response, "incomplete_details", None)
                        reason = getattr(details, "reason", None) or "unknown reason"
                        raise SemanticChunkingError(
                            f"OpenAI returned an incomplete semantic result: {reason}."
                        )
                    raise SemanticChunkingError("OpenAI returned no parsed semantic result.")
                result = (
                    parsed
                    if isinstance(parsed, SemanticPageResult)
                    else SemanticPageResult.model_validate(parsed)
                )
                if result.page_number != page.location_number:
                    raise SemanticChunkingError(
                        f"Location mismatch: expected {page.location_number}, received {result.page_number}."
                    )
                return result
            except SemanticChunkingRefusal:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
        raise SemanticChunkingError(
            f"Semantic chunking failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


def _extract_refusal(response: object) -> str:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)[:500]
    return ""
