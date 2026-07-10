"""Provider usage events shared by model adapters and application services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelUsage:
    """One provider-reported usage event for a named application stage."""

    stage: str
    provider: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    request_id: str | None = None
    usage_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


UsageCallback = Callable[[ModelUsage], None]


def usage_from_response(
    response: object,
    *,
    stage: str,
    fallback_model: str,
    provider: str = "openai",
    metadata: dict[str, Any] | None = None,
) -> ModelUsage:
    """Normalize Responses API or embeddings usage without estimating tokens."""

    usage = getattr(response, "usage", None)
    model = str(getattr(response, "model", None) or fallback_model)
    request_id = getattr(response, "id", None)
    if usage is None:
        return ModelUsage(
            stage=stage,
            provider=provider,
            model=model,
            request_id=str(request_id) if request_id else None,
            usage_available=False,
            metadata=metadata or {},
        )

    input_tokens = _integer_attr(usage, "input_tokens", "prompt_tokens")
    output_tokens = _integer_attr(usage, "output_tokens", "completion_tokens")
    total_tokens = _integer_attr(usage, "total_tokens")
    input_details = getattr(usage, "input_tokens_details", None) or getattr(
        usage, "prompt_tokens_details", None
    )
    output_details = getattr(usage, "output_tokens_details", None) or getattr(
        usage, "completion_tokens_details", None
    )
    cached_input_tokens = _integer_attr(input_details, "cached_tokens")
    reasoning_tokens = _integer_attr(output_details, "reasoning_tokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return ModelUsage(
        stage=stage,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        request_id=str(request_id) if request_id else None,
        metadata=metadata or {},
    )


def emit_usage(callback: UsageCallback | None, usage: ModelUsage) -> None:
    """Emit observability data without allowing telemetry to break model work."""

    if callback is None:
        return
    try:
        callback(usage)
    except Exception:
        # Application persistence failures are handled at the service boundary.
        return


def _integer_attr(value: object | None, *names: str) -> int:
    if value is None:
        return 0
    for name in names:
        raw = getattr(value, name, None)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0
    return 0
