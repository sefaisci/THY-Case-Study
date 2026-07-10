"""Versioned pricing registry and non-invented USD cost calculation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model.usage import ModelUsage


@dataclass(frozen=True)
class PriceResult:
    cost_usd: float | None
    pricing_version: str
    pricing_status: str


class PricingRegistry:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version = str(payload["version"])
        self.models = dict(payload.get("models", {}))
        self.aliases = dict(payload.get("model_aliases", {}))

    def calculate(self, usage: ModelUsage) -> PriceResult:
        if not usage.usage_available:
            return PriceResult(None, self.version, "usage_unavailable")
        registry_model = self.aliases.get(usage.model, usage.model)
        rates = self.models.get(registry_model)
        if rates is None:
            return PriceResult(None, self.version, "unpriced_model")
        input_rate = rates.get("input")
        cached_rate = rates.get("cached_input")
        output_rate = rates.get("output")
        if input_rate is None or output_rate is None:
            return PriceResult(None, self.version, "unpriced_model")
        cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
        uncached_tokens = max(0, usage.input_tokens - cached_tokens)
        if cached_tokens and cached_rate is None:
            return PriceResult(None, self.version, "cached_input_unpriced")
        cost = (
            uncached_tokens * float(input_rate)
            + cached_tokens * float(cached_rate or 0.0)
            + usage.output_tokens * float(output_rate)
        ) / 1_000_000
        return PriceResult(round(cost, 10), self.version, "priced")
