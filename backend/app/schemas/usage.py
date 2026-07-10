"""Token and cost observability response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UsageRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    operation: str
    stage: str
    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: float | None = None
    pricing_version: str | None = None
    pricing_status: str
    created_at: datetime


class UsageTotals(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_record_count: int = 0


class UsageSummaryResponse(BaseModel):
    request: UsageTotals | None = None
    session: UsageTotals | None = None
    total: UsageTotals
    records: list[UsageRecordResponse] = Field(default_factory=list)
