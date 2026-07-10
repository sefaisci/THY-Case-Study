"""Persist real provider usage and calculate request/session/total aggregates."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy.orm import Session

from model.usage import ModelUsage

from ..models import UsageRecord
from ..repositories import UsageRepository
from ..schemas.usage import UsageTotals
from .pricing import PricingRegistry


class UsageService:
    def __init__(self, session: Session, pricing: PricingRegistry) -> None:
        self.session = session
        self.repository = UsageRepository(session)
        self.pricing = pricing

    def persist_events(
        self,
        events: Iterable[ModelUsage],
        *,
        user_id: str,
        operation: str,
        reasoning_effort: str | None = None,
        document_id: str | None = None,
        ingestion_job_id: str | None = None,
        chat_session_id: str | None = None,
        chat_message_id: str | None = None,
    ) -> list[UsageRecord]:
        records = []
        for event in events:
            price = self.pricing.calculate(event)
            records.append(
                self.repository.add(
                    UsageRecord(
                        user_id=user_id,
                        document_id=document_id,
                        ingestion_job_id=ingestion_job_id,
                        chat_session_id=chat_session_id,
                        chat_message_id=chat_message_id,
                        operation=operation,
                        stage=event.stage,
                        provider=event.provider,
                        model=event.model or None,
                        reasoning_effort=reasoning_effort,
                        input_tokens=event.input_tokens,
                        cached_input_tokens=event.cached_input_tokens,
                        output_tokens=event.output_tokens,
                        reasoning_tokens=event.reasoning_tokens,
                        total_tokens=event.total_tokens,
                        cost_usd=price.cost_usd,
                        pricing_version=price.pricing_version,
                        pricing_status=price.pricing_status,
                        provider_request_id=event.request_id,
                        details=event.metadata,
                    )
                )
            )
        return records

    def record_not_applicable(
        self,
        *,
        user_id: str,
        operation: str,
        stage: str,
        document_id: str | None = None,
        ingestion_job_id: str | None = None,
        chat_session_id: str | None = None,
        chat_message_id: str | None = None,
    ) -> UsageRecord:
        return self.repository.add(
            UsageRecord(
                user_id=user_id,
                document_id=document_id,
                ingestion_job_id=ingestion_job_id,
                chat_session_id=chat_session_id,
                chat_message_id=chat_message_id,
                operation=operation,
                stage=stage,
                provider=None,
                model=None,
                pricing_status="not_applicable",
                cost_usd=0.0,
            )
        )

    @staticmethod
    def totals(records: Iterable[UsageRecord]) -> UsageTotals:
        items = list(records)
        return UsageTotals(
            input_tokens=sum(item.input_tokens for item in items),
            cached_input_tokens=sum(item.cached_input_tokens for item in items),
            output_tokens=sum(item.output_tokens for item in items),
            reasoning_tokens=sum(item.reasoning_tokens for item in items),
            total_tokens=sum(item.total_tokens for item in items),
            cost_usd=round(sum(item.cost_usd or 0.0 for item in items), 10),
            unpriced_record_count=sum(
                item.pricing_status
                not in {"priced", "not_applicable"}
                for item in items
            ),
        )
