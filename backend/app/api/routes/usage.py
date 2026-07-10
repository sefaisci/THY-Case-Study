"""Request, session, and total token/cost observability route."""

from fastapi import APIRouter, Query

from ...exceptions import NotFoundError
from ...repositories import ChatRepository, UsageRepository
from ...schemas.usage import UsageRecordResponse, UsageSummaryResponse
from ...services import PricingRegistry, UsageService
from ..dependencies import ApplicationSettings, CurrentUser, DatabaseSession

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=UsageSummaryResponse, summary="Get owner-scoped usage totals")
async def get_usage(
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    session_id: str | None = Query(default=None),
    message_id: str | None = Query(default=None),
) -> UsageSummaryResponse:
    if session_id and ChatRepository(session).get_owned(user.id, session_id) is None:
        raise NotFoundError("Chat session not found.", code="chat_session_not_found")
    repository = UsageRepository(session)
    service = UsageService(session, PricingRegistry(settings.pricing_registry_path))
    total_records = repository.list_for_user(user.id, limit=10_000)
    session_records = (
        repository.list_for_user(user.id, session_id=session_id, limit=10_000)
        if session_id
        else []
    )
    request_records = (
        repository.list_for_user(user.id, message_id=message_id, limit=1_000)
        if message_id
        else []
    )
    displayed = request_records or session_records or total_records[:200]
    return UsageSummaryResponse(
        request=service.totals(request_records) if message_id else None,
        session=service.totals(session_records) if session_id else None,
        total=service.totals(total_records),
        records=[UsageRecordResponse.model_validate(item) for item in displayed],
    )
