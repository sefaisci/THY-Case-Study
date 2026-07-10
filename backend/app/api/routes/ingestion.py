"""Persistent background ingestion lifecycle routes."""

from fastapi import APIRouter, BackgroundTasks, status

from ...database import SessionLocal
from ...schemas.documents import (
    IngestionBatchResponse,
    IngestionJobResponse,
    IngestionStartRequest,
)
from ...services import IngestionService
from ..dependencies import ApplicationSettings, CurrentUser, DatabaseSession

router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion"])


def _job_response(job) -> IngestionJobResponse:
    message = None
    if job.status == "completed":
        message = "Ingestion completed successfully."
    elif job.status == "failed":
        message = "Ingestion failed. Review the failure details and retry."
    elif job.status == "processing":
        message = "Ingestion is processing."
    else:
        message = "Ingestion is pending."
    return IngestionJobResponse.model_validate(job).model_copy(update={"message": message})


@router.post(
    "",
    response_model=IngestionBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start ingestion for one or more pending documents",
)
async def start_ingestion(
    request: IngestionStartRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
) -> IngestionBatchResponse:
    service = IngestionService(
        session,
        settings,
        session_factory=SessionLocal,
    )
    jobs = service.start(user_id=user.id, document_ids=request.document_ids)
    for job in jobs:
        background_tasks.add_task(service.process_job, job.id)
    return IngestionBatchResponse(
        jobs=[_job_response(job) for job in jobs],
        message="Ingestion started.",
    )


@router.get(
    "/{job_id}",
    response_model=IngestionJobResponse,
    summary="Poll one owner-scoped ingestion job",
)
async def get_ingestion_job(
    job_id: str,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
) -> IngestionJobResponse:
    job = IngestionService(session, settings).get_job(user_id=user.id, job_id=job_id)
    return _job_response(job)
