"""Persistent background ingestion lifecycle routes."""

from fastapi import APIRouter, BackgroundTasks, status

from ...database import SessionLocal
from ...schemas.documents import (
    IngestionBatchResponse,
    IngestionJobResponse,
    IngestionStartRequest,
    IngestionStatusRequest,
    IngestionStatusResponse,
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
    total_pages = max(0, job.total_pages)
    processed_pages = min(max(0, job.processed_pages), total_pages)
    progress_percent = (
        min(100, round(processed_pages * 100 / total_pages))
        if total_pages
        else 0
    )
    return IngestionJobResponse.model_validate(job).model_copy(
        update={
            "processed_pages": processed_pages,
            "progress_percent": progress_percent,
            "message": message,
        }
    )


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
    jobs = await service.start(user_id=user.id, document_ids=request.document_ids)
    background_tasks.add_task(service.process_jobs, [job.id for job in jobs])
    return IngestionBatchResponse(
        jobs=[_job_response(job) for job in jobs],
        message="Ingestion started.",
    )


@router.post(
    "/status",
    response_model=IngestionStatusResponse,
    summary="Poll multiple owner-scoped ingestion jobs",
)
async def get_ingestion_jobs(
    request: IngestionStatusRequest,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
) -> IngestionStatusResponse:
    jobs = await IngestionService(session, settings).get_jobs(
        user_id=user.id,
        job_ids=request.job_ids,
    )
    return IngestionStatusResponse(jobs=[_job_response(job) for job in jobs])


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
    job = await IngestionService(session, settings).get_job(user_id=user.id, job_id=job_id)
    return _job_response(job)
