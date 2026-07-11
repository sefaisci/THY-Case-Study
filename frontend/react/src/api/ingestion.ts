import { apiRequest, jsonBody } from "./client";
import type {
  IngestionBatchResponse,
  IngestionJobResponse,
  IngestionStartRequest,
  IngestionStatusBatchRequest,
  IngestionStatusBatchResponse,
} from "./contracts";

export function startIngestion(
  username: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<IngestionBatchResponse> {
  const payload: IngestionStartRequest = { document_ids: documentIds };
  return apiRequest<IngestionBatchResponse>("/ingestion-jobs", {
    method: "POST",
    username,
    body: jsonBody(payload),
    signal,
  });
}

export function getIngestionJob(
  username: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<IngestionJobResponse> {
  return apiRequest<IngestionJobResponse>(
    `/ingestion-jobs/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
      username,
      signal,
    },
  );
}

export function getIngestionJobs(
  username: string,
  jobIds: string[],
  signal?: AbortSignal,
): Promise<IngestionStatusBatchResponse> {
  const payload: IngestionStatusBatchRequest = { job_ids: jobIds };
  return apiRequest<IngestionStatusBatchResponse>("/ingestion-jobs/status", {
    method: "POST",
    username,
    body: jsonBody(payload),
    signal,
  });
}
