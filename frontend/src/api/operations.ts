import { apiRequest, getOperationsBaseUrl } from "./client";
import type { HealthResponse, ReadinessResponse } from "./contracts";

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health", {
    method: "GET",
    baseUrl: getOperationsBaseUrl(),
    signal,
  });
}

export function getReadiness(signal?: AbortSignal): Promise<ReadinessResponse> {
  return apiRequest<ReadinessResponse>("/ready", {
    method: "GET",
    baseUrl: getOperationsBaseUrl(),
    signal,
  });
}
