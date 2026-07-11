import { QueryClient } from "@tanstack/react-query";

import { ApiError } from "../api/client";

const RETRYABLE_STATUS_CODES = new Set([408, 429, 502, 503, 504]);

export function shouldRetryQuery(failureCount: number, error: unknown): boolean {
  if (failureCount >= 2) return false;
  if (!(error instanceof ApiError)) return failureCount < 1;
  if (error.status === null) return true;
  return RETRYABLE_STATUS_CODES.has(error.status);
}

export function createAppQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
        retry: shouldRetryQuery,
      },
      mutations: {
        retry: false,
      },
    },
  });
}
