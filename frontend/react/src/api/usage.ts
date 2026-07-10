import { apiRequest } from "./client";
import type { UsageQuery, UsageSummaryResponse } from "./contracts";

export function getUsage(
  username: string,
  query: UsageQuery = {},
  signal?: AbortSignal,
): Promise<UsageSummaryResponse> {
  return apiRequest<UsageSummaryResponse>("/usage", {
    method: "GET",
    username,
    query: {
      session_id: query.sessionId,
      message_id: query.messageId,
    },
    signal,
  });
}
