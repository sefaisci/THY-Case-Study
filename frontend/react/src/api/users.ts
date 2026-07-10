import { apiRequest, jsonBody } from "./client";
import type { UserResolveRequest, UserResponse } from "./contracts";

export function resolveUser(username: string, signal?: AbortSignal): Promise<UserResponse> {
  const payload: UserResolveRequest = { username };
  return apiRequest<UserResponse>("/users/resolve", {
    method: "POST",
    body: jsonBody(payload),
    signal,
  });
}
