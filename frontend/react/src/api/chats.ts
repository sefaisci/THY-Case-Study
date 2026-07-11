import { apiRequest, jsonBody } from "./client";
import type {
  ChatMessageRequest,
  ChatMessageResponse,
  ChatSessionCreate,
  ChatSessionResponse,
  ChatTurnResponse,
} from "./contracts";

export function createChatSession(
  username: string,
  request: ChatSessionCreate = {},
  signal?: AbortSignal,
): Promise<ChatSessionResponse> {
  return apiRequest<ChatSessionResponse>("/chat/sessions", {
    method: "POST",
    username,
    body: jsonBody(request),
    signal,
  });
}

export function listChatSessions(
  username: string,
  signal?: AbortSignal,
): Promise<ChatSessionResponse[]> {
  return apiRequest<ChatSessionResponse[]>("/chat/sessions", {
    method: "GET",
    username,
    signal,
  });
}

export function listChatMessages(
  username: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<ChatMessageResponse[]> {
  return apiRequest<ChatMessageResponse[]>(
    `/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: "GET",
      username,
      signal,
    },
  );
}

export function sendChatMessage(options: {
  username: string;
  sessionId: string;
  request: ChatMessageRequest;
  signal?: AbortSignal;
}): Promise<ChatTurnResponse> {
  return apiRequest<ChatTurnResponse>(
    `/chat/sessions/${encodeURIComponent(options.sessionId)}/messages`,
    {
      method: "POST",
      username: options.username,
      body: jsonBody(options.request),
      signal: options.signal,
      timeoutMs: 180_000,
    },
  );
}
