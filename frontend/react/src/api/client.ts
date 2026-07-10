import type { ErrorResponse } from "./contracts";

const DEFAULT_API_BASE_URL = "/api/v1";
const JSON_CONTENT_TYPE = "application/json";

type QueryValue = string | number | boolean | null | undefined;

export interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: BodyInit | null;
  username?: string;
  query?: Record<string, QueryValue>;
  baseUrl?: string;
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number | null;
  readonly requestId: string;
  readonly details: Record<string, unknown>;

  constructor(options: {
    message: string;
    code?: string;
    status?: number | null;
    requestId: string;
    details?: Record<string, unknown>;
    cause?: unknown;
  }) {
    super(options.message, { cause: options.cause });
    this.name = "ApiError";
    this.code = options.code ?? "api_error";
    this.status = options.status ?? null;
    this.requestId = options.requestId;
    this.details = options.details ?? {};
  }
}

function trimTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  return trimTrailingSlashes(configured || DEFAULT_API_BASE_URL);
}

export function getOperationsBaseUrl(): string {
  const apiBase = getApiBaseUrl();
  return apiBase.endsWith("/api/v1") ? apiBase.slice(0, -7) : "";
}

function createRequestId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildUrl(
  path: string,
  query: Record<string, QueryValue> | undefined,
  baseUrl: string,
): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const baseWithPath = `${trimTrailingSlashes(baseUrl)}${normalizedPath}` || "/";
  const url = new URL(baseWithPath, window.location.origin);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== null && value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

async function parseJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes(JSON_CONTENT_TYPE)) {
    return null;
  }
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function isErrorResponse(value: unknown): value is ErrorResponse {
  if (!value || typeof value !== "object" || !("error" in value)) return false;
  const error = (value as { error?: unknown }).error;
  return Boolean(
    error &&
      typeof error === "object" &&
      "message" in error &&
      typeof (error as { message?: unknown }).message === "string",
  );
}

export async function apiRequest<T>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const requestId = createRequestId();
  const {
    body,
    username,
    query,
    baseUrl = getApiBaseUrl(),
    headers: initialHeaders,
    ...requestInit
  } = options;
  const headers = new Headers(initialHeaders);
  headers.set("Accept", JSON_CONTENT_TYPE);
  headers.set("X-Request-ID", requestId);
  if (username) headers.set("X-Username", username);
  if (body && !(body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", JSON_CONTENT_TYPE);
  }

  let response: Response;
  try {
    response = await fetch(buildUrl(path, query, baseUrl), {
      ...requestInit,
      body,
      headers,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError({
      message: "The FastAPI backend is unavailable. Verify that it is running and reachable.",
      code: "backend_unavailable",
      requestId,
      cause,
    });
  }

  const payload = await parseJson(response);
  if (response.ok) return payload as T;

  const responseRequestId = response.headers.get("X-Request-ID");
  if (isErrorResponse(payload)) {
    throw new ApiError({
      message: payload.error.message,
      code: payload.error.code,
      status: response.status,
      requestId: payload.error.request_id ?? responseRequestId ?? requestId,
      details: payload.error.details,
    });
  }

  throw new ApiError({
    message: `Backend request failed with status ${response.status}.`,
    status: response.status,
    requestId: responseRequestId ?? requestId,
  });
}

export function jsonBody(value: unknown): string {
  return JSON.stringify(value);
}
