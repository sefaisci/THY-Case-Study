import { apiRequest } from "./client";
import type {
  DocumentDeleteResponse,
  DocumentResponse,
  MultiUploadResponse,
  UploadConfiguration,
} from "./contracts";

export function listDocuments(
  username: string,
  signal?: AbortSignal,
): Promise<DocumentResponse[]> {
  return apiRequest<DocumentResponse[]>("/documents", {
    method: "GET",
    username,
    signal,
  });
}

export function uploadDocuments(options: {
  username: string;
  files: File[];
  configuration: UploadConfiguration;
  signal?: AbortSignal;
}): Promise<MultiUploadResponse> {
  const formData = new FormData();
  for (const file of options.files) formData.append("files", file);
  formData.append("ingestion_method", options.configuration.ingestionMethod);
  if (options.configuration.ingestionMethod === "semantic") {
    formData.append("semantic_model", options.configuration.semanticModel);
    formData.append(
      "semantic_reasoning_effort",
      options.configuration.semanticReasoningEffort,
    );
  }
  return apiRequest<MultiUploadResponse>("/documents/upload", {
    method: "POST",
    username: options.username,
    body: formData,
    signal: options.signal,
    timeoutMs: 300_000,
  });
}

export function deleteDocument(
  username: string,
  documentId: string,
  signal?: AbortSignal,
): Promise<DocumentDeleteResponse> {
  return apiRequest<DocumentDeleteResponse>(
    `/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
      username,
      signal,
      timeoutMs: 120_000,
    },
  );
}
