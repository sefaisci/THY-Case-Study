import { apiRequest } from "./client";
import type { ModelCatalogResponse } from "./contracts";

export function getModelCatalog(
  options: { refresh?: boolean; signal?: AbortSignal } = {},
): Promise<ModelCatalogResponse> {
  return apiRequest<ModelCatalogResponse>("/models", {
    method: "GET",
    query: { refresh: options.refresh ?? false },
    signal: options.signal,
  });
}
