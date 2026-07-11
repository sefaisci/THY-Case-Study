export type ReasoningEffort = "low" | "medium" | "high";
export type CollectionScope = "semantic" | "docling" | "both";
export type IngestionMethod = "semantic" | "docling";

export interface ErrorBody {
  code: string;
  message: string;
  request_id: string | null;
  details: Record<string, unknown>;
}

export interface ErrorResponse {
  error: ErrorBody;
}

export interface HealthResponse {
  status: "ok";
  service: string;
}

export interface ReadinessCheck {
  ready: boolean;
  error?: string;
  mode?: string;
}

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  checks: {
    postgresql: ReadinessCheck;
    qdrant: ReadinessCheck;
    openai: ReadinessCheck;
  };
}

export interface UserResolveRequest {
  username: string;
}

export interface UserResponse {
  username: string;
  created_at: string;
  created: boolean;
}

export type ModelReleaseStage = "general_availability" | "preview";

export interface CatalogModel {
  id: string;
  display_name: string | null;
  family: string | null;
  variant: string | null;
  release_stage: ModelReleaseStage;
  description: string | null;
  documentation_url: string | null;
  reasoning_efforts: ReasoningEffort[];
}

export interface UnavailableModel extends CatalogModel {
  unavailable_reason: string;
}

export interface ModelCatalogResponse {
  provider: "openai";
  provider_available: boolean;
  models: CatalogModel[];
  unavailable_models: UnavailableModel[];
  error: string | null;
  refreshed_at: string;
}

export type DocumentStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "deletion_pending"
  | "deleted";

export interface DocumentResponse {
  id: string;
  filename: string;
  mime_type: string;
  file_extension: string;
  file_size_bytes: number;
  sha256: string;
  uploaded_at: string;
  ingestion_method: IngestionMethod;
  semantic_model: string | null;
  semantic_reasoning_effort: string | null;
  collection_name: string;
  status: DocumentStatus;
  error_message: string | null;
}

export interface MultiUploadResponse {
  documents: DocumentResponse[];
  message: string;
}

export interface SemanticUploadConfiguration {
  ingestionMethod: "semantic";
  semanticModel: string;
  semanticReasoningEffort: ReasoningEffort;
}

export interface DoclingUploadConfiguration {
  ingestionMethod: "docling";
}

export type UploadConfiguration =
  | SemanticUploadConfiguration
  | DoclingUploadConfiguration;

export interface IngestionStartRequest {
  document_ids: string[];
}

export type IngestionJobStatus = "pending" | "processing" | "completed" | "failed";

export interface IngestionJobResponse {
  id: string;
  document_id: string;
  status: IngestionJobStatus;
  total_pages: number;
  processed_pages: number;
  progress_percent: number;
  chunk_count: number;
  point_count: number;
  started_at: string | null;
  completed_at: string | null;
  failure_message: string | null;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  cost_usd: number;
  message: string | null;
}

export interface IngestionBatchResponse {
  jobs: IngestionJobResponse[];
  message: string;
}

export interface IngestionStatusBatchRequest {
  job_ids: string[];
}

export interface IngestionStatusBatchResponse {
  jobs: IngestionJobResponse[];
}

export interface DocumentDeleteResponse {
  document_id: string;
  status: "deleted" | "deletion_pending";
  deleted_points: number;
  message: string;
}

export interface ChatSessionCreate {
  title?: string | null;
}

export interface ChatSessionResponse {
  id: string;
  session_identifier: string;
  title: string;
  created_at: string;
  last_activity_at: string;
}

export interface CitationResponse {
  filename: string;
  document_id: string;
  page_number: number | null;
  slide_number: number | null;
  chunk_id: string;
  source_excerpt: string;
  retrieval_score: number;
  ingestion_method: IngestionMethod;
  source_collection: "semantic_chunks" | "docling_fixed_chunks";
  source_pipeline: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessageResponse {
  id: string;
  role: ChatRole;
  content: string;
  citations: CitationResponse[];
  model: string | null;
  reasoning_effort: string | null;
  created_at: string;
}

export interface ChatMessageRequest {
  question: string;
  chat_model: string;
  chat_reasoning_effort: ReasoningEffort;
  collection_scope: CollectionScope;
}

export interface UsageTotals {
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  cost_usd: number;
  unpriced_record_count: number;
}

export interface ChatTurnResponse {
  user_message: ChatMessageResponse;
  assistant_message: ChatMessageResponse;
  no_answer: boolean;
  checked_collections: string[];
  request_usage: UsageTotals;
  session_usage: UsageTotals;
  total_usage: UsageTotals;
}

export interface UsageRecordResponse {
  id: string;
  operation: string;
  stage: string;
  provider: string | null;
  model: string | null;
  reasoning_effort: string | null;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  pricing_version: string | null;
  pricing_status: string;
  created_at: string;
}

export interface UsageSummaryResponse {
  request: UsageTotals | null;
  session: UsageTotals | null;
  total: UsageTotals;
  records: UsageRecordResponse[];
}

export interface UsageQuery {
  sessionId?: string;
  messageId?: string;
}
