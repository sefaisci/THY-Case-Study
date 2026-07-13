import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  File,
  FileText,
  LoaderCircle,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDropzone } from "react-dropzone";

import { ApiError } from "../../api/client";
import type {
  CatalogModel,
  CollectionScope,
  DocumentResponse,
  IngestionMethod,
  IngestionJobResponse,
  ModelCatalogResponse,
  ReasoningEffort,
  UploadConfiguration,
} from "../../api/contracts";
import { deleteDocument, uploadDocuments } from "../../api/documents";
import { getIngestionJobs, startIngestion } from "../../api/ingestion";
import { normalizeUsernameKey, queryKeys } from "../../app/queryKeys";
import { formatBytes, formatTimestamp } from "../../lib/format";
import {
  useWorkspaceStore,
  type ActiveIngestionJob,
} from "../../state/workspaceStore";
import type { WorkspaceNotice } from "../workspace/types";

interface DocumentInspectorProps {
  username: string | null;
  catalog: ModelCatalogResponse | undefined;
  documents: DocumentResponse[];
  isLoadingDocuments: boolean;
  isRefreshingModels: boolean;
  mobileOpen: boolean;
  semanticModel: string | null;
  semanticReasoningEffort: ReasoningEffort | null;
  chatModel: string | null;
  chatReasoningEffort: ReasoningEffort | null;
  collectionScope: CollectionScope;
  onSemanticSelection: (model: string | null, effort: ReasoningEffort | null) => void;
  onChatSelection: (model: string | null, effort: ReasoningEffort | null) => void;
  onCollectionScopeChange: (scope: CollectionScope) => void;
  onRefreshModels: () => void;
  onCloseMobile: () => void;
  onNotice: (notice: WorkspaceNotice) => void;
}

const MAX_FILE_BYTES = 100 * 1024 * 1024;

class IngestionStartAfterUploadError extends Error {
  readonly causeError: unknown;
  readonly uploadedDocuments: DocumentResponse[];

  constructor(causeError: unknown, uploadedDocuments: DocumentResponse[]) {
    super("Files were uploaded, but their ingestion jobs could not be started.");
    this.name = "IngestionStartAfterUploadError";
    this.causeError = causeError;
    this.uploadedDocuments = uploadedDocuments;
  }
}

interface IngestionSubmission {
  username: string;
  files: File[];
  configuration: UploadConfiguration;
}

interface DocumentMutationVariables {
  username: string;
  document: DocumentResponse;
}

function mergeDocuments(
  current: DocumentResponse[] | undefined,
  incoming: DocumentResponse[],
): DocumentResponse[] {
  const incomingIds = new Set(incoming.map((document) => document.id));
  return [...incoming, ...(current ?? []).filter((document) => !incomingIds.has(document.id))];
}

function errorNotice(error: unknown, title: string): WorkspaceNotice {
  if (error instanceof ApiError) {
    return {
      kind: "error",
      title,
      message: error.message,
      requestId: error.requestId,
    };
  }
  return { kind: "error", title, message: "The operation could not be completed." };
}

function ModelSelect({
  id,
  label,
  models,
  value,
  effort,
  disabled,
  onChange,
}: {
  id: string;
  label: string;
  models: CatalogModel[];
  value: string | null;
  effort: ReasoningEffort | null;
  disabled: boolean;
  onChange: (model: string | null, effort: ReasoningEffort | null) => void;
}) {
  const selected = models.find((model) => model.id === value);
  const efforts = selected?.reasoning_efforts ?? [];
  return (
    <div className="model-control-grid">
      <label>
        <span>{label} model</span>
        <select
          id={`${id}-model`}
          value={value ?? ""}
          disabled={disabled || models.length === 0}
          onChange={(event) => {
            const model = models.find((item) => item.id === event.target.value);
            onChange(model?.id ?? null, model?.reasoning_efforts[0] ?? null);
          }}
        >
          {models.length === 0 ? <option value="">No available model</option> : null}
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.display_name ?? model.id}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>{label} reasoning effort</span>
        <select
          id={`${id}-effort`}
          value={effort ?? ""}
          disabled={disabled || !selected}
          onChange={(event) => onChange(value, event.target.value as ReasoningEffort)}
        >
          {efforts.map((item) => (
            <option key={item} value={item}>
              {item.charAt(0).toUpperCase() + item.slice(1)}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function ActiveJobRow({
  job,
  remoteJob,
}: {
  job: ActiveIngestionJob;
  remoteJob: IngestionJobResponse | undefined;
}) {
  const status = remoteJob?.status ?? job.status;
  const totalPages = remoteJob?.total_pages ?? 0;
  const processedPages = Math.min(remoteJob?.processed_pages ?? 0, totalPages);
  const progressPercent = remoteJob?.progress_percent ?? 0;
  const hasPageCount = totalPages > 0;
  const progressLabel =
    status === "pending"
      ? "Queued for processing"
      : hasPageCount
        ? `${processedPages} of ${totalPages} pages processed`
        : "Determining page count";
  return (
    <div className="active-job-row">
      <div className="active-job-row__icon">
        <LoaderCircle className="spin" size={17} aria-hidden="true" />
      </div>
      <div>
        <strong>{job.filename}</strong>
        <div className="active-job-row__progress-label">
          <span>{progressLabel}</span>
          <span>{hasPageCount ? `${progressPercent}%` : "Preparing…"}</span>
        </div>
        <div
          className={`indeterminate-progress${hasPageCount ? " indeterminate-progress--determinate" : ""}`}
          role="progressbar"
          aria-label={`${job.filename} ingestion progress`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={hasPageCount ? progressPercent : undefined}
        >
          <span style={hasPageCount ? { width: `${progressPercent}%` } : undefined} />
        </div>
      </div>
    </div>
  );
}

function ActiveJobsPanel({
  username,
  jobs,
  onNotice,
}: {
  username: string;
  jobs: ActiveIngestionJob[];
  onNotice: (notice: WorkspaceNotice) => void;
}) {
  const queryClient = useQueryClient();
  const removeActiveJob = useWorkspaceStore((state) => state.removeActiveJob);
  const jobIds = useMemo(() => jobs.map((job) => job.id).sort(), [jobs]);
  const reportedTerminalJobs = useRef(new Set<string>());
  const reportedPollingFailure = useRef(false);
  const jobsQuery = useQuery({
    queryKey: queryKeys.ingestion.jobs(username, jobIds),
    queryFn: ({ signal }) => getIngestionJobs(username, jobIds, signal),
    enabled: jobIds.length > 0,
    refetchInterval: (query) => {
      const statuses = query.state.data?.jobs.map((job) => job.status) ?? [];
      return statuses.length > 0 && statuses.every((status) => status === "completed" || status === "failed")
        ? false
        : 1750;
    },
    staleTime: 0,
  });
  const remoteById = useMemo(
    () => new Map((jobsQuery.data?.jobs ?? []).map((job) => [job.id, job])),
    [jobsQuery.data?.jobs],
  );

  useEffect(() => {
    const terminal = (jobsQuery.data?.jobs ?? []).filter(
      (job) => job.status === "completed" || job.status === "failed",
    );
    if (terminal.length === 0) return;
    for (const job of terminal) {
      removeActiveJob(username, job.id);
      if (reportedTerminalJobs.current.has(job.id)) continue;
      reportedTerminalJobs.current.add(job.id);
      onNotice(
        job.status === "completed"
          ? {
              kind: "success",
              title: "Ingestion completed",
              message: "Ingestion completed successfully.",
            }
          : {
              kind: "error",
              title: "Ingestion failed",
              message:
                job.failure_message ??
                "Ingestion failed. Review the document status and retry.",
            },
      );
    }
    void queryClient.invalidateQueries({ queryKey: queryKeys.documents.list(username) });
  }, [jobsQuery.data?.jobs, onNotice, queryClient, removeActiveJob, username]);

  useEffect(() => {
    if (!jobsQuery.isError || reportedPollingFailure.current) return;
    reportedPollingFailure.current = true;
    onNotice(errorNotice(jobsQuery.error, "Ingestion status could not be refreshed"));
  }, [jobsQuery.error, jobsQuery.isError, onNotice]);

  return (
    <section className="inspector-section active-jobs" aria-live="polite">
      <div className="section-heading-row">
        <h3>Ingestion jobs</h3>
        <span>{jobs.length} active</span>
      </div>
      {jobs.map((job) => (
        <ActiveJobRow
          key={job.id}
          job={job}
          remoteJob={remoteById.get(job.id)}
        />
      ))}
    </section>
  );
}

function statusLabel(status: DocumentResponse["status"]): string {
  return status.replaceAll("_", " ");
}

export function DocumentInspector({
  username,
  catalog,
  documents,
  isLoadingDocuments,
  isRefreshingModels,
  mobileOpen,
  semanticModel,
  semanticReasoningEffort,
  chatModel,
  chatReasoningEffort,
  collectionScope,
  onSemanticSelection,
  onChatSelection,
  onCollectionScopeChange,
  onRefreshModels,
  onCloseMobile,
  onNotice,
}: DocumentInspectorProps) {
  const queryClient = useQueryClient();
  const [files, setFiles] = useState<File[]>([]);
  const [ingestionMethod, setIngestionMethod] = useState<IngestionMethod>("semantic");
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search.trim().toLocaleLowerCase("en-US"));
  const activeJobsByUsername = useWorkspaceStore((state) => state.activeJobsByUsername);
  const upsertActiveJob = useWorkspaceStore((state) => state.upsertActiveJob);
  const activeUsernameRef = useRef(username);
  activeUsernameRef.current = username;
  const activeJobs = useMemo(
    () =>
      username
        ? Object.values(activeJobsByUsername[normalizeUsernameKey(username)] ?? {})
        : [],
    [activeJobsByUsername, username],
  );
  const activeDocumentIds = useMemo(
    () => new Set(activeJobs.map((job) => job.documentId)),
    [activeJobs],
  );
  const models = catalog?.models ?? [];
  const visibleDocuments = useMemo(
    () =>
      deferredSearch
        ? documents.filter((document) =>
            document.filename.toLocaleLowerCase("en-US").includes(deferredSearch),
          )
        : documents,
    [deferredSearch, documents],
  );

  useEffect(() => {
    setFiles([]);
  }, [username]);

  const onDropAccepted = useCallback((acceptedFiles: File[]) => {
    setFiles((current) => {
      const known = new Set(current.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
      return [
        ...current,
        ...acceptedFiles.filter(
          (file) => !known.has(`${file.name}:${file.size}:${file.lastModified}`),
        ),
      ];
    });
  }, []);
  const onDropRejected = useCallback(() => {
    onNotice({
      kind: "error",
      title: "File not accepted",
      message: "Select PDF, DOCX, or PPTX files no larger than 100 MB each.",
    });
  }, [onNotice]);
  const uploadMutation = useMutation({
    mutationFn: async (submission: IngestionSubmission) => {
      const uploaded = await uploadDocuments({
        username: submission.username,
        files: submission.files,
        configuration: submission.configuration,
      });
      let started: Awaited<ReturnType<typeof startIngestion>>;
      try {
        started = await startIngestion(
          submission.username,
          uploaded.documents.map((document) => document.id),
        );
      } catch (error) {
        throw new IngestionStartAfterUploadError(error, uploaded.documents);
      }
      return { uploaded, started };
    },
    onSuccess: ({ uploaded, started }, submission) => {
      const filenames = new Map(uploaded.documents.map((document) => [document.id, document.filename]));
      for (const job of started.jobs) {
        upsertActiveJob(submission.username, {
          id: job.id,
          documentId: job.document_id,
          filename: filenames.get(job.document_id) ?? "Uploaded document",
          status: job.status === "processing" ? "processing" : "pending",
          createdAt: new Date().toISOString(),
        });
      }
      queryClient.setQueryData<DocumentResponse[]>(
        queryKeys.documents.list(submission.username),
        (current) => mergeDocuments(current, uploaded.documents),
      );
      void queryClient.invalidateQueries({
        queryKey: queryKeys.documents.list(submission.username),
      });
      if (activeUsernameRef.current === submission.username) {
        const submittedFiles = new Set(submission.files);
        setFiles((current) => current.filter((file) => !submittedFiles.has(file)));
        onNotice({ kind: "info", title: "Ingestion started", message: started.message });
      }
    },
    onError: (error, submission) => {
      if (error instanceof IngestionStartAfterUploadError) {
        queryClient.setQueryData<DocumentResponse[]>(
          queryKeys.documents.list(submission.username),
          (current) => mergeDocuments(current, error.uploadedDocuments),
        );
        void queryClient.invalidateQueries({
          queryKey: queryKeys.documents.list(submission.username),
        });
        if (activeUsernameRef.current !== submission.username) {
          return;
        }
        const submittedFiles = new Set(submission.files);
        setFiles((current) => current.filter((file) => !submittedFiles.has(file)));
        const providerNotice = errorNotice(
          error.causeError,
          "Files uploaded; ingestion not started",
        );
        onNotice({
          ...providerNotice,
          message: `${providerNotice.message} Use Retry ingestion on the pending documents; do not upload them again.`,
        });
        return;
      }
      if (activeUsernameRef.current === submission.username) {
        onNotice(errorNotice(error, "Upload or ingestion failed"));
      }
    },
  });

  const retryMutation = useMutation({
    mutationFn: async ({ username: mutationUsername, document }: DocumentMutationVariables) => {
      const started = await startIngestion(mutationUsername, [document.id]);
      return { document, started };
    },
    onSuccess: ({ document, started }, variables) => {
      for (const job of started.jobs) {
        upsertActiveJob(variables.username, {
          id: job.id,
          documentId: document.id,
          filename: document.filename,
          status: job.status === "processing" ? "processing" : "pending",
          createdAt: new Date().toISOString(),
        });
      }
      void queryClient.invalidateQueries({
        queryKey: queryKeys.documents.list(variables.username),
      });
      if (activeUsernameRef.current === variables.username) {
        onNotice({ kind: "info", title: "Ingestion started", message: started.message });
      }
    },
    onError: (error, variables) => {
      if (activeUsernameRef.current === variables.username) {
        onNotice(errorNotice(error, "Ingestion could not be started"));
      }
    },
  });

  const deleteMutation = useMutation({
    mutationFn: ({ username: mutationUsername, document }: DocumentMutationVariables) => {
      return deleteDocument(mutationUsername, document.id);
    },
    onSuccess: (result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.documents.list(variables.username),
      });
      if (activeUsernameRef.current === variables.username) {
        onNotice(
          result.status === "deleted"
            ? { kind: "success", title: "Document deleted", message: result.message }
            : {
                kind: "error",
                title: "Deletion requires retry",
                message: result.message,
              },
        );
      }
    },
    onError: (error, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.documents.list(variables.username),
      });
      if (activeUsernameRef.current === variables.username) {
        onNotice(errorNotice(error, "Document deletion failed"));
      }
    },
  });

  const dropzone = useDropzone({
    onDropAccepted,
    onDropRejected,
    maxSize: MAX_FILE_BYTES,
    disabled: !username || uploadMutation.isPending,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    },
  });

  const canIngest = Boolean(
    username &&
      files.length > 0 &&
      !uploadMutation.isPending &&
      (ingestionMethod === "docling" || (semanticModel && semanticReasoningEffort)),
  );

  function submitIngestion() {
    if (!username || files.length === 0) return;
    let configuration: UploadConfiguration;
    if (ingestionMethod === "semantic") {
      if (!semanticModel || !semanticReasoningEffort) return;
      configuration = {
        ingestionMethod: "semantic",
        semanticModel,
        semanticReasoningEffort,
      };
    } else {
      configuration = { ingestionMethod: "docling" };
    }
    uploadMutation.mutate({ username, files: [...files], configuration });
  }

  return (
    <>
      <aside
        className={`document-inspector${mobileOpen ? " document-inspector--open" : ""}`}
        role={mobileOpen ? "dialog" : undefined}
        aria-modal={mobileOpen ? true : undefined}
        aria-label={mobileOpen ? "Documents and ingestion" : undefined}
      >
        <header className="inspector-header">
          <div>
            <p className="eyebrow">Knowledge base</p>
            <h2>Documents &amp; ingestion</h2>
          </div>
          <button
            type="button"
            className="icon-button inspector-header__close"
            aria-label="Close documents and ingestion"
            onClick={onCloseMobile}
          >
            <X size={19} aria-hidden="true" />
          </button>
        </header>

        <div className="inspector-scroll-region">
          <section className="inspector-section inspector-section--upload">
            <div
              {...dropzone.getRootProps({
                className: `upload-dropzone${dropzone.isDragActive ? " upload-dropzone--active" : ""}`,
              })}
            >
              <input
                {...dropzone.getInputProps({
                  id: "source-documents",
                  name: "source-documents",
                  "aria-label": "Choose source documents",
                })}
              />
              <span className="upload-dropzone__icon">
                <UploadCloud size={22} aria-hidden="true" />
              </span>
              <strong>{dropzone.isDragActive ? "Drop files to add them" : "Upload source documents"}</strong>
              <span>Drag and drop or browse</span>
              <small>PDF, DOCX, PPTX · 100 MB per file</small>
            </div>

            {files.length > 0 ? (
              <div className="pending-files">
                <div className="section-heading-row">
                  <h3>Pending files</h3>
                  <span>{files.length}</span>
                </div>
                {files.map((file) => (
                  <div key={`${file.name}:${file.lastModified}`} className="pending-file-row">
                    <FileText size={16} aria-hidden="true" />
                    <div>
                      <strong>{file.name}</strong>
                      <span>{formatBytes(file.size)}</span>
                    </div>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`Remove ${file.name}`}
                      disabled={uploadMutation.isPending}
                      onClick={() =>
                        setFiles((current) => current.filter((candidate) => candidate !== file))
                      }
                    >
                      <X size={15} aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <section className="inspector-section">
            <div className="section-heading-row">
              <h3>Ingestion configuration</h3>
              <span>{ingestionMethod === "semantic" ? "Semantic" : "Fixed"}</span>
            </div>
            <div className="segmented-control" aria-label="Ingestion method">
              <button
                type="button"
                className={ingestionMethod === "semantic" ? "selected" : ""}
                disabled={uploadMutation.isPending}
                onClick={() => setIngestionMethod("semantic")}
              >
                Semantic chunking
              </button>
              <button
                type="button"
                className={ingestionMethod === "docling" ? "selected" : ""}
                disabled={uploadMutation.isPending}
                onClick={() => setIngestionMethod("docling")}
              >
                Docling fixed
              </button>
            </div>
            {ingestionMethod === "semantic" ? (
              <ModelSelect
                id="semantic"
                label="Semantic chunking"
                models={models}
                value={semanticModel}
                effort={semanticReasoningEffort}
                disabled={!username || uploadMutation.isPending}
                onChange={onSemanticSelection}
              />
            ) : (
              <p className="configuration-note">
                Docling fixed chunking does not call a semantic chunking model.
              </p>
            )}
          </section>

          <section className="inspector-section">
            <div className="section-heading-row">
              <h3>Answer configuration</h3>
              <span>Independent</span>
            </div>
            <ModelSelect
              id="chat"
              label="Chat"
              models={models}
              value={chatModel}
              effort={chatReasoningEffort}
              disabled={!username}
              onChange={onChatSelection}
            />
            <label className="stacked-control">
              <span>Collection scope</span>
              <select
                id="collection-scope"
                name="collection-scope"
                value={collectionScope}
                onChange={(event) => onCollectionScopeChange(event.target.value as CollectionScope)}
              >
                <option value="both">Both collections</option>
                <option value="semantic">Semantic chunks</option>
                <option value="docling">Docling fixed chunks</option>
              </select>
            </label>
            <button
              type="button"
              className="secondary-button model-refresh-button"
              onClick={onRefreshModels}
              disabled={isRefreshingModels}
            >
              <RefreshCw className={isRefreshingModels ? "spin" : ""} size={15} aria-hidden="true" />
              Refresh model access
            </button>
          </section>

          <div className="ingest-action-bar">
            <div>
              <strong>{files.length || "No"} file(s) selected</strong>
              <span>
                Target: {ingestionMethod === "semantic" ? "semantic_chunks" : "docling_fixed_chunks"}
              </span>
            </div>
            <button
              type="button"
              className="primary-button"
              disabled={!canIngest}
              onClick={submitIngestion}
            >
              {uploadMutation.isPending ? (
                <LoaderCircle className="spin" size={16} aria-hidden="true" />
              ) : (
                <UploadCloud size={16} aria-hidden="true" />
              )}
              Ingest
            </button>
          </div>

          {username && activeJobs.length > 0 ? (
            <ActiveJobsPanel username={username} jobs={activeJobs} onNotice={onNotice} />
          ) : null}

          <section className="inspector-section document-library">
            <div className="section-heading-row">
              <h3>Documents</h3>
              <span>{documents.length}</span>
            </div>
            <div className="document-search">
              <Search size={15} aria-hidden="true" />
              <input
                id="document-search"
                name="document-search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search documents"
                aria-label="Search documents"
                disabled={!username}
              />
              <button
                type="button"
                className="icon-button"
                aria-label="Refresh documents"
                onClick={() =>
                  username
                    ? void queryClient.invalidateQueries({
                        queryKey: queryKeys.documents.list(username),
                      })
                    : undefined
                }
              >
                <RefreshCw size={14} aria-hidden="true" />
              </button>
            </div>

            {!username ? (
              <p className="inspector-empty">Enter a username to load documents.</p>
            ) : isLoadingDocuments ? (
              <p className="inspector-empty">
                <LoaderCircle className="spin" size={16} aria-hidden="true" />
                Loading documents…
              </p>
            ) : visibleDocuments.length === 0 ? (
              <p className="inspector-empty">
                {search ? "No documents match your search." : "No documents uploaded yet."}
              </p>
            ) : (
              <div className="document-list">
                {visibleDocuments.map((document) => (
                  <article key={document.id} className="document-row">
                    <div className="document-row__icon">
                      <File size={18} aria-hidden="true" />
                    </div>
                    <div className="document-row__body">
                      <div className="document-row__title">
                        <strong>{document.filename}</strong>
                        <span className={`status-badge status-badge--${document.status}`}>
                          {document.status === "completed" ? (
                            <CheckCircle2 size={11} aria-hidden="true" />
                          ) : null}
                          {statusLabel(document.status)}
                        </span>
                      </div>
                      <span>
                        {document.file_extension.toUpperCase()} · {formatBytes(document.file_size_bytes)} · {document.ingestion_method}
                      </span>
                      <span>{document.collection_name}</span>
                      {document.semantic_model ? (
                        <span>
                          {document.semantic_model} · {document.semantic_reasoning_effort} reasoning
                        </span>
                      ) : null}
                      <time dateTime={document.uploaded_at}>
                        Uploaded {formatTimestamp(document.uploaded_at)}
                      </time>
                      {document.error_message ? <p>{document.error_message}</p> : null}
                    </div>
                    <div className="document-row__actions">
                      {(document.status === "pending" || document.status === "failed") &&
                      !activeDocumentIds.has(document.id) ? (
                        <button
                          type="button"
                          className="text-button"
                          disabled={retryMutation.isPending}
                          onClick={() =>
                            username
                              ? retryMutation.mutate({ username, document })
                              : undefined
                          }
                        >
                          Retry ingestion
                        </button>
                      ) : null}
                      {document.status === "deletion_pending" ? (
                        <button
                          type="button"
                          className="text-button"
                          disabled={deleteMutation.isPending}
                          onClick={() =>
                            username
                              ? deleteMutation.mutate({ username, document })
                              : undefined
                          }
                        >
                          Retry deletion
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="icon-button icon-button--danger"
                          aria-label={`Delete ${document.filename}`}
                          disabled={
                            document.status === "processing" || deleteMutation.isPending
                          }
                          onClick={() => {
                            if (
                              username &&
                              window.confirm(
                                `Delete ${document.filename}? This action removes its vectors and source file.`,
                              )
                            ) {
                              deleteMutation.mutate({ username, document });
                            }
                          }}
                        >
                          <Trash2 size={15} aria-hidden="true" />
                        </button>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      </aside>
      {mobileOpen ? (
        <button
          type="button"
          className="drawer-backdrop drawer-backdrop--right"
          aria-label="Close documents and ingestion"
          onClick={onCloseMobile}
        />
      ) : null}
    </>
  );
}
