import { CheckCircle2, Info, TriangleAlert, X } from "lucide-react";

import type { WorkspaceNotice } from "../features/workspace/types";

interface ApiErrorNoticeProps {
  notice: WorkspaceNotice;
  onDismiss: () => void;
}

function NoticeIcon({ kind }: { kind: WorkspaceNotice["kind"] }) {
  if (kind === "success") return <CheckCircle2 size={18} aria-hidden="true" />;
  if (kind === "info") return <Info size={18} aria-hidden="true" />;
  return <TriangleAlert size={18} aria-hidden="true" />;
}

export function ApiErrorNotice({ notice, onDismiss }: ApiErrorNoticeProps) {
  return (
    <div className={`workspace-notice workspace-notice--${notice.kind}`} role="status">
      <NoticeIcon kind={notice.kind} />
      <div>
        <strong>{notice.title}</strong>
        <p>{notice.message}</p>
        {notice.requestId ? (
          <button
            type="button"
            className="workspace-notice__request"
            onClick={() => {
              void navigator.clipboard?.writeText(notice.requestId ?? "");
            }}
            title="Copy request ID"
          >
            Request ID: {notice.requestId}
          </button>
        ) : null}
      </div>
      <button
        type="button"
        className="icon-button workspace-notice__close"
        aria-label="Dismiss notification"
        onClick={onDismiss}
      >
        <X size={17} aria-hidden="true" />
      </button>
    </div>
  );
}
