import { Check, Copy, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface MessageActionsProps {
  content: string;
}

type CopyStatus = "idle" | "copied" | "failed";

const STATUS_RESET_DELAY_MS = 1_800;

export function MessageActions({ content }: MessageActionsProps) {
  const [status, setStatus] = useState<CopyStatus>("idle");
  const resetTimerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    },
    [],
  );

  async function copyAnswer() {
    if (resetTimerRef.current !== null) window.clearTimeout(resetTimerRef.current);
    try {
      await navigator.clipboard.writeText(content);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
    resetTimerRef.current = window.setTimeout(() => {
      setStatus("idle");
      resetTimerRef.current = null;
    }, STATUS_RESET_DELAY_MS);
  }

  const statusLabel = status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : "";

  return (
    <div className="message-actions">
      <button type="button" aria-label="Copy answer" onClick={() => void copyAnswer()}>
        {status === "copied" ? (
          <Check size={14} aria-hidden="true" />
        ) : status === "failed" ? (
          <TriangleAlert size={14} aria-hidden="true" />
        ) : (
          <Copy size={14} aria-hidden="true" />
        )}
        <span>{statusLabel || "Copy"}</span>
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {statusLabel}
      </span>
    </div>
  );
}
