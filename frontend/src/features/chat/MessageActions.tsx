import { Check, Copy, Timer, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { formatLatency } from "../../lib/format";

interface MessageActionsProps {
  content: string;
  latencyMs: number | null;
}

type CopyStatus = "idle" | "copied" | "failed";

const STATUS_RESET_DELAY_MS = 1_800;

async function writeClipboardText(content: string): Promise<void> {
  let clipboardFailure: unknown;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(content);
      return;
    } catch (error) {
      clipboardFailure = error;
    }
  }

  if (typeof document.execCommand !== "function") {
    throw clipboardFailure instanceof Error
      ? clipboardFailure
      : new Error("Clipboard access is unavailable.");
  }

  const activeElement = document.activeElement instanceof HTMLElement
    ? document.activeElement
    : null;
  const textarea = document.createElement("textarea");
  textarea.value = content;
  textarea.readOnly = true;
  textarea.dataset.copyFallback = "true";
  Object.assign(textarea.style, {
    position: "fixed",
    top: "-1000px",
    left: "-1000px",
    opacity: "0",
  });
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  activeElement?.focus();
  if (!copied) throw new Error("The browser rejected the copy operation.");
}

export function MessageActions({ content, latencyMs }: MessageActionsProps) {
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
      await writeClipboardText(content);
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
  const latencyLabel = latencyMs === null
    ? "Response time unavailable"
    : `Response time ${formatLatency(latencyMs)}`;

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
      <span
        className="message-actions__latency"
        title="End-to-end RAG processing time"
      >
        <Timer size={13} aria-hidden="true" />
        <span>{latencyLabel}</span>
      </span>
    </div>
  );
}
