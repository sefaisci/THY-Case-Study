import { FileText } from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";

import type { CitationResponse } from "../../api/contracts";
import {
  formatCitationAccessibleLabel,
  formatCitationLocation,
  formatRetrievalScore,
} from "../../lib/format";

export interface InlineCitationProps {
  index: number;
  citation: CitationResponse;
}

interface PreviewPosition {
  left: number;
  top: number;
  placement: "above" | "below";
}

const PREVIEW_WIDTH = 352;
const PREVIEW_HEIGHT_FALLBACK = 180;
const VIEWPORT_GUTTER = 12;
const PREVIEW_GAP = 10;

export function InlineCitation({ index, citation }: InlineCitationProps) {
  const previewId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const previewRef = useRef<HTMLElement>(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [position, setPosition] = useState<PreviewPosition>({
    left: VIEWPORT_GUTTER,
    top: VIEWPORT_GUTTER,
    placement: "below",
  });
  const visible = hovered || focused || pinned;

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;

    const triggerRect = trigger.getBoundingClientRect();
    const previewRect = previewRef.current?.getBoundingClientRect();
    const width =
      previewRect?.width || Math.min(PREVIEW_WIDTH, window.innerWidth - VIEWPORT_GUTTER * 2);
    const height = previewRect?.height || PREVIEW_HEIGHT_FALLBACK;
    const halfWidth = Math.max(0, width / 2);
    const minimumCenter = VIEWPORT_GUTTER + halfWidth;
    const maximumCenter = Math.max(
      minimumCenter,
      window.innerWidth - VIEWPORT_GUTTER - halfWidth,
    );
    const triggerCenter = triggerRect.left + triggerRect.width / 2;
    const left = Math.min(Math.max(triggerCenter, minimumCenter), maximumCenter);
    const placement = triggerRect.top >= height + PREVIEW_GAP + VIEWPORT_GUTTER
      ? "above"
      : "below";

    setPosition({
      left,
      top: placement === "above" ? triggerRect.top : triggerRect.bottom,
      placement,
    });
  }, []);

  useLayoutEffect(() => {
    if (!visible) return undefined;
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [updatePosition, visible]);

  useEffect(() => {
    if (!visible) return undefined;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setHovered(false);
      setPinned(false);
      triggerRef.current?.blur();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [visible]);

  useEffect(() => {
    if (!pinned) return undefined;
    const closeOutside = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (triggerRef.current?.contains(target) || previewRef.current?.contains(target)) return;
      setPinned(false);
    };
    document.addEventListener("pointerdown", closeOutside, true);
    return () => document.removeEventListener("pointerdown", closeOutside, true);
  }, [pinned]);

  const previewStyle: CSSProperties = {
    left: position.left,
    top: position.top,
  };
  const label = formatCitationAccessibleLabel(index, citation.filename);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="inline-citation"
        aria-label={label}
        aria-describedby={visible ? previewId : undefined}
        aria-expanded={pinned}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onClick={() => setPinned((current) => !current)}
      >
        [{index}]
      </button>
      {visible
        ? createPortal(
            <aside
              ref={previewRef}
              id={previewId}
              role="tooltip"
              className={`citation-preview citation-preview--${position.placement}`}
              style={previewStyle}
            >
              <div className="citation-preview__header">
                <span>Source {index}</span>
                <strong>
                  <FileText size={15} aria-hidden="true" />
                  {citation.filename}
                </strong>
              </div>
              <div className="citation-preview__metadata">
                <span>{formatCitationLocation(citation)}</span>
                <span>Score {formatRetrievalScore(citation.retrieval_score)}</span>
                <span>
                  {citation.ingestion_method === "semantic" ? "Semantic" : "Docling fixed"}
                </span>
              </div>
              <p>{citation.source_excerpt}</p>
            </aside>,
            document.body,
          )
        : null}
    </>
  );
}
