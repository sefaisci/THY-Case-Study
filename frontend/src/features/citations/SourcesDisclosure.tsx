import { ChevronDown, FileText, Layers3 } from "lucide-react";
import { useId, useState } from "react";

import type { CitationResponse } from "../../api/contracts";
import {
  formatCitationAccessibleLabel,
  formatCitationLocation,
  formatRetrievalScore,
} from "../../lib/format";

interface SourcesDisclosureProps {
  citations: CitationResponse[];
}

export function SourcesDisclosure({ citations }: SourcesDisclosureProps) {
  const disclosureId = useId();
  const [open, setOpen] = useState(false);
  const [expandedExcerpts, setExpandedExcerpts] = useState<Set<string>>(() => new Set());
  if (citations.length === 0) return null;

  return (
    <section className={`sources-disclosure${open ? " sources-disclosure--open" : ""}`}>
      <button
        type="button"
        className="sources-disclosure__trigger"
        aria-expanded={open}
        aria-controls={`${disclosureId}-list`}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="source-stack" aria-hidden="true">
          {citations.slice(0, 3).map((citation, index) => (
            <span key={citation.chunk_id} style={{ zIndex: citations.length - index }}>
              {index + 1}
            </span>
          ))}
        </span>
        <strong>Sources · {citations.length}</strong>
        <span>Evidence used for this answer</span>
        <ChevronDown size={17} aria-hidden="true" />
      </button>
      {open ? (
        <div id={`${disclosureId}-list`} className="sources-list">
          {citations.map((citation, index) => {
            const excerptExpanded = expandedExcerpts.has(citation.chunk_id);
            return (
            <article
              key={citation.chunk_id}
              id={`${disclosureId}-${citation.chunk_id.replace(/[^A-Za-z0-9_-]/g, "-")}`}
              className="source-card"
              aria-label={formatCitationAccessibleLabel(index + 1, citation.filename)}
            >
              <div className="source-card__index">{index + 1}</div>
              <div className="source-card__body">
                <div className="source-card__heading">
                  <FileText size={16} aria-hidden="true" />
                  <strong>{citation.filename}</strong>
                </div>
                <div className="source-card__metadata">
                  <span>{formatCitationLocation(citation)}</span>
                  <span>Score {formatRetrievalScore(citation.retrieval_score)}</span>
                  <span>{citation.ingestion_method === "semantic" ? "Semantic" : "Docling fixed"}</span>
                  <span className="source-card__collection">
                    <Layers3 size={12} aria-hidden="true" />
                    {citation.source_collection}
                  </span>
                </div>
                <p className={excerptExpanded ? "source-card__excerpt--expanded" : ""}>
                  {citation.source_excerpt}
                </p>
                {citation.source_excerpt.length > 360 ? (
                  <button
                    type="button"
                    className="source-card__excerpt-toggle"
                    aria-expanded={excerptExpanded}
                    onClick={() =>
                      setExpandedExcerpts((current) => {
                        const next = new Set(current);
                        if (next.has(citation.chunk_id)) next.delete(citation.chunk_id);
                        else next.add(citation.chunk_id);
                        return next;
                      })
                    }
                  >
                    {excerptExpanded ? "Show less" : "Show full excerpt"}
                  </button>
                ) : null}
              </div>
            </article>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
