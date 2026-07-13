import "katex/dist/katex.min.css";
import { memo, type ComponentPropsWithoutRef } from "react";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import ReactMarkdown, { type Components } from "react-markdown";

import { normalizeLatexDelimiters } from "../../lib/markdownMath";
import { normalizeEvidenceExcerpt } from "./evidenceNormalization";

interface EvidenceExcerptProps {
  children: string;
  compact?: boolean;
  expanded?: boolean;
}

function SafeLink({ href, children, title }: ComponentPropsWithoutRef<"a">) {
  const external = Boolean(href && /^https?:\/\//i.test(href));
  return (
    <a
      href={href}
      title={title}
      rel={external ? "noreferrer noopener" : undefined}
      target={external ? "_blank" : undefined}
    >
      {children}
    </a>
  );
}

const evidenceComponents: Components = {
  a: SafeLink,
  // Source excerpts are textual evidence. Rendering document-provided image URLs
  // would add tracking and layout risks without improving citation traceability.
  img: () => null,
};

export const EvidenceExcerpt = memo(function EvidenceExcerpt({
  children,
  compact = false,
  expanded = false,
}: EvidenceExcerptProps) {
  const className = [
    "evidence-excerpt",
    compact ? "evidence-excerpt--compact" : "",
    expanded ? "evidence-excerpt--expanded" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className}>
      <ReactMarkdown
        components={evidenceComponents}
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          rehypeRaw,
          rehypeSanitize,
          [rehypeKatex, { throwOnError: false, strict: false }],
        ]}
      >
        {normalizeLatexDelimiters(normalizeEvidenceExcerpt(children))}
      </ReactMarkdown>
    </div>
  );
});
