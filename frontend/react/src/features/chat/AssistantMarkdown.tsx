import "katex/dist/katex.min.css";
import { memo } from "react";
import rehypeKatex from "rehype-katex";
import rehypeSanitize from "rehype-sanitize";
import remarkMath from "remark-math";
import ReactMarkdown from "react-markdown";

import { normalizeLatexDelimiters } from "../../lib/markdownMath";

interface AssistantMarkdownProps {
  children: string;
}

export const AssistantMarkdown = memo(function AssistantMarkdown({
  children,
}: AssistantMarkdownProps) {
  return (
    <div className="assistant-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        // KaTeX runs after sanitization so its generated MathML and classes are
        // preserved without allowing raw model-provided HTML through.
        rehypePlugins={[rehypeSanitize, [rehypeKatex, { throwOnError: false, strict: false }]]}
      >
        {normalizeLatexDelimiters(children)}
      </ReactMarkdown>
    </div>
  );
});
