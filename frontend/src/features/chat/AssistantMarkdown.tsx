import "katex/dist/katex.min.css";
import { createContext, memo, useContext, type ReactNode } from "react";
import rehypeKatex from "rehype-katex";
import rehypeSanitize from "rehype-sanitize";
import remarkMath from "remark-math";
import ReactMarkdown, { type Components } from "react-markdown";

import type { CitationResponse } from "../../api/contracts";
import { normalizeLatexDelimiters } from "../../lib/markdownMath";
import { CitationAwareText } from "../citations/CitationAwareText";

interface AssistantMarkdownProps {
  children: string;
  citations?: CitationResponse[];
}

const EMPTY_CITATIONS: CitationResponse[] = [];
const CitationContext = createContext<CitationResponse[]>(EMPTY_CITATIONS);

function CitationText({ children }: { children: ReactNode }) {
  const citations = useContext(CitationContext);
  return <CitationAwareText citations={citations}>{children}</CitationAwareText>;
}

const markdownComponents: Components = {
  p: ({ children }) => <p><CitationText>{children}</CitationText></p>,
  li: ({ children }) => <li><CitationText>{children}</CitationText></li>,
  blockquote: ({ children }) => <blockquote><CitationText>{children}</CitationText></blockquote>,
  td: ({ children }) => <td><CitationText>{children}</CitationText></td>,
  th: ({ children }) => <th><CitationText>{children}</CitationText></th>,
  h1: ({ children }) => <h1><CitationText>{children}</CitationText></h1>,
  h2: ({ children }) => <h2><CitationText>{children}</CitationText></h2>,
  h3: ({ children }) => <h3><CitationText>{children}</CitationText></h3>,
  h4: ({ children }) => <h4><CitationText>{children}</CitationText></h4>,
  h5: ({ children }) => <h5><CitationText>{children}</CitationText></h5>,
  h6: ({ children }) => <h6><CitationText>{children}</CitationText></h6>,
  strong: ({ children }) => <strong><CitationText>{children}</CitationText></strong>,
  em: ({ children }) => <em><CitationText>{children}</CitationText></em>,
};

export const AssistantMarkdown = memo(function AssistantMarkdown({
  children,
  citations = EMPTY_CITATIONS,
}: AssistantMarkdownProps) {
  return (
    <div className="assistant-markdown">
      <CitationContext.Provider value={citations}>
        <ReactMarkdown
          components={markdownComponents}
          remarkPlugins={[remarkMath]}
          // KaTeX runs after sanitization so its generated MathML and classes are
          // preserved without allowing raw model-provided HTML through.
          rehypePlugins={[rehypeSanitize, [rehypeKatex, { throwOnError: false, strict: false }]]}
        >
          {normalizeLatexDelimiters(children)}
        </ReactMarkdown>
      </CitationContext.Provider>
    </div>
  );
});
