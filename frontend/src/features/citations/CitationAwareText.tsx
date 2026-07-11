import { Children, type ReactNode } from "react";

import type { CitationResponse } from "../../api/contracts";
import { InlineCitation } from "./InlineCitation";

interface CitationAwareTextProps {
  children: ReactNode;
  citations: CitationResponse[];
}

function decorateText(text: string, citations: CitationResponse[]): ReactNode[] {
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(/\[(\d+)]/g)) {
    const offset = match.index;
    const index = Number(match[1]);
    const citation = citations[index - 1];
    if (offset > cursor) nodes.push(text.slice(cursor, offset));
    nodes.push(
      citation ? (
        <InlineCitation key={`${offset}-${index}`} index={index} citation={citation} />
      ) : (
        match[0]
      ),
    );
    cursor = offset + match[0].length;
  }

  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

export function CitationAwareText({ children, citations }: CitationAwareTextProps) {
  return (
    <>
      {Children.map(children, (child) =>
        typeof child === "string" ? decorateText(child, citations) : child,
      )}
    </>
  );
}
