const OPENING_FENCE_PATTERN = /^( {0,3})(`{3,}|~{3,})[^\r\n]*(?:\r?\n|$)/;
const CLOSING_FENCE_PATTERN = /^( {0,3})(`{3,}|~{3,})[ \t]*(?:\r?\n|$)/;

function isEscaped(source: string, index: number): boolean {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && source[cursor] === "\\"; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function findClosingDelimiter(source: string, delimiter: string, fromIndex: number): number {
  let cursor = source.indexOf(delimiter, fromIndex);
  while (cursor !== -1) {
    if (!isEscaped(source, cursor)) return cursor;
    cursor = source.indexOf(delimiter, cursor + delimiter.length);
  }
  return -1;
}

function inlineCodeEnd(source: string, startIndex: number): number | null {
  let openingLength = 1;
  while (source[startIndex + openingLength] === "`") openingLength += 1;

  let cursor = startIndex + openingLength;
  while (cursor < source.length) {
    const candidate = source.indexOf("`", cursor);
    if (candidate === -1) return null;

    let candidateLength = 1;
    while (source[candidate + candidateLength] === "`") candidateLength += 1;
    if (candidateLength === openingLength) return candidate + candidateLength;
    cursor = candidate + candidateLength;
  }
  return null;
}

function fencedCodeEnd(source: string, startIndex: number): number | null {
  if (startIndex > 0 && source[startIndex - 1] !== "\n") return null;

  const opening = OPENING_FENCE_PATTERN.exec(source.slice(startIndex));
  const marker = opening?.[2];
  if (!opening || !marker) return null;

  const markerCharacter = marker[0];
  if (!markerCharacter) return null;

  let cursor = startIndex + opening[0].length;
  while (cursor < source.length) {
    const closing = CLOSING_FENCE_PATTERN.exec(source.slice(cursor));
    const closingMarker = closing?.[2];
    if (
      closing &&
      closingMarker?.[0] === markerCharacter &&
      closingMarker.length >= marker.length
    ) {
      return cursor + closing[0].length;
    }

    const nextLine = source.indexOf("\n", cursor);
    if (nextLine === -1) return source.length;
    cursor = nextLine + 1;
  }

  return source.length;
}

function displayMathReplacement(source: string, startIndex: number, endIndex: number): string {
  const content = source.slice(startIndex + 2, endIndex).trim();
  const lineStart = source.lastIndexOf("\n", startIndex - 1) + 1;
  const nextLineBreak = source.indexOf("\n", endIndex + 2);
  const lineEnd = nextLineBreak === -1 ? source.length : nextLineBreak;
  const beginsOwnLine = source.slice(lineStart, startIndex).trim().length === 0;
  const endsOwnLine = source.slice(endIndex + 2, lineEnd).trim().length === 0;
  const rendered = `$$\n${content}\n$$`;

  return beginsOwnLine && endsOwnLine ? rendered : `\n\n${rendered}\n\n`;
}

/**
 * remark-math intentionally follows Markdown's dollar-delimiter syntax. OpenAI
 * responses also commonly use LaTeX's \(...\) and \[...\] delimiters, which
 * Markdown otherwise treats as escaped punctuation. Normalize only paired
 * delimiters outside code so examples and source snippets remain verbatim.
 */
export function normalizeLatexDelimiters(source: string): string {
  let output = "";
  let cursor = 0;

  while (cursor < source.length) {
    const fenceEnd = fencedCodeEnd(source, cursor);
    if (fenceEnd !== null) {
      output += source.slice(cursor, fenceEnd);
      cursor = fenceEnd;
      continue;
    }

    if (source[cursor] === "`") {
      const codeEnd = inlineCodeEnd(source, cursor);
      if (codeEnd !== null) {
        output += source.slice(cursor, codeEnd);
        cursor = codeEnd;
        continue;
      }
    }

    if (source[cursor] === "\\" && !isEscaped(source, cursor)) {
      const opening = source[cursor + 1];
      if (opening === "(" || opening === "[") {
        const closingDelimiter = opening === "(" ? "\\)" : "\\]";
        const endIndex = findClosingDelimiter(source, closingDelimiter, cursor + 2);
        if (endIndex !== -1) {
          const content = source.slice(cursor + 2, endIndex).trim();
          output +=
            opening === "("
              ? `$${content}$`
              : displayMathReplacement(source, cursor, endIndex);
          cursor = endIndex + closingDelimiter.length;
          continue;
        }
      }
    }

    output += source[cursor];
    cursor += 1;
  }

  return output;
}
