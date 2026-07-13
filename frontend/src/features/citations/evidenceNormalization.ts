export function normalizeEvidenceExcerpt(value: string): string {
  let normalized = value.replace(/\r\n?/g, "\n").trim();
  const fences = normalized.match(/```/g) ?? [];

  if (fences.length % 2 === 1 && normalized.startsWith("```")) {
    const firstLineEnd = normalized.indexOf("\n");
    const firstLine = firstLineEnd === -1 ? normalized : normalized.slice(0, firstLineEnd);
    const remainder = firstLineEnd === -1 ? "" : normalized.slice(firstLineEnd + 1);
    const afterFence = firstLine.slice(3).trim();
    const looksLikeLanguageTag = /^[A-Za-z0-9_+-]+$/.test(afterFence);
    normalized = [looksLikeLanguageTag ? "" : afterFence, remainder]
      .filter(Boolean)
      .join("\n")
      .trim();
  } else if (fences.length % 2 === 1 && normalized.endsWith("```")) {
    normalized = normalized.slice(0, -3).trim();
  }

  let insideFence = false;
  return normalized
    .split("\n")
    .map((line) => {
      if (line.trimStart().startsWith("```")) {
        insideFence = !insideFence;
        return line;
      }
      if (insideFence || line.includes("`")) return line;
      return line.replace(/([^\n])\s+(#{1,6})\s+(?=\S)/g, "$1\n\n$2 ");
    })
    .join("\n");
}
