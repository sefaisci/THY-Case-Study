import type { CitationResponse } from "../api/contracts";

const integerFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function formatInteger(value: number): string {
  return integerFormatter.format(value);
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "Not available";
  if (bytes < 1024) return `${formatInteger(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"] as const;
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: value >= 10 ? 1 : 2,
  })} ${units[unitIndex]}`;
}

export function formatKnownCost(value: number): string {
  const fractionDigits = value > 0 && value < 0.01 ? 6 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: 6,
  }).format(value);
}

export function formatRetrievalScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "Not available";
}

export function formatLatency(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "Not available";
  if (value < 1_000) return `${Math.round(value)} ms`;
  const totalSeconds = value / 1_000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toLocaleString("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} s`;
  }
  const roundedTotalSeconds = Math.round(totalSeconds);
  const minutes = Math.floor(roundedTotalSeconds / 60);
  const seconds = roundedTotalSeconds % 60;
  return `${minutes} min ${seconds} s`;
}

export function formatCitationLocation(
  citation: Pick<CitationResponse, "page_number" | "slide_number">,
): string {
  if (citation.page_number !== null) return `Page ${citation.page_number}`;
  if (citation.slide_number !== null) return `Slide ${citation.slide_number}`;
  return "Location unavailable";
}

export function formatCitationAccessibleLabel(index: number, filename: string): string {
  return `Source ${index}: ${filename}`;
}

export function formatTimestamp(value: string, locale = "en-US"): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}
