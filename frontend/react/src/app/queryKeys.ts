import type { UsageQuery } from "../api/contracts";

export function normalizeUsernameKey(username: string): string {
  return username.normalize("NFKC").trim().toLocaleLowerCase("en-US");
}

function ownerKey(username: string) {
  return ["owner", normalizeUsernameKey(username)] as const;
}

export const queryKeys = {
  operations: {
    health: () => ["operations", "health"] as const,
    readiness: () => ["operations", "readiness"] as const,
  },
  models: {
    catalog: () => ["models", "catalog"] as const,
  },
  owner: (username: string) => ownerKey(username),
  documents: {
    list: (username: string) => [...ownerKey(username), "documents"] as const,
  },
  ingestion: {
    job: (username: string, jobId: string) =>
      [...ownerKey(username), "ingestion-job", jobId] as const,
  },
  chats: {
    sessions: (username: string) => [...ownerKey(username), "chat-sessions"] as const,
    messages: (username: string, sessionId: string) =>
      [...ownerKey(username), "chat-session", sessionId, "messages"] as const,
  },
  usage: {
    summary: (username: string, query: UsageQuery = {}) =>
      [
        ...ownerKey(username),
        "usage",
        query.sessionId ?? null,
        query.messageId ?? null,
      ] as const,
  },
};
