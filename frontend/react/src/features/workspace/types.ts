import type { ChatTurnResponse } from "../../api/contracts";

export interface WorkspaceNotice {
  kind: "error" | "success" | "info";
  title: string;
  message: string;
  requestId?: string;
}

export type TurnUsageByMessage = Record<string, ChatTurnResponse>;
