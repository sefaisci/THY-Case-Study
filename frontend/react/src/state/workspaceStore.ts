import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type {
  CollectionScope,
  IngestionJobStatus,
  ReasoningEffort,
} from "../api/contracts";
import { normalizeUsernameKey } from "../app/queryKeys";
import { safeSessionStorage } from "../lib/sessionStorage";

export const WORKSPACE_STORAGE_KEY = "thy-document-intelligence-workspace";

export interface ActiveIngestionJob {
  id: string;
  documentId: string;
  filename: string;
  status: Extract<IngestionJobStatus, "pending" | "processing">;
  createdAt: string;
}

export interface WorkspacePreferences {
  semanticModel: string | null;
  semanticReasoningEffort: ReasoningEffort | null;
  chatModel: string | null;
  chatReasoningEffort: ReasoningEffort | null;
  collectionScope: CollectionScope;
}

type JobsByUsername = Record<string, Record<string, ActiveIngestionJob>>;

export interface WorkspaceState extends WorkspacePreferences {
  username: string | null;
  activeChatId: string | null;
  activeJobsByUsername: JobsByUsername;
  setResolvedUsername: (username: string | null) => void;
  setActiveChatId: (chatId: string | null) => void;
  setSemanticSelection: (model: string | null, effort: ReasoningEffort | null) => void;
  setChatSelection: (model: string | null, effort: ReasoningEffort | null) => void;
  setCollectionScope: (scope: CollectionScope) => void;
  upsertActiveJob: (username: string, job: ActiveIngestionJob) => void;
  removeActiveJob: (username: string, jobId: string) => void;
  clearActiveJobs: (username: string) => void;
  resetWorkspace: () => void;
}

const defaultPreferences: WorkspacePreferences = {
  semanticModel: null,
  semanticReasoningEffort: null,
  chatModel: null,
  chatReasoningEffort: null,
  collectionScope: "both",
};

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      username: null,
      activeChatId: null,
      activeJobsByUsername: {},
      ...defaultPreferences,
      setResolvedUsername: (rawUsername) =>
        set((state) => {
          const username = rawUsername ? normalizeUsernameKey(rawUsername) : null;
          return {
            username,
            activeChatId: username === state.username ? state.activeChatId : null,
          };
        }),
      setActiveChatId: (activeChatId) => set({ activeChatId }),
      setSemanticSelection: (semanticModel, semanticReasoningEffort) =>
        set({ semanticModel, semanticReasoningEffort }),
      setChatSelection: (chatModel, chatReasoningEffort) =>
        set({ chatModel, chatReasoningEffort }),
      setCollectionScope: (collectionScope) => set({ collectionScope }),
      upsertActiveJob: (rawUsername, job) =>
        set((state) => {
          const username = normalizeUsernameKey(rawUsername);
          return {
            activeJobsByUsername: {
              ...state.activeJobsByUsername,
              [username]: {
                ...state.activeJobsByUsername[username],
                [job.id]: job,
              },
            },
          };
        }),
      removeActiveJob: (rawUsername, jobId) =>
        set((state) => {
          const username = normalizeUsernameKey(rawUsername);
          const currentJobs = state.activeJobsByUsername[username];
          if (!currentJobs?.[jobId]) return state;
          const remainingJobs = { ...currentJobs };
          delete remainingJobs[jobId];
          return {
            activeJobsByUsername: {
              ...state.activeJobsByUsername,
              [username]: remainingJobs,
            },
          };
        }),
      clearActiveJobs: (rawUsername) =>
        set((state) => {
          const username = normalizeUsernameKey(rawUsername);
          return {
            activeJobsByUsername: {
              ...state.activeJobsByUsername,
              [username]: {},
            },
          };
        }),
      resetWorkspace: () =>
        set({
          username: null,
          activeChatId: null,
          ...defaultPreferences,
        }),
    }),
    {
      name: WORKSPACE_STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => safeSessionStorage),
      partialize: (state) => ({
        username: state.username,
        activeChatId: state.activeChatId,
        semanticModel: state.semanticModel,
        semanticReasoningEffort: state.semanticReasoningEffort,
        chatModel: state.chatModel,
        chatReasoningEffort: state.chatReasoningEffort,
        collectionScope: state.collectionScope,
        activeJobsByUsername: state.activeJobsByUsername,
      }),
    },
  ),
);

export function selectActiveJobsForUsername(
  state: WorkspaceState,
  username: string | null,
): ActiveIngestionJob[] {
  if (!username) return [];
  return Object.values(state.activeJobsByUsername[normalizeUsernameKey(username)] ?? {});
}
