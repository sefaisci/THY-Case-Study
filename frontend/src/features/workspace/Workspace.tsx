import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError } from "../../api/client";
import { createChatSession, listChatMessages, listChatSessions, sendChatMessage } from "../../api/chats";
import type {
  CatalogModel,
  ChatMessageResponse,
  CollectionScope,
  ReasoningEffort,
} from "../../api/contracts";
import { listDocuments } from "../../api/documents";
import { getModelCatalog } from "../../api/models";
import { resolveUser } from "../../api/users";
import { normalizeUsernameKey, queryKeys } from "../../app/queryKeys";
import { ApiErrorNotice } from "../../components/ApiErrorNotice";
import { useWorkspaceStore } from "../../state/workspaceStore";
import { ChatWorkspace } from "../chat/ChatWorkspace";
import { DocumentInspector } from "../documents/DocumentInspector";
import { ConversationRail } from "../navigation/ConversationRail";
import type { TurnUsageByMessage, WorkspaceNotice } from "./types";

interface ChatSubmission {
  username: string;
  question: string;
  sessionId: string | null;
  model: string;
  reasoningEffort: ReasoningEffort;
  collectionScope: CollectionScope;
}

function selectModel(
  models: CatalogModel[],
  requestedModel: string | null,
  requestedEffort: ReasoningEffort | null,
): { model: string | null; effort: ReasoningEffort | null } {
  const model = models.find((candidate) => candidate.id === requestedModel) ?? models[0];
  if (!model) return { model: null, effort: null };
  const effort =
    requestedEffort && model.reasoning_efforts.includes(requestedEffort)
      ? requestedEffort
      : (model.reasoning_efforts[0] ?? null);
  return { model: model.id, effort };
}

function mutationError(error: unknown, title: string): WorkspaceNotice {
  if (error instanceof ApiError) {
    return {
      kind: "error",
      title,
      message: error.message,
      requestId: error.requestId,
    };
  }
  return { kind: "error", title, message: "The operation could not be completed." };
}

export function Workspace() {
  const queryClient = useQueryClient();
  const username = useWorkspaceStore((state) => state.username);
  const activeChatId = useWorkspaceStore((state) => state.activeChatId);
  const storedSemanticModel = useWorkspaceStore((state) => state.semanticModel);
  const storedSemanticEffort = useWorkspaceStore((state) => state.semanticReasoningEffort);
  const storedChatModel = useWorkspaceStore((state) => state.chatModel);
  const storedChatEffort = useWorkspaceStore((state) => state.chatReasoningEffort);
  const collectionScope = useWorkspaceStore((state) => state.collectionScope);
  const activeJobsByUsername = useWorkspaceStore((state) => state.activeJobsByUsername);
  const setResolvedUsername = useWorkspaceStore((state) => state.setResolvedUsername);
  const setActiveChatId = useWorkspaceStore((state) => state.setActiveChatId);
  const setSemanticSelection = useWorkspaceStore((state) => state.setSemanticSelection);
  const setChatSelection = useWorkspaceStore((state) => state.setChatSelection);
  const setCollectionScope = useWorkspaceStore((state) => state.setCollectionScope);

  const [usernameDraft, setUsernameDraft] = useState(username ?? "");
  const [notice, setNotice] = useState<WorkspaceNotice | null>(null);
  const [leftMobileOpen, setLeftMobileOpen] = useState(false);
  const [rightMobileOpen, setRightMobileOpen] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [turnUsageByMessage, setTurnUsageByMessage] = useState<TurnUsageByMessage>({});

  const modelQuery = useQuery({
    queryKey: queryKeys.models.catalog(),
    queryFn: ({ signal }) => getModelCatalog({ signal }),
    staleTime: 5 * 60_000,
  });
  const hasActiveIngestionJobs = Boolean(
    username &&
      Object.keys(activeJobsByUsername[normalizeUsernameKey(username)] ?? {}).length > 0,
  );
  const documentQuery = useQuery({
    queryKey: username ? queryKeys.documents.list(username) : ["owner", "none", "documents"],
    queryFn: ({ signal }) => listDocuments(username ?? "", signal),
    enabled: Boolean(username),
    refetchInterval: (query) =>
      hasActiveIngestionJobs ||
      query.state.data?.some((document) => document.status === "processing")
        ? 3000
        : false,
  });
  const sessionQuery = useQuery({
    queryKey: username ? queryKeys.chats.sessions(username) : ["owner", "none", "sessions"],
    queryFn: ({ signal }) => listChatSessions(username ?? "", signal),
    enabled: Boolean(username),
  });

  const models = modelQuery.data?.models ?? [];
  const semanticSelection = selectModel(models, storedSemanticModel, storedSemanticEffort);
  const chatSelection = selectModel(models, storedChatModel, storedChatEffort);
  const sessions = sessionQuery.data ?? [];
  const effectiveChatId =
    sessions.find((session) => session.id === activeChatId)?.id ?? sessions[0]?.id ?? null;
  const messageQuery = useQuery({
    queryKey:
      username && effectiveChatId
        ? queryKeys.chats.messages(username, effectiveChatId)
        : ["owner", "none", "messages"],
    queryFn: ({ signal }) => listChatMessages(username ?? "", effectiveChatId ?? "", signal),
    enabled: Boolean(username && effectiveChatId),
  });

  const showNotice = useCallback((nextNotice: WorkspaceNotice) => {
    setNotice(nextNotice);
  }, []);

  useEffect(() => {
    if (!notice || notice.kind === "error") return undefined;
    const timeout = window.setTimeout(() => setNotice(null), 6000);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  useEffect(() => {
    if (!leftMobileOpen && !rightMobileOpen) return undefined;
    const closeDrawer = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setLeftMobileOpen(false);
      setRightMobileOpen(false);
    };
    window.addEventListener("keydown", closeDrawer);
    return () => window.removeEventListener("keydown", closeDrawer);
  }, [leftMobileOpen, rightMobileOpen]);

  const resolveMutation = useMutation({
    mutationFn: (value: string) => resolveUser(value),
    onSuccess: (resolved) => {
      const previousUsername = username;
      if (previousUsername && previousUsername !== resolved.username) {
        void queryClient.cancelQueries({ queryKey: queryKeys.owner(previousUsername) });
        queryClient.removeQueries({ queryKey: queryKeys.owner(previousUsername) });
      }
      setResolvedUsername(resolved.username);
      setUsernameDraft(resolved.username);
      setTurnUsageByMessage({});
      setPendingQuestion(null);
      showNotice({
        kind: "success",
        title: resolved.created ? "Workspace created" : "Workspace loaded",
        message: resolved.created
          ? `A private workspace was created for ${resolved.username}.`
          : `Documents and chats for ${resolved.username} are now available.`,
      });
    },
    onError: (error) => showNotice(mutationError(error, "Username could not be resolved")),
  });

  const createChatMutation = useMutation({
    mutationFn: (mutationUsername: string) => createChatSession(mutationUsername),
    onSuccess: (session, mutationUsername) => {
      queryClient.setQueryData(queryKeys.chats.sessions(mutationUsername), (current: typeof sessions | undefined) => [
        session,
        ...(current ?? []).filter((item) => item.id !== session.id),
      ]);
      queryClient.setQueryData(queryKeys.chats.messages(mutationUsername, session.id), []);
      if (useWorkspaceStore.getState().username === mutationUsername) {
        setActiveChatId(session.id);
        setLeftMobileOpen(false);
      }
    },
    onError: (error, mutationUsername) => {
      if (useWorkspaceStore.getState().username === mutationUsername) {
        showNotice(mutationError(error, "Chat could not be created"));
      }
    },
  });

  const sendMutation = useMutation({
    mutationFn: async (submission: ChatSubmission) => {
      const sessionId =
        submission.sessionId ?? (await createChatSession(submission.username)).id;
      const turn = await sendChatMessage({
        username: submission.username,
        sessionId,
        request: {
          question: submission.question,
          chat_model: submission.model,
          chat_reasoning_effort: submission.reasoningEffort,
          collection_scope: submission.collectionScope,
        },
      });
      return { submission, sessionId, turn };
    },
    onSuccess: ({ submission, sessionId, turn }) => {
      if (useWorkspaceStore.getState().username === submission.username) {
        setActiveChatId(sessionId);
      }
      queryClient.setQueryData<ChatMessageResponse[]>(
        queryKeys.chats.messages(submission.username, sessionId),
        (current) => {
          const existingIds = new Set((current ?? []).map((message) => message.id));
          return [
            ...(current ?? []),
            ...[turn.user_message, turn.assistant_message].filter(
              (message) => !existingIds.has(message.id),
            ),
          ];
        },
      );
      setTurnUsageByMessage((current) => ({
        ...current,
        [turn.assistant_message.id]: turn,
      }));
      void queryClient.invalidateQueries({
        queryKey: queryKeys.chats.sessions(submission.username),
      });
    },
    onError: (error, submission) => {
      if (useWorkspaceStore.getState().username === submission.username) {
        showNotice(mutationError(error, "Answer generation failed"));
      }
    },
    onSettled: () => setPendingQuestion(null),
  });

  const refreshModelsMutation = useMutation({
    mutationFn: () => getModelCatalog({ refresh: true }),
    onSuccess: (catalog) => {
      queryClient.setQueryData(queryKeys.models.catalog(), catalog);
      showNotice({
        kind: "success",
        title: "Model access refreshed",
        message: `${catalog.models.length} model(s) are available to this OpenAI project.`,
      });
    },
    onError: (error) => showNotice(mutationError(error, "Model access could not be refreshed")),
  });

  const handleSemanticSelection = useCallback(
    (model: string | null, effort: ReasoningEffort | null) => setSemanticSelection(model, effort),
    [setSemanticSelection],
  );
  const handleChatSelection = useCallback(
    (model: string | null, effort: ReasoningEffort | null) => setChatSelection(model, effort),
    [setChatSelection],
  );
  const documents = useMemo(() => documentQuery.data ?? [], [documentQuery.data]);
  const completedDocumentCount = useMemo(
    () => documents.reduce((count, document) => count + (document.status === "completed" ? 1 : 0), 0),
    [documents],
  );

  function handleResolveUsername() {
    const value = usernameDraft.trim();
    if (value.length >= 2) resolveMutation.mutate(value);
  }

  function handleSend(question: string) {
    if (
      sendMutation.isPending ||
      !username ||
      !chatSelection.model ||
      !chatSelection.effort
    ) {
      return;
    }
    setPendingQuestion(question);
    sendMutation.mutate({
      username,
      question,
      sessionId: effectiveChatId,
      model: chatSelection.model,
      reasoningEffort: chatSelection.effort,
      collectionScope,
    });
  }

  return (
    <div className="app-shell">
      <ConversationRail
        username={username}
        usernameDraft={usernameDraft}
        sessions={sessions}
        activeChatId={effectiveChatId}
        isResolvingUser={resolveMutation.isPending}
        isLoadingSessions={sessionQuery.isLoading}
        isCreatingChat={createChatMutation.isPending}
        mobileOpen={leftMobileOpen}
        onUsernameDraftChange={setUsernameDraft}
        onResolveUsername={handleResolveUsername}
        onCreateChat={() => {
          if (username) createChatMutation.mutate(username);
        }}
        onSelectChat={(sessionId) => {
          setActiveChatId(sessionId);
          setLeftMobileOpen(false);
        }}
        onCloseMobile={() => setLeftMobileOpen(false)}
      />

      <ChatWorkspace
        username={username}
        sessionId={effectiveChatId}
        messages={messageQuery.data ?? []}
        isLoadingMessages={messageQuery.isLoading}
        isSending={sendMutation.isPending}
        pendingQuestion={pendingQuestion}
        chatModel={chatSelection.model}
        chatReasoningEffort={chatSelection.effort}
        collectionScope={collectionScope}
        turnUsageByMessage={turnUsageByMessage}
        completedDocumentCount={completedDocumentCount}
        isObscured={leftMobileOpen || rightMobileOpen}
        onSend={handleSend}
        onOpenConversations={() => setLeftMobileOpen(true)}
        onOpenDocuments={() => setRightMobileOpen(true)}
      />

      <DocumentInspector
        username={username}
        catalog={modelQuery.data}
        documents={documents}
        isLoadingDocuments={documentQuery.isLoading}
        isRefreshingModels={refreshModelsMutation.isPending}
        mobileOpen={rightMobileOpen}
        semanticModel={semanticSelection.model}
        semanticReasoningEffort={semanticSelection.effort}
        chatModel={chatSelection.model}
        chatReasoningEffort={chatSelection.effort}
        collectionScope={collectionScope}
        onSemanticSelection={handleSemanticSelection}
        onChatSelection={handleChatSelection}
        onCollectionScopeChange={setCollectionScope}
        onRefreshModels={() => refreshModelsMutation.mutate()}
        onCloseMobile={() => setRightMobileOpen(false)}
        onNotice={showNotice}
      />

      {notice ? <ApiErrorNotice notice={notice} onDismiss={() => setNotice(null)} /> : null}
    </div>
  );
}
