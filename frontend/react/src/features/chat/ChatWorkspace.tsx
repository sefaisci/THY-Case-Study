import {
  Database,
  FileSearch,
  LoaderCircle,
  Menu,
  PanelRightOpen,
  Send,
  ShieldCheck,
} from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";

import type {
  ChatMessageResponse,
  CollectionScope,
  ReasoningEffort,
} from "../../api/contracts";
import { BrandMark } from "../../components/Brand";
import { MessageTimeline } from "./MessageTimeline";
import type { TurnUsageByMessage } from "../workspace/types";

interface ChatWorkspaceProps {
  username: string | null;
  sessionId: string | null;
  messages: ChatMessageResponse[];
  isLoadingMessages: boolean;
  isSending: boolean;
  pendingQuestion: string | null;
  chatModel: string | null;
  chatReasoningEffort: ReasoningEffort | null;
  collectionScope: CollectionScope;
  turnUsageByMessage: TurnUsageByMessage;
  completedDocumentCount: number;
  isObscured: boolean;
  onSend: (question: string) => void;
  onOpenConversations: () => void;
  onOpenDocuments: () => void;
}

const scopeLabels: Record<CollectionScope, string> = {
  semantic: "Semantic collection",
  docling: "Docling fixed collection",
  both: "Both collections",
};

export function ChatWorkspace({
  username,
  sessionId,
  messages,
  isLoadingMessages,
  isSending,
  pendingQuestion,
  chatModel,
  chatReasoningEffort,
  collectionScope,
  turnUsageByMessage,
  completedDocumentCount,
  isObscured,
  onSend,
  onOpenConversations,
  onOpenDocuments,
}: ChatWorkspaceProps) {
  const [question, setQuestion] = useState("");
  const canSend = Boolean(
    username && chatModel && chatReasoningEffort && question.trim() && !isSending,
  );

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const value = question.trim();
    if (!value || !canSend) return;
    setQuestion("");
    onSend(value);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <main className="chat-workspace" inert={isObscured ? true : undefined}>
      <header className="chat-header">
        <button
          type="button"
          className="icon-button chat-header__mobile-action"
          aria-label="Open conversations"
          onClick={onOpenConversations}
        >
          <Menu size={20} aria-hidden="true" />
        </button>
        <div className="chat-header__title">
          <h1>Cabin Knowledge Assistant</h1>
          <span>
            <ShieldCheck size={13} aria-hidden="true" />
            Private workspace
          </span>
        </div>
        <div className="chat-header__context">
          <span>
            <Database size={14} aria-hidden="true" />
            {scopeLabels[collectionScope]}
          </span>
          <span>{completedDocumentCount} ready document(s)</span>
        </div>
        <button
          type="button"
          className="icon-button chat-header__mobile-action"
          aria-label="Open documents and ingestion"
          onClick={onOpenDocuments}
        >
          <PanelRightOpen size={20} aria-hidden="true" />
        </button>
      </header>

      <div className="chat-scroll-region">
        {!username ? (
          <div className="workspace-gate">
            <BrandMark size="large" />
            <p className="eyebrow">Document-grounded intelligence</p>
            <h2>Open your private workspace</h2>
            <p>
              Enter a username in the conversation panel. Existing users recover their own
              documents and sessions; new usernames create an isolated workspace.
            </p>
          </div>
        ) : sessionId ? (
          <MessageTimeline
            username={username}
            sessionId={sessionId}
            messages={messages}
            isLoading={isLoadingMessages}
            pendingQuestion={pendingQuestion}
            turnUsageByMessage={turnUsageByMessage}
          />
        ) : (
          <div className="workspace-gate workspace-gate--ready">
            <div className="workspace-gate__icon">
              <FileSearch size={25} aria-hidden="true" />
            </div>
            <p className="eyebrow">Workspace ready</p>
            <h2>Start a grounded conversation</h2>
            <p>
              Create a new chat or ask your first question below. Each chat starts with clean
              short-term memory and retrieves only this user&apos;s completed documents.
            </p>
          </div>
        )}
      </div>

      <div className="composer-shell">
        <form className="chat-composer" onSubmit={submit}>
          <label htmlFor="document-question" className="sr-only">
            Ask a question about your documents
          </label>
          <textarea
            id="document-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              username
                ? "Ask a question about your documents…"
                : "Enter a username to begin…"
            }
            disabled={!username || isSending}
            rows={2}
            maxLength={20_000}
          />
          <div className="chat-composer__footer">
            <div className="composer-context">
              <span>
                <Database size={13} aria-hidden="true" />
                {scopeLabels[collectionScope]}
              </span>
              <span>{chatModel ?? "Select a chat model"}</span>
              {chatReasoningEffort ? <span>{chatReasoningEffort} reasoning</span> : null}
            </div>
            <span className="composer-hint">Enter to send · Shift + Enter for a new line</span>
            <button
              type="submit"
              className="composer-send"
              disabled={!canSend}
              aria-label="Send message"
            >
              {isSending ? (
                <LoaderCircle className="spin" size={18} aria-hidden="true" />
              ) : (
                <Send size={18} aria-hidden="true" />
              )}
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
