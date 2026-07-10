import {
  ChevronLeft,
  LoaderCircle,
  MessageSquareText,
  PanelLeftClose,
  Plus,
  Search,
  UserRound,
} from "lucide-react";
import { useDeferredValue, useMemo, useState, type FormEvent } from "react";

import type { ChatSessionResponse } from "../../api/contracts";
import { BrandLockup } from "../../components/Brand";

interface ConversationRailProps {
  username: string | null;
  usernameDraft: string;
  sessions: ChatSessionResponse[];
  activeChatId: string | null;
  isResolvingUser: boolean;
  isLoadingSessions: boolean;
  isCreatingChat: boolean;
  mobileOpen: boolean;
  onUsernameDraftChange: (value: string) => void;
  onResolveUsername: () => void;
  onCreateChat: () => void;
  onSelectChat: (sessionId: string) => void;
  onCloseMobile: () => void;
}

function isRecent(value: string): boolean {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) && Date.now() - timestamp < 24 * 60 * 60 * 1000;
}

function SessionButton({
  session,
  selected,
  onSelect,
}: {
  session: ChatSessionResponse;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`conversation-item${selected ? " conversation-item--active" : ""}`}
      onClick={onSelect}
      aria-current={selected ? "page" : undefined}
    >
      <MessageSquareText size={16} aria-hidden="true" />
      <span>{session.title}</span>
      <time dateTime={session.last_activity_at}>
        {new Date(session.last_activity_at).toLocaleTimeString("en-US", {
          hour: "numeric",
          minute: "2-digit",
        })}
      </time>
    </button>
  );
}

export function ConversationRail({
  username,
  usernameDraft,
  sessions,
  activeChatId,
  isResolvingUser,
  isLoadingSessions,
  isCreatingChat,
  mobileOpen,
  onUsernameDraftChange,
  onResolveUsername,
  onCreateChat,
  onSelectChat,
  onCloseMobile,
}: ConversationRailProps) {
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search.trim().toLocaleLowerCase("en-US"));
  const visibleSessions = useMemo(
    () =>
      deferredSearch
        ? sessions.filter((session) =>
            session.title.toLocaleLowerCase("en-US").includes(deferredSearch),
          )
        : sessions,
    [deferredSearch, sessions],
  );
  const today = visibleSessions.filter((session) => isRecent(session.last_activity_at));
  const earlier = visibleSessions.filter((session) => !isRecent(session.last_activity_at));

  function submitUsername(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onResolveUsername();
  }

  return (
    <>
      <aside
        className={`conversation-rail${mobileOpen ? " conversation-rail--open" : ""}`}
        role={mobileOpen ? "dialog" : undefined}
        aria-modal={mobileOpen ? true : undefined}
        aria-label={mobileOpen ? "Conversations and workspace identity" : undefined}
      >
        <div className="conversation-rail__brand-row">
          <BrandLockup />
          <button
            type="button"
            className="icon-button conversation-rail__mobile-close"
            aria-label="Close conversations"
            onClick={onCloseMobile}
          >
            <ChevronLeft size={19} aria-hidden="true" />
          </button>
          <PanelLeftClose className="conversation-rail__desktop-mark" size={17} aria-hidden="true" />
        </div>

        <form className="username-form" onSubmit={submitUsername}>
          <UserRound size={16} aria-hidden="true" />
          <input
            id="workspace-username"
            name="username"
            value={usernameDraft}
            onChange={(event) => onUsernameDraftChange(event.target.value)}
            placeholder="Username"
            aria-label="Username"
            autoComplete="username"
            spellCheck="false"
          />
          <button
            type="submit"
            className="username-form__submit"
            disabled={isResolvingUser || usernameDraft.trim().length < 2}
            aria-label="Load private workspace"
          >
            {isResolvingUser ? (
              <LoaderCircle className="spin" size={16} aria-hidden="true" />
            ) : (
              <span>Load</span>
            )}
          </button>
        </form>

        <button
          type="button"
          className="primary-button new-chat-button"
          onClick={onCreateChat}
          disabled={!username || isCreatingChat}
        >
          {isCreatingChat ? (
            <LoaderCircle className="spin" size={17} aria-hidden="true" />
          ) : (
            <Plus size={17} aria-hidden="true" />
          )}
          New chat
        </button>

        <div className="conversation-search">
          <Search size={16} aria-hidden="true" />
          <input
            id="conversation-search"
            name="conversation-search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            disabled={!username}
          />
        </div>

        <nav className="conversation-list" aria-label="Chat sessions">
          {!username ? (
            <p className="rail-empty">Enter a username to open a private document workspace.</p>
          ) : isLoadingSessions ? (
            <div className="rail-loading" aria-label="Loading conversations">
              <span />
              <span />
              <span />
            </div>
          ) : visibleSessions.length === 0 ? (
            <p className="rail-empty">
              {search ? "No conversations match your search." : "No chats yet. Start a new conversation."}
            </p>
          ) : (
            <>
              {today.length > 0 ? <h2>Today</h2> : null}
              {today.map((session) => (
                <SessionButton
                  key={session.id}
                  session={session}
                  selected={session.id === activeChatId}
                  onSelect={() => onSelectChat(session.id)}
                />
              ))}
              {earlier.length > 0 ? <h2>Previous 7 days</h2> : null}
              {earlier.map((session) => (
                <SessionButton
                  key={session.id}
                  session={session}
                  selected={session.id === activeChatId}
                  onSelect={() => onSelectChat(session.id)}
                />
              ))}
            </>
          )}
        </nav>

        <div className="conversation-rail__footer">
          <span className="user-avatar">{username?.slice(0, 1).toUpperCase() ?? "?"}</span>
          <div>
            <strong>{username ?? "No workspace"}</strong>
            <span>{username ? "Private workspace" : "Username required"}</span>
          </div>
        </div>
      </aside>
      {mobileOpen ? (
        <button
          type="button"
          className="drawer-backdrop"
          aria-label="Close conversations"
          onClick={onCloseMobile}
        />
      ) : null}
    </>
  );
}
