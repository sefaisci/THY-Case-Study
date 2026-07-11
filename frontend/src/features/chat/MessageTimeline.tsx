import { LoaderCircle, UserRound } from "lucide-react";
import { lazy, Suspense, useEffect, useRef } from "react";

import type { ChatMessageResponse } from "../../api/contracts";
import { BrandMark } from "../../components/Brand";
import { SourcesDisclosure } from "../citations/SourcesDisclosure";
import { UsageDisclosure } from "../usage/UsageDisclosure";
import type { TurnUsageByMessage } from "../workspace/types";
import { MessageActions } from "./MessageActions";

const AssistantMarkdown = lazy(async () => {
  const module = await import("./AssistantMarkdown");
  return { default: module.AssistantMarkdown };
});

interface MessageTimelineProps {
  username: string;
  sessionId: string;
  messages: ChatMessageResponse[];
  isLoading: boolean;
  pendingQuestion: string | null;
  turnUsageByMessage: TurnUsageByMessage;
}

export function MessageTimeline({
  username,
  sessionId,
  messages,
  isLoading,
  pendingQuestion,
  turnUsageByMessage,
}: MessageTimelineProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages.length, pendingQuestion]);

  if (isLoading) {
    return (
      <div className="chat-loading" role="status">
        <LoaderCircle className="spin" size={20} aria-hidden="true" />
        Loading conversation…
      </div>
    );
  }

  if (messages.length === 0 && !pendingQuestion) {
    return (
      <div className="chat-welcome">
        <BrandMark size="large" />
        <p className="eyebrow">Private document workspace</p>
        <h2>What would you like to learn from your documents?</h2>
        <p>
          Ask a focused question. Answers use only evidence retrieved from documents owned by
          this workspace, with visible sources and measured provider usage.
        </p>
      </div>
    );
  }

  return (
    <div className="message-timeline" aria-live="polite">
      {messages.map((message) =>
        message.role === "user" ? (
          <article key={message.id} className="message message--user">
            <div className="message__avatar message__avatar--user">
              <UserRound size={18} aria-hidden="true" />
            </div>
            <div className="message__content">
              <span className="message__author">You</span>
              <p>{message.content}</p>
            </div>
          </article>
        ) : (
          <article key={message.id} className="message message--assistant">
            <div className="message__avatar message__avatar--assistant">
              <BrandMark size="small" />
            </div>
            <div className="message__content">
              <span className="message__author">Cabin Knowledge Assistant</span>
              <Suspense
                fallback={
                  <div
                    className="assistant-markdown assistant-markdown--loading"
                    aria-hidden="true"
                  >
                    {message.content}
                  </div>
                }
              >
                <AssistantMarkdown citations={message.citations}>{message.content}</AssistantMarkdown>
              </Suspense>
              <SourcesDisclosure citations={message.citations} />
              <UsageDisclosure
                username={username}
                sessionId={sessionId}
                message={message}
                turn={turnUsageByMessage[message.id]}
              />
              <MessageActions content={message.content} />
            </div>
          </article>
        ),
      )}
      {pendingQuestion ? (
        <>
          <article className="message message--user message--pending">
            <div className="message__avatar message__avatar--user">
              <UserRound size={18} aria-hidden="true" />
            </div>
            <div className="message__content">
              <span className="message__author">You</span>
              <p>{pendingQuestion}</p>
            </div>
          </article>
          <article className="message message--assistant message--thinking">
            <div className="message__avatar message__avatar--assistant">
              <BrandMark size="small" />
            </div>
            <div className="message__content">
              <span className="message__author">Cabin Knowledge Assistant</span>
              <p>
                <LoaderCircle className="spin" size={16} aria-hidden="true" />
                Retrieving evidence and grounding the answer…
              </p>
            </div>
          </article>
        </>
      ) : null}
      <div ref={endRef} />
    </div>
  );
}
