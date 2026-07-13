"""Owner-scoped chat session and message queries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, user_id: str, title: str, identifier: str) -> ChatSession:
        chat = ChatSession(user_id=user_id, title=title, session_identifier=identifier)
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def list_sessions(self, user_id: str) -> list[ChatSession]:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.last_activity_at.desc(), ChatSession.id)
        )
        return list(await self.session.scalars(statement))

    async def get_owned(self, user_id: str, session_id: str) -> ChatSession | None:
        return await self.session.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )

    async def list_messages(self, session_id: str, *, limit: int | None = None) -> list[ChatMessage]:
        if limit == 0:
            return []
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
        )
        if limit is None:
            return list(
                await self.session.scalars(
                    statement.order_by(ChatMessage.created_at, ChatMessage.id)
                )
            )
        newest_first = list(
            await self.session.scalars(
                statement.order_by(
                    ChatMessage.created_at.desc(),
                    ChatMessage.id.desc(),
                ).limit(limit)
            )
        )
        newest_first.reverse()
        return newest_first

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        latency_ms: int | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations or [],
            model=model,
            reasoning_effort=reasoning_effort,
            latency_ms=latency_ms,
        )
        self.session.add(message)
        await self.session.flush()
        return message
