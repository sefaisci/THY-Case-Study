"""Usage record persistence and owner-scoped retrieval."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import UsageRecord


class UsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, record: UsageRecord) -> UsageRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def list_for_user(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        message_id: str | None = None,
        limit: int = 200,
    ) -> list[UsageRecord]:
        statement = select(UsageRecord).where(UsageRecord.user_id == user_id)
        if session_id:
            statement = statement.where(UsageRecord.chat_session_id == session_id)
        if message_id:
            statement = statement.where(UsageRecord.chat_message_id == message_id)
        statement = statement.order_by(UsageRecord.created_at.desc()).limit(limit)
        return list(await self.session.scalars(statement))
