"""Usage record persistence and owner-scoped retrieval."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import UsageRecord


class UsageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, record: UsageRecord) -> UsageRecord:
        self.session.add(record)
        self.session.flush()
        return record

    def list_for_user(
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
        return list(self.session.scalars(statement))
