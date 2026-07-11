"""User persistence queries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_username(self, username: str) -> User | None:
        return await self.session.scalar(select(User).where(User.username == username))

    async def create(self, username: str) -> User:
        user = User(username=username)
        self.session.add(user)
        await self.session.flush()
        return user
