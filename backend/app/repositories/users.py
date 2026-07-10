"""User persistence queries."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import User


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_username(self, username: str) -> User | None:
        return self.session.scalar(select(User).where(User.username == username))

    def create(self, username: str) -> User:
        user = User(username=username)
        self.session.add(user)
        self.session.flush()
        return user
