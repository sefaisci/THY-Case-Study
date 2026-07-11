"""Username normalization and get-or-create behavior."""

from __future__ import annotations

import unicodedata

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import AppError
from ..repositories import UserRepository
from ..schemas.users import UserResponse


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = UserRepository(session)

    @staticmethod
    def normalize_username(raw_username: str) -> str:
        username = unicodedata.normalize("NFKC", raw_username).strip().casefold()
        if not 2 <= len(username) <= 80:
            raise AppError(
                "Username must contain between 2 and 80 characters.",
                code="invalid_username",
                status_code=422,
            )
        if not username[0].isalnum() or any(
            not (character.isalnum() or character in "._-") for character in username
        ):
            raise AppError(
                "Username may contain letters, numbers, periods, underscores, and hyphens, and must start with a letter or number.",
                code="invalid_username",
                status_code=422,
            )
        return username

    async def resolve(self, raw_username: str) -> tuple[object, bool]:
        username = self.normalize_username(raw_username)
        existing = await self.repository.get_by_username(username)
        if existing is not None:
            return existing, False
        try:
            user = await self.repository.create(username)
            await self.session.commit()
            return user, True
        except IntegrityError:
            await self.session.rollback()
            concurrent = await self.repository.get_by_username(username)
            if concurrent is None:
                raise
            return concurrent, False

    def response(self, user: object, created: bool) -> UserResponse:
        return UserResponse.model_validate(user).model_copy(update={"created": created})
