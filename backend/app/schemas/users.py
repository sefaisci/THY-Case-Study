"""Logical username resolution schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserResolveRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str
    created_at: datetime
    created: bool = False
