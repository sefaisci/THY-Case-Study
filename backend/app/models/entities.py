"""SQLAlchemy models for users, documents, jobs, chats, and usage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def uuid_string() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)

    documents: Mapped[list["Document"]] = relationship(back_populates="user")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="user")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
        Index("ix_documents_user_sha256", "user_id", "sha256"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(160), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(12), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    ingestion_method: Mapped[str] = mapped_column(String(24), nullable=False)
    semantic_model: Mapped[str | None] = mapped_column(String(120))
    semantic_reasoning_effort: Mapped[str | None] = mapped_column(String(16))
    collection_name: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="documents")
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="document")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="document")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        Index(
            "uq_ingestion_jobs_active_document",
            "document_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'processing')"),
            sqlite_where=text("status IN ('pending', 'processing')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    total_pages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_pages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_message: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    document: Mapped[Document] = relationship(back_populates="ingestion_jobs")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="ingestion_job")


class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("session_identifier", name="uq_chat_session_identifier"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    session_identifier: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(200), default="New chat", nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="chat_session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_chat_messages_latency_ms_nonnegative",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    model: Mapped[str | None] = mapped_column(String(120))
    reasoning_effort: Mapped[str | None] = mapped_column(String(16))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="chat_message")


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_user_created", "user_id", "created_at"),
        Index("ix_usage_session_created", "chat_session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )
    ingestion_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="SET NULL"), index=True
    )
    chat_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"), index=True
    )
    chat_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="SET NULL"), index=True
    )
    operation: Mapped[str] = mapped_column(String(24), nullable=False)
    stage: Mapped[str] = mapped_column(String(80), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(120))
    reasoning_effort: Mapped[str | None] = mapped_column(String(16))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    pricing_version: Mapped[str | None] = mapped_column(String(80))
    pricing_status: Mapped[str] = mapped_column(String(32), default="not_applicable")
    provider_request_id: Mapped[str | None] = mapped_column(String(160))
    details: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="usage_records")
    document: Mapped[Document | None] = relationship(back_populates="usage_records")
    ingestion_job: Mapped[IngestionJob | None] = relationship(back_populates="usage_records")
    chat_session: Mapped[ChatSession | None] = relationship(back_populates="usage_records")
    chat_message: Mapped[ChatMessage | None] = relationship(back_populates="usage_records")
