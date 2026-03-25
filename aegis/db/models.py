"""SQLAlchemy ORM models for Project Aegis.

Schema matches dev_plan.md Section 9.2.
Phase 1 tables: users, sessions, conversations, messages,
tool_executions, rate_limits, nim_usage, alerts, portfolio_suggestions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # admin | named_guest | anon_guest
    clearance: Mapped[str | None] = mapped_column(String, nullable=True)  # alpha | beta | null
    owner_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    company: Mapped[str | None] = mapped_column(String, nullable=True)
    totp_secret: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    last_active: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    sessions: Mapped[list[ChatSession]] = relationship(back_populates="user")
    conversations: Mapped[list[Conversation]] = relationship(back_populates="user")


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped[User] = relationship(back_populates="sessions")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)  # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    parameters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    status: Mapped[str | None] = mapped_column(String, nullable=True)  # success|failed|confirmed|rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class RateLimit(Base):
    __tablename__ = "rate_limits"

    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), primary_key=True
    )
    window: Mapped[str] = mapped_column(
        String, primary_key=True  # minute | hour | day
    )
    count: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class NIMUsage(Base):
    __tablename__ = "nim_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class UploadQueue(Base):
    __tablename__ = "upload_queue"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    local_path: Mapped[str] = mapped_column(String, nullable=False)
    drive_path: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|uploading|complete|failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GuestActivity(Base):
    __tablename__ = "guest_activity"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    guest_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    owner_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str | None] = mapped_column(String, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    severity: Mapped[str] = mapped_column(String, nullable=False)  # info | warning | critical
    message: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    emailed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PortfolioSuggestion(Base):
    __tablename__ = "portfolio_suggestions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_generate_uuid)
    owner_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    guest_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending | added | dismissed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PolicyDecision(Base):
    """Audit log for policy engine decisions (ID-04)."""

    __tablename__ = "policy_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    resource: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)  # allow | deny
    rule_matched: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class AuditEvent(Base):
    """Immutable audit log with hash chain (ID-18)."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    actor_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    resource: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    prev_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    entry_hash: Mapped[str] = mapped_column(String, nullable=False)
