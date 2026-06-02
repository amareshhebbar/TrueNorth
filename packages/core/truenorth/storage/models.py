"""SQLAlchemy ORM models."""

from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Float, Boolean, Integer, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import uuid


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    goal_id: Mapped[str] = mapped_column(String(100), index=True)
    user_id: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)

    # State (stored as JSON)
    profile: Mapped[dict] = mapped_column(JSON, default=dict)
    conversation: Mapped[list] = mapped_column(JSON, default=list)
    missing_required: Mapped[list] = mapped_column(JSON, default=list)
    missing_optional: Mapped[list] = mapped_column(JSON, default=list)

    # Intelligence signals
    emotion_state: Mapped[str] = mapped_column(String(50), default="neutral")
    conversation_quality_score: Mapped[float] = mapped_column(Float, default=1.0)

    # Output
    output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_version: Mapped[int] = mapped_column(Integer, default=0)

    # Lifecycle
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Cost
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                   onupdate=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    event_type: Mapped[str] = mapped_column(String(100))  # field_set, llm_call, session_created
    field_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    value_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # SHA-256, never raw
    log_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
