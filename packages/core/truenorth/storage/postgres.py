"""Async PostgreSQL storage."""

from __future__ import annotations
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from truenorth.storage.models import Base, Session as SessionModel
from truenorth.core.graph_state import GraphState, FieldValue


class PostgresStore:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_state(self, state: GraphState) -> None:
        async with self.session_factory() as db:
            existing = await db.get(SessionModel, state.session_id)

            # Serialize profile (FieldValue → dict)
            profile_data = {
                k: {"value": v.value, "confidence": v.confidence,
                    "source": v.source, "raw_text": v.raw_text,
                    "privacy_level": v.privacy_level}
                for k, v in state.profile.items()
            }

            if existing:
                existing.profile = profile_data
                existing.conversation = [
                    {"role": t.role, "content": t.content,
                     "timestamp": t.timestamp.isoformat()}
                    for t in state.conversation
                ]
                existing.missing_required = state.missing_required
                existing.missing_optional = state.missing_optional
                existing.emotion_state = state.emotion_state
                existing.output = state.output
                existing.completed = state.completed
                existing.escalated = state.escalated
                existing.cost_usd = state.cost_usd
                existing.tokens_used = state.tokens_used
                existing.updated_at = datetime.utcnow()
            else:
                db.add(SessionModel(
                    id=state.session_id,
                    goal_id=state.goal_id,
                    user_id=state.user_id,
                    profile=profile_data,
                    conversation=[{"role": t.role, "content": t.content} for t in state.conversation],
                    missing_required=state.missing_required,
                    missing_optional=state.missing_optional,
                    emotion_state=state.emotion_state,
                    completed=state.completed,
                    cost_usd=state.cost_usd,
                    tokens_used=state.tokens_used,
                ))
            await db.commit()

    async def load_state(self, session_id: str, config: dict) -> GraphState | None:
        async with self.session_factory() as db:
            row = await db.get(SessionModel, session_id)
            if not row:
                return None

            state = GraphState(
                session_id=row.id,
                goal_id=row.goal_id,
                config=config,
                user_id=row.user_id,
                emotion_state=row.emotion_state,
                completed=row.completed,
                escalated=row.escalated,
                cost_usd=row.cost_usd,
                tokens_used=row.tokens_used,
                output=row.output,
                missing_required=row.missing_required,
                missing_optional=row.missing_optional,
                resumed=True,
            )

            # Deserialize profile
            for k, v in (row.profile or {}).items():
                state.profile[k] = FieldValue(
                    value=v["value"], confidence=v.get("confidence", 0.7),
                    source=v.get("source", "user_stated"),
                    raw_text=v.get("raw_text", ""),
                    privacy_level=v.get("privacy_level", "low"),
                )

            return state
