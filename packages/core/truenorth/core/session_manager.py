"""Session lifecycle management."""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timedelta
from truenorth.core.graph_state import GraphState, FieldValue, ConversationTurn
from truenorth.storage.postgres import PostgresStore
from truenorth.storage.redis_store import RedisStore

logger = logging.getLogger(__name__)
SESSION_CACHE_TTL = 3600


class SessionManager:
    def __init__(self, postgres: PostgresStore, redis: RedisStore):
        self.pg = postgres
        self.redis = redis

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    async def create(self, goal_id: str, user_id: str | None = None) -> str:
        session_id = self.new_session_id()
        await self.pg.create_session(session_id=session_id, goal_id=goal_id, user_id=user_id)
        logger.info(f"Created session {session_id}")
        return session_id

    async def load(self, session_id: str) -> GraphState | None:
        cached = await self.redis.get(f"session:{session_id}")
        if cached:
            return self._deserialize(cached)
        row = await self.pg.get_session(session_id)
        if not row:
            return None
        state = self._row_to_state(row)
        await self.redis.set(f"session:{session_id}", self._serialize(state), ttl=SESSION_CACHE_TTL)
        return state

    async def save(self, state: GraphState) -> None:
        serialized = self._serialize(state)
        await self.redis.set(f"session:{state.session_id}", serialized, ttl=SESSION_CACHE_TTL)
        await self.pg.update_session(
            session_id=state.session_id,
            profile=state.get_profile_values(),
            state_json=serialized,
            emotion_state=state.emotion_state,
            completed=state.completed,
            escalated=state.escalated,
            cost_usd=state.cost_usd,
            tokens_used=state.tokens_used,
        )

    async def is_resumable(self, session_id: str, ttl_hours: int = 168) -> bool:
        row = await self.pg.get_session(session_id)
        if not row:
            return False
        cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
        return row["updated_at"] > cutoff and not row.get("completed", False)

    async def count_today(self, user_id: str, goal_id: str) -> int:
        return await self.pg.count_sessions_today(user_id=user_id, goal_id=goal_id)

    def _serialize(self, state: GraphState) -> str:
        return json.dumps({
            "session_id": state.session_id,
            "goal_id": state.goal_id,
            "user_id": state.user_id,
            "config": state.config,
            "profile": {
                k: {"value": v.value, "confidence": v.confidence, "source": v.source,
                    "raw_text": v.raw_text, "timestamp": v.timestamp.isoformat(),
                    "privacy_level": v.privacy_level, "needs_confirmation": v.needs_confirmation}
                for k, v in state.profile.items()
            },
            "conversation": [
                {"role": t.role, "content": t.content, "timestamp": t.timestamp.isoformat(),
                 "emotion_state": t.emotion_state, "tokens_used": t.tokens_used, "cost_usd": t.cost_usd}
                for t in state.conversation
            ],
            "missing_required": state.missing_required,
            "missing_optional": state.missing_optional,
            "emotion_state": state.emotion_state,
            "conversation_quality_score": state.conversation_quality_score,
            "abandonment_risk": state.abandonment_risk,
            "consecutive_confusion_turns": state.consecutive_confusion_turns,
            "completed": state.completed,
            "escalated": state.escalated,
            "escalation_reason": state.escalation_reason,
            "output": state.output,
            "detected_language": state.detected_language,
            "tokens_used": state.tokens_used,
            "cost_usd": state.cost_usd,
            "budget_exceeded": state.budget_exceeded,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        })

    def _deserialize(self, data: str) -> GraphState:
        d = json.loads(data)
        state = GraphState(session_id=d["session_id"], goal_id=d["goal_id"],
                           user_id=d.get("user_id"), config=d.get("config", {}))
        state.profile = {
            k: FieldValue(value=v["value"], confidence=v["confidence"], source=v["source"],
                          raw_text=v.get("raw_text", ""),
                          timestamp=datetime.fromisoformat(v["timestamp"]),
                          privacy_level=v.get("privacy_level", "low"),
                          needs_confirmation=v.get("needs_confirmation", False))
            for k, v in d.get("profile", {}).items()
        }
        state.conversation = [
            ConversationTurn(role=t["role"], content=t["content"],
                             timestamp=datetime.fromisoformat(t["timestamp"]),
                             emotion_state=t.get("emotion_state"),
                             tokens_used=t.get("tokens_used", 0), cost_usd=t.get("cost_usd", 0.0))
            for t in d.get("conversation", [])
        ]
        for attr in ["missing_required", "missing_optional", "emotion_state",
                     "conversation_quality_score", "abandonment_risk", "consecutive_confusion_turns",
                     "completed", "escalated", "escalation_reason", "output", "detected_language",
                     "tokens_used", "cost_usd", "budget_exceeded"]:
            if attr in d:
                setattr(state, attr, d[attr])
        state.created_at = datetime.fromisoformat(d["created_at"])
        state.updated_at = datetime.fromisoformat(d["updated_at"])
        return state

    def _row_to_state(self, row: dict) -> GraphState:
        if row.get("state_json"):
            return self._deserialize(row["state_json"])
        return GraphState(session_id=row["session_id"], goal_id=row["goal_id"],
                          user_id=row.get("user_id"))
