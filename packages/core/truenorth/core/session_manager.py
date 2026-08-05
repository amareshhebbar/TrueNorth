"""
truenorth/core/session_manager.py

Manages the full lifecycle of TrueNorth sessions:
  - Create new session (assign ID, load goal config, init GraphState)
  - Load existing session from storage (Postgres + Redis cache)
  - Save session state after every turn
  - Resume interrupted session (skip already-collected fields)
  - Expire / archive completed sessions

SessionManager is the single source of truth for session state. The engine
never touches storage directly — it goes through here.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class SessionStatus:
    ACTIVE    = "active"
    PAUSED    = "paused"
    COMPLETE  = "complete"
    FAILED    = "failed"
    EXPIRED   = "expired"

class SessionEnvelope:
    """
    Everything needed to restore a session from cold storage.
    GraphState holds the live runtime state; SessionEnvelope is the serializable shell.
    """

    def __init__(
        self,
        session_id: str,
        goal_id: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
        status: str,
        state_data: dict,
        metadata: dict,
        created_at: datetime,
        updated_at: datetime,
    ):
        self.session_id  = session_id
        self.goal_id     = goal_id
        self.user_id     = user_id
        self.tenant_id   = tenant_id
        self.status      = status
        self.state_data  = state_data
        self.metadata    = metadata
        self.created_at  = created_at
        self.updated_at  = updated_at

    def to_dict(self) -> dict:
        return {
            "session_id" : self.session_id,
            "goal_id"    : self.goal_id,
            "user_id"    : self.user_id,
            "tenant_id"  : self.tenant_id,
            "status"     : self.status,
            "state_data" : self.state_data,
            "metadata"   : self.metadata,
            "created_at" : self.created_at.isoformat(),
            "updated_at" : self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEnvelope":
        return cls(
            session_id  = data["session_id"],
            goal_id     = data["goal_id"],
            user_id     = data.get("user_id"),
            tenant_id   = data.get("tenant_id"),
            status      = data.get("status", SessionStatus.ACTIVE),
            state_data  = data.get("state_data", {}),
            metadata    = data.get("metadata", {}),
            created_at  = datetime.fromisoformat(data["created_at"]),
            updated_at  = datetime.fromisoformat(data["updated_at"]),
        )

class SessionManager:
    """
    Async session manager. Requires Postgres and (optionally) Redis.

    Usage:
        sm = SessionManager(postgres=pg, redis=redis_client)
        session_id = await sm.create(goal_id="fitness_plan", user_id="u123")
        state = await sm.load(session_id)
        await sm.save(session_id, state)
        await sm.complete(session_id, output={"report": "..."})
    """

    REDIS_TTL: int = 3600
    RESUME_WINDOW: int = 86400 * 7

    def __init__(
        self,
        postgres=None,
        redis=None,
        config: Optional[dict] = None,
    ):
        self._pg    = postgres
        self._redis = redis
        self._cfg   = config or {}

        self._mem_store: dict[str, dict] = {}

    async def create(
        self,
        goal_id: str,
        user_id:   Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata:  Optional[dict] = None,
        state_data: Optional[dict] = None,
    ) -> str:
        """
        Create a new session. Returns the session_id.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        envelope = SessionEnvelope(
            session_id  = session_id,
            goal_id     = goal_id,
            user_id     = user_id,
            tenant_id   = tenant_id,
            status      = SessionStatus.ACTIVE,
            state_data  = state_data or {},
            metadata    = metadata or {},
            created_at  = now,
            updated_at  = now,
        )
        await self._persist(envelope)
        logger.info("session=%s created goal=%s user=%s", session_id, goal_id, user_id)
        return session_id

    async def load(self, session_id: str) -> Optional[dict]:
        """
        Load session state data. Returns state_data dict or None if not found.
        Checks Redis cache first, then Postgres.
        """

        cached = await self._redis_get(session_id)
        if cached:
            logger.debug("session=%s loaded from Redis cache", session_id)
            return cached.get("state_data", {})

        envelope = await self._pg_get(session_id)
        if envelope is None:
            logger.warning("session=%s not found", session_id)
            return None

        await self._redis_set(envelope)
        return envelope.state_data

    async def save(self, session_id: str, state_data: dict) -> None:
        """
        Persist updated state after a conversation turn.
        Writes to Redis immediately (fast path) and Postgres (durable).
        """
        envelope = await self._get_envelope(session_id)
        if envelope is None:
            logger.error("session=%s cannot save — envelope not found", session_id)
            return
        envelope.state_data = state_data
        envelope.updated_at = datetime.now(timezone.utc)
        await self._persist(envelope)
        logger.debug("session=%s state saved (%d fields)", session_id, len(state_data))

    async def pause(self, session_id: str) -> None:
        """Mark session as paused (user left mid-conversation)."""
        await self._set_status(session_id, SessionStatus.PAUSED)
        logger.info("session=%s paused", session_id)

    async def resume(self, session_id: str) -> Optional[dict]:
        """
        Resume a paused session. Loads state; skips already-collected fields.
        Returns state_data with 'resumed': True marker.
        """
        envelope = await self._get_envelope(session_id)
        if envelope is None:
            return None
        if envelope.status not in (SessionStatus.PAUSED, SessionStatus.ACTIVE):
            logger.warning("session=%s cannot resume — status=%s", session_id, envelope.status)
            return None

        envelope.status = SessionStatus.ACTIVE
        envelope.updated_at = datetime.now(timezone.utc)
        envelope.state_data["resumed"] = True
        await self._persist(envelope)
        logger.info("session=%s resumed", session_id)
        return envelope.state_data

    async def complete(self, session_id: str, output: Optional[dict] = None) -> None:
        """Mark session as complete and store the final output."""
        envelope = await self._get_envelope(session_id)
        if envelope is None:
            return
        envelope.status = SessionStatus.COMPLETE
        envelope.updated_at = datetime.now(timezone.utc)
        if output:
            envelope.state_data["final_output"] = output
        await self._persist(envelope)
        logger.info("session=%s completed", session_id)

    async def fail(self, session_id: str, error: str) -> None:
        """Mark session as failed."""
        envelope = await self._get_envelope(session_id)
        if envelope is None:
            return
        envelope.status = SessionStatus.FAILED
        envelope.updated_at = datetime.now(timezone.utc)
        envelope.state_data["error"] = error
        await self._persist(envelope)
        logger.error("session=%s failed: %s", session_id, error)

    async def get_status(self, session_id: str) -> Optional[str]:
        envelope = await self._get_envelope(session_id)
        return envelope.status if envelope else None

    async def list_sessions(
        self,
        user_id:   Optional[str] = None,
        tenant_id: Optional[str] = None,
        status:    Optional[str] = None,
        limit:     int = 50,
        offset:    int = 0,
    ) -> list[dict]:
        """List sessions with optional filters. Returns list of envelope dicts."""
        if self._pg:
            return await self._pg_list(user_id=user_id, tenant_id=tenant_id,
                                       status=status, limit=limit, offset=offset)
        results = []
        for env_dict in self._mem_store.values():
            if user_id   and env_dict.get("user_id")   != user_id:   continue
            if tenant_id and env_dict.get("tenant_id") != tenant_id: continue
            if status    and env_dict.get("status")    != status:     continue
            results.append(env_dict)
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results[offset: offset + limit]

    async def delete(self, session_id: str) -> bool:
        """Hard-delete a session (GDPR erasure). Returns True if deleted."""
        key = f"session:{session_id}"
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass
        if self._pg:
            try:
                await self._pg.execute(
                    "DELETE FROM sessions WHERE session_id = $1", session_id
                )
                return True
            except Exception as e:
                logger.error("session=%s delete failed: %s", session_id, e)
                return False

        if session_id in self._mem_store:
            del self._mem_store[session_id]
            return True
        return False

    async def _redis_get(self, session_id: str) -> Optional[dict]:
        if not self._redis:
            return None
        try:
            import json
            key = f"session:{session_id}"
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Redis get failed for session=%s: %s", session_id, e)
            return None

    async def _redis_set(self, envelope: SessionEnvelope) -> None:
        if not self._redis:
            return
        try:
            import json
            key = f"session:{envelope.session_id}"
            await self._redis.set(key, json.dumps(envelope.to_dict()), ex=self.REDIS_TTL)
        except Exception as e:
            logger.warning("Redis set failed for session=%s: %s", envelope.session_id, e)

    async def _redis_del(self, session_id: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.delete(f"session:{session_id}")
        except Exception:
            pass

    async def _pg_get(self, session_id: str) -> Optional[SessionEnvelope]:
        if not self._pg:

            data = self._mem_store.get(session_id)
            return SessionEnvelope.from_dict(data) if data else None
        try:
            row = await self._pg.fetchrow(
                "SELECT * FROM sessions WHERE session_id = $1", session_id
            )
            if not row:
                return None
            import json
            data = dict(row)
            data["state_data"] = json.loads(data.get("state_data") or "{}")
            data["metadata"]   = json.loads(data.get("metadata")   or "{}")
            return SessionEnvelope.from_dict(data)
        except Exception as e:
            logger.error("Postgres get failed for session=%s: %s", session_id, e)
            return None

    async def _pg_upsert(self, envelope: SessionEnvelope) -> None:
        if not self._pg:
            return
        try:
            import json
            await self._pg.execute(
                """
                INSERT INTO sessions
                  (session_id, goal_id, user_id, tenant_id, status,
                   state_data, metadata, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (session_id) DO UPDATE SET
                  status=$5, state_data=$6, metadata=$7, updated_at=$9
                """,
                envelope.session_id,
                envelope.goal_id,
                envelope.user_id,
                envelope.tenant_id,
                envelope.status,
                json.dumps(envelope.state_data),
                json.dumps(envelope.metadata),
                envelope.created_at,
                envelope.updated_at,
            )
        except Exception as e:
            logger.error("Postgres upsert failed for session=%s: %s", envelope.session_id, e)

    async def _pg_list(self, **filters) -> list[dict]:
        if not self._pg:
            return []
        try:
            import json
            where, args, idx = [], [], 1
            for col in ("user_id", "tenant_id", "status"):
                if filters.get(col):
                    where.append(f"{col} = ${idx}")
                    args.append(filters[col])
                    idx += 1
            where_clause = ("WHERE " + " AND ".join(where)) if where else ""
            limit  = filters.get("limit", 50)
            offset = filters.get("offset", 0)
            args += [limit, offset]
            rows = await self._pg.fetch(
                f"SELECT * FROM sessions {where_clause} "
                f"ORDER BY updated_at DESC LIMIT ${idx} OFFSET ${idx+1}",
                *args,
            )
            result = []
            for row in rows:
                d = dict(row)
                d["state_data"] = json.loads(d.get("state_data") or "{}")
                d["metadata"]   = json.loads(d.get("metadata")   or "{}")
                result.append(d)
            return result
        except Exception as e:
            logger.error("Postgres list failed: %s", e)
            return []

    async def _get_envelope(self, session_id: str) -> Optional[SessionEnvelope]:
        cached = await self._redis_get(session_id)
        if cached:
            return SessionEnvelope.from_dict(cached)
        return await self._pg_get(session_id)

    async def _persist(self, envelope: SessionEnvelope) -> None:
        """Write to both Redis (fast) and Postgres (durable) + in-memory fallback."""
        await asyncio.gather(
            self._redis_set(envelope),
            self._pg_upsert(envelope),
            return_exceptions=True,
        )
        self._mem_store[envelope.session_id] = envelope.to_dict()

    async def _set_status(self, session_id: str, status: str) -> None:
        envelope = await self._get_envelope(session_id)
        if envelope is None:
            return
        envelope.status     = status
        envelope.updated_at = datetime.now(timezone.utc)
        await self._persist(envelope)
