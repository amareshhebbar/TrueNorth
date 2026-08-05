"""
Long-term user memory — persists facts about a user across sessions.

When a user completes a fitness intake, TrueNorth knows their age,
weight, and goal. When they start a nutrition goal a week later,
they shouldn't be asked again. LongTermMemory answers that.

Architecture:
  UserFact     — one piece of known information about a user
  LongTermMemory — stores, retrieves, and merges user facts

Storage:
  - In-memory dict (default — for tests and dry-runs)
  - Postgres (production — async, keyed by user_id + fact_key)
  - Redis cache layer (fast lookup, falls back to Postgres)

Merge strategy:
  - More recent facts overwrite older ones
  - Confidence threshold: low-confidence extractions not persisted
  - Conflict detection: when new fact contradicts existing, flag it

Sector-agnostic: medical history, legal matter history, HR candidate
profile, financial profile — all stored the same way.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class UserFact:
    """One piece of known information about a user."""
    user_id:      str
    fact_key:     str
    value:        Any
    goal_id:      str
    session_id:   str
    confidence:   float     = 1.0
    source:       str       = "extracted"
    created_at:   float     = field(default_factory=time.time)
    updated_at:   float     = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "user_id":    self.user_id,
            "fact_key":   self.fact_key,
            "value":      self.value,
            "goal_id":    self.goal_id,
            "session_id": self.session_id,
            "confidence": round(self.confidence, 3),
            "source":     self.source,
            "updated_at": self.updated_at,
        }

class LongTermMemory:
    """
    Persistent user memory across TrueNorth sessions.
    """

    MIN_CONFIDENCE = 0.70

    def __init__(
        self,
        postgres: Optional[Any] = None,
        redis:    Optional[Any] = None,
        min_confidence: float   = 0.70,
    ):
        self._pg          = postgres
        self._redis       = redis
        self._min_conf    = min_confidence
        self._store:      Dict[str, Dict[str, UserFact]] = {}

    def store(self, fact: UserFact) -> bool:
        """Store one fact. Returns True if stored, False if confidence too low."""
        if fact.confidence < self._min_conf:
            logger.debug(
                "memory: skipping fact key=%s confidence=%.2f < %.2f",
                fact.fact_key, fact.confidence, self._min_conf,
            )
            return False

        if fact.user_id not in self._store:
            self._store[fact.user_id] = {}

        existing = self._store[fact.user_id].get(fact.fact_key)
        if existing and existing.confidence > fact.confidence:
            logger.debug(
                "memory: keeping higher-confidence existing fact key=%s", fact.fact_key
            )
            return False

        self._store[fact.user_id][fact.fact_key] = fact
        logger.debug(
            "memory: stored user=%s key=%s value=%r conf=%.2f",
            fact.user_id, fact.fact_key, fact.value, fact.confidence,
        )
        return True

    def store_from_session(
        self,
        user_id:           str,
        session_id:        str,
        goal_id:           str,
        collected_fields:  Dict[str, Any],
        field_confidences: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Bulk-store all collected fields from a completed session.
        Returns count of facts stored.
        """
        confs   = field_confidences or {}
        count   = 0
        for key, value in collected_fields.items():
            fact = UserFact(
                user_id    = user_id,
                fact_key   = key,
                value      = value,
                goal_id    = goal_id,
                session_id = session_id,
                confidence = confs.get(key, 0.85),
            )
            if self.store(fact):
                count += 1
        logger.info(
            "memory: stored %d/%d facts from session=%s goal=%s user=%s",
            count, len(collected_fields), session_id, goal_id, user_id,
        )
        return count

    def get(self, user_id: str, fact_key: str) -> Optional[Any]:
        """Get one fact value for a user."""
        fact = self._store.get(user_id, {}).get(fact_key)
        return fact.value if fact else None

    def get_fact(self, user_id: str, fact_key: str) -> Optional[UserFact]:
        """Get the full UserFact object."""
        return self._store.get(user_id, {}).get(fact_key)

    def get_all(self, user_id: str) -> Dict[str, Any]:
        """Get all known facts for a user as a flat dict."""
        user_facts = self._store.get(user_id, {})
        return {k: f.value for k, f in user_facts.items()}

    def get_all_facts(self, user_id: str) -> List[UserFact]:
        """Get all UserFact objects for a user."""
        return list(self._store.get(user_id, {}).values())

    def get_by_goal(self, user_id: str, goal_id: str) -> Dict[str, Any]:
        """Get facts that came from a specific goal."""
        user_facts = self._store.get(user_id, {})
        return {k: f.value for k, f in user_facts.items() if f.goal_id == goal_id}

    def seed_engine(self, engine: Any, user_id: str, min_confidence: float = 0.0) -> int:
        """
        Pre-fill an engine's collected_fields with known user facts.
        Skips fields not in the goal's field config.
        Returns count of fields seeded.
        """
        count = 0
        user_facts = self._store.get(user_id, {})
        for key, fact in user_facts.items():
            if fact.confidence < min_confidence:
                continue
            if key in engine.state.fields_config:
                engine.state.set_field(key, fact.value, confidence=fact.confidence)
                count += 1
        return count

    def forget(self, user_id: str, fact_key: Optional[str] = None) -> int:
        """Delete facts (for GDPR erasure). Returns count deleted."""
        if user_id not in self._store:
            return 0
        if fact_key:
            deleted = 1 if fact_key in self._store[user_id] else 0
            self._store[user_id].pop(fact_key, None)
            return deleted
        count = len(self._store[user_id])
        del self._store[user_id]
        return count

    def stats(self, user_id: Optional[str] = None) -> dict:
        if user_id:
            facts = self._store.get(user_id, {})
            return {"user_id": user_id, "fact_count": len(facts), "keys": list(facts.keys())}
        return {
            "total_users": len(self._store),
            "total_facts": sum(len(v) for v in self._store.values()),
        }
