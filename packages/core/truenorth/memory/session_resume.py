"""
Session resume — continue interrupted conversations.

When a user drops off mid-intake, TrueNorth saves the progress.
When they return (hours or days later), the engine picks up exactly
where it left off — without re-asking questions already answered.

CLI: truenorth resume SESSION_ID

Integration:
  - engine.py calls session_manager.save() after every turn (already done)
  - This module handles the RESUME path: load state, skip known fields,
    craft a re-engagement message
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResumeResult:
    """Result of attempting to resume a session."""
    session_id:        str
    resumable:         bool
    collected_fields:  Dict[str, Any]
    missing_fields:    List[str]
    completion_pct:    float
    turns_completed:   int
    re_engagement_msg: Optional[str] = None
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "resumable":        self.resumable,
            "collected_count":  len(self.collected_fields),
            "missing_count":    len(self.missing_fields),
            "completion_pct":   round(self.completion_pct, 1),
            "turns_completed":  self.turns_completed,
        }


class SessionResume:
    """
    Handles resumption of interrupted sessions.
    """

    def __init__(
        self,
        session_manager: Any,
        router:          Optional[Any] = None,
    ):
        self._sm     = session_manager
        self._router = router

    async def check(self, session_id: str) -> ResumeResult:
        """
        Check if a session is resumable and load its state.
        """
        try:
            state_data = await self._sm.load(session_id)
        except Exception as e:
            return ResumeResult(
                session_id=session_id, resumable=False,
                collected_fields={}, missing_fields=[], completion_pct=0,
                turns_completed=0, error=str(e),
            )

        if state_data is None:
            return ResumeResult(
                session_id=session_id, resumable=False,
                collected_fields={}, missing_fields=[], completion_pct=0,
                turns_completed=0, error="Session not found",
            )

        collected  = state_data.get("collected_fields", {})
        missing    = state_data.get("missing_required", [])
        pct        = state_data.get("completion_pct", 0.0)
        turns      = state_data.get("current_turn", 0)
        final_out  = state_data.get("final_output")

        if final_out:
            return ResumeResult(
                session_id=session_id, resumable=False,
                collected_fields=collected, missing_fields=[],
                completion_pct=100.0, turns_completed=turns,
                error="Session already completed",
            )

        msg = self._re_engagement_message(collected, missing, turns)
        return ResumeResult(
            session_id        = session_id,
            resumable         = True,
            collected_fields  = collected,
            missing_fields    = missing,
            completion_pct    = pct,
            turns_completed   = turns,
            re_engagement_msg = msg,
        )

    @staticmethod
    def _re_engagement_message(
        collected: Dict[str, Any],
        missing:   List[str],
        turns:     int,
    ) -> str:
        """Build a re-engagement message from prior session state."""
        name = collected.get("name") or collected.get("first_name") or ""
        greeting = f"Welcome back{', ' + name if name else ''}!"
        n_collected = len(collected)
        n_missing   = len(missing)

        if n_collected == 0:
            return f"{greeting} Let's pick up where we left off."

        return (
            f"{greeting} You completed {n_collected} question(s) last time. "
            f"We just need {n_missing} more to finish your profile. "
            f"Ready to continue?"
        )