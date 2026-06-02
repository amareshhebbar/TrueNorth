"""
Monitors conversation quality in real time. Produces a ConversationQualityReport
after every turn. The engine uses this to decide whether to adapt its strategy.

Metrics tracked:
  - Clarity score        : how clear / unambiguous the user's answer was
  - Engagement score     : is the user engaged or giving one-word answers
  - Frustration signal   : rising frustration pattern over last N turns
  - Progress rate        : fields collected per turn (efficiency)
  - Abandonment risk     : probability the user will drop off
  - Response length trend: shortening responses signal disengagement

All scoring is heuristic + pattern-based. No LLM calls here — this must be
fast (< 5ms) to run on every single turn.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversationQualityReport:
    """Output of one quality check. Attached to graph state after each turn."""

    turn_number:        int
    clarity_score:      float   # 0.0–1.0  (how clear was this answer)
    engagement_score:   float   # 0.0–1.0  (how engaged is the user)
    frustration_signal: float   # 0.0–1.0  (risk of frustration / dropout)
    progress_rate:      float   # fields_collected / turns_so_far
    abandonment_risk:   float   # 0.0–1.0  (model's best guess at dropout)
    flags:              List[str] = field(default_factory=list)  # warning codes
    suggestions:        List[str] = field(default_factory=list)  # action hints for planner

    @property
    def is_healthy(self) -> bool:
        return (
            self.abandonment_risk < 0.40
            and self.frustration_signal < 0.50
            and self.engagement_score > 0.30
        )

    def to_dict(self) -> dict:
        return {
            "turn":              self.turn_number,
            "clarity":           round(self.clarity_score, 3),
            "engagement":        round(self.engagement_score, 3),
            "frustration":       round(self.frustration_signal, 3),
            "progress_rate":     round(self.progress_rate, 3),
            "abandonment_risk":  round(self.abandonment_risk, 3),
            "flags":             self.flags,
            "suggestions":       self.suggestions,
            "healthy":           self.is_healthy,
        }


# ---------------------------------------------------------------------------
# Heuristic signals
# ---------------------------------------------------------------------------

_CONFUSION_PATTERNS = re.compile(
    r"\b(what|huh|why|don'?t understand|not sure|confused|what do you mean|unclear|i don'?t know)\b",
    re.IGNORECASE,
)

_FRUSTRATION_WORDS = re.compile(
    r"\b(annoying|stop|enough|whatever|forget it|this is stupid|why are you|pointless|waste of time)\b",
    re.IGNORECASE,
)

_SHORT_ANSWER_LIMIT = 12 

_FILLER_PATTERNS = re.compile(
    r"^(ok|okay|sure|fine|yes|no|idk|dunno|maybe|hmm+|uh+|um+|alright)\.?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ConversationQualityMonitor
# ---------------------------------------------------------------------------

class ConversationQualityMonitor:
    """
    Stateless quality monitor. Call check() once per turn, passing the recent
    turn history and current session statistics.

    Example:
        monitor = ConversationQualityMonitor()
        report = monitor.check(
            turn_number=4,
            user_message="idk",
            turn_history=[...],
            fields_collected=2,
            total_required_fields=8,
        )
        if not report.is_healthy:
            print(report.suggestions)
    """

    TREND_WINDOW = 4

    def check(
        self,
        turn_number:          int,
        user_message:         str,
        turn_history:         List[dict], 
        fields_collected:     int = 0,
        total_required_fields: int = 1,
    ) -> ConversationQualityReport:
        """
        Evaluate the quality of the conversation up to this turn.

        Args:
            turn_number:           Current turn index (1-based).
            user_message:          The user's most recent message.
            turn_history:          Full turn history so far.
            fields_collected:      How many required fields have been collected.
            total_required_fields: Total required fields in this goal.

        Returns:
            ConversationQualityReport
        """
        flags:       List[str] = []
        suggestions: List[str] = []

        # --- per-message scores ---
        clarity     = self._score_clarity(user_message, flags, suggestions)
        engagement  = self._score_engagement(user_message, turn_history, flags, suggestions)
        frustration = self._score_frustration(user_message, turn_history, flags, suggestions)

        # --- session-level scores ---
        progress = fields_collected / max(total_required_fields, 1)
        abandonment = self._score_abandonment(
            frustration, engagement, turn_number, progress, flags, suggestions
        )

        report = ConversationQualityReport(
            turn_number        = turn_number,
            clarity_score      = clarity,
            engagement_score   = engagement,
            frustration_signal = frustration,
            progress_rate      = progress,
            abandonment_risk   = abandonment,
            flags              = flags,
            suggestions        = suggestions,
        )

        if not report.is_healthy:
            logger.info(
                "turn=%d quality warning: abandonment=%.2f frustration=%.2f engagement=%.2f flags=%s",
                turn_number, abandonment, frustration, engagement, flags,
            )

        return report

    # ------------------------------------------------------------------
    # Individual scorers
    # ------------------------------------------------------------------

    def _score_clarity(self, msg: str, flags: list, suggestions: list) -> float:
        """How clear and specific was this answer?"""
        msg = msg.strip()

        if not msg:
            flags.append("EMPTY_RESPONSE")
            return 0.0

        if len(msg) < _SHORT_ANSWER_LIMIT:
            if _FILLER_PATTERNS.match(msg):
                flags.append("FILLER_ANSWER")
                suggestions.append("rephrase_question_simpler")
                return 0.15
            return 0.40 
        
        if _CONFUSION_PATTERNS.search(msg):
            flags.append("CONFUSION_DETECTED")
            suggestions.append("add_context_or_example")
            return 0.30

        score = min(1.0, 0.60 + len(msg) / 800)
        return round(score, 3)

    def _score_engagement(self, msg: str, history: List[dict], flags: list, suggestions: list) -> float:
        """Is the user actively engaged? Look at response length trend."""
        user_msgs = [
            t["content"] for t in history if t.get("role") == "user"
        ][-self.TREND_WINDOW:]

        if not user_msgs:
            return 0.70  

        avg_len = sum(len(m) for m in user_msgs) / len(user_msgs)

        current_len = len(msg.strip())
        if avg_len > 40 and current_len < avg_len * 0.35:
            flags.append("DECLINING_RESPONSE_LENGTH")
            suggestions.append("acknowledge_effort_before_asking")

        # All recent messages are very short
        if all(len(m.strip()) < _SHORT_ANSWER_LIMIT for m in user_msgs):
            flags.append("CONSISTENTLY_SHORT_RESPONSES")
            suggestions.append("switch_to_yes_no_questions")
            return 0.20

        score = min(1.0, current_len / 120)
        return round(score, 3)

    def _score_frustration(self, msg: str, history: List[dict], flags: list, suggestions: list) -> float:
        """Detect rising frustration pattern."""
        score = 0.0

        if _FRUSTRATION_WORDS.search(msg):
            flags.append("FRUSTRATION_WORDS_DETECTED")
            suggestions.append("empathize_and_reduce_friction")
            score += 0.60

        recent_user_msgs = [
            t["content"] for t in history if t.get("role") == "user"
        ][-self.TREND_WINDOW:]

        frustration_in_history = sum(
            1 for m in recent_user_msgs if _FRUSTRATION_WORDS.search(m)
        )
        score += frustration_in_history * 0.15

        if len(recent_user_msgs) >= 2:
            stripped = [m.strip().lower() for m in recent_user_msgs]
            if len(set(stripped)) == 1 and stripped[0]:
                flags.append("REPEATED_IDENTICAL_ANSWER")
                suggestions.append("clarify_question_differently")
                score += 0.35

        return round(min(score, 1.0), 3)

    def _score_abandonment(
        self,
        frustration:  float,
        engagement:   float,
        turn_number:  int,
        progress:     float,
        flags:        list,
        suggestions:  list,
    ) -> float:
        """
        Weighted abandonment risk model.
        Higher frustration + lower engagement + low progress + high turn count = high risk.
        """
        base = frustration * 0.45 + (1.0 - engagement) * 0.30
        if turn_number > 10 and progress < 0.40:
            base += 0.20
            flags.append("LONG_SESSION_LOW_PROGRESS")
            suggestions.append("consider_skipping_optional_fields")
        if turn_number > 0:
            efficiency = progress / turn_number
            if efficiency < 0.05 and turn_number > 5:
                base += 0.10
                flags.append("LOW_EFFICIENCY")

        risk = round(min(base, 1.0), 3)

        if risk > 0.65:
            suggestions.append("offer_to_save_and_resume_later")

        return risk

    # ------------------------------------------------------------------
    # Trend helpers
    # ------------------------------------------------------------------

    def compare_reports(
        self, reports: List[ConversationQualityReport]
    ) -> dict:
        """
        Compare a list of reports (e.g. last 5 turns) and return trend summary.
        """
        if not reports:
            return {}
        engagements   = [r.engagement_score   for r in reports]
        frustrations  = [r.frustration_signal for r in reports]
        engag_trend   = engagements[-1] - engagements[0]   if len(engagements) > 1 else 0.0
        frust_trend   = frustrations[-1] - frustrations[0] if len(frustrations) > 1 else 0.0
        return {
            "turns_analysed":     len(reports),
            "engagement_trend":   round(engag_trend, 3),   # positive = improving
            "frustration_trend":  round(frust_trend, 3),   # positive = worsening
            "avg_abandonment":    round(sum(r.abandonment_risk for r in reports) / len(reports), 3),
            "healthy_turns":      sum(1 for r in reports if r.is_healthy),
        }