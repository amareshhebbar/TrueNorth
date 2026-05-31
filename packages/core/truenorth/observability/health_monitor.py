"""
HealthMonitor — goal-level completion and quality metrics.

Answers the questions operators actually care about:
  - What % of sessions complete the goal?
  - Which fields are users most likely to abandon on?
  - How many turns does it usually take?
  - Which fields get skipped most often?
  - Where in the conversation do users drop off?
  - Is today's completion rate worse than yesterday?

Data source: session traces from TrueNorthTracer.

Metrics computed:
  completion_rate   — sessions that reached 100% / total started
  avg_turns         — average turns per completed session
  field_skip_rate   — per-field skip rate (skipped / attempts)
  abandonment_map   — {turn_number: count_of_sessions_that_abandoned}
  avg_cost_per_session — average USD cost for completed sessions
  p95_latency_ms    — 95th percentile turn latency
  pii_rate          — fraction of turns with PII detected
  conflict_rate     — avg conflicts per session
  hallucination_rate — fraction of turns with firewall hits

Usage:
    monitor = HealthMonitor(tracer=tracer)
    report  = monitor.goal_report("medical_intake")
    print(report["completion_rate"])   # 0.82
    print(report["abandonment_map"])   # {3: 12, 5: 7, ...}
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.observability.tracer import TrueNorthTracer, SessionTrace


@dataclass
class GoalHealthReport:
    """Health metrics for one goal over a time window."""
    goal_id:            str
    window_hours:       int
    session_count:      int
    completed_count:    int
    abandoned_count:    int
    completion_rate:    float          # 0–1
    avg_turns:          float
    avg_cost_usd:       float
    p50_latency_ms:     int
    p95_latency_ms:     int
    field_skip_rates:   Dict[str, float]   # field_name → 0..1
    abandonment_map:    Dict[int, int]     # turn_number → sessions abandoned
    pii_rate:           float
    conflict_rate:      float
    hallucination_rate: float
    generated_at:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "goal_id":          self.goal_id,
            "window_hours":     self.window_hours,
            "session_count":    self.session_count,
            "completed_count":  self.completed_count,
            "abandoned_count":  self.abandoned_count,
            "completion_rate":  round(self.completion_rate, 4),
            "avg_turns":        round(self.avg_turns, 1),
            "avg_cost_usd":     round(self.avg_cost_usd, 6),
            "p50_latency_ms":   self.p50_latency_ms,
            "p95_latency_ms":   self.p95_latency_ms,
            "field_skip_rates": {k: round(v, 4) for k, v in self.field_skip_rates.items()},
            "abandonment_map":  self.abandonment_map,
            "pii_rate":         round(self.pii_rate, 4),
            "conflict_rate":    round(self.conflict_rate, 4),
            "hallucination_rate":round(self.hallucination_rate, 4),
            "generated_at":     self.generated_at,
        }


class HealthMonitor:
    """
    Computes goal-level health metrics from session trace data.

    Sector-agnostic — same metrics for a medical intake, a legal
    case intake, an HR screening, a financial plan, a fitness goal.
    The fields and completion criteria are goal-specific but the
    metrics framework is identical.

    """

    DEFAULT_THRESHOLDS = {
        "completion_rate_min":     0.60,  
        "avg_turns_max":           12,    
        "p95_latency_ms_max":      8000,  
        "hallucination_rate_max":  0.05,  
    }

    def __init__(
        self,
        tracer:     "TrueNorthTracer",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self._tracer     = tracer
        self._thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def goal_report(
        self,
        goal_id:      str,
        window_hours: int = 24,
    ) -> GoalHealthReport:
        """
        Compute a complete health report for one goal.

        Args:
            goal_id:      The goal YAML id to report on.
            window_hours: Only include sessions from the last N hours.
        """
        cutoff   = time.time() - (window_hours * 3600)
        sessions = self._sessions_for_goal(goal_id, cutoff)

        if not sessions:
            return self._empty_report(goal_id, window_hours)

        completed  = [s for s in sessions if self._is_complete(s)]
        abandoned  = [s for s in sessions if not self._is_complete(s) and s.finished_at]

        completion_rate = len(completed) / len(sessions)
        avg_turns       = (
            sum(s.turn_count for s in completed) / len(completed)
            if completed else 0.0
        )
        avg_cost = (
            sum(s.total_cost_usd for s in completed) / len(completed)
            if completed else 0.0
        )

        all_latencies  = self._collect_latencies(sessions)
        p50, p95       = self._percentiles(all_latencies)
        field_skips    = self._field_skip_rates(sessions)
        abandon_map    = self._abandonment_map(abandoned)
        pii_rate       = self._pii_rate(sessions)
        conflict_rate  = self._conflict_rate(sessions)
        fw_rate        = self._hallucination_rate(sessions)

        return GoalHealthReport(
            goal_id            = goal_id,
            window_hours       = window_hours,
            session_count      = len(sessions),
            completed_count    = len(completed),
            abandoned_count    = len(abandoned),
            completion_rate    = completion_rate,
            avg_turns          = avg_turns,
            avg_cost_usd       = avg_cost,
            p50_latency_ms     = p50,
            p95_latency_ms     = p95,
            field_skip_rates   = field_skips,
            abandonment_map    = abandon_map,
            pii_rate           = pii_rate,
            conflict_rate      = conflict_rate,
            hallucination_rate = fw_rate,
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def check_alerts(self, report: GoalHealthReport) -> List[dict]:
        """
        Compare report metrics against thresholds.
        Returns a list of alert dicts (empty = all healthy).
        """
        alerts = []
        t      = self._thresholds

        if report.completion_rate < t["completion_rate_min"]:
            alerts.append({
                "metric":    "completion_rate",
                "value":     round(report.completion_rate, 4),
                "threshold": t["completion_rate_min"],
                "severity":  "high",
                "message":   f"Completion rate {report.completion_rate:.0%} below {t['completion_rate_min']:.0%}",
            })

        if report.avg_turns > t["avg_turns_max"]:
            alerts.append({
                "metric":    "avg_turns",
                "value":     round(report.avg_turns, 1),
                "threshold": t["avg_turns_max"],
                "severity":  "medium",
                "message":   f"Average {report.avg_turns:.1f} turns exceeds {t['avg_turns_max']} max",
            })

        if report.p95_latency_ms > t["p95_latency_ms_max"]:
            alerts.append({
                "metric":    "p95_latency_ms",
                "value":     report.p95_latency_ms,
                "threshold": int(t["p95_latency_ms_max"]),
                "severity":  "high",
                "message":   f"p95 latency {report.p95_latency_ms}ms exceeds {int(t['p95_latency_ms_max'])}ms",
            })

        if report.hallucination_rate > t["hallucination_rate_max"]:
            alerts.append({
                "metric":    "hallucination_rate",
                "value":     round(report.hallucination_rate, 4),
                "threshold": t["hallucination_rate_max"],
                "severity":  "critical",
                "message":   f"Hallucination rate {report.hallucination_rate:.1%} exceeds {t['hallucination_rate_max']:.1%}",
            })

        # High-abandon fields
        for field_name, rate in report.field_skip_rates.items():
            if rate > 0.30:
                alerts.append({
                    "metric":    "field_skip_rate",
                    "field":     field_name,
                    "value":     round(rate, 4),
                    "threshold": 0.30,
                    "severity":  "medium",
                    "message":   f"Field '{field_name}' skipped by {rate:.0%} of users",
                })

        return alerts

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_periods(
        self,
        goal_id:        str,
        current_hours:  int = 24,
        previous_hours: int = 24,
    ) -> dict:
        """
        Compare current period vs previous period metrics.
        Returns {metric: {current, previous, change_pct}}.
        """
        now_cutoff  = time.time() - current_hours * 3600
        prev_cutoff = now_cutoff - previous_hours * 3600

        curr_sessions = self._sessions_for_goal(goal_id, now_cutoff)
        prev_sessions = self._sessions_for_goal(goal_id, prev_cutoff,
                                                  until=now_cutoff)

        def _rate(sessions):
            if not sessions:
                return 0.0
            return sum(1 for s in sessions if self._is_complete(s)) / len(sessions)

        curr_rate = _rate(curr_sessions)
        prev_rate = _rate(prev_sessions)

        def _pct_change(curr, prev):
            if prev == 0:
                return None
            return round((curr - prev) / prev * 100, 1)

        return {
            "goal_id":    goal_id,
            "completion_rate": {
                "current":    round(curr_rate, 4),
                "previous":   round(prev_rate, 4),
                "change_pct": _pct_change(curr_rate, prev_rate),
            },
            "session_count": {
                "current":    len(curr_sessions),
                "previous":   len(prev_sessions),
                "change_pct": _pct_change(len(curr_sessions), len(prev_sessions)),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sessions_for_goal(
        self,
        goal_id: str,
        since:   float,
        until:   Optional[float] = None,
    ) -> List["SessionTrace"]:
        all_ids = self._tracer.all_session_ids()
        result  = []
        for sid in all_ids:
            sess = self._tracer.get_session(sid)
            if not sess or sess.goal_id != goal_id:
                continue
            if sess.started_at < since:
                continue
            if until and sess.started_at > until:
                continue
            result.append(sess)
        return result

    @staticmethod
    def _is_complete(session: "SessionTrace") -> bool:
        """A session is complete if it has any turns and all fields extracted."""
        return session.turn_count > 0 and session.finished_at is not None

    @staticmethod
    def _collect_latencies(sessions: List["SessionTrace"]) -> List[int]:
        latencies = []
        for sess in sessions:
            for turn in sess.turns:
                if turn.latency_ms > 0:
                    latencies.append(turn.latency_ms)
        return latencies

    @staticmethod
    def _percentiles(latencies: List[int]) -> tuple:
        if not latencies:
            return 0, 0
        s   = sorted(latencies)
        p50 = s[len(s) // 2]
        p95 = s[int(len(s) * 0.95)]
        return p50, p95

    @staticmethod
    def _field_skip_rates(sessions: List["SessionTrace"]) -> Dict[str, float]:
        attempts: Dict[str, int] = defaultdict(int)
        skipped:  Dict[str, int] = defaultdict(int)
        for sess in sessions:
            for turn in sess.turns:
                for ext in turn.extractions:
                    fname = ext.get("field", "")
                    if fname:
                        attempts[fname] += 1
                        if not ext.get("success"):
                            skipped[fname] += 1
        rates = {}
        for fname, total in attempts.items():
            rates[fname] = skipped[fname] / total
        return rates

    @staticmethod
    def _abandonment_map(abandoned: List["SessionTrace"]) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for sess in abandoned:
            turn = sess.turn_count
            counts[turn] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _pii_rate(sessions: List["SessionTrace"]) -> float:
        all_turns = 0
        pii_turns = 0
        for sess in sessions:
            all_turns += sess.turn_count
            pii_turns += sess.pii_turn_count
        return pii_turns / max(all_turns, 1)

    @staticmethod
    def _conflict_rate(sessions: List["SessionTrace"]) -> float:
        if not sessions:
            return 0.0
        return sum(s.conflict_count for s in sessions) / len(sessions)

    @staticmethod
    def _hallucination_rate(sessions: List["SessionTrace"]) -> float:
        all_turns = sum(s.turn_count for s in sessions)
        fw_hits   = sum(
            1 for s in sessions for t in s.turns
            if t.fw_verdict in ("FLAGGED", "BLOCKED")
        )
        return fw_hits / max(all_turns, 1)

    @staticmethod
    def _empty_report(goal_id: str, window_hours: int) -> GoalHealthReport:
        return GoalHealthReport(
            goal_id=goal_id, window_hours=window_hours,
            session_count=0, completed_count=0, abandoned_count=0,
            completion_rate=0.0, avg_turns=0.0, avg_cost_usd=0.0,
            p50_latency_ms=0, p95_latency_ms=0,
            field_skip_rates={}, abandonment_map={},
            pii_rate=0.0, conflict_rate=0.0, hallucination_rate=0.0,
        )