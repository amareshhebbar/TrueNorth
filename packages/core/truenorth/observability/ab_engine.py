"""
A/B Engine — split-test two versions of a goal YAML.

Real use cases:
  - Test a new field order that might reduce abandonment
  - Test a more empathetic persona prompt
  - Test removing an optional field to see if completion rate improves
  - Test a different follow_up schedule
  - Test routing output to Sonnet vs Haiku on quality vs cost

The engine splits incoming sessions between variant A and B at the
configured ratio, tracks completion rates per variant, and computes
statistical significance when enough data is collected.

"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ABVariant(str, Enum):
    A = "A"
    B = "B"


class ABStatus(str, Enum):
    RUNNING       = "running"        
    SIGNIFICANT   = "significant"     
    INCONCLUSIVE  = "inconclusive"    
    STOPPED       = "stopped"        


@dataclass
class VariantStats:
    """Running stats for one variant."""
    variant:       ABVariant
    sessions:      int   = 0
    completions:   int   = 0
    total_cost:    float = 0.0
    total_turns:   float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.completions / max(self.sessions, 1)

    @property
    def avg_cost(self) -> float:
        return self.total_cost / max(self.completions, 1)

    @property
    def avg_turns(self) -> float:
        return self.total_turns / max(self.completions, 1)

    def to_dict(self) -> dict:
        return {
            "variant":         self.variant.value,
            "sessions":        self.sessions,
            "completions":     self.completions,
            "completion_rate": round(self.completion_rate, 4),
            "avg_cost_usd":    round(self.avg_cost, 6),
            "avg_turns":       round(self.avg_turns, 2),
        }


@dataclass
class ABResult:
    """Final A/B test result."""
    test_id:       str
    status:        ABStatus
    winner:        Optional[ABVariant]
    stats_a:       VariantStats
    stats_b:       VariantStats
    p_value:       Optional[float]
    lift_pct:      Optional[float]    
    confidence:    Optional[float]
    min_sessions:  int
    created_at:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "test_id":      self.test_id,
            "status":       self.status.value,
            "winner":       self.winner.value if self.winner else None,
            "variant_a":    self.stats_a.to_dict(),
            "variant_b":    self.stats_b.to_dict(),
            "p_value":      round(self.p_value, 4) if self.p_value else None,
            "lift_pct":     round(self.lift_pct, 2) if self.lift_pct else None,
            "confidence":   round(self.confidence, 4) if self.confidence else None,
            "min_sessions": self.min_sessions,
        }


class ABEngine:
    """
    Splits sessions between two goal YAML configurations and
    tracks completion outcomes per variant.

    Assignment is deterministic per session_id — the same session
    always gets the same variant (no flicker).

    Split ratio: split_ratio = 0.30 means 30% of sessions go to B,
    70% stay on A.
    """

    def __init__(
        self,
        test_id:          str,
        variant_a_config: dict,
        variant_b_config: dict,
        split_ratio:      float = 0.50,    
        min_sessions:     int   = 50,      
        success_metric:   str   = "completion_rate",
    ):
        self._test_id       = test_id
        self._configs       = {ABVariant.A: variant_a_config, ABVariant.B: variant_b_config}
        self._split_ratio   = max(0.0, min(1.0, split_ratio))
        self._min_sessions  = min_sessions
        self._success_metric = success_metric
        self._status        = ABStatus.RUNNING

        self._stats = {
            ABVariant.A: VariantStats(variant=ABVariant.A),
            ABVariant.B: VariantStats(variant=ABVariant.B),
        }
        self._assignments: Dict[str, ABVariant] = {}

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(self, session_id: str) -> dict:
        """
        Assign a session to a variant and return its goal config.
        Assignment is deterministic — same session_id → same variant.

        Returns the goal config dict for the assigned variant.
        """
        if session_id in self._assignments:
            variant = self._assignments[session_id]
        else:
            variant = self._hash_assign(session_id)
            self._assignments[session_id] = variant
            self._stats[variant].sessions += 1

        return self._configs[variant]

    def get_variant(self, session_id: str) -> Optional[ABVariant]:
        """Return the variant assigned to a session (None if not assigned)."""
        return self._assignments.get(session_id)

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        session_id: str,
        completed:  bool,
        cost_usd:   float = 0.0,
        turns:      int   = 0,
    ) -> None:
        """Record the outcome of a session."""
        variant = self._assignments.get(session_id)
        if variant is None:
            return
        stats = self._stats[variant]
        if completed:
            stats.completions += 1
            stats.total_cost  += cost_usd
            stats.total_turns += turns

    # ------------------------------------------------------------------
    # Result and significance
    # ------------------------------------------------------------------

    def result(self) -> ABResult:
        """
        Compute current A/B result with statistical significance test.
        Uses a two-proportion z-test on completion rates.
        """
        stats_a = self._stats[ABVariant.A]
        stats_b = self._stats[ABVariant.B]
        if (stats_a.sessions < self._min_sessions
                or stats_b.sessions < self._min_sessions):
            return ABResult(
                test_id      = self._test_id,
                status       = self._status,
                winner       = None,
                stats_a      = stats_a,
                stats_b      = stats_b,
                p_value      = None,
                lift_pct     = None,
                confidence   = None,
                min_sessions = self._min_sessions,
            )

        p_a  = stats_a.completion_rate
        p_b  = stats_b.completion_rate
        n_a  = stats_a.sessions
        n_b  = stats_b.sessions
        p_pool = (stats_a.completions + stats_b.completions) / (n_a + n_b)

        se  = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        if se == 0:
            z = 0.0
        else:
            z = (p_b - p_a) / se

        p_value    = 2 * (1 - self._normal_cdf(abs(z)))   # two-tailed
        confidence = 1 - p_value
        significant = p_value < 0.05

        lift_pct = ((p_b - p_a) / max(p_a, 1e-10)) * 100 if p_a > 0 else None

        if significant:
            winner = ABVariant.B if p_b > p_a else ABVariant.A
            status = ABStatus.SIGNIFICANT
        else:
            winner = None
            status = ABStatus.INCONCLUSIVE

        return ABResult(
            test_id      = self._test_id,
            status       = status,
            winner       = winner,
            stats_a      = stats_a,
            stats_b      = stats_b,
            p_value      = p_value,
            lift_pct     = lift_pct,
            confidence   = confidence,
            min_sessions = self._min_sessions,
        )

    def stop(self) -> None:
        """Manually stop the test (e.g. when a safety issue is found)."""
        self._status = ABStatus.STOPPED

    def current_stats(self) -> Dict[str, dict]:
        return {v.value: s.to_dict() for v, s in self._stats.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hash_assign(self, session_id: str) -> ABVariant:
        """Deterministically assign a variant via hash of session_id."""
        digest = int(hashlib.md5(
            f"{self._test_id}:{session_id}".encode()
        ).hexdigest(), 16)
        bucket = (digest % 10_000) / 10_000.0    # 0..1
        return ABVariant.B if bucket < self._split_ratio else ABVariant.A

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate CDF of the standard normal distribution."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────────
#  ABRegistry — manage multiple concurrent A/B tests
# ─────────────────────────────────────────────────────────────────────────────

class ABRegistry:
    """
    Manages multiple concurrent A/B tests across different goals.
    """

    def __init__(self):
        self._tests: Dict[str, ABEngine] = {}

    def register(self, engine: ABEngine) -> None:
        self._tests[engine._test_id] = engine

    def assign(self, test_id: str, session_id: str) -> Optional[dict]:
        engine = self._tests.get(test_id)
        return engine.assign(session_id) if engine else None

    def record_outcome(
        self,
        test_id:    str,
        session_id: str,
        completed:  bool,
        **kwargs,
    ) -> None:
        engine = self._tests.get(test_id)
        if engine:
            engine.record_outcome(session_id, completed, **kwargs)

    def result(self, test_id: str) -> Optional[ABResult]:
        engine = self._tests.get(test_id)
        return engine.result() if engine else None

    def all_results(self) -> Dict[str, dict]:
        return {tid: e.result().to_dict() for tid, e in self._tests.items()}

    def list_tests(self) -> List[str]:
        return list(self._tests.keys())