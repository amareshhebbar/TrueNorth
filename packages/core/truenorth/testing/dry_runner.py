"""
truenorth/testing/dry_runner.py

DryRunner — runs a full simulated conversation without any API calls or human input.

Two modes:
  1. Scenario mode    — replay predefined user answers from a JSON scenario file
  2. Auto mode        — automatically generate plausible answers from field configs
                        (useful for smoke-testing a new goal YAML instantly)

Output: a DryRunReport with field collection results, cost estimate, and pass/fail.

Usage (CLI):
    truenorth dry-run --goal examples/goals/fitness_plan.yaml
    truenorth dry-run --goal examples/goals/fitness_plan.yaml --scenario tests/fixtures/scenarios/fitness_happy_path.json

Usage (Python):
    runner = DryRunner("examples/goals/fitness_plan.yaml")
    report = await runner.run()
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn:       int
    agent:      str
    user:       str
    extracted:  List[str]      
    action:     str
    latency_ms: int = 0


@dataclass
class DryRunReport:
    goal_id:          str
    goal_path:        str
    scenario_path:    Optional[str]
    passed:           bool
    total_turns:      int
    collected_fields: Dict[str, Any]
    missing_required: List[str]
    field_confidences: Dict[str, float]
    turns:            List[TurnRecord]
    total_cost_usd:   float
    elapsed_sec:      float
    errors:           List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines  = [
            "",
            f"{'─'*55}",
            f"  DRY RUN REPORT — {self.goal_id}",
            f"{'─'*55}",
            f"  Status        : {status}",
            f"  Turns         : {self.total_turns}",
            f"  Elapsed       : {self.elapsed_sec:.2f}s",
            f"  Estimated cost: ${self.total_cost_usd:.4f}",
            "",
            "  COLLECTED FIELDS:",
        ]
        for k, v in self.collected_fields.items():
            conf = self.field_confidences.get(k, 0)
            lines.append(f"    ✓  {k:<28} = {v!r}  (conf: {conf:.2f})")
        if self.missing_required:
            lines.append("")
            lines.append("  MISSING REQUIRED:")
            for m in self.missing_required:
                lines.append(f"    ✗  {m}")
        if self.errors:
            lines.append("")
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    !  {e}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "goal_id":          self.goal_id,
            "passed":           self.passed,
            "total_turns":      self.total_turns,
            "collected":        self.collected_fields,
            "missing":          self.missing_required,
            "cost_usd":         self.total_cost_usd,
            "elapsed_sec":      self.elapsed_sec,
            "errors":           self.errors,
        }


# ---------------------------------------------------------------------------
# Auto-answer generator
# ---------------------------------------------------------------------------

_AUTO_ANSWERS: Dict[str, Any] = {
    "name":           "Alex",
    "age":            28,
    "weight":         72,
    "height":         175,
    "goal":           "lose weight",
    "activity":       "moderately active",
    "days":           4,
    "duration":       45,
    "minutes":        45,
    "equipment":      "I have dumbbells at home and can go to a gym twice a week",
    "injury":         "No injuries",
    "injuries":       "No injuries",
    "diet":           "no restrictions",
    "gender":         "male",
    "complaint":      "I have been having lower back pain for the past week",
    "symptom":        "about one week",
    "pain":           5,
    "condition":      "No existing conditions",
    "medication":     "None",
    "allergy":        "None",
    "allergies":      "No known allergies",
}


def _auto_answer(field_name: str, field_config: dict) -> str:
    """Generate a plausible auto-answer string for a field."""
    name_lower = field_name.lower()

    for kw, val in _AUTO_ANSWERS.items():
        if kw in name_lower:
            return str(val)

    ftype = field_config.get("type", "text")
    allowed = field_config.get("allowed_values", [])

    if allowed:
        return str(allowed[0])
    if ftype in ("integer", "number"):
        mn = field_config.get("min", 1)
        mx = field_config.get("max", 100)
        return str((mn + mx) // 2)
    if ftype == "boolean":
        return "yes"

    return "Yes, sounds good"


# ---------------------------------------------------------------------------
# DryRunner
# ---------------------------------------------------------------------------

class DryRunner:
    """
    Runs a complete simulated conversation against a goal YAML.

    Args:
        goal_path:     Path to goal YAML
        scenario_path: Optional path to JSON scenario file (predefined answers)
        mock:          Use mock LLM (default True for dry-run — no API cost)
        verbose:       Print each turn to stdout
    """

    MAX_TURNS = 30  # safety limit

    def __init__(
        self,
        goal_path:     Union[str, Path],
        scenario_path: Optional[Union[str, Path]] = None,
        mock:          bool = True,
        verbose:       bool = True,
    ):
        self.goal_path     = str(goal_path)
        self.scenario_path = str(scenario_path) if scenario_path else None
        self.mock          = mock
        self.verbose       = verbose
        self._scenario:    Optional[Dict[str, Any]] = None

    def _load_scenario(self) -> Optional[Dict[str, str]]:
        """Load predefined user answers from scenario JSON file."""
        if not self.scenario_path:
            return None
        path = Path(self.scenario_path)
        if not path.exists():
            logger.warning("dry_runner: scenario not found: %s", path)
            return None
        with open(path) as f:
            data = json.load(f)
        # Support two formats:
        # 1. {"turns": [{"user": "..."}]}
        # 2. {"answers": {"field_name": "value"}}
        if "turns" in data:
            return {"turns": [t.get("user", "") if isinstance(t, dict) else t for t in data["turns"]]}
        if "answers" in data:
            return {"answers": data["answers"]}
        return data

    async def run(self) -> DryRunReport:
        """
        Run the full dry-run simulation. Returns a DryRunReport.
        """
        start_time = time.perf_counter()
        errors:     List[str] = []
        turns:      List[TurnRecord] = []

        # ── Load scenario ──────────────────────────────────────────────
        scenario = self._load_scenario()

        # ── Build engine ───────────────────────────────────────────────
        try:
            engine = await self._build_engine()
        except Exception as e:
            errors.append(f"Engine init failed: {e}")
            return DryRunReport(
                goal_id="?", goal_path=self.goal_path,
                scenario_path=self.scenario_path, passed=False,
                total_turns=0, collected_fields={}, missing_required=[],
                field_confidences={}, turns=[], total_cost_usd=0.0,
                elapsed_sec=0.0, errors=errors,
            )

        goal_id = engine.state.goal_id

        # ── Start conversation ─────────────────────────────────────────
        try:
            start_resp = await engine.start()
            if self.verbose:
                self._print_turn(0, "", start_resp.text, [], start_resp.action)
        except Exception as e:
            errors.append(f"Engine start failed: {e}")
            start_resp = None

        # ── Build answer queue ─────────────────────────────────────────
        answer_queue: List[str] = []
        if scenario and "turns" in scenario:
            answer_queue = list(scenario["turns"])
        elif scenario and "answers" in scenario:
            answer_queue = []  # will use _get_answer_for_field instead

        field_answer_map: Dict[str, str] = {}
        if scenario and "answers" in scenario:
            field_answer_map = {k: str(v) for k, v in scenario["answers"].items()}

        # ── Conversation loop ──────────────────────────────────────────
        last_target_field = start_resp.target_field if start_resp else None

        turn_num = 0
        while not engine.state.is_complete and turn_num < self.MAX_TURNS:
            turn_num += 1

            user_answer = self._get_next_answer(
                turn_num         = turn_num,
                target_field     = last_target_field,
                fields_config    = engine.state.fields_config,
                answer_queue     = answer_queue,
                field_answer_map = field_answer_map,
            )

            try:
                t0 = time.perf_counter()
                response = await engine.process_message(user_answer)
                latency  = int((time.perf_counter() - t0) * 1000)

                # Update target_field for NEXT turn from this response
                last_target_field = response.target_field

                record = TurnRecord(
                    turn       = turn_num,
                    agent      = response.text,
                    user       = user_answer,
                    extracted  = list(engine.state.collected_fields.keys()),
                    action     = response.action,
                    latency_ms = latency,
                )
                turns.append(record)

                if self.verbose:
                    self._print_turn(
                        turn_num, user_answer, response.text,
                        list(engine.state.collected_fields.keys()), response.action,
                    )

                if response.is_complete:
                    break

            except Exception as e:
                errors.append(f"Turn {turn_num} failed: {e}")
                logger.exception("dry_runner turn %d failed", turn_num)
                break

        elapsed = time.perf_counter() - start_time

        # ── Build report ───────────────────────────────────────────────
        collected = engine.get_collected_fields()
        missing   = engine.get_missing_fields()
        passed    = len(missing) == 0 and len(errors) == 0

        report = DryRunReport(
            goal_id           = goal_id,
            goal_path         = self.goal_path,
            scenario_path     = self.scenario_path,
            passed            = passed,
            total_turns       = turn_num,
            collected_fields  = collected,
            missing_required  = missing,
            field_confidences = engine.state.field_confidences,
            turns             = turns,
            total_cost_usd    = engine.state.total_cost_usd,
            elapsed_sec       = round(elapsed, 3),
            errors            = errors,
        )

        if self.verbose:
            print(report.summary())

        return report

    # ------------------------------------------------------------------
    # Answer selection
    # ------------------------------------------------------------------

    def _get_next_answer(
        self,
        turn_num:        int,
        target_field:    Optional[str],
        fields_config:   Dict[str, dict],
        answer_queue:    List[str],
        field_answer_map: Dict[str, str],
    ) -> str:
        if field_answer_map and target_field and target_field in field_answer_map:
            return field_answer_map[target_field]

        if answer_queue:
            return answer_queue.pop(0) if answer_queue else "yes"

        if target_field and target_field in fields_config:
            return _auto_answer(target_field, fields_config[target_field])

        return "yes"

    @staticmethod
    def _last_target(turns: List[TurnRecord]) -> Optional[str]:
        """Get the last target field from turn records."""
        return None  # reasoner tracks this internally

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    async def _build_engine(self):
        """Build the engine — mock LLM for dry-run, real for live mode."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        if self.mock:
            # Build a mock router that returns scripted answers
            mock_client = MockLLMClient(
                responses={
                    "extract":   '{"extractions": []}',
                    "converse":  "Got it! Tell me more.",
                    "classify":  '{"label": "neutral", "score": 0.6}',
                },
                default="Understood, thank you.",
            )
            router = LLMRouter()
            for model in [
                "gemini-1.5-flash",
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-20250514",
            ]:
                router.register_client(model, mock_client)
        else:
            router = LLMRouter.from_env()

        engine = await TrueNorthEngine.from_yaml(self.goal_path, router=router)
        return engine

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_turn(
        turn: int, user: str, agent: str,
        collected: List[str], action: str,
    ) -> None:
        if turn == 0:
            print(f"\n{'─'*55}")
            print(f"  🤖  {agent}")
            return
        print(f"\n  👤  {user}")
        print(f"  🤖  {agent}")
        print(f"       ↳ action={action}  collected={len(collected)} fields")