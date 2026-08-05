"""
Reminder & Async AI — core scheduling engine.

This is the feature no other AI agent framework has.
Every framework builds agents that RESPOND to users.
TrueNorth can INITIATE contact with users on a schedule.

Real examples:
  - "You said you'd exercise Tuesday. How did it go?"
  - "Your follow-up appointment is in 3 days. Have you booked it?"
  - "It's been 2 weeks since your intake. How are you feeling?"
  - "Your Q1 budget review is overdue. Want to pick it up now?"

Architecture:
  FollowUpRule    — a single rule from the goal YAML follow_up: block
  ScheduledReminder — one pending or fired reminder
  ReminderEngine  — evaluates rules against session state, schedules
                    reminders via APScheduler (or in-memory for testing)

Sector-agnostic: medical follow-ups, legal status updates, HR
onboarding check-ins, fitness weekly reviews — all use the same engine.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class ReminderStatus(str, Enum):
    PENDING   = "pending"
    TRIGGERED = "triggered"
    DELIVERED = "delivered"
    FAILED    = "failed"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"

class TriggerType(str, Enum):
    AFTER_HOURS  = "after_hours"
    AFTER_DAYS   = "after_days"
    ON_DATE      = "on_date"
    WEEKLY       = "weekly"
    MONTHLY      = "monthly"

@dataclass
class FollowUpRule:
    """
    One follow-up rule from a goal YAML.
    Immutable once created — all state lives in ScheduledReminder.
    """
    rule_id:         str
    trigger:         str
    channel:         str
    message_prompt:  str
    check_field:     Optional[str]   = None
    check_value:     Optional[Any]   = None
    condition:       Optional[dict]  = None
    recurring:       bool            = False
    max_fires:       int             = 1

    @classmethod
    def from_yaml(cls, raw: dict, rule_index: int = 0) -> "FollowUpRule":
        """Parse one entry from the YAML follow_up: list."""
        return cls(
            rule_id        = raw.get("id", f"rule_{rule_index}"),
            trigger        = raw.get("trigger", "after 1 day"),
            channel        = raw.get("channel", "email"),
            message_prompt = raw.get("message_prompt", ""),
            check_field    = raw.get("check_field"),
            check_value    = raw.get("check_value"),
            condition      = raw.get("condition"),
            recurring      = raw.get("recurring", False),
            max_fires      = raw.get("max_fires", 1),
        )

    def parse_trigger(self, base_time: datetime) -> Optional[datetime]:
        """
        Convert trigger string to a concrete datetime.
        base_time: when the session completed (UTC).
        """
        t = self.trigger.strip().lower()
        m = re.match(r"after\s+(\d+(?:\.\d+)?)\s+(hour|hours|day|days|week|weeks)", t)
        if m:
            amount = float(m.group(1))
            unit   = m.group(2).rstrip("s")
            delta  = {
                "hour":  timedelta(hours=amount),
                "day":   timedelta(days=amount),
                "week":  timedelta(weeks=amount),
            }[unit]
            return base_time + delta

        if t == "weekly":
            return base_time + timedelta(weeks=1)

        if t == "monthly":
            return base_time + timedelta(days=30)

        m2 = re.match(r"on\s+(\d{4}-\d{2}-\d{2})", t)
        if m2:
            try:
                return datetime.fromisoformat(m2.group(1)).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        logger.warning("reminder_engine: cannot parse trigger %r", self.trigger)
        return None

    def condition_met(self, collected_fields: dict) -> bool:
        """Return True if this rule's condition is satisfied for the current state."""

        if self.check_field:
            field_value = collected_fields.get(self.check_field)
            if self.check_value is None:
                if field_value is not None:
                    return False
            else:
                if field_value != self.check_value:
                    return False

        if self.condition:
            for field_name, expected in self.condition.items():
                if collected_fields.get(field_name) != expected:
                    return False

        return True

@dataclass
class ScheduledReminder:
    """One pending or completed reminder for one session."""
    reminder_id:     str
    rule_id:         str
    session_id:      str
    user_id:         Optional[str]
    goal_id:         str
    channel:         str
    fire_at:         datetime
    status:          ReminderStatus = ReminderStatus.PENDING
    fired_at:        Optional[datetime] = None
    delivery_result: Optional[dict]     = None
    message_text:    Optional[str]      = None
    fire_count:      int                = 0
    created_at:      float = field(default_factory=time.time)

    def is_due(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return (
            self.status == ReminderStatus.PENDING
            and self.fire_at <= now
        )

    def to_dict(self) -> dict:
        return {
            "reminder_id":  self.reminder_id,
            "rule_id":      self.rule_id,
            "session_id":   self.session_id,
            "goal_id":      self.goal_id,
            "channel":      self.channel,
            "fire_at":      self.fire_at.isoformat(),
            "status":       self.status.value,
            "fired_at":     self.fired_at.isoformat() if self.fired_at else None,
            "message_text": self.message_text,
            "fire_count":   self.fire_count,
        }

class ReminderEngine:
    """
    Evaluates follow-up rules and manages the scheduled reminder queue.

    Works without APScheduler (in-memory mode for tests and dry-runs).
    With APScheduler, reminders survive process restarts via Redis/Postgres.
    """

    def __init__(
        self,
        delivery:     Optional[Any]      = None,
        planner:      Optional[Any]      = None,
        redis:        Optional[Any]      = None,
        poll_interval: float             = 60.0,
    ):
        self._delivery      = delivery
        self._planner       = planner
        self._redis         = redis
        self._poll_interval = poll_interval
        self._reminders:    Dict[str, ScheduledReminder] = {}
        self._running       = False

    def schedule_all(
        self,
        rules:           List[FollowUpRule],
        session_id:      str,
        user_id:         Optional[str],
        goal_id:         str,
        collected_fields: dict,
        completed_at:    Optional[datetime] = None,
    ) -> List[ScheduledReminder]:
        """
        Evaluate all follow-up rules for a completed session.
        Creates ScheduledReminder objects for rules whose conditions are met.
        Returns the list of scheduled reminders.
        """
        base_time = completed_at or datetime.now(timezone.utc)
        scheduled: List[ScheduledReminder] = []

        for rule in rules:
            if not rule.condition_met(collected_fields):
                logger.debug(
                    "reminder_engine: rule=%s skipped (condition not met)",
                    rule.rule_id,
                )
                continue

            fire_at = rule.parse_trigger(base_time)
            if fire_at is None:
                continue

            reminder = ScheduledReminder(
                reminder_id = str(uuid.uuid4())[:12],
                rule_id     = rule.rule_id,
                session_id  = session_id,
                user_id     = user_id,
                goal_id     = goal_id,
                channel     = rule.channel,
                fire_at     = fire_at,
            )
            self._reminders[reminder.reminder_id] = reminder
            scheduled.append(reminder)

            logger.info(
                "reminder_engine: scheduled reminder=%s rule=%s session=%s "
                "channel=%s fire_at=%s",
                reminder.reminder_id, rule.rule_id, session_id,
                rule.channel, fire_at.isoformat(),
            )

        return scheduled

    def schedule_one(
        self,
        rule:            FollowUpRule,
        session_id:      str,
        user_id:         Optional[str],
        goal_id:         str,
        fire_at:         datetime,
    ) -> ScheduledReminder:
        """Schedule a single reminder explicitly."""
        reminder = ScheduledReminder(
            reminder_id = str(uuid.uuid4())[:12],
            rule_id     = rule.rule_id,
            session_id  = session_id,
            user_id     = user_id,
            goal_id     = goal_id,
            channel     = rule.channel,
            fire_at     = fire_at,
        )
        self._reminders[reminder.reminder_id] = reminder
        return reminder

    def get_due(self, now: Optional[datetime] = None) -> List[ScheduledReminder]:
        """Return all reminders that are currently due."""
        now = now or datetime.now(timezone.utc)
        return [r for r in self._reminders.values() if r.is_due(now)]

    def get_pending(self, session_id: Optional[str] = None) -> List[ScheduledReminder]:
        """Return all pending reminders, optionally filtered by session."""
        reminders = [
            r for r in self._reminders.values()
            if r.status == ReminderStatus.PENDING
        ]
        if session_id:
            reminders = [r for r in reminders if r.session_id == session_id]
        return reminders

    def get_all(self, session_id: Optional[str] = None) -> List[dict]:
        """Return all reminders as dicts for display/serialisation."""
        reminders = list(self._reminders.values())
        if session_id:
            reminders = [r for r in reminders if r.session_id == session_id]
        return [r.to_dict() for r in reminders]

    def mark_triggered(self, reminder_id: str) -> None:
        r = self._reminders.get(reminder_id)
        if r:
            r.status   = ReminderStatus.TRIGGERED
            r.fired_at = datetime.now(timezone.utc)
            r.fire_count += 1

    def mark_delivered(
        self,
        reminder_id:     str,
        delivery_result: Optional[dict] = None,
        message_text:    Optional[str]  = None,
    ) -> None:
        r = self._reminders.get(reminder_id)
        if r:
            r.status          = ReminderStatus.DELIVERED
            r.delivery_result = delivery_result
            r.message_text    = message_text

    def mark_failed(self, reminder_id: str, error: str = "") -> None:
        r = self._reminders.get(reminder_id)
        if r:
            r.status = ReminderStatus.FAILED
            r.delivery_result = {"error": error}

    def cancel(self, reminder_id: str) -> bool:
        r = self._reminders.get(reminder_id)
        if r and r.status == ReminderStatus.PENDING:
            r.status = ReminderStatus.CANCELLED
            return True
        return False

    def cancel_all_for_session(self, session_id: str) -> int:
        count = 0
        for r in self._reminders.values():
            if r.session_id == session_id and r.status == ReminderStatus.PENDING:
                r.status = ReminderStatus.CANCELLED
                count += 1
        return count

    async def start(self) -> None:
        """Start the background polling loop. Call from your async startup."""
        self._running = True
        logger.info("reminder_engine: background loop started (interval=%.0fs)", self._poll_interval)
        while self._running:
            await self._tick()
            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("reminder_engine: background loop stopped")

    async def _tick(self) -> None:
        """One polling cycle — fire all due reminders."""
        due = self.get_due()
        if not due:
            return

        logger.info("reminder_engine: %d reminder(s) due", len(due))
        for reminder in due:
            self.mark_triggered(reminder.reminder_id)
            try:
                if self._planner and self._delivery:

                    msg_text = await self._planner.compose(reminder)
                    reminder.message_text = msg_text

                    result = await self._delivery.send(reminder)
                    self.mark_delivered(reminder.reminder_id, result, msg_text)
                else:
                    logger.warning(
                        "reminder_engine: no delivery/planner configured — "
                        "marking triggered reminder=%s",
                        reminder.reminder_id,
                    )
            except Exception as e:
                logger.error(
                    "reminder_engine: delivery failed reminder=%s: %s",
                    reminder.reminder_id, e,
                )
                self.mark_failed(reminder.reminder_id, str(e))

    def stats(self) -> dict:
        all_r = list(self._reminders.values())
        by_status: Dict[str, int] = {}
        for r in all_r:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "total":    len(all_r),
            "by_status": by_status,
        }
