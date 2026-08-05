"""
FollowUpPlanner — uses the LLM to compose personalised reminder messages.

Instead of template strings, TrueNorth uses the LLM to write the message.
This means the follow-up feels like it came from a thoughtful assistant
who remembers what the user said, not a cron job.

Context injected into the LLM:
  - The message_prompt from the YAML rule
  - All collected fields from the original session
  - How much time has passed since the session
  - The user's name (if collected)
  - The goal that was completed

Output:
  A short, warm, personalised message appropriate for the channel.
  WhatsApp → conversational, max 160 chars
  Email    → slightly longer, with subject line
  SMS      → max 160 chars, no formatting

Sector examples:

  Medical: "Hi Priya! It's been 3 days since your intake appointment.
            Dr. Mehta mentioned to follow up about your back pain. How are
            you feeling today?"

  HR: "Hi Alex! Just checking in — have you had a chance to review the
       onboarding docs we sent over? Happy to answer any questions."

  Fitness: "Hey! You mentioned Tuesday was your workout day. How did it go?
            Still on track for your weight loss goal?"

  Legal: "Hi Rahul, following up on your case review from last week.
          Do you have any additional documents to share?"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.scheduler.reminder_engine import ScheduledReminder

logger = logging.getLogger(__name__)

_CHANNEL_MAX_CHARS = {
    "whatsapp": 4096,
    "sms":      160,
    "email":    2000,
    "push":     150,
    "console":  None,
}

_SYSTEM_PROMPT = """You are TrueNorth's follow-up assistant.
Your job is to write a short, warm, personalised follow-up message.

Rules:
  - Sound human, not like a bot or cron job
  - Use the user's name if known
  - Reference specific things they shared in their session
  - Keep it brief — this is a check-in, not a report
  - Match the tone to the channel (WhatsApp = casual, Email = slightly formal)
  - Do NOT add greetings like "Dear" or sign-offs — keep it conversational
  - Do NOT make up information not in the context
  - Output ONLY the message text, nothing else
"""

class FollowUpPlanner:
    """
    Composes personalised follow-up messages using the LLM.

    Usage:
        planner = FollowUpPlanner(router=router)
        message = await planner.compose(
            reminder         = scheduled_reminder,
            original_state   = session_state_dict,
            rule_prompt      = rule.message_prompt,
        )
    """

    def __init__(
        self,
        router:       Any,
        model:        Optional[str] = None,
        max_tokens:   int           = 200,
    ):
        self._router    = router
        self._model     = model
        self._max_tokens = max_tokens

    async def compose(
        self,
        reminder:       "ScheduledReminder",
        original_state: Optional[dict] = None,
        rule_prompt:    Optional[str]  = None,
    ) -> str:
        """
        Compose a personalised follow-up message.

        Args:
            reminder:       The ScheduledReminder being fired
            original_state: The session's to_dict() from when it completed
            rule_prompt:    The message_prompt from the YAML rule

        Returns:
            The message text to send, trimmed to channel length limits.
        """
        context = self._build_context(reminder, original_state, rule_prompt)
        channel = reminder.channel

        from truenorth.llm.base  import Message
        from truenorth.llm.router import TASK_CONVERSE

        messages = [Message(role="user", content=context)]
        try:
            resp = await self._router.generate(
                task        = TASK_CONVERSE,
                messages    = messages,
                system      = _SYSTEM_PROMPT,
                max_tokens  = self._max_tokens,
                temperature = 0.7,
                model       = self._model,
            )
            text = resp.content.strip()
        except Exception as e:
            logger.error("follow_up_planner: LLM call failed: %s", e)
            text = self._fallback_message(reminder, original_state)

        max_chars = _CHANNEL_MAX_CHARS.get(channel)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        logger.info(
            "follow_up_planner: composed message reminder=%s channel=%s chars=%d",
            reminder.reminder_id, channel, len(text),
        )
        return text

    def _build_context(
        self,
        reminder:       "ScheduledReminder",
        original_state: Optional[dict],
        rule_prompt:    Optional[str],
    ) -> str:
        """Build the LLM prompt context from session data."""
        lines = []

        if rule_prompt:
            lines.append(f"Follow-up instruction: {rule_prompt}")
            lines.append("")

        lines.append(f"Channel: {reminder.channel}")
        lines.append(f"Goal: {reminder.goal_id}")

        if original_state:
            collected = original_state.get("collected_fields", {})
            if collected:
                lines.append("\nUser's collected information:")
                for field_name, value in collected.items():
                    label = field_name.replace("_", " ").title()
                    lines.append(f"  {label}: {value}")

            created_at = original_state.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, (int, float)):
                        session_time = datetime.fromtimestamp(created_at, tz=timezone.utc)
                    else:
                        session_time = datetime.fromisoformat(str(created_at))
                    elapsed = datetime.now(timezone.utc) - session_time
                    days    = elapsed.days
                    if days == 0:
                        lines.append(f"\nTime since session: {elapsed.seconds // 3600} hours ago")
                    else:
                        lines.append(f"\nTime since session: {days} day(s) ago")
                except Exception:
                    pass

        lines.append(
            f"\nWrite a brief, warm {reminder.channel} message for this follow-up."
        )
        return "\n".join(lines)

    @staticmethod
    def _fallback_message(
        reminder:       "ScheduledReminder",
        original_state: Optional[dict],
    ) -> str:
        """Fallback template if LLM call fails."""
        name = ""
        if original_state:
            fields = original_state.get("collected_fields", {})
            name   = fields.get("name") or fields.get("first_name") or ""
        greeting = f"Hi {name}!" if name else "Hi!"
        return f"{greeting} Just checking in on your {reminder.goal_id.replace('_', ' ')} progress. How are you doing?"
