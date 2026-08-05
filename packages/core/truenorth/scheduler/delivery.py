"""
Delivery channel adapters for TrueNorth reminders.

Channels:
  WhatsAppAdapter  — Meta WhatsApp Business API
  EmailAdapter     — SMTP / SendGrid / SES
  SMSAdapter       — Twilio SMS
  PushAdapter      — FCM / APNs push notifications
  ConsoleAdapter   — stdout (dry-run / testing)

All adapters implement DeliveryChannel (abstract base).
The ReminderEngine calls delivery.send(reminder) — which adapter
is used is determined by reminder.channel.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.scheduler.reminder_engine import ScheduledReminder

logger = logging.getLogger(__name__)

@dataclass
class DeliveryResult:
    success:    bool
    channel:    str
    message_id: Optional[str] = None
    error:      Optional[str] = None
    latency_ms: int           = 0
    raw:        Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "success":    self.success,
            "channel":    self.channel,
            "message_id": self.message_id,
            "error":      self.error,
            "latency_ms": self.latency_ms,
        }

class DeliveryChannel(ABC):
    """Abstract delivery channel. Implement send() for each provider."""

    channel_name: str = "unknown"

    @abstractmethod
    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        """Send the reminder. Returns DeliveryResult."""

    async def health_check(self) -> bool:
        """Check if the channel is reachable. Override in subclasses."""
        return True

class WhatsAppAdapter(DeliveryChannel):
    """
    Meta WhatsApp Business API adapter.

    Sends messages via the WhatsApp Cloud API.
    Requires a verified WhatsApp Business account and phone number.

    Config:
        phone_number_id : Meta phone number ID (from dashboard)
        access_token    : permanent system user token
        api_version     : "v19.0" (or newer)

    Note: For proactive messages (business-initiated), WhatsApp requires
    using pre-approved message templates. The message_text is sent as
    template variable {1} in a "follow_up" template.
    If you have free-form messaging enabled (within 24h window),
    the message sends directly.
    """

    channel_name = "whatsapp"
    BASE_URL     = "https://graph.facebook.com"

    def __init__(
        self,
        phone_number_id: str,
        access_token:    str,
        api_version:     str   = "v19.0",
        template_name:   str   = "follow_up",
        template_lang:   str   = "en",
    ):
        self._phone_id   = phone_number_id
        self._token      = access_token
        self._version    = api_version
        self._template   = template_name
        self._lang       = template_lang

    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        t0 = time.perf_counter()
        to = reminder.user_id or ""
        if not to:
            return DeliveryResult(
                success=False, channel=self.channel_name,
                error="No user phone number in user_id field"
            )

        payload = {
            "messaging_product": "whatsapp",
            "to":                to,
            "type":              "template",
            "template": {
                "name":     self._template,
                "language": {"code": self._lang},
                "components": [{
                    "type": "body",
                    "parameters": [{"type": "text", "text": reminder.message_text or ""}],
                }],
            },
        }

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.BASE_URL}/{self._version}/{self._phone_id}/messages",
                    json    = payload,
                    headers = {
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type":  "application/json",
                    },
                    timeout = 15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                msg_id = data.get("messages", [{}])[0].get("id")
                latency = int((time.perf_counter() - t0) * 1000)
                logger.info(
                    "whatsapp: sent reminder=%s to=%s msg_id=%s latency=%dms",
                    reminder.reminder_id, to, msg_id, latency,
                )
                return DeliveryResult(
                    success=True, channel=self.channel_name,
                    message_id=msg_id, latency_ms=latency, raw=data,
                )
        except Exception as e:
            return DeliveryResult(
                success=False, channel=self.channel_name,
                error=str(e), latency_ms=int((time.perf_counter() - t0) * 1000),
            )

class EmailAdapter(DeliveryChannel):
    """
    SMTP email adapter.
    Also supports SendGrid and Amazon SES via compatible SMTP.

    Config:
        smtp_host   : e.g. "smtp.gmail.com"
        smtp_port   : 587 (TLS) or 465 (SSL)
        username    : SMTP username
        password    : SMTP password or app password
        from_addr   : sender address
        from_name   : sender display name
    """

    channel_name = "email"

    def __init__(
        self,
        smtp_host:  str,
        smtp_port:  int            = 587,
        username:   str            = "",
        password:   str            = "",
        from_addr:  str            = "noreply@truenorth.ai",
        from_name:  str            = "TrueNorth",
        use_tls:    bool           = True,
        subject_template: str      = "Your TrueNorth follow-up",
    ):
        self._host     = smtp_host
        self._port     = smtp_port
        self._user     = username
        self._pass     = password
        self._from     = from_addr
        self._name     = from_name
        self._tls      = use_tls
        self._subject  = subject_template

    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        t0      = time.perf_counter()
        to_addr = reminder.user_id or ""
        if not to_addr or "@" not in to_addr:
            return DeliveryResult(
                success=False, channel=self.channel_name,
                error="Invalid or missing email address in user_id",
            )
        body = reminder.message_text or ""

        try:
            import smtplib
            from email.mime.text    import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg          = MIMEMultipart("alternative")
            msg["From"]  = f"{self._name} <{self._from}>"
            msg["To"]    = to_addr
            msg["Subject"] = self._subject
            msg.attach(MIMEText(body, "plain"))

            loop = asyncio.get_event_loop()
            def _send():
                if self._tls:
                    server = smtplib.SMTP(self._host, self._port)
                    server.starttls()
                else:
                    server = smtplib.SMTP_SSL(self._host, self._port)
                if self._user:
                    server.login(self._user, self._pass)
                server.sendmail(self._from, [to_addr], msg.as_string())
                server.quit()

            await loop.run_in_executor(None, _send)
            latency = int((time.perf_counter() - t0) * 1000)
            logger.info("email: sent reminder=%s to=%s latency=%dms",
                        reminder.reminder_id, to_addr, latency)
            return DeliveryResult(success=True, channel=self.channel_name, latency_ms=latency)
        except Exception as e:
            return DeliveryResult(
                success=False, channel=self.channel_name,
                error=str(e), latency_ms=int((time.perf_counter() - t0) * 1000),
            )

class SMSAdapter(DeliveryChannel):
    """Twilio SMS adapter."""

    channel_name = "sms"

    def __init__(
        self,
        account_sid: str,
        auth_token:  str,
        from_number: str,
    ):
        self._sid  = account_sid
        self._tok  = auth_token
        self._from = from_number

    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        t0 = time.perf_counter()
        to = reminder.user_id or ""
        body = reminder.message_text or ""

        try:
            import httpx
            url  = f"https://api.twilio.com/2010-04-01/Accounts/{self._sid}/Messages.json"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    data={"From": self._from, "To": to, "Body": body[:1600]},
                    auth=(self._sid, self._tok),
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                latency = int((time.perf_counter() - t0) * 1000)
                return DeliveryResult(
                    success=True, channel=self.channel_name,
                    message_id=data.get("sid"), latency_ms=latency, raw=data,
                )
        except Exception as e:
            return DeliveryResult(
                success=False, channel=self.channel_name,
                error=str(e), latency_ms=int((time.perf_counter() - t0) * 1000),
            )

class ConsoleAdapter(DeliveryChannel):
    """
    Prints reminders to stdout. For dry-runs, testing, local dev.
    Always succeeds.
    """
    channel_name = "console"

    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        print(
            f"\n[TrueNorth Reminder]\n"
            f"  Session : {reminder.session_id}\n"
            f"  Goal    : {reminder.goal_id}\n"
            f"  Channel : {reminder.channel}\n"
            f"  Message : {reminder.message_text or '(no message)'}\n"
        )
        return DeliveryResult(
            success=True, channel=self.channel_name, message_id="console"
        )

class MultiChannelDelivery:
    """
    Routes reminder delivery to the correct channel adapter.

    Usage:
        delivery = MultiChannelDelivery({
            "whatsapp": WhatsAppAdapter(...),
            "email":    EmailAdapter(...),
            "console":  ConsoleAdapter(),
        })
        result = await delivery.send(reminder)
    """

    def __init__(
        self,
        adapters:        Dict[str, DeliveryChannel],
        fallback_channel: Optional[str] = "console",
    ):
        self._adapters = adapters
        self._fallback = fallback_channel

    def register(self, channel_name: str, adapter: DeliveryChannel) -> None:
        self._adapters[channel_name] = adapter

    async def send(self, reminder: "ScheduledReminder") -> DeliveryResult:
        channel  = reminder.channel
        adapter  = self._adapters.get(channel)

        if adapter is None:
            if self._fallback:
                adapter = self._adapters.get(self._fallback)
            if adapter is None:
                return DeliveryResult(
                    success=False, channel=channel,
                    error=f"No adapter configured for channel {channel!r}",
                )

        return await adapter.send(reminder)

    async def health_check(self) -> Dict[str, bool]:
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception:
                results[name] = False
        return results
