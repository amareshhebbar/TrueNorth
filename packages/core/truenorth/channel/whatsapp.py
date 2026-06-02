"""
WhatsApp native conversation channel for TrueNorth.

This is not the delivery adapter (that's in scheduler/delivery.py).
This is the INBOUND channel — TrueNorth runs a full conversation
inside WhatsApp. The user messages via WhatsApp; TrueNorth responds
via WhatsApp. The entire field collection happens in the chat.

Architecture:
  WhatsAppMessage  — normalised inbound message from Meta webhook
  WhatsAppChannel  — wraps TrueNorthEngine; processes webhook payloads

Webhook setup (FastAPI example):
    channel = WhatsAppChannel(
        engine_factory = lambda: TrueNorthEngine(goal_config=config),
        verify_token   = "my_webhook_token",
        access_token   = "EAAGm...",
        phone_id       = "123456789",
    )
    app.post("/webhook/whatsapp")(channel.handle_webhook)
    app.get("/webhook/whatsapp")(channel.verify_webhook)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WhatsAppMessage:
    """Normalised inbound WhatsApp message from Meta webhook payload."""
    message_id:  str
    from_number: str           
    text:        str        
    timestamp:   float
    message_type: str = "text" # text | image | audio | document | interactive
    media_url:   Optional[str] = None
    button_reply: Optional[str] = None   

    @classmethod
    def from_webhook(cls, payload: dict) -> Optional["WhatsAppMessage"]:
        """Parse a Meta WhatsApp webhook entry into a WhatsAppMessage."""
        try:
            entry   = payload["entry"][0]
            changes = entry["changes"][0]["value"]
            msgs    = changes.get("messages", [])
            if not msgs:
                return None
            msg = msgs[0]
            text  = msg.get("text", {}).get("body", "")
            mtype = msg.get("type", "text")

            # Interactive button replies
            if mtype == "interactive":
                btn = msg.get("interactive", {})
                if btn.get("type") == "button_reply":
                    text = btn["button_reply"].get("title", "")

            return cls(
                message_id   = msg["id"],
                from_number  = msg["from"],
                text         = text,
                timestamp    = float(msg.get("timestamp", time.time())),
                message_type = mtype,
            )
        except (KeyError, IndexError, TypeError):
            return None

    def to_dict(self) -> dict:
        return {
            "message_id":  self.message_id,
            "from_number": self.from_number,
            "text":        self.text,
            "type":        self.message_type,
            "timestamp":   self.timestamp,
        }


class WhatsAppChannel:
    """
    WhatsApp Business API conversation channel.

    Maintains one TrueNorthEngine per phone number (session).
    Processes inbound webhooks and sends responses via the API.

    Usage with FastAPI:
        from fastapi import FastAPI, Request
        from truenorth.channel.whatsapp import WhatsAppChannel
        from truenorth.core.yaml_loader import YAMLLoader

        goal = YAMLLoader.load("medical_intake.yaml")
        channel = WhatsAppChannel(
            engine_factory = lambda sid: TrueNorthEngine(
                goal_config=goal, session_id=sid
            ),
            verify_token = os.environ["WA_VERIFY_TOKEN"],
            access_token = os.environ["WA_ACCESS_TOKEN"],
            phone_id     = os.environ["WA_PHONE_NUMBER_ID"],
        )

        app = FastAPI()
        app.get("/webhook")(channel.verify_webhook)
        app.post("/webhook")(channel.handle_webhook)
    """

    BASE_URL = "https://graph.facebook.com9.0"

    def __init__(
        self,
        engine_factory: Callable[[str], Any],    
        verify_token:   str,
        access_token:   str,
        phone_id:       str,
        app_secret:     Optional[str] = None,    # for webhook signature verification
    ):
        self._factory      = engine_factory
        self._verify_token = verify_token
        self._token        = access_token
        self._phone_id     = phone_id
        self._secret       = app_secret
        self._sessions:    Dict[str, Any] = {}   

    # ------------------------------------------------------------------
    # Webhook verification (GET)
    # ------------------------------------------------------------------

    def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """
        Handle Meta's webhook verification challenge.

        FastAPI usage:
            @app.get("/webhook")
            def verify(
                mode=Query(None, alias="hub.mode"),
                token=Query(None, alias="hub.verify_token"),
                challenge=Query(None, alias="hub.challenge"),
            ):
                return channel.verify_webhook(mode, token, challenge)
        """
        if mode == "subscribe" and token == self._verify_token:
            logger.info("whatsapp_channel: webhook verified")
            return challenge
        logger.warning("whatsapp_channel: webhook verification failed")
        return None

    # ------------------------------------------------------------------
    # Inbound webhook (POST)
    # ------------------------------------------------------------------

    async def handle_webhook(self, payload: dict, signature: str = "") -> dict:
        """
        Process an inbound webhook payload from Meta.
        Returns a dict with status and any errors.

        FastAPI usage:
            @app.post("/webhook")
            async def webhook(request: Request):
                body = await request.json()
                sig  = request.headers.get("X-Hub-Signature-256", "")
                return await channel.handle_webhook(body, sig)
        """
        if self._secret and signature:
            if not self._verify_signature(payload, signature):
                logger.warning("whatsapp_channel: invalid webhook signature")
                return {"status": "error", "reason": "invalid signature"}

        msg = WhatsAppMessage.from_webhook(payload)
        if msg is None:
            return {"status": "ok", "processed": False}

        logger.info(
            "whatsapp_channel: inbound from=%s text=%r",
            msg.from_number, msg.text[:60],
        )

        engine = await self._get_engine(msg.from_number)
        response = await engine.process_message(msg.text)

        await self._send_text(msg.from_number, response.text or "")

        # If engine is complete, send final output and clean up
        if engine.state and engine.state.completion_pct >= 100:
            logger.info(
                "whatsapp_channel: session complete for %s", msg.from_number
            )

        return {"status": "ok", "processed": True}

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def _send_text(self, to: str, text: str) -> dict:
        """Send a text message via WhatsApp Cloud API."""
        payload = {
            "messaging_product": "whatsapp",
            "to":   to,
            "type": "text",
            "text": {"body": text[:4096]},
        }
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.BASE_URL}/{self._phone_id}/messages",
                    json    = payload,
                    headers = {
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type":  "application/json",
                    },
                    timeout = 10.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("whatsapp_channel: send failed to=%s: %s", to, e)
            return {"error": str(e)}

    async def send_quick_replies(self, to: str, text: str, options: List[str]) -> dict:
        """Send a message with quick reply buttons (max 3 buttons)."""
        buttons = [
            {"type": "reply", "reply": {"id": str(i), "title": opt[:20]}}
            for i, opt in enumerate(options[:3])
        ]
        payload = {
            "messaging_product": "whatsapp",
            "to":   to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": text[:1024]},
                "action": {"buttons": buttons},
            },
        }
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.BASE_URL}/{self._phone_id}/messages",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=10.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_engine(self, from_number: str) -> Any:
        """Get existing engine or start a new session for this phone number."""
        if from_number not in self._sessions:
            session_id = f"wa:{from_number}"
            engine     = self._factory(session_id)
            await engine.start()
            self._sessions[from_number] = engine
            logger.info(
                "whatsapp_channel: new session=%s for=%s", session_id, from_number
            )
        return self._sessions[from_number]

    def _verify_signature(self, payload: dict, signature: str) -> bool:
        """Verify Meta webhook signature."""
        import json
        if not signature.startswith("sha256="):
            return False
        expected = hmac.new(
            self._secret.encode(), json.dumps(payload).encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature[7:])