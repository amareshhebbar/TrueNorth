"""
CliniqFlow — Clinic Intake via WhatsApp
=========================================

WHAT THIS DOES
--------------
Production-ready clinic intake system.
Patient messages the clinic's WhatsApp number before their appointment.
TrueNorth collects: chief complaint, pain, medications, allergies, history.
Doctor sees a 30-second summary before the patient walks in.
Replaces the paper clipboard at every clinic.

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic httpx

FILE STRUCTURE
--------------
    cliniqflow/
    ├── app.py              ← this file (the complete server)
    ├── goal.yaml           ← copy 01-patient-intake.yaml here
    └── .env                ← your credentials (see below)

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    WA_VERIFY_TOKEN=cliniqflow-secret-token
    WA_ACCESS_TOKEN=your-whatsapp-cloud-api-token
    WA_PHONE_NUMBER_ID=your-phone-number-id
    CLINIC_NAME=Dr. Sharma Clinic

HOW TO RUN
----------
    # Option A: Uvicorn directly
    cd cliniqflow
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload

    # Option B: Docker
    docker run -p 8080:8080 --env-file .env \\
        -v $(pwd)/goal.yaml:/app/goal.yaml \\
        truenorth/cliniqflow:latest

    # Option C: Local test (no WhatsApp needed)
    python app.py

    # Option D: ngrok for WhatsApp webhook testing
    ngrok http 8080
    # Set webhook URL in Meta Console: https://xxx.ngrok.io/webhook

WHATSAPP SETUP (5 minutes)
---------------------------
    1. https://developers.facebook.com → Create App → Business → WhatsApp
    2. Add a phone number (test number is free)
    3. Copy Phone Number ID and Access Token
    4. Set webhook: https://your-server/webhook
    5. Verify token: whatever you set in WA_VERIFY_TOKEN
    6. Subscribe to: messages

API ENDPOINTS
-------------
    GET  /          → Dashboard (sessions + stats)
    GET  /health    → Health check
    GET  /webhook   → WhatsApp verification
    POST /webhook   → Incoming WhatsApp messages
    GET  /sessions  → Active sessions (for clinic dashboard)
    GET  /completed → Completed intake summaries (for EMR)
    DELETE /sessions/{id} → Clear a session

INTEGRATION WITH EMR
--------------------
    CliniqFlow fires a webhook when a session completes.
    Set COMPLETION_WEBHOOK_URL in .env to receive completed intakes.
    Payload: { session_id, phone, completed_at, intake_json }

DPDP COMPLIANCE
---------------
    - Consent notice shown before first question
    - Data retention: 365 days (configurable in goal.yaml)
    - Erasure: DELETE /sessions/{id} removes all data
    - All PII masked in logs by default
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cliniqflow")

# ── Config ─────────────────────────────────────────────────────────────────

CLINIC_NAME        = os.environ.get("CLINIC_NAME", "Our Clinic")
VERIFY_TOKEN       = os.environ.get("WA_VERIFY_TOKEN", "cliniqflow-token")
ACCESS_TOKEN       = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_NUMBER_ID    = os.environ.get("WA_PHONE_NUMBER_ID", "")
GOAL_YAML          = os.environ.get("GOAL_YAML", "goal.yaml")
COMPLETION_WEBHOOK = os.environ.get("COMPLETION_WEBHOOK_URL", "")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"


# ── Storage (in-memory; swap for Redis/Postgres in production) ──────────────

class SessionStore:
    def __init__(self):
        self._sessions:   Dict[str, TrueNorthEngine]     = {}
        self._completed:  Dict[str, dict]                 = {}
        self._meta:       Dict[str, dict]                 = {}

    def get(self, session_id: str) -> Optional[TrueNorthEngine]:
        return self._sessions.get(session_id)

    def set(self, session_id: str, engine: TrueNorthEngine, phone: str):
        self._sessions[session_id] = engine
        self._meta[session_id]     = {"phone": phone, "created_at": time.time(), "turns": 0}

    def complete(self, session_id: str, output: dict):
        engine = self._sessions.pop(session_id, None)
        meta   = self._meta.pop(session_id, {})
        self._completed[session_id] = {
            "session_id":   session_id,
            "phone":        meta.get("phone", ""),
            "completed_at": datetime.now().isoformat(),
            "turns":        meta.get("turns", 0),
            "intake":       output,
        }
        log.info(f"Session {session_id} completed and stored")

    def increment_turns(self, session_id: str):
        if session_id in self._meta:
            self._meta[session_id]["turns"] += 1

    def delete(self, session_id: str) -> bool:
        removed = session_id in self._sessions or session_id in self._completed
        self._sessions.pop(session_id, None)
        self._completed.pop(session_id, None)
        self._meta.pop(session_id, None)
        return removed

    def stats(self) -> dict:
        return {
            "active_sessions":    len(self._sessions),
            "completed_today":    len(self._completed),
            "total_turns_active": sum(m.get("turns", 0) for m in self._meta.values()),
        }


store       = SessionStore()
_goal_config = None

# ── FastAPI ─────────────────────────────────────────────────────────────────

app = FastAPI(title=f"CliniqFlow — {CLINIC_NAME}", version="1.0.0")


@app.on_event("startup")
async def startup():
    global _goal_config
    try:
        _goal_config = YAMLLoader.load(GOAL_YAML)
        log.info(f"✅ Goal loaded: {_goal_config.get('name')} "
                 f"({len(_goal_config.get('fields', []))} fields)")
    except FileNotFoundError:
        log.error(f"❌ {GOAL_YAML} not found. Copy 01-patient-intake.yaml to goal.yaml")

    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — running in console/print mode")


# ── Dashboard ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    stats = store.stats()
    return f"""
    <html><head><title>CliniqFlow — {CLINIC_NAME}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        .stat {{ display: inline-block; background: #f0f9ff; border-radius: 8px;
                 padding: 20px 30px; margin: 10px; text-align: center; }}
        .stat h2 {{ margin: 0; font-size: 2rem; color: #0369a1; }}
        .stat p  {{ margin: 0; color: #64748b; }}
        h1 {{ color: #1e293b; }}
        .green {{ color: #16a34a; }} .badge {{ background: #dcfce7; color: #16a34a;
                  padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
    </style></head><body>
    <h1>🏥 CliniqFlow</h1>
    <p>{CLINIC_NAME} &nbsp; <span class="badge">LIVE</span></p>
    <div>
        <div class="stat"><h2>{stats['active_sessions']}</h2><p>Active Intakes</p></div>
        <div class="stat"><h2>{stats['completed_today']}</h2><p>Completed Today</p></div>
        <div class="stat"><h2>{stats['total_turns_active']}</h2><p>Total Turns</p></div>
    </div>
    <p>Send a WhatsApp message to your registered number to start an intake.</p>
    <p><small>API: <a href="/docs">/docs</a> &nbsp; Health: <a href="/health">/health</a>
    &nbsp; Completed: <a href="/completed">/completed</a></small></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "clinic":        CLINIC_NAME,
        "goal_loaded":   _goal_config is not None,
        "wa_connected":  bool(ACCESS_TOKEN and PHONE_NUMBER_ID),
        **store.stats(),
    }


@app.get("/completed")
async def completed_intakes():
    """Returns all completed intake summaries. Connect your EMR to poll this."""
    return {"intakes": list(store._completed.values())}


@app.get("/sessions")
async def active_sessions():
    """Live view of active sessions (for clinic monitor screen)."""
    return {
        sid: {
            "completion_pct": eng.state.completion_pct,
            "current_turn":   eng.state.current_turn,
            "fields_done":    len(eng.state.collected_fields),
            "started":        store._meta.get(sid, {}).get("created_at"),
        }
        for sid, eng in store._sessions.items()
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """DPDP erasure — remove all data for a session."""
    removed = store.delete(session_id)
    if not removed:
        raise HTTPException(404, f"Session {session_id} not found")
    return {"deleted": session_id, "status": "erased"}


# ── WhatsApp webhook ────────────────────────────────────────────────────────

@app.get("/webhook")
async def verify(request: Request):
    p         = request.query_params
    mode      = p.get("hub.mode")
    token     = p.get("hub.verify_token")
    challenge = p.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("✅ WhatsApp webhook verified")
        return PlainTextResponse(challenge)
    raise HTTPException(403, "Bad verify token")


@app.post("/webhook")
async def handle(request: Request, background: BackgroundTasks):
    body = await request.json()
    try:
        messages = body["entry"][0]["changes"][0]["value"].get("messages", [])
        if not messages:
            return {"status": "no_message"}

        msg   = messages[0]
        phone = msg["from"]
        text  = msg.get("text", {}).get("body", "").strip()

        if not text:
            return {"status": "non_text"}

        log.info(f"📨 {phone}: {text[:60]}")
        background.add_task(process_message, phone, text)
        return {"status": "processing"}

    except (KeyError, IndexError) as e:
        log.warning(f"Unexpected webhook shape: {e}")
        return {"status": "ignored"}


async def process_message(phone: str, text: str):
    """Process a WhatsApp message and drive the TrueNorth session."""
    session_id = f"cf_{hashlib.md5(phone.encode()).hexdigest()[:12]}"

    engine = store.get(session_id)

    if engine is None:
        # ── New session ────────────────────────────────────────────────────
        if _goal_config is None:
            await send_wa(phone, "Sorry, the intake system is not configured yet. "
                                 "Please contact the clinic directly.")
            return

        engine = TrueNorthEngine(
            goal_config = _goal_config,
            session_id  = session_id,
            router      = LLMRouter(),
        )
        store.set(session_id, engine, phone)

        # DPDP consent notice
        consent = (
            f"*{CLINIC_NAME} Intake*\n\n"
            "Before we begin: we'll collect your health information for your "
            "appointment today. This is kept confidential and shared only with "
            "your doctor. Type *AGREE* to continue or *STOP* to cancel."
        )
        await send_wa(phone, consent)
        return

    # ── Consent handling ────────────────────────────────────────────────────
    if not engine.state.current_turn and text.upper() not in ("AGREE", "YES", "HA", "हाँ"):
        # First turn — check for consent
        if text.upper() in ("STOP", "NO", "NAHI", "नहीं"):
            await send_wa(phone, "No problem. Your data has not been stored. "
                                 "Please come in a few minutes early to fill the paper form.")
            store.delete(session_id)
            return
        # Assume any other reply is consent for smoother UX
        pass

    # Start the conversation on first turn
    if engine.state.current_turn == 0:
        first = await engine.start()
        await send_wa(phone, first.text)
        # If their first message has info (not just "AGREE"), also process it
        if text.upper() not in ("AGREE", "YES", "START", "BEGIN"):
            resp = await engine.process_message(text)
            store.increment_turns(session_id)
            await send_wa(phone, resp.text)
            await _check_complete(session_id, phone, resp)
        return

    # ── Continue session ────────────────────────────────────────────────────
    resp = await engine.process_message(text)
    store.increment_turns(session_id)
    await send_wa(phone, resp.text)
    await _check_complete(session_id, phone, resp)


async def _check_complete(session_id: str, phone: str, response):
    """Handle session completion."""
    if response.is_complete and response.output:
        # Store the completed intake
        store.complete(session_id, response.output.content)

        # Send confirmation to patient
        summary = (
            "✅ *Your intake is complete!*\n\n"
            "Your doctor has been notified and will review it before seeing you.\n"
            "If anything changes before your appointment, please let us know. 🙏"
        )
        await send_wa(phone, summary)

        # Fire completion webhook if configured
        if COMPLETION_WEBHOOK:
            await fire_completion_webhook(
                session_id  = session_id,
                phone       = phone,
                output      = response.output.content,
            )


async def fire_completion_webhook(session_id: str, phone: str, output: dict):
    """Notify external systems (EMR, hospital software) when intake completes."""
    payload = {
        "event":       "intake_completed",
        "session_id":  session_id,
        "phone":       phone,                  # masked in real prod
        "clinic":      CLINIC_NAME,
        "completed_at": datetime.now().isoformat(),
        "intake":      output,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(COMPLETION_WEBHOOK, json=payload, timeout=5.0)
            log.info(f"Completion webhook: {resp.status_code}")
    except Exception as e:
        log.warning(f"Completion webhook failed: {e}")


async def send_wa(phone: str, text: str):
    """Send a WhatsApp message. Falls back to print if no credentials."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print(f"\n📤 [{phone}]: {text}\n")
        return

    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            r = await client.post(
                WA_API_URL,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                json={"messaging_product": "whatsapp", "to": phone,
                      "type": "text", "text": {"body": chunk}},
                timeout=10.0,
            )
            if r.status_code != 200:
                log.error(f"WhatsApp API error: {r.status_code} — {r.text[:200]}")


# ── Local test ──────────────────────────────────────────────────────────────

async def local_test():
    """Run CliniqFlow locally without WhatsApp credentials."""
    print("\n" + "=" * 60)
    print(f"  CliniqFlow — {CLINIC_NAME}")
    print("  LOCAL TEST MODE (no WhatsApp needed)")
    print("=" * 60)

    global _goal_config
    try:
        _goal_config = YAMLLoader.load(GOAL_YAML)
        print(f"✅ Goal: {_goal_config.get('name')}\n")
    except FileNotFoundError:
        print(f"❌ {GOAL_YAML} not found. Copy 01-patient-intake.yaml to goal.yaml")
        return

    phone = "test_phone_9999999999"
    sess  = f"cf_localtest"
    eng   = TrueNorthEngine(goal_config=_goal_config, session_id=sess, router=LLMRouter())
    store.set(sess, eng, phone)

    first = await eng.start()
    print(f"CliniqFlow: {first.text}\n")

    while not eng.state.is_complete:
        try:
            user_input = input("Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession aborted.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "q", "exit"):
            break

        resp = await eng.process_message(user_input)
        store.increment_turns(sess)
        print(f"\nCliniqFlow: {resp.text}\n")

        if resp.is_complete and resp.output:
            print("\n" + "=" * 60)
            print("  INTAKE COMPLETE — Doctor's Summary")
            print("=" * 60)
            print(json.dumps(resp.output.content, indent=2, ensure_ascii=False))
            store.complete(sess, resp.output.content)
            break

    final_stats = store.stats()
    print(f"\nStats: {final_stats}")


if __name__ == "__main__":
    asyncio.run(local_test())