"""
GarageBot — Two-Wheeler Service Booking via WhatsApp
======================================================

WHAT THIS DOES
--------------
Replaces phone calls and WhatsApp chaos for a bike workshop.
Customer messages the workshop's WhatsApp number.
TrueNorth collects: bike details, complaint, preferred slot.
Workshop gets a structured job card before the bike arrives.
Works for cars too (set VEHICLE_TYPE=car in .env).

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic httpx

FILE STRUCTURE
--------------
    garagebot/
    ├── app.py                ← this file (complete server)
    ├── goal.yaml             ← booking goal (created automatically on first run)
    └── .env                  ← your credentials

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    WA_VERIFY_TOKEN=garagebot-secret
    WA_ACCESS_TOKEN=your-whatsapp-token
    WA_PHONE_NUMBER_ID=your-phone-number-id
    GARAGE_NAME=Speed Motors
    VEHICLE_TYPE=bike             # bike or car
    SLOTS=9am,11am,2pm,4pm        # available booking slots
    TECHNICIAN_WHATSAPP=917xxxxxxxxx   # gets job card notification

HOW TO RUN
----------
    # Terminal 1: start the server
    cd garagebot
    uvicorn app:app --host 0.0.0.0 --port 8080

    # Terminal 2: expose for WhatsApp (development)
    ngrok http 8080

    # Local test (no WhatsApp needed)
    python app.py

WHAT THE CONVERSATION LOOKS LIKE
---------------------------------
    Customer: Hi, need to service my bike
    GarageBot: Welcome to Speed Motors! What is the make and model
               of your bike?
    Customer:  Honda Activa 6G
    GarageBot: What year is it?
    Customer:  2022
    GarageBot: What is the current kilometer reading?
    Customer:  8,200 km
    GarageBot: What is the main issue or service you need?
               (e.g. regular service, brake noise, engine trouble)
    Customer:  Regular service is due and the brakes feel a bit soft
    GarageBot: Got it! We have slots available: 9am, 11am, 2pm, 4pm.
               Which works for you tomorrow?
    Customer:  11am please
    GarageBot: ✅ Booked! Honda Activa 6G (2022) — 11am tomorrow.
               Your job card number is JC-20241215-003.
               We will send a reminder tonight.

JOB CARD FORMAT (sent to technician)
-------------------------------------
    JC-20241215-003 | 11:00 AM
    ────────────────────────────
    Vehicle:    Honda Activa 6G 2022
    KM:         8,200
    Customer:   Ravi Kumar (+91 98765 43210)
    Complaint:  Regular service + soft brakes
    Priority:   NORMAL
    Notes:      —

API ENDPOINTS
-------------
    GET  /           → Dashboard with today's job cards
    GET  /health     → Health check
    GET  /webhook    → WhatsApp verification
    POST /webhook    → Incoming WhatsApp messages
    GET  /jobcards   → All job cards (for workshop management system)
    GET  /jobcards/{date} → Job cards for a specific date

SCALE TO MULTIPLE WORKSHOPS
----------------------------
    Deploy one instance per workshop with different .env files.
    Or pass workshop_id as a query param and route in the webhook handler.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("garagebot")

GARAGE_NAME        = os.environ.get("GARAGE_NAME", "Speed Motors")
VEHICLE_TYPE       = os.environ.get("VEHICLE_TYPE", "bike")
SLOTS              = os.environ.get("SLOTS", "9am,11am,2pm,4pm").split(",")
VERIFY_TOKEN       = os.environ.get("WA_VERIFY_TOKEN", "garagebot-token")
ACCESS_TOKEN       = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_NUMBER_ID    = os.environ.get("WA_PHONE_NUMBER_ID", "")
TECH_WA            = os.environ.get("TECHNICIAN_WHATSAPP", "")
GOAL_YAML          = os.environ.get("GOAL_YAML", "goal.yaml")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

DEFAULT_GOAL = f"""
id: garagebot_{VEHICLE_TYPE}_booking
name: "{GARAGE_NAME} Booking"
version: "1.0.0"

persona:
  name: "{GARAGE_NAME} Assistant"
  tone: "friendly and efficient"
  empathy_level: medium
  language: auto
  greeting: >
    Welcome to {GARAGE_NAME}! I can help you book your {VEHICLE_TYPE} service.
    This will take about 2 minutes.

fields:
  - name: customer_name
    type: text
    required: true
    question: "What is your name?"

  - name: vehicle_make_model
    type: text
    required: true
    question: "What is the make and model of your {VEHICLE_TYPE}?"

  - name: vehicle_year
    type: integer
    required: true
    min: 2000
    max: 2025
    question: "What year is the {VEHICLE_TYPE}?"

  - name: kilometer_reading
    type: integer
    required: true
    min: 0
    max: 500000
    question: "What is the current kilometer reading on the odometer?"

  - name: service_complaint
    type: text
    required: true
    question: "What is the main issue or what service do you need? (e.g. regular service, brake issue, engine noise)"

  - name: last_service_km
    type: integer
    required: false
    question: "When did you last get it serviced? (km reading or say 'not sure')"

  - name: preferred_slot
    type: text
    required: true
    allowed_values: {json.dumps(SLOTS)}
    question: "We have these slots for tomorrow: {', '.join(SLOTS)}. Which works for you?"

  - name: pickup_needed
    type: boolean
    required: true
    question: "Do you need us to pick up the {VEHICLE_TYPE}, or will you drop it off? (pickup/dropoff)"

output:
  format: json
  template: >
    Create a workshop job card for the following booking.
    Customer: {{customer_name}}. Vehicle: {{vehicle_make_model}} {{vehicle_year}}.
    KM: {{kilometer_reading}}. Last service at: {{last_service_km}} km.
    Complaint: {{service_complaint}}.
    Slot: {{preferred_slot}}. Pickup needed: {{pickup_needed}}.
    Return JSON with: job_card_id (generate like JC-YYYYMMDD-NNN),
    vehicle_summary, work_required (list of tasks), priority (LOW/NORMAL/HIGH),
    estimated_duration_hours, customer_confirmation_message (short WhatsApp-ready text)
"""

def ensure_goal_yaml():
    """Create goal.yaml if it doesn't exist."""
    if not os.path.exists(GOAL_YAML):
        with open(GOAL_YAML, "w") as f:
            f.write(DEFAULT_GOAL)
        log.info(f"✅ Created {GOAL_YAML} with default booking goal")

_sessions:  Dict[str, TrueNorthEngine] = {}
_job_cards: List[dict]                  = []
_meta:      Dict[str, dict]             = {}
_goal_config = None

app = FastAPI(title=f"GarageBot — {GARAGE_NAME}", version="1.0.0")

@app.on_event("startup")
async def startup():
    global _goal_config
    ensure_goal_yaml()
    _goal_config = YAMLLoader.load(GOAL_YAML)
    log.info(f"✅ Loaded: {_goal_config.get('name')}")
    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — console mode")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    today_cards = [j for j in _job_cards if j.get("date") == datetime.now().strftime("%Y-%m-%d")]
    rows = "".join(
        f"<tr><td>{j.get('time','?')}</td><td><b>{j.get('vehicle','?')}</b></td>"
        f"<td>{j.get('complaint','?')[:40]}</td><td>{j.get('priority','?')}</td></tr>"
        for j in sorted(today_cards, key=lambda x: x.get("time", ""))
    ) or "<tr><td colspan=4>No bookings yet today</td></tr>"

    return f"""
    <html><head><title>GarageBot — {GARAGE_NAME}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1e293b; }} table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #1e40af; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
        tr:hover {{ background: #f8fafc; }}
        .stat {{ display: inline-block; background: #fff7ed; border-radius: 8px;
                 padding: 15px 25px; margin: 5px; text-align: center; border: 1px solid #fed7aa; }}
        .stat h2 {{ margin: 0; color: #ea580c; font-size: 1.8rem; }}
    </style></head><body>
    <h1>🔧 GarageBot — {GARAGE_NAME}</h1>
    <div>
        <div class="stat"><h2>{len(_sessions)}</h2><p>Active chats</p></div>
        <div class="stat"><h2>{len(today_cards)}</h2><p>Bookings today</p></div>
        <div class="stat"><h2>{len(_job_cards)}</h2><p>Total bookings</p></div>
    </div>
    <h2>Today's Job Cards</h2>
    <table><thead><tr><th>Time</th><th>Vehicle</th><th>Complaint</th><th>Priority</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <p><small><a href="/jobcards">All job cards (JSON)</a> &nbsp;|&nbsp; <a href="/health">Health</a></small></p>
    </body></html>
    """

@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "garage":      GARAGE_NAME,
        "vehicle":     VEHICLE_TYPE,
        "slots":       SLOTS,
        "active_chats": len(_sessions),
        "bookings_today": len([j for j in _job_cards
                               if j.get("date") == datetime.now().strftime("%Y-%m-%d")]),
    }

@app.get("/jobcards")
async def all_job_cards():
    return {"job_cards": _job_cards, "total": len(_job_cards)}

@app.get("/jobcards/{date}")
async def job_cards_for_date(date: str):
    cards = [j for j in _job_cards if j.get("date") == date]
    return {"date": date, "job_cards": cards, "total": len(cards)}

@app.get("/webhook")
async def verify(request: Request):
    p         = request.query_params
    mode      = p.get("hub.mode")
    token     = p.get("hub.verify_token")
    challenge = p.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("✅ WhatsApp webhook verified")
        return PlainTextResponse(challenge)
    raise HTTPException(403, "Bad token")

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
        if text:
            background.add_task(process_message, phone, text)
        return {"status": "ok"}
    except (KeyError, IndexError):
        return {"status": "ignored"}

async def process_message(phone: str, text: str):
    session_id = f"gb_{hashlib.md5(phone.encode()).hexdigest()[:12]}"
    engine     = _sessions.get(session_id)

    if engine is None:
        if _goal_config is None:
            await send_wa(phone, "Service booking system loading — please try again in a moment.")
            return
        engine = TrueNorthEngine(
            goal_config = _goal_config,
            session_id  = session_id,
            router      = LLMRouter(),
        )
        _sessions[session_id] = engine
        _meta[session_id]     = {"phone": phone, "started": time.time()}

        first = await engine.start()
        await send_wa(phone, first.text)
        if text.lower() not in ("hi", "hello", "hey", "start", "book"):
            resp = await engine.process_message(text)
            await send_wa(phone, resp.text)
            await _check_done(session_id, phone, resp)
        return

    resp = await engine.process_message(text)
    await send_wa(phone, resp.text)
    await _check_done(session_id, phone, resp)

async def _check_done(session_id: str, phone: str, response):
    if not (response.is_complete and response.final_output):
        return

    content = response.final_output.content or {}

    job_card = {
        "session_id":  session_id,
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "time":        content.get("preferred_slot", "?"),
        "vehicle":     content.get("vehicle_summary", "?"),
        "complaint":   content.get("work_required", ["?"])[0] if isinstance(
                           content.get("work_required"), list) else str(content.get("work_required", "?")),
        "priority":    content.get("priority", "NORMAL"),
        "customer":    phone,
        "full_card":   content,
    }
    _job_cards.append(job_card)
    log.info(f"✅ Job card created: {content.get('job_card_id', session_id)}")

    confirmation = content.get(
        "customer_confirmation_message",
        f"✅ Booking confirmed! See you tomorrow. Reference: {content.get('job_card_id', session_id)}"
    )
    await send_wa(phone, f"*{GARAGE_NAME}* — {confirmation}")

    if TECH_WA:
        tech_msg = (
            f"🔧 *New Job Card*\n"
            f"ID: {content.get('job_card_id', session_id)}\n"
            f"Time: {content.get('preferred_slot', '?')}\n"
            f"Vehicle: {content.get('vehicle_summary', '?')}\n"
            f"Work: {', '.join(content.get('work_required', ['?']))}\n"
            f"Priority: {content.get('priority', 'NORMAL')}"
        )
        await send_wa(TECH_WA, tech_msg)

    del _sessions[session_id]

async def send_wa(phone: str, text: str):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print(f"\n📤 [{phone}]: {text}\n")
        return
    async with httpx.AsyncClient() as client:
        r = await client.post(
            WA_API_URL,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json={"messaging_product": "whatsapp", "to": phone,
                  "type": "text", "text": {"body": text[:4000]}},
            timeout=10.0,
        )
        if r.status_code != 200:
            log.error(f"WhatsApp error: {r.status_code}")

async def local_test():
    print("\n" + "=" * 60)
    print(f"  GarageBot — {GARAGE_NAME}")
    print("  LOCAL TEST MODE")
    print("=" * 60)

    global _goal_config
    ensure_goal_yaml()
    _goal_config = YAMLLoader.load(GOAL_YAML)
    print(f"✅ Goal: {_goal_config.get('name')}\n")

    phone     = "test_customer_9999"
    sess_id   = "gb_localtest"
    engine    = TrueNorthEngine(goal_config=_goal_config, session_id=sess_id, router=LLMRouter())
    _sessions[sess_id] = engine

    first = await engine.start()
    print(f"GarageBot: {first.text}\n")

    while not engine.state.is_complete:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "q"):
            break

        resp = await engine.process_message(user_input)
        print(f"\nGarageBot: {resp.text}\n")

        if resp.is_complete and resp.final_output:
            print("\n" + "=" * 60)
            print("  JOB CARD CREATED")
            print("=" * 60)
            print(json.dumps(resp.final_output.content, indent=2, ensure_ascii=False))
            break

if __name__ == "__main__":
    asyncio.run(local_test())
