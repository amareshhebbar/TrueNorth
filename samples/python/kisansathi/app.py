"""
KisanSathi — AI Farm Advisor in Local Languages via WhatsApp
=============================================================

WHAT THIS DOES
--------------
An AI agriculture advisor that speaks the farmer's language.
Farmer messages a WhatsApp number describing their crop problem.
KisanSathi collects: location, crop, symptom, land size, weather.
Returns advice in the farmer's own language (Hindi, Kannada, Telugu, etc.)
matching ICAR guidelines.

This solves a real problem:
  120M+ small farmers in India
  Need expert agronomist advice
  Can't afford a consultant
  Speak in local languages, not English
  Already use WhatsApp every day

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic httpx

FILE STRUCTURE
--------------
    kisansathi/
    ├── app.py              ← this file
    ├── goal.yaml           ← copy 03-crop-advisory.yaml here
    └── .env

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    WA_VERIFY_TOKEN=kisansathi-secret
    WA_ACCESS_TOKEN=your-whatsapp-token
    WA_PHONE_NUMBER_ID=your-phone-number-id
    APP_NAME=KisanSathi
    DEFAULT_LANGUAGE=hindi
    EXTENSION_WORKER_WA=917xxxxxxxxx   # govt extension worker notified for critical cases

HOW TO RUN
----------
    # Local test (no WhatsApp needed)
    python app.py

    # Production
    cd kisansathi
    uvicorn app:app --host 0.0.0.0 --port 8080

    # With ngrok for WhatsApp testing
    ngrok http 8080

SAMPLE CONVERSATION (in Hindi)
--------------------------------
    Farmer: नमस्ते
    KisanSathi: नमस्ते! मैं KisanSathi हूँ — आपका खेती सलाहकार।
                आप किस राज्य से हैं और कौन सी फसल उगा रहे हैं?
    Farmer: मध्यप्रदेश से हूँ, मक्का की फसल है
    KisanSathi: कितनी जमीन पर? और क्या समस्या आ रही है?
    Farmer: 2 बीघा में है, पत्तियाँ पीली पड़ रही हैं
    KisanSathi: कब से यह हो रहा है? और खाद कौन सी डाली थी?
    Farmer: 10 दिन से, DAP डाला था बुवाई पर
    ...
    KisanSathi: ✅ आपकी मक्का की समस्या नाइट्रोजन की कमी लगती है।
                उपाय: 1 बीघे पर 25 किलो यूरिया डालें...

CRITICAL CASE ESCALATION
--------------------------
    If the agent detects an urgent crop emergency (80%+ crop loss risk),
    it notifies the EXTENSION_WORKER_WA with the farmer's details.
    The extension worker can call back within 24 hours.

SCALE
------
    This same app handles farmers across all Indian states.
    Language detected automatically per farmer.
    1 WhatsApp number serves all regions.
    Advice localised per state (different state-specific schemes, prices).
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("kisansathi")

# ── Config ───────────────────────────────────────────────────────────────────

APP_NAME        = os.environ.get("APP_NAME", "KisanSathi")
VERIFY_TOKEN    = os.environ.get("WA_VERIFY_TOKEN", "kisansathi-token")
ACCESS_TOKEN    = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WA_PHONE_NUMBER_ID", "")
GOAL_YAML       = os.environ.get("GOAL_YAML", "goal.yaml")
EXT_WORKER_WA   = os.environ.get("EXTENSION_WORKER_WA", "")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ── Inline goal (fallback if goal.yaml not found) ────────────────────────────

INLINE_GOAL = {
    "id": "crop_advisory",
    "name": f"{APP_NAME} Crop Advisory",
    "persona": {
        "name": APP_NAME,
        "tone": "patient, simple language — farmer-friendly, no technical jargon",
        "empathy_level": "high",
        "language": "auto",
        "greeting": (
            "नमस्ते! / Hello! / ನಮಸ್ಕಾರ! / வணக்கம்!\n"
            f"I am {APP_NAME}, your farm advisor.\n"
            "Please tell me about your crop problem — in any language you are comfortable with."
        ),
    },
    "fields": [
        {"name": "state_district", "type": "text", "required": True,
         "question": "Which state and district is your farm in?"},
        {"name": "crop_name",      "type": "text", "required": True,
         "question": "Which crop are you growing?"},
        {"name": "land_size",      "type": "text", "required": True,
         "question": "How much land? (in bigha, acre, or hectare — any unit)"},
        {"name": "sowing_date",    "type": "text", "required": True,
         "question": "When did you sow? (approximate date or 'X weeks ago')"},
        {"name": "problem",        "type": "text", "required": True,
         "question": "What problem are you seeing? Describe it in detail — leaf colour, which part, how many plants affected."},
        {"name": "problem_start",  "type": "text", "required": True,
         "question": "When did this start? Is it spreading fast?"},
        {"name": "fertiliser",     "type": "text", "required": False,
         "question": "What fertiliser have you applied and when? (or say 'none')"},
        {"name": "soil_type",      "type": "text", "required": False,
         "allowed_values": ["black cotton", "red soil", "sandy loam", "alluvial", "clay", "not sure"],
         "question": "What type of soil? (black cotton / red / sandy loam / alluvial / not sure)"},
    ],
    "output": {
        "format": "json",
        "template": (
            "You are an expert agricultural advisor trained on ICAR guidelines.\n"
            "Farmer from {state_district} grows {crop_name} on {land_size}, "
            "sown {sowing_date}.\n"
            "Problem: {problem}. Started: {problem_start}.\n"
            "Fertiliser: {fertiliser}. Soil: {soil_type}.\n\n"
            "Return JSON with:\n"
            "- likely_cause\n"
            "- urgency (URGENT/MODERATE/CAN_WAIT)\n"
            "- immediate_action (within 3 days)\n"
            "- treatment (step-by-step, quantities per acre)\n"
            "- products (generic names + Indian brand names)\n"
            "- cost_estimate_inr (per acre)\n"
            "- advisory_in_farmer_language (full advice in the language they used)\n"
            "- advisory_english (same in English for extension worker)\n"
            "- red_flag (true if crop loss >50% risk — triggers escalation)"
        ),
    },
}

# ── Storage ───────────────────────────────────────────────────────────────────

_sessions:  Dict[str, TrueNorthEngine] = {}
_advisories: List[dict]                 = []
_meta:       Dict[str, dict]            = {}
_goal_config = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title=f"{APP_NAME}", version="1.0.0")


@app.on_event("startup")
async def startup():
    global _goal_config
    try:
        _goal_config = YAMLLoader.load(GOAL_YAML)
        log.info(f"✅ Loaded: {_goal_config.get('name')}")
    except FileNotFoundError:
        _goal_config = INLINE_GOAL
        log.warning(f"goal.yaml not found — using inline goal ({APP_NAME})")

    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — console mode")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    today = datetime.now().strftime("%d %B %Y")
    today_adv = [a for a in _advisories if a.get("date") == datetime.now().strftime("%Y-%m-%d")]

    # Crop distribution
    crop_counts: Dict[str, int] = {}
    for a in _advisories:
        crop = a.get("crop", "other")
        crop_counts[crop] = crop_counts.get(crop, 0) + 1

    crop_rows = "".join(
        f"<tr><td>{c}</td><td>{n}</td></tr>"
        for c, n in sorted(crop_counts.items(), key=lambda x: -x[1])[:8]
    ) or "<tr><td colspan=2>No data yet</td></tr>"

    urgent = sum(1 for a in _advisories if a.get("urgency") == "URGENT")

    return f"""
    <html><head><title>{APP_NAME}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px;
                background: #f0fdf4; }}
        h1 {{ color: #15803d; }} .stat {{ display: inline-block; background: white;
             border-radius: 10px; padding: 20px; margin: 8px; text-align: center;
             border: 2px solid #86efac; min-width: 120px; }}
        .stat h2 {{ margin: 0; font-size: 2rem; color: #15803d; }}
        .stat p  {{ margin: 4px 0 0; color: #64748b; font-size: 0.9rem; }}
        .urgent {{ color: #dc2626 !important; }}
        table {{ width: 100%; border-collapse: collapse; background: white;
                 border-radius: 8px; overflow: hidden; }}
        th {{ background: #15803d; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #dcfce7; }}
        .green {{ color: #15803d; font-weight: bold; }}
    </style></head><body>
    <h1>🌾 {APP_NAME} — Farm Advisor Dashboard</h1>
    <p>{today}</p>
    <div>
        <div class="stat"><h2>{len(_sessions)}</h2><p>Active chats</p></div>
        <div class="stat"><h2>{len(today_adv)}</h2><p>Today's advisories</p></div>
        <div class="stat"><h2>{len(_advisories)}</h2><p>Total advisories</p></div>
        <div class="stat"><h2 class="{'urgent' if urgent > 0 else ''}">{urgent}</h2><p>Urgent cases</p></div>
    </div>
    <h2>Top Crops by Advisory Count</h2>
    <table><thead><tr><th>Crop</th><th>Advisory count</th></tr></thead>
    <tbody>{crop_rows}</tbody></table>
    <br>
    <p><small>
        <a href="/advisories">All advisories (JSON)</a> &nbsp;|&nbsp;
        <a href="/health">Health check</a>
    </small></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "app":             APP_NAME,
        "active_sessions": len(_sessions),
        "total_advisories": len(_advisories),
        "goal_loaded":     _goal_config is not None,
        "wa_connected":    bool(ACCESS_TOKEN and PHONE_NUMBER_ID),
    }


@app.get("/advisories")
async def all_advisories():
    return {"advisories": _advisories, "total": len(_advisories)}


@app.get("/webhook")
async def verify(request: Request):
    p         = request.query_params
    mode      = p.get("hub.mode")
    token     = p.get("hub.verify_token")
    challenge = p.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
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
    session_id = f"ks_{hashlib.md5(phone.encode()).hexdigest()[:12]}"
    engine     = _sessions.get(session_id)

    if engine is None:
        engine = TrueNorthEngine(
            goal_config = _goal_config or INLINE_GOAL,
            session_id  = session_id,
            router      = LLMRouter(),
        )
        _sessions[session_id] = engine
        _meta[session_id]     = {"phone": phone, "started": time.time()}

        first = await engine.start()
        await send_wa(phone, first.text)

        # Process their first message if it has content
        if text.lower() not in ("hi", "hello", "हेलो", "नमस्ते", "ನಮಸ್ಕಾರ", "start"):
            resp = await engine.process_message(text)
            await send_wa(phone, resp.text)
            await _check_complete(session_id, phone, resp)
        return

    resp = await engine.process_message(text)
    await send_wa(phone, resp.text)
    await _check_complete(session_id, phone, resp)


async def _check_complete(session_id: str, phone: str, response):
    if not (response.is_complete and response.final_output):
        return

    content  = response.final_output.content or {}
    lang     = _sessions.get(session_id)
    detected = lang.state.detected_language if lang else "unknown"

    # Store advisory
    advisory = {
        "session_id":     session_id,
        "date":           datetime.now().strftime("%Y-%m-%d"),
        "language":       detected,
        "crop":           content.get("crop_name", "unknown"),
        "urgency":        content.get("urgency", "MODERATE"),
        "likely_cause":   content.get("likely_cause", ""),
        "full_advisory":  content,
    }
    _advisories.append(advisory)

    # Send advice to farmer in their language
    farmer_advice = content.get("advisory_in_farmer_language", "")
    if not farmer_advice:
        # Fallback: structured summary
        farmer_advice = (
            f"✅ *{APP_NAME} Advisory*\n\n"
            f"*Problem:* {content.get('likely_cause', 'See details')}\n"
            f"*Urgency:* {content.get('urgency', 'MODERATE')}\n\n"
            f"*Immediate action:*\n{content.get('immediate_action', '—')}\n\n"
            f"*Treatment:*\n{content.get('treatment', '—')}\n\n"
            f"*Estimated cost:* ₹{content.get('cost_estimate_inr', '?')} per acre"
        )

    await send_wa(phone, farmer_advice)

    # Escalate URGENT cases to extension worker
    if content.get("red_flag") and EXT_WORKER_WA:
        eng_advice = content.get("advisory_english", "No English advisory")
        alert = (
            f"🚨 *{APP_NAME} URGENT ESCALATION*\n"
            f"Phone: {phone}\n"
            f"Language: {detected}\n"
            f"Problem: {content.get('likely_cause', '?')}\n\n"
            f"English advisory:\n{eng_advice[:800]}"
        )
        await send_wa(EXT_WORKER_WA, alert)
        log.warning(f"URGENT case escalated for {phone}")

    del _sessions[session_id]
    log.info(f"Advisory delivered to {phone}")


async def send_wa(phone: str, text: str):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print(f"\n📤 [{phone}]: {text[:200]}\n")
        return
    async with httpx.AsyncClient() as client:
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            r = await client.post(
                WA_API_URL,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                json={"messaging_product": "whatsapp", "to": phone,
                      "type": "text", "text": {"body": chunk}},
                timeout=10.0,
            )
            if r.status_code != 200:
                log.error(f"WhatsApp error {r.status_code}")


# ── Local test ────────────────────────────────────────────────────────────────

async def local_test():
    print("\n" + "=" * 60)
    print(f"  {APP_NAME} — Farm Advisor")
    print("  LOCAL TEST (type in Hindi, Kannada, or English)")
    print("=" * 60)

    global _goal_config
    try:
        _goal_config = YAMLLoader.load(GOAL_YAML)
    except FileNotFoundError:
        _goal_config = INLINE_GOAL
        print(f"Using inline goal (copy 03-crop-advisory.yaml to goal.yaml for full version)\n")

    sess   = "local_ks_test"
    engine = TrueNorthEngine(goal_config=_goal_config, session_id=sess, router=LLMRouter())
    _sessions[sess] = engine

    first = await engine.start()
    print(f"\n{APP_NAME}: {first.text}\n")

    while not engine.state.is_complete:
        try:
            user_input = input("Farmer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break

        resp = await engine.process_message(user_input)
        detected = resp.detected_language or "auto"
        print(f"\n{APP_NAME} [{detected}]: {resp.text}\n")

        if resp.is_complete and resp.final_output:
            print("=" * 60)
            print("ADVISORY GENERATED:")
            print("=" * 60)
            content = resp.final_output.content
            print(json.dumps(content, indent=2, ensure_ascii=False))

            if content.get("red_flag"):
                print("\n⚠️  RED FLAG: Critical case — extension worker would be notified")
            break


if __name__ == "__main__":
    asyncio.run(local_test())