"""
ScholarFinder — Scholarship Eligibility for Indian Students
=============================================================

WHAT THIS DOES
--------------
Conversational scholarship eligibility finder for Indian students.
Student answers questions about their background.
ScholarFinder returns a personalised ranked list of scholarships
they qualify for — central government, state, and private.

The problem this solves:
  ₹40,000 crore+ in scholarships go unclaimed every year in India.
  Students don't know which schemes apply to them.
  Govt portals are complicated, English-only, and form-heavy.
  ScholarFinder: 5 minutes, any language, maximum eligibility found.

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic

FILE STRUCTURE
--------------
    scholarfinder/
    ├── app.py              ← this file
    ├── goal.yaml           ← copy 03-crop-advisory adjusted for education,
    │                         or use inline goal (auto-created)
    └── .env

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    APP_NAME=ScholarFinder
    WA_VERIFY_TOKEN=scholar-secret
    WA_ACCESS_TOKEN=your-whatsapp-token
    WA_PHONE_NUMBER_ID=your-phone-number-id

HOW TO RUN
----------
    # Local interactive test
    python app.py

    # Web server
    uvicorn app:app --host 0.0.0.0 --port 8080

    # Local test in Hindi
    DEMO_LANG=hindi python app.py

SAMPLE CONVERSATION
--------------------
    ScholarFinder: Namaste! I am here to find scholarships for you.
                   Let us start — what is your name?
    Student:       Anjali Patel
    ScholarFinder: Great Anjali! Which state are you from?
    Student:       Gujarat
    ScholarFinder: What category? (SC/ST/OBC/General/Minority)
    Student:       OBC
    ScholarFinder: What are your Class 12 marks percentage?
    Student:       82%
    ScholarFinder: Annual family income?
    Student:       Around 2.5 lakh
    ScholarFinder: Which course are you pursuing?
    Student:       B.Tech first year
    ...
    ScholarFinder: ✅ Anjali, you qualify for 8 scholarships!
                   Top matches:
                   1. Post-Matric Scholarship (OBC) — ₹10,000/year
                   2. NSP Central Sector Scholarship — ₹12,000/year
                   3. Gujarat State OBC Scholarship — ₹8,000/year
                   ...

HOW TO ADD MORE SCHOLARSHIPS
------------------------------
    Edit the SCHOLARSHIP_DATABASE list in this file.
    Each entry has: name, amount, eligibility criteria.
    The LLM matches student profile to eligibility criteria.

    For a production version: connect to National Scholarship Portal (NSP) API.
    NSP API: https://scholarships.gov.in/fresh/newstudentregfresh

WHATSAPP DEPLOYMENT
--------------------
    Set WA_* env vars and run as a FastAPI server.
    Students message your WhatsApp number.
    Session runs automatically.
    Results sent back as a formatted WhatsApp message.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scholarfinder")

# ── Config ───────────────────────────────────────────────────────────────────

APP_NAME        = os.environ.get("APP_NAME", "ScholarFinder")
VERIFY_TOKEN    = os.environ.get("WA_VERIFY_TOKEN", "scholar-token")
ACCESS_TOKEN    = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WA_PHONE_NUMBER_ID", "")
DEMO_LANG       = os.environ.get("DEMO_LANG", "english")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ── Scholarship database (sample — expand with real data) ────────────────────

SCHOLARSHIP_DATABASE = [
    {
        "name":        "NSP Post-Matric Scholarship (SC/ST)",
        "amount":      "₹7,000-₹23,400/year",
        "portal":      "scholarships.gov.in",
        "deadline":    "31 October",
        "eligibility": {
            "categories":  ["SC", "ST"],
            "min_marks":   0,
            "max_income":  250000,
            "courses":     ["any"],
        },
    },
    {
        "name":        "NSP Post-Matric Scholarship (OBC)",
        "amount":      "₹1,000-₹10,000/year",
        "portal":      "scholarships.gov.in",
        "deadline":    "31 October",
        "eligibility": {
            "categories":  ["OBC"],
            "min_marks":   0,
            "max_income":  100000,
            "courses":     ["any"],
        },
    },
    {
        "name":        "Central Sector Scholarship (Merit-based)",
        "amount":      "₹10,000-₹20,000/year",
        "portal":      "scholarships.gov.in",
        "deadline":    "31 October",
        "eligibility": {
            "categories":  ["SC", "ST", "OBC", "General", "Minority"],
            "min_marks":   80,
            "max_income":  450000,
            "courses":     ["degree"],
        },
    },
    {
        "name":        "Inspire Scholarship (Science)",
        "amount":      "₹80,000/year",
        "portal":      "online-inspire.gov.in",
        "deadline":    "November",
        "eligibility": {
            "categories":  ["any"],
            "min_marks":   75,
            "max_income":  999999,
            "courses":     ["BSc", "BScHons", "integrated", "medicine"],
        },
    },
    {
        "name":        "PM Scholarship for Central Armed Police Forces",
        "amount":      "₹25,000-₹30,000/year",
        "portal":      "desw.gov.in",
        "deadline":    "October",
        "eligibility": {
            "categories":  ["any"],
            "min_marks":   60,
            "max_income":  999999,
            "courses":     ["degree", "diploma", "technical"],
            "special":     "parent in armed forces",
        },
    },
    {
        "name":        "Begum Hazrat Mahal National Scholarship (Minority Girls)",
        "amount":      "₹10,000-₹12,000/year",
        "portal":      "scholarships.gov.in",
        "deadline":    "September",
        "eligibility": {
            "categories":  ["Minority"],
            "gender":      "female",
            "min_marks":   50,
            "max_income":  200000,
            "courses":     ["Class 9 to 12"],
        },
    },
    {
        "name":        "Aicte Pragati Scholarship (Technical, Girls)",
        "amount":      "₹50,000/year",
        "portal":      "aicte-india.org",
        "deadline":    "October",
        "eligibility": {
            "categories":  ["any"],
            "gender":      "female",
            "min_marks":   0,
            "max_income":  800000,
            "courses":     ["B.Tech", "B.Arch", "B.Pharma", "hotel management", "applied arts"],
        },
    },
    {
        "name":        "Tata Capital Pankh Scholarship",
        "amount":      "₹12,000/year",
        "portal":      "buddy4study.com",
        "deadline":    "September",
        "eligibility": {
            "categories":  ["any"],
            "min_marks":   60,
            "max_income":  400000,
            "courses":     ["degree", "diploma", "vocational"],
        },
    },
    {
        "name":        "Sitaram Jindal Foundation Scholarship",
        "amount":      "₹1,500-₹3,000/month",
        "portal":      "sitaramjindalfoundation.org",
        "deadline":    "October",
        "eligibility": {
            "categories":  ["any"],
            "min_marks":   55,
            "max_income":  250000,
            "courses":     ["degree", "diploma", "vocational", "medical", "engineering"],
        },
    },
    {
        "name":        "Vigyan Pragati Scholarship (Science girls, rural)",
        "amount":      "₹3,000-₹5,000/month",
        "portal":      "buddy4study.com",
        "deadline":    "August",
        "eligibility": {
            "categories":  ["any"],
            "gender":      "female",
            "min_marks":   65,
            "max_income":  300000,
            "courses":     ["BSc", "BTech", "BPharm", "medical"],
        },
    },
]

# ── Inline goal ───────────────────────────────────────────────────────────────

INLINE_GOAL = {
    "id": "scholarship_finder",
    "name": f"{APP_NAME}",
    "persona": {
        "name": APP_NAME,
        "tone": "encouraging and clear — speak simply, like talking to a school student",
        "empathy_level": "high",
        "language": "auto",
        "greeting": (
            f"Namaste! I am {APP_NAME}.\n"
            "I will help you find scholarships you qualify for — in just 5 minutes.\n"
            "Thousands of students miss scholarships they deserve because they never applied.\n"
            "Let us make sure that does not happen to you! Ready?"
        ),
    },
    "fields": [
        {"name": "student_name",      "type": "text",    "required": True,
         "question": "What is your name?"},
        {"name": "state",             "type": "text",    "required": True,
         "question": "Which state are you from?"},
        {"name": "gender",            "type": "text",    "required": True,
         "allowed_values": ["male", "female", "other"],
         "question": "What is your gender? (male / female / other)"},
        {"name": "category",          "type": "text",    "required": True,
         "allowed_values": ["SC", "ST", "OBC", "General", "Minority", "EWS"],
         "question": "What is your category? (SC / ST / OBC / General / Minority / EWS)"},
        {"name": "class12_percent",   "type": "number",  "required": True, "min": 0, "max": 100,
         "question": "What was your Class 12 percentage? (or say 'currently in Class 12')"},
        {"name": "annual_income_lpa", "type": "number",  "required": True, "min": 0, "max": 50,
         "question": "What is your approximate family annual income in lakhs? (e.g. 2.5 means ₹2.5 lakh)"},
        {"name": "current_course",    "type": "text",    "required": True,
         "question": "What course are you currently pursuing? (e.g. B.Tech, BSc, BA, MBBS, Diploma, Class 11, etc.)"},
        {"name": "college_type",      "type": "text",    "required": True,
         "allowed_values": ["government", "private", "aided", "not yet admitted"],
         "question": "Is your college government, private, or aided?"},
        {"name": "disability",        "type": "boolean", "required": True,
         "question": "Do you have any disability? (yes/no)"},
        {"name": "parent_in_forces",  "type": "boolean", "required": False,
         "question": "Is any family member in the Indian Armed Forces or Police? (yes/no)"},
        {"name": "rural_area",        "type": "boolean", "required": False,
         "question": "Do you live in a rural area or village? (yes/no)"},
    ],
    "output": {
        "format": "json",
        "template": (
            f"You are a scholarship advisor for Indian students.\n"
            "Student: {student_name}, {gender}, {category} category, from {state}.\n"
            "Class 12: {class12_percent}%. Family income: {annual_income_lpa} lakh/year.\n"
            "Course: {current_course} at {college_type} college.\n"
            "Disability: {disability}. Parent in forces: {parent_in_forces}. Rural: {rural_area}.\n\n"
            f"Available scholarships: {json.dumps(SCHOLARSHIP_DATABASE, ensure_ascii=False)}\n\n"
            "Based on the student's profile, return JSON with:\n"
            "- matched_scholarships (list, each with: name, amount, deadline, portal, why_eligible)\n"
            "- total_potential_amount (sum of all matched amounts, give a range like ₹30,000-₹80,000/year)\n"
            "- top_recommendation (the single best scholarship to apply to first)\n"
            "- application_tip (one specific tip for this student's profile)\n"
            "- response_in_student_language (the full result as a WhatsApp-ready message in the language they used)\n"
            "- next_steps (3 action items with deadlines)"
        ),
    },
}

# ── Storage ───────────────────────────────────────────────────────────────────

_sessions:  Dict[str, TrueNorthEngine] = {}
_results:   List[dict]                  = []
_goal_config = None

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title=APP_NAME)


@app.on_event("startup")
async def startup():
    global _goal_config
    try:
        _goal_config = YAMLLoader.load("goal.yaml")
        log.info(f"Goal loaded: {_goal_config.get('name')}")
    except FileNotFoundError:
        _goal_config = INLINE_GOAL
        log.info(f"Using inline goal for {APP_NAME}")
    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — console mode")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    total_amount = 0  # rough count
    return f"""
    <html><head><title>{APP_NAME}</title>
    <style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;}}
    h1{{color:#1d4ed8;}} .stat{{display:inline-block;background:#eff6ff;border-radius:8px;
    padding:20px 30px;margin:8px;text-align:center;border:1px solid #bfdbfe;}}
    .stat h2{{margin:0;color:#1d4ed8;font-size:2rem;}} .stat p{{margin:4px 0 0;color:#64748b;}}</style>
    </head><body>
    <h1>🎓 {APP_NAME} — Scholarship Finder</h1>
    <div>
        <div class="stat"><h2>{len(_sessions)}</h2><p>Active sessions</p></div>
        <div class="stat"><h2>{len(_results)}</h2><p>Students helped</p></div>
        <div class="stat"><h2>{len(SCHOLARSHIP_DATABASE)}</h2><p>Scholarships in database</p></div>
    </div>
    <p>Students WhatsApp the number to find scholarships they qualify for.</p>
    <p><small><a href="/results">Results (JSON)</a> | <a href="/health">Health</a></small></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "app": APP_NAME,
            "active": len(_sessions), "helped": len(_results)}


@app.get("/results")
async def results():
    return {"results": _results}


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
    import hashlib
    session_id = f"sf_{hashlib.md5(phone.encode()).hexdigest()[:12]}"
    engine     = _sessions.get(session_id)

    if engine is None:
        engine = TrueNorthEngine(
            goal_config = _goal_config or INLINE_GOAL,
            session_id  = session_id,
            router      = LLMRouter(),
        )
        _sessions[session_id] = engine
        first = await engine.start()
        await send_wa(phone, first.text)
        if text.lower() not in ("hi", "hello", "start", "yes", "ready"):
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

    content = response.final_output.content or {}
    matched = content.get("matched_scholarships", [])
    amount  = content.get("total_potential_amount", "unknown")

    _results.append({
        "session_id":  session_id,
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "matched":     len(matched),
        "amount":      amount,
        "top_pick":    content.get("top_recommendation", {}).get("name", "?") if isinstance(
                          content.get("top_recommendation"), dict) else content.get("top_recommendation", "?"),
    })

    # Send result in student's language
    student_msg = content.get("response_in_student_language", "")
    if not student_msg:
        lines = [f"✅ *{APP_NAME} Results*\n"]
        lines.append(f"You qualify for *{len(matched)} scholarships!*")
        lines.append(f"Total potential: *{amount}/year*\n")
        for i, s in enumerate(matched[:5], 1):
            name   = s.get("name", "?")
            amount = s.get("amount", "?")
            why    = s.get("why_eligible", "")
            lines.append(f"{i}. *{name}*\n   Amount: {amount}\n   Why you qualify: {why}\n")
        steps = content.get("next_steps", [])
        if steps:
            lines.append("*Next steps:*")
            for step in steps[:3]:
                lines.append(f"• {step}")
        student_msg = "\n".join(lines)

    await send_wa(phone, student_msg)
    del _sessions[session_id]
    log.info(f"Scholar results delivered to {phone} — {len(matched)} scholarships matched")


async def send_wa(phone: str, text: str):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print(f"\n📤 [{phone}]: {text[:300]}\n")
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
            log.error(f"WA error {r.status_code}: {r.text[:100]}")


# ── Local test ────────────────────────────────────────────────────────────────

async def local_test():
    print("\n" + "=" * 60)
    print(f"  {APP_NAME} — Find your scholarships")
    print(f"  Language: {DEMO_LANG} (set DEMO_LANG= to change)")
    print("=" * 60)

    global _goal_config
    _goal_config = INLINE_GOAL

    # Demo answers per language
    demo_turns = {
        "english": [
            "Anjali Patel", "Gujarat", "female", "OBC",
            "82", "2.5", "B.Tech first year", "private",
            "no", "no", "no",
        ],
        "hindi": [
            "अंजली पटेल", "गुजरात", "female", "OBC",
            "82", "2.5", "बी.टेक प्रथम वर्ष", "private",
            "नहीं", "नहीं", "हाँ",
        ],
    }.get(DEMO_LANG, [
        "Anjali Patel", "Gujarat", "female", "OBC",
        "82", "2.5", "B.Tech first year", "private", "no", "no", "no",
    ])

    sess   = "sf_local_test"
    engine = TrueNorthEngine(goal_config=_goal_config, session_id=sess, router=LLMRouter())
    _sessions[sess] = engine

    first = await engine.start()
    print(f"\n{APP_NAME}: {first.text}\n")

    if os.environ.get("INTERACTIVE", "0") == "1":
        # Interactive mode
        while not engine.state.is_complete:
            try:
                user_input = input("Student: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("q", "quit"):
                break
            resp = await engine.process_message(user_input)
            print(f"\n{APP_NAME}: {resp.text}\n")
            if resp.is_complete and resp.final_output:
                print("=" * 60)
                print(json.dumps(resp.final_output.content, indent=2, ensure_ascii=False))
                break
    else:
        # Scripted demo
        for ans in demo_turns:
            if engine.state.is_complete:
                break
            print(f"Student: {ans}")
            resp = await engine.process_message(ans)
            print(f"\n{APP_NAME}: {resp.text}\n")
            if resp.is_complete and resp.final_output:
                print("=" * 60)
                print("SCHOLARSHIP RESULTS:")
                print("=" * 60)
                content = resp.final_output.content
                print(json.dumps(content, indent=2, ensure_ascii=False))

                matched = content.get("matched_scholarships", [])
                amount  = content.get("total_potential_amount", "?")
                print(f"\n✅ {len(matched)} scholarships matched — {amount}/year potential")
                break

    print("\nRun with INTERACTIVE=1 to type your own profile:")
    print(f"  INTERACTIVE=1 python app.py")


if __name__ == "__main__":
    asyncio.run(local_test())