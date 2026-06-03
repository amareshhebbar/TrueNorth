"""
PaisaCoach — Personal Finance Advisor for First-Time Earners
=============================================================

WHAT THIS DOES
--------------
A personal finance AI for India's 23-35 year-old first-time earners.

The problem:
  Most young professionals have no idea how to budget, invest, or
  get insurance. Banks don't explain. Parents give generic advice.
  Financial advisors want a minimum corpus of ₹50 lakh.

PaisaCoach gives everyone the advice that used to require a ₹5,000/hr
financial advisor — for free. In plain Hindi or English.

It collects:
  - Income, expenses, existing savings/debt
  - Financial goals (emergency fund, marriage, house, retirement)
  - Risk appetite (explained in simple terms — no jargon)
  - Tax regime and investment history

It outputs:
  - Month-wise savings plan
  - Specific investment recommendations (PPF, ELSS, NPS, FD)
  - Insurance gap analysis (term life, health)
  - Step-by-step action plan for the next 30 days

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic

FILE STRUCTURE
--------------
    paisacoach/
    ├── app.py    ← this file
    └── .env

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    APP_NAME=PaisaCoach
    WA_VERIFY_TOKEN=paisa-secret
    WA_ACCESS_TOKEN=your-whatsapp-token
    WA_PHONE_NUMBER_ID=your-phone-number-id

HOW TO RUN
----------
    python app.py                    # local interactive test
    DEMO_LANG=hindi python app.py    # demo in Hindi
    uvicorn app:app --host 0.0.0.0 --port 8080

SAMPLE CONVERSATION
--------------------
    PaisaCoach: Namaste! I am PaisaCoach.
                I will help you build a simple, realistic plan
                for your money. No jargon. No judgment.
                Your monthly take-home salary?
    User:       55,000
    PaisaCoach: Great. How much do you spend on rent?
    User:       18,000
    PaisaCoach: Any loan EMIs?
    User:       Car loan, 8,000 per month
    PaisaCoach: What is your most important financial goal?
    User:       Build an emergency fund first
    ...
    PaisaCoach: ✅ Here is your plan, Rahul:

    SAMPLE OUTPUT
    ─────────────────────────────────────────────────────────
    Monthly surplus:   ₹12,000 (after rent, EMI, basic expenses)
    Emergency fund:    Target ₹1.65L (3 months expenses)
                       Save ₹5,000/month → ready in 11 months
    Investments:       ₹5,000/month → ELSS SIP (80C + growth)
    Insurance gap:     Term life ₹50L cover needed (₹400/month)
    30-day actions:    1. Open PPF at post office
                       2. Start ₹500/week SIP in NIFTY 50 index
                       3. Get term insurance quote on PolicyBazaar

LEGAL DISCLAIMER
-----------------
    PaisaCoach provides general financial education, not registered
    investment advice under SEBI. Always consult a SEBI-registered
    financial advisor before making investment decisions.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paisacoach")

APP_NAME        = os.environ.get("APP_NAME", "PaisaCoach")
VERIFY_TOKEN    = os.environ.get("WA_VERIFY_TOKEN", "paisa-token")
ACCESS_TOKEN    = os.environ.get("WA_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("WA_PHONE_NUMBER_ID", "")
DEMO_LANG       = os.environ.get("DEMO_LANG", "english")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ── Goal config ───────────────────────────────────────────────────────────────

GOAL = {
    "id": "paisa_coach",
    "name": f"{APP_NAME}",
    "persona": {
        "name": APP_NAME,
        "tone": "friendly and honest — like a knowledgeable friend, not a salesperson",
        "empathy_level": "high",
        "language": "auto",
        "greeting": (
            f"Namaste! / Hello! I am {APP_NAME}.\n"
            "I will help you build a simple, realistic plan for your money.\n"
            "No jargon. No judgment. No selling.\n"
            "Let us start. What is your monthly take-home salary (after tax)?"
        ),
    },
    "fields": [
        {"name": "name",               "type": "text",    "required": True,
         "question": "What is your name?"},
        {"name": "age",                "type": "integer",  "required": True, "min": 18, "max": 70,
         "question": "How old are you?"},
        {"name": "monthly_income",     "type": "integer",  "required": True, "min": 0, "max": 10000000,
         "question": "Monthly take-home salary in ₹ (after all tax deductions)?"},
        {"name": "rent",               "type": "integer",  "required": True, "min": 0,
         "question": "Monthly rent or home loan EMI? (0 if living with parents)"},
        {"name": "other_emis",         "type": "integer",  "required": True, "min": 0,
         "question": "Any other loan EMIs per month? (personal loan, car loan, credit card — total)"},
        {"name": "monthly_expenses",   "type": "integer",  "required": True, "min": 0,
         "question": "Rough estimate of all other monthly expenses? (food, transport, utilities, phone — total excluding rent and EMIs)"},
        {"name": "current_savings",    "type": "integer",  "required": True, "min": 0,
         "question": "Total savings and investments right now? (FD, savings account, mutual funds, stocks — approximate total in ₹)"},
        {"name": "emergency_fund_months","type": "integer", "required": False, "min": 0, "max": 24,
         "question": "How many months of expenses do you have saved as an emergency fund? (0 if you do not have one)"},
        {"name": "primary_goal",       "type": "text",    "required": True,
         "allowed_values": ["emergency fund", "marriage", "house down payment",
                            "retirement", "child education", "car", "travel", "general savings"],
         "question": "What is your most important financial goal right now? (emergency fund / marriage / house / retirement / child education / car / travel / general savings)"},
        {"name": "goal_timeline_years","type": "number",  "required": True, "min": 0.5, "max": 40,
         "question": "By when do you want to achieve that goal? (in years — e.g. 2, 5, 10)"},
        {"name": "goal_amount",        "type": "integer",  "required": False, "min": 0,
         "question": "Approximate target amount for that goal in ₹? (rough estimate — say 'not sure' if you do not know)"},
        {"name": "risk_appetite",      "type": "text",    "required": True,
         "allowed_values": ["very conservative", "conservative", "moderate", "aggressive"],
         "question": "If you invested ₹1 lakh and it dropped to ₹80,000 in one year, what would you do?\nA) Panic and sell → very conservative\nB) Worry but hold → conservative\nC) Hold calmly → moderate\nD) Buy more → aggressive"},
        {"name": "has_term_insurance", "type": "boolean", "required": True,
         "question": "Do you have term life insurance? (yes/no)"},
        {"name": "has_health_insurance","type": "boolean","required": True,
         "question": "Do you have health/medical insurance beyond employer cover? (yes/no)"},
        {"name": "tax_regime",         "type": "text",    "required": False,
         "allowed_values": ["old regime", "new regime", "not sure"],
         "question": "Which income tax regime are you on? (old / new / not sure)"},
        {"name": "existing_investments","type": "text",   "required": False,
         "question": "Any existing investments? (PF, PPF, ELSS, FD, stocks, gold — list or say 'none')"},
    ],
    "output": {
        "format": "json",
        "template": (
            "You are a certified financial planner for India.\n"
            "IMPORTANT: Provide GENERAL FINANCIAL EDUCATION only. Not SEBI-registered advice.\n\n"
            "Client: {name}, age {age}.\n"
            "Income: ₹{monthly_income}/month.\n"
            "Expenses: rent ₹{rent} + EMIs ₹{other_emis} + others ₹{monthly_expenses}.\n"
            "Savings: ₹{current_savings} total. Emergency fund: {emergency_fund_months} months.\n"
            "Goal: {primary_goal} in {goal_timeline_years} years (target: ₹{goal_amount}).\n"
            "Risk: {risk_appetite}. Insurance: term={has_term_insurance}, health={has_health_insurance}.\n"
            "Tax regime: {tax_regime}. Current investments: {existing_investments}.\n\n"
            "Return JSON with:\n"
            "- monthly_surplus (income minus all expenses)\n"
            "- surplus_allocation (how to split surplus: emergency/goal/investments — percentages and amounts)\n"
            "- emergency_fund_plan (months to build it, monthly amount, where to keep — liquid FD/savings)\n"
            "- primary_goal_plan (monthly savings needed, investment vehicle, timeline reality check)\n"
            "- investment_recommendations (list: each with instrument, amount, reason, tax benefit, risk)\n"
            "  Include: ELSS, PPF, NPS, index funds, FD — whatever fits their profile\n"
            "- insurance_gaps (term life recommendation: cover amount + monthly cost; health gap if any)\n"
            "- tax_saving_opportunities (80C, 80D, NPS 80CCD1B — specific amounts they can save)\n"
            "- thirty_day_action_plan (5 specific actions with exact steps and links)\n"
            "- whatsapp_summary (the full plan as a clean WhatsApp message in the user's language)\n"
            "- disclaimer (one-line: not SEBI-registered advice)"
        ),
    },
}

# ── Storage ───────────────────────────────────────────────────────────────────

_sessions: Dict[str, TrueNorthEngine] = {}
_plans:    List[dict]                  = []


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title=APP_NAME)


@app.on_event("startup")
async def startup():
    log.info(f"✅ {APP_NAME} ready")
    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — console mode")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    total_surplus = sum(p.get("monthly_surplus", 0) for p in _plans if isinstance(p.get("monthly_surplus"), (int, float)))
    avg_surplus   = total_surplus / len(_plans) if _plans else 0

    return f"""
    <html><head><title>{APP_NAME}</title>
    <style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#fffbeb;}}
    h1{{color:#92400e;}} .s{{display:inline-block;background:white;border-radius:10px;padding:20px;
    margin:8px;border:2px solid #fde68a;text-align:center;}}
    .s h2{{margin:0;color:#d97706;font-size:1.8rem;}} .s p{{margin:4px 0 0;color:#64748b;font-size:.85rem;}}</style>
    </head><body>
    <h1>💰 {APP_NAME} — Personal Finance Advisor</h1>
    <p>Helping first-time earners build financial health. Free. No selling.</p>
    <div>
        <div class="s"><h2>{len(_sessions)}</h2><p>Active chats</p></div>
        <div class="s"><h2>{len(_plans)}</h2><p>Plans created</p></div>
        <div class="s"><h2>₹{avg_surplus:,.0f}</h2><p>Avg monthly surplus</p></div>
    </div>
    <p style="color:#6b7280;font-size:.85rem;margin-top:30px">
    ⚠️ Disclaimer: {APP_NAME} provides general financial education, not SEBI-registered investment advice.
    </p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "app": APP_NAME, "active": len(_sessions), "plans": len(_plans)}


@app.get("/webhook")
async def verify(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(p.get("hub.challenge"))
    raise HTTPException(403, "Bad token")


@app.post("/webhook")
async def handle(request: Request, background: BackgroundTasks):
    body = await request.json()
    try:
        messages = body["entry"][0]["changes"][0]["value"].get("messages", [])
        if messages:
            msg  = messages[0]
            phone = msg["from"]
            text  = msg.get("text", {}).get("body", "").strip()
            if text:
                background.add_task(process_message, phone, text)
        return {"status": "ok"}
    except (KeyError, IndexError):
        return {"status": "ignored"}


async def process_message(phone: str, text: str):
    sid    = f"pc_{hashlib.md5(phone.encode()).hexdigest()[:12]}"
    engine = _sessions.get(sid)

    if engine is None:
        engine = TrueNorthEngine(goal_config=GOAL, session_id=sid, router=LLMRouter())
        _sessions[sid] = engine
        first = await engine.start()
        await send_wa(phone, first.text)
        if text.lower() not in ("hi", "hello", "start", "namaste", "नमस्ते"):
            resp = await engine.process_message(text)
            await send_wa(phone, resp.text)
            await _check_done(sid, phone, resp)
        return

    resp = await engine.process_message(text)
    await send_wa(phone, resp.text)
    await _check_done(sid, phone, resp)


async def _check_done(sid: str, phone: str, response):
    if not (response.is_complete and response.output):
        return

    content = response.output.content or {}
    eng     = _sessions.get(sid)

    _plans.append({
        "session_id":     sid,
        "date":           datetime.now().strftime("%Y-%m-%d"),
        "name":           eng.state.collected_fields.get("name", "?") if eng else "?",
        "monthly_surplus": content.get("monthly_surplus", 0),
        "primary_goal":   eng.state.collected_fields.get("primary_goal", "?") if eng else "?",
    })

    # Send plan in user's language
    summary = content.get("whatsapp_summary", "")
    if not summary:
        surplus   = content.get("monthly_surplus", 0)
        inv_list  = content.get("investment_recommendations", [])
        actions   = content.get("thirty_day_action_plan", [])
        disclaimer= content.get("disclaimer", "Not SEBI-registered advice.")

        lines = [f"💰 *{APP_NAME} — Your Personal Finance Plan*\n"]
        lines.append(f"Monthly surplus: *₹{surplus:,}*\n")
        if inv_list:
            lines.append("*Recommended investments:*")
            for inv in inv_list[:3]:
                name   = inv.get("instrument", "?")
                amount = inv.get("amount", "?")
                reason = inv.get("reason", "")[:80]
                lines.append(f"• {name}: ₹{amount}/month — {reason}")
        if actions:
            lines.append("\n*30-day action plan:*")
            for a in actions[:3]:
                lines.append(f"✅ {a}")
        lines.append(f"\n_{disclaimer}_")
        summary = "\n".join(lines)

    await send_wa(phone, summary)
    del _sessions[sid]
    log.info(f"Plan delivered to {phone}")


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
            log.error(f"WA error {r.status_code}")


# ── Local test ────────────────────────────────────────────────────────────────

DEMO_INPUTS = {
    "english": [
        "Rahul",             # name
        "27",                # age
        "55000",             # monthly_income
        "18000",             # rent
        "8000",              # other_emis (car loan)
        "12000",             # monthly_expenses
        "40000",             # current_savings
        "0",                 # emergency_fund_months
        "emergency fund",    # primary_goal
        "1",                 # goal_timeline_years
        "165000",            # goal_amount (3 months × 55k)
        "moderate",          # risk_appetite
        "no",                # has_term_insurance
        "no",                # has_health_insurance
        "new regime",        # tax_regime
        "none",              # existing_investments
    ],
    "hindi": [
        "राहुल",
        "27",
        "55000",
        "18000",
        "8000",
        "12000",
        "40000",
        "0",
        "emergency fund",
        "1",
        "165000",
        "moderate",
        "नहीं",
        "नहीं",
        "new regime",
        "कुछ नहीं",
    ],
}


async def local_test():
    print("\n" + "=" * 60)
    print(f"  {APP_NAME} — Personal Finance Advisor")
    print(f"  Language: {DEMO_LANG}")
    print("=" * 60)

    turns    = DEMO_INPUTS.get(DEMO_LANG, DEMO_INPUTS["english"])
    sess     = "pc_local_test"
    engine   = TrueNorthEngine(goal_config=GOAL, session_id=sess, router=LLMRouter())
    _sessions[sess] = engine

    first = await engine.start()
    print(f"\n{APP_NAME}: {first.text}\n")

    interactive = os.environ.get("INTERACTIVE", "0") == "1"

    for ans in turns:
        if engine.state.is_complete:
            break

        if interactive:
            try:
                ans = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not ans or ans.lower() in ("q", "quit"):
                break
        else:
            print(f"You: {ans}")

        resp = await engine.process_message(ans)
        lang = resp.detected_language or "auto"
        print(f"\n{APP_NAME} [{lang}]: {resp.text}\n")

        if resp.is_complete and resp.output:
            print("=" * 60)
            print("  YOUR FINANCE PLAN")
            print("=" * 60)
            content = resp.output.content
            print(f"  Monthly surplus:  ₹{content.get('monthly_surplus', '?'):,}")
            print(f"  Primary goal:     {content.get('primary_goal_plan', {}).get('target', '?') if isinstance(content.get('primary_goal_plan'), dict) else '?'}")
            print()
            print(json.dumps(content, indent=2, ensure_ascii=False))
            break

    print(f"\nRun with INTERACTIVE=1 for your own profile:")
    print(f"  INTERACTIVE=1 python app.py")


if __name__ == "__main__":
    asyncio.run(local_test())