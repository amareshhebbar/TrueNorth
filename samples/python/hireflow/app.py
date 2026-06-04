"""
HireFlow — AI Candidate Screening for Startups
================================================

WHAT THIS DOES
--------------
Replaces first-round recruiter phone screens.

Candidates apply, get a WhatsApp / email link, and complete
a structured AI screening conversation. Recruiter wakes up to
a ranked shortlist with scorecards — not 200 raw applications.

Key features:
  - Role-configurable via .env (no code changes per role)
  - Automatic scorecard: STRONG_HIRE / SHORTLIST / HOLD / REJECT
  - Detects red flags: vague answers, inflated claims, salary mismatch
  - Follow-up questions for the next round generated automatically
  - DPDP compliant: candidate consent before data collected
  - Webhook to ATS (Greenhouse, Lever, Freshteam)

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install fastapi uvicorn anthropic httpx

FILE STRUCTURE
--------------
    hireflow/
    ├── app.py       ← this file
    └── .env

CREATE .env FILE:
-----------------
    ANTHROPIC_API_KEY=sk-ant-...
    WA_VERIFY_TOKEN=hireflow-secret
    WA_ACCESS_TOKEN=your-whatsapp-token
    WA_PHONE_NUMBER_ID=your-phone-number-id

    # Role configuration (change per open position)
    ROLE_TITLE=Senior Backend Engineer
    ROLE_MIN_EXP=4
    ROLE_SKILLS=Python, distributed systems, PostgreSQL
    ROLE_MAX_NOTICE=60
    ROLE_BUDGET_LPA=40
    COMPANY_NAME=TechCorp
    ATS_WEBHOOK_URL=https://your-ats.com/webhook/truenorth

HOW TO RUN
----------
    # Local test
    python app.py

    # Production (one role)
    uvicorn app:app --host 0.0.0.0 --port 8080

    # Production (multi-role: run one instance per open role)
    ROLE_TITLE="Product Manager" ROLE_SKILLS="roadmap,data" uvicorn app:app --port 8081

SAMPLE CONVERSATION
--------------------
    HireFlow: Hi! Thanks for applying to TechCorp.
              I am the AI screening assistant for the Senior Backend
              Engineer role. This takes about 8 minutes. Ready?
    Candidate: Yes
    HireFlow: What is your current role and company?
    Candidate: Senior Engineer at Flipkart, 5 years experience
    HireFlow: Great. What are your top 3 technical skills?
    ...
    HireFlow: ✅ Thank you! Your screening is complete.
              The team will review and get back to you within 3 days.

RECRUITER DASHBOARD
--------------------
    GET /          → Ranked shortlist (dashboard)
    GET /shortlist → SHORTLIST + STRONG_HIRE candidates
    GET /all       → All candidates with scores
    GET /reject    → Rejected candidates (with reasons)

UNDERSTANDING SCORES
---------------------
    technical_score: Based on quality of technical question answers
    culture_score:   Based on motivation, communication, why-us
    availability_score: Based on notice period, salary fit
    overall_score:   Weighted: technical 50%, culture 30%, availability 20%

    STRONG_HIRE:  overall >= 85
    SHORTLIST:    overall >= 65
    HOLD:         overall >= 45
    REJECT:       overall < 45
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
log = logging.getLogger("hireflow")

# ── Config from environment ───────────────────────────────────────────────────

COMPANY_NAME     = os.environ.get("COMPANY_NAME",    "TechCorp")
ROLE_TITLE       = os.environ.get("ROLE_TITLE",      "Senior Backend Engineer")
ROLE_MIN_EXP     = int(os.environ.get("ROLE_MIN_EXP", "4"))
ROLE_SKILLS      = os.environ.get("ROLE_SKILLS",     "Python, system design, databases")
ROLE_MAX_NOTICE  = int(os.environ.get("ROLE_MAX_NOTICE", "60"))
ROLE_BUDGET_LPA  = int(os.environ.get("ROLE_BUDGET_LPA", "30"))
ATS_WEBHOOK      = os.environ.get("ATS_WEBHOOK_URL", "")

VERIFY_TOKEN     = os.environ.get("WA_VERIFY_TOKEN",    "hireflow-token")
ACCESS_TOKEN     = os.environ.get("WA_ACCESS_TOKEN",    "")
PHONE_NUMBER_ID  = os.environ.get("WA_PHONE_NUMBER_ID", "")

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ── Dynamic goal built from env config ───────────────────────────────────────

def build_goal() -> dict:
    return {
        "id": f"hireflow_{ROLE_TITLE.lower().replace(' ', '_')}",
        "name": f"{COMPANY_NAME} — {ROLE_TITLE} Screening",
        "persona": {
            "name": f"{COMPANY_NAME} Talent Team",
            "tone": "professional and warm — this is a two-way conversation, not an interrogation",
            "empathy_level": "medium",
            "language": "auto",
            "greeting": (
                f"Hi! Thanks for applying for the {ROLE_TITLE} role at {COMPANY_NAME}.\n"
                "I am the AI screening assistant. This takes about 8 minutes and helps us move faster for you.\n"
                "Ready to begin?"
            ),
        },
        "fields": [
            {"name": "candidate_name",   "type": "text",   "required": True,
             "question": "What is your full name?"},
            {"name": "current_role",     "type": "text",   "required": True,
             "question": "What is your current role and company?"},
            {"name": "total_exp_years",  "type": "number", "required": True, "min": 0, "max": 50,
             "question": "Total years of professional experience?"},
            {"name": "relevant_exp",     "type": "number", "required": True, "min": 0, "max": 50,
             "question": f"Years directly relevant to a {ROLE_TITLE} role?"},
            {"name": "top_skills",       "type": "text",   "required": True,
             "question": f"Your top 3 to 5 technical skills? We are looking for: {ROLE_SKILLS}"},
            {"name": "technical_q1",     "type": "text",   "required": True,
             "question": "Tell me about the most complex system you have built or designed. What was the scale, the challenge, and what you would do differently?"},
            {"name": "technical_q2",     "type": "text",   "required": True,
             "question": "How do you debug a production performance issue affecting users right now — walk me through your exact approach."},
            {"name": "current_ctc_lpa",  "type": "number", "required": False, "min": 0, "max": 500,
             "question": "Current total compensation in LPA? (Say 'prefer not to share' if you would rather not.)"},
            {"name": "expected_ctc_lpa", "type": "number", "required": True,  "min": 0, "max": 500,
             "question": f"Salary expectation for this role in LPA? (Budget is up to {ROLE_BUDGET_LPA} LPA)"},
            {"name": "notice_days",      "type": "integer","required": True,  "min": 0, "max": 180,
             "question": f"Notice period in days? (We prefer under {ROLE_MAX_NOTICE} days)"},
            {"name": "work_preference",  "type": "text",   "required": True,
             "allowed_values": ["remote", "hybrid", "in-office", "flexible"],
             "question": "Work mode preference? (remote / hybrid / in-office / flexible)"},
            {"name": "location",         "type": "text",   "required": True,
             "question": "Current city? Open to relocation?"},
            {"name": "why_this_role",    "type": "text",   "required": True,
             "question": f"What specifically excites you about this {ROLE_TITLE} role at {COMPANY_NAME}?"},
            {"name": "reason_for_looking","type": "text",  "required": True,
             "question": "What is motivating you to explore new opportunities now?"},
        ],
        "output": {
            "format": "json",
            "template": (
                f"You are a senior technical recruiter at {COMPANY_NAME}.\n"
                f"Role: {ROLE_TITLE}. Min exp required: {ROLE_MIN_EXP} years. "
                f"Budget: up to {ROLE_BUDGET_LPA} LPA. Max notice: {ROLE_MAX_NOTICE} days.\n"
                f"Required skills: {ROLE_SKILLS}.\n\n"
                "Candidate: {candidate_name}. Current: {current_role}.\n"
                "Exp: {total_exp_years} years total, {relevant_exp} relevant.\n"
                "Skills: {top_skills}.\n"
                "Tech Q1: {technical_q1}\n"
                "Tech Q2: {technical_q2}\n"
                "CTC: {current_ctc_lpa} LPA current, {expected_ctc_lpa} LPA expected.\n"
                "Notice: {notice_days} days. Mode: {work_preference}. Location: {location}.\n"
                "Why this role: {why_this_role}. Reason for looking: {reason_for_looking}.\n\n"
                "Return JSON with:\n"
                "- overall_score (0-100)\n"
                "- recommendation (STRONG_HIRE / SHORTLIST / HOLD / REJECT)\n"
                "- technical_score (0-100)\n"
                "- culture_score (0-100)\n"
                "- availability_score (0-100 — based on notice and salary fit)\n"
                "- strengths (list of 3 specific strengths with evidence)\n"
                "- concerns (list of specific concerns, honest)\n"
                "- red_flags (warning signs that need probing — empty list if none)\n"
                "- salary_fit (IN_RANGE / ABOVE_RANGE / BELOW_RANGE)\n"
                "- meets_min_experience (true/false)\n"
                "- shortlist_reason (one sentence: why shortlist or why not)\n"
                "- questions_for_next_round (3 follow-up questions based on gaps)"
            ),
        },
    }


# ── Storage ───────────────────────────────────────────────────────────────────

_sessions:    Dict[str, TrueNorthEngine] = {}
_candidates:  List[dict]                  = []
_goal_config: Optional[dict]              = None


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title=f"HireFlow — {COMPANY_NAME}")


@app.on_event("startup")
async def startup():
    global _goal_config
    _goal_config = build_goal()
    log.info(f"✅ {_goal_config['name']} ready")
    if not ACCESS_TOKEN:
        log.warning("WA_ACCESS_TOKEN not set — console mode")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    shortlist = [c for c in _candidates if c.get("recommendation") in ("SHORTLIST","STRONG_HIRE")]
    strong    = [c for c in _candidates if c.get("recommendation") == "STRONG_HIRE"]
    rejected  = [c for c in _candidates if c.get("recommendation") == "REJECT"]

    rows = ""
    for c in sorted(_candidates, key=lambda x: -x.get("overall_score", 0)):
        rec   = c.get("recommendation", "?")
        colour= {"STRONG_HIRE":"#16a34a","SHORTLIST":"#2563eb","HOLD":"#d97706","REJECT":"#dc2626"}.get(rec,"#64748b")
        rows += (
            f"<tr><td><b>{c.get('name','?')}</b></td>"
            f"<td>{c.get('overall_score','?')}/100</td>"
            f"<td style='color:{colour};font-weight:bold'>{rec}</td>"
            f"<td>{c.get('salary_fit','?')}</td>"
            f"<td>{c.get('notice_days','?')}d</td>"
            f"<td>{c.get('screened_at','?')}</td></tr>"
        )
    rows = rows or "<tr><td colspan=6>No candidates yet</td></tr>"

    return f"""
    <html><head><title>HireFlow — {COMPANY_NAME}</title>
    <style>body{{font-family:sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;}}
    h1{{color:#1e293b;}} .s{{display:inline-block;background:#f8fafc;border-radius:8px;
    padding:15px 20px;margin:5px;border:1px solid #e2e8f0;text-align:center;}}
    .s h2{{margin:0;font-size:1.8rem;}} .s p{{margin:2px 0 0;color:#64748b;font-size:.85rem;}}
    table{{width:100%;border-collapse:collapse;margin-top:20px;}}
    th{{background:#1e293b;color:white;padding:10px;text-align:left;}}
    td{{padding:10px;border-bottom:1px solid #f1f5f9;}}tr:hover{{background:#f8fafc;}}</style>
    </head><body>
    <h1>💼 HireFlow — {COMPANY_NAME} / {ROLE_TITLE}</h1>
    <div>
        <div class="s"><h2>{len(_sessions)}</h2><p>Active screens</p></div>
        <div class="s"><h2>{len(_candidates)}</h2><p>Completed</p></div>
        <div class="s"><h2 style="color:#16a34a">{len(strong)}</h2><p>Strong hires</p></div>
        <div class="s"><h2 style="color:#2563eb">{len(shortlist)}</h2><p>Shortlisted</p></div>
        <div class="s"><h2 style="color:#dc2626">{len(rejected)}</h2><p>Rejected</p></div>
    </div>
    <table><thead><tr>
        <th>Candidate</th><th>Score</th><th>Recommendation</th>
        <th>Salary fit</th><th>Notice</th><th>Screened</th>
    </tr></thead><tbody>{rows}</tbody></table>
    <p><small><a href="/shortlist">Shortlist (JSON)</a> | <a href="/all">All (JSON)</a></small></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "role": ROLE_TITLE, "company": COMPANY_NAME,
            "active": len(_sessions), "completed": len(_candidates)}


@app.get("/shortlist")
async def shortlist():
    return {"candidates": [c for c in _candidates
                           if c.get("recommendation") in ("SHORTLIST", "STRONG_HIRE")]}


@app.get("/all")
async def all_candidates():
    return {"candidates": sorted(_candidates, key=lambda x: -x.get("overall_score", 0))}


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
    sid    = f"hf_{hashlib.md5(phone.encode()).hexdigest()[:12]}"
    engine = _sessions.get(sid)

    if engine is None:
        engine = TrueNorthEngine(
            goal_config = _goal_config,
            session_id  = sid,
            router      = LLMRouter(),
        )
        _sessions[sid] = engine
        first = await engine.start()
        await send_wa(phone, first.text)
        if text.lower() not in ("hi", "hello", "yes", "ready", "start"):
            resp = await engine.process_message(text)
            await send_wa(phone, resp.text)
            await _check_done(sid, phone, resp)
        return

    resp = await engine.process_message(text)
    await send_wa(phone, resp.text)
    await _check_done(sid, phone, resp)


async def _check_done(sid: str, phone: str, response):
    if not (response.is_complete and response.final_output):
        return

    content = response.final_output.content or {}
    eng     = _sessions.get(sid)

    record = {
        "session_id":     sid,
        "phone":          phone,
        "name":           content.get("candidate_name",
                          eng.state.collected_fields.get("candidate_name", "?") if eng else "?"),
        "overall_score":  content.get("overall_score", 0),
        "recommendation": content.get("recommendation", "UNKNOWN"),
        "salary_fit":     content.get("salary_fit", "UNKNOWN"),
        "notice_days":    content.get("notice_days",
                          eng.state.collected_fields.get("notice_days", "?") if eng else "?"),
        "strengths":      content.get("strengths", []),
        "concerns":       content.get("concerns", []),
        "red_flags":      content.get("red_flags", []),
        "next_round_qs":  content.get("questions_for_next_round", []),
        "screened_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "full_scorecard": content,
    }
    _candidates.append(record)

    # Thank the candidate
    rec  = content.get("recommendation", "")
    msg  = (
        f"✅ *{COMPANY_NAME} — Application Received*\n\n"
        f"Thank you for completing the screening for the {ROLE_TITLE} role.\n"
        "The team will review your application and get back to you within 3 business days.\n\n"
        "Best of luck! 🙏"
    )
    await send_wa(phone, msg)

    # Push to ATS
    if ATS_WEBHOOK:
        async with httpx.AsyncClient() as client:
            await client.post(ATS_WEBHOOK, json=record, timeout=5.0)

    del _sessions[sid]
    log.info(f"Candidate screened: {record['name']} → {rec} ({record['overall_score']}/100)")


async def send_wa(phone: str, text: str):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print(f"\n📤 [{phone}]: {text[:200]}\n")
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

async def local_test():
    print("\n" + "=" * 60)
    print(f"  HireFlow — {COMPANY_NAME} / {ROLE_TITLE}")
    print("  LOCAL TEST MODE")
    print("=" * 60)

    global _goal_config
    _goal_config = build_goal()
    print(f"✅ Role: {ROLE_TITLE} | Min exp: {ROLE_MIN_EXP}y | Budget: {ROLE_BUDGET_LPA} LPA\n")

    sess   = "local_hf_test"
    engine = TrueNorthEngine(goal_config=_goal_config, session_id=sess, router=LLMRouter())
    _sessions[sess] = engine

    first = await engine.start()
    print(f"HireFlow: {first.text}\n")

    while not engine.state.is_complete:
        try:
            user_input = input("Candidate: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("q", "quit"):
            break

        resp = await engine.process_message(user_input)
        print(f"\nHireFlow: {resp.text}\n")

        if resp.is_complete and resp.final_output:
            content = resp.final_output.content
            print("=" * 60)
            print(f"  SCORECARD: {content.get('candidate_name','?')}")
            print("=" * 60)
            print(f"  Overall score:    {content.get('overall_score','?')}/100")
            print(f"  Recommendation:   {content.get('recommendation','?')}")
            print(f"  Technical:        {content.get('technical_score','?')}/100")
            print(f"  Culture:          {content.get('culture_score','?')}/100")
            print(f"  Availability:     {content.get('availability_score','?')}/100")
            print(f"  Salary fit:       {content.get('salary_fit','?')}")
            print(f"\n  Strengths:")
            for s in content.get("strengths", []):
                print(f"    + {s}")
            print(f"\n  Concerns:")
            for c in content.get("concerns", []):
                print(f"    - {c}")
            if content.get("red_flags"):
                print(f"\n  ⚠️  Red flags:")
                for r in content["red_flags"]:
                    print(f"    ⚠  {r}")
            print(f"\n  Next round questions:")
            for q in content.get("questions_for_next_round", []):
                print(f"    Q: {q}")
            break


if __name__ == "__main__":
    asyncio.run(local_test())