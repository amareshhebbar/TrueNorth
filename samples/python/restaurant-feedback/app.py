"""
Restaurant Feedback — Flask Web Chat
======================================

WHAT THIS IS
------------
A web-based conversational feedback collector for restaurants.
Guest scans a QR code at the table, opens a chat in their browser,
and answers questions naturally. Manager sees real-time dashboard.

No app download. No form. Works on any smartphone browser.

PROJECT STRUCTURE
-----------------
    restaurant-feedback/
    ├── app.py              ← this file (Flask server)
    ├── goal.yaml           ← feedback questionnaire
    ├── requirements.txt
    └── templates/
        └── index.html      ← chat UI (served by Flask)

requirements.txt:
-----------------
    flask>=3.0.0
    anthropic>=0.49.0

INSTALL
-------
    cd packages/core && pip install -e .
    pip install flask anthropic

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-ant-...
    cd sample-projects/restaurant-feedback
    python app.py

    # Open in browser
    open http://localhost:5000

    # Dashboard (manager view)
    open http://localhost:5000/dashboard

QR CODE FOR TABLE
-----------------
    # Generate a QR code pointing to your server URL
    pip install qrcode
    python -c "import qrcode; qrcode.make('http://YOUR_IP:5000').save('table-qr.png')"
    # Print and place on each table

WHAT MANAGERS SEE (dashboard)
------------------------------
    Today's Feedback — The Spice Garden
    ─────────────────────────────────────
    23 responses | Avg score: 7.8/10 | NPS: +42

    ⚠ Priority issues:
      • 3 guests mentioned slow service between 7-8pm
      • 2 complaints about biryani portion size

    ✅ Staff highlights:
      • "The waiter Suresh was incredibly attentive"
      • "Best dal makhani we've ever had"

    NPS breakdown: 14 Promoters | 6 Passive | 3 Detractors
"""

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, render_template, request, session

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

# ── Config ────────────────────────────────────────────────────────────────────

RESTAURANT_NAME = os.environ.get("RESTAURANT_NAME", "The Spice Garden")
GOAL_YAML       = os.environ.get("GOAL_YAML",       "goal.yaml")
SECRET_KEY      = os.environ.get("SECRET_KEY",      uuid.uuid4().hex)

goal_config  = YAMLLoader.load(GOAL_YAML)
active_sessions: dict[str, TrueNorthEngine] = {}
feedback_log:    list[dict]                  = []
loop = asyncio.new_event_loop()

def run_async(coro):
    """Run async code from Flask's sync context."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)

threading.Thread(target=loop.run_forever, daemon=True).start()

# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = SECRET_KEY


@app.route("/")
def index():
    session["session_id"] = session.get("session_id") or str(uuid.uuid4())[:12]
    return render_template("index.html", restaurant=RESTAURANT_NAME)


@app.route("/api/start", methods=["POST"])
def start():
    sid    = session.get("session_id") or str(uuid.uuid4())[:12]
    engine = TrueNorthEngine(
        goal_config = goal_config,
        session_id  = sid,
        router      = LLMRouter(),
    )
    active_sessions[sid] = engine
    resp = run_async(engine.start())
    return jsonify({"text": resp.text, "session_id": sid, "completion": 0})


@app.route("/api/message", methods=["POST"])
def message():
    data = request.json or {}
    sid  = data.get("session_id") or session.get("session_id", "")
    text = (data.get("text") or "").strip()

    if not sid or not text:
        return jsonify({"error": "missing session_id or text"}), 400

    engine = active_sessions.get(sid)
    if not engine:
        return jsonify({"error": "session not found"}), 404

    resp = run_async(engine.process_message(text))

    result: dict[str, Any] = {
        "text":       resp.text,
        "completion": resp.completion_pct,
        "is_complete": resp.is_complete,
    }

    if resp.is_complete and resp.final_output:
        content = resp.final_output.content or {}
        feedback_log.append({
            "session_id":  sid,
            "timestamp":   datetime.now().isoformat(),
            "feedback":    content,
            "fields":      engine.state.collected_fields,
        })
        del active_sessions[sid]
        result["output"] = content
        result["thank_you"] = content.get(
            "response_message",
            f"Thank you for your feedback! See you again at {RESTAURANT_NAME}. 😊"
        )

    return jsonify(result)


@app.route("/dashboard")
def dashboard():
    if not feedback_log:
        return f"<h2>No feedback yet — share the QR code!</h2>"

    scores   = [f["feedback"].get("overall_score", 0) for f in feedback_log if f.get("feedback")]
    nps_vals = [f["fields"].get("nps_score", 0)       for f in feedback_log if f.get("fields")]
    avg      = sum(scores) / len(scores) if scores else 0
    promoters = sum(1 for n in nps_vals if n >= 9)
    passives  = sum(1 for n in nps_vals if 7 <= n <= 8)
    detractors= sum(1 for n in nps_vals if n <= 6)
    nps_score = ((promoters - detractors) / len(nps_vals) * 100) if nps_vals else 0

    priority_issues = [
        f["feedback"].get("priority_issue", "")
        for f in feedback_log if f.get("feedback") and f["feedback"].get("manager_action_required")
    ]
    compliments = [
        f["feedback"].get("compliment_to_share", "")
        for f in feedback_log if f.get("feedback") and f["feedback"].get("compliment_to_share")
    ]

    issues_html = "".join(f"<li>⚠️ {i}</li>" for i in priority_issues[:5]) or "<li>None today ✅</li>"
    praise_html = "".join(f"<li>✅ {c}</li>" for c in compliments[:5]) or "<li>Keep up the work!</li>"

    return f"""
    <!DOCTYPE html><html><head><title>Dashboard — {RESTAURANT_NAME}</title>
    <style>body{{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#fffbf0;}}
    h1{{color:#92400e;}}h2{{color:#78350f;}}
    .stat{{display:inline-block;background:white;border-radius:12px;padding:20px 30px;
           margin:8px;text-align:center;border:2px solid #fde68a;min-width:130px;}}
    .stat h2{{margin:0;font-size:2rem;color:#d97706;}}
    .stat p{{margin:4px 0 0;color:#64748b;font-size:.85rem;}}
    ul{{line-height:2;}} li{{margin-bottom:4px;}}
    .section{{background:white;border-radius:12px;padding:20px;margin:20px 0;
              border:1px solid #fde68a;}}
    </style></head><body>
    <h1>🍽️ {RESTAURANT_NAME} — Feedback Dashboard</h1>
    <p>{datetime.now().strftime('%d %B %Y')}</p>
    <div>
      <div class="stat"><h2>{len(feedback_log)}</h2><p>Responses today</p></div>
      <div class="stat"><h2>{avg:.1f}/10</h2><p>Avg score</p></div>
      <div class="stat"><h2>{nps_score:+.0f}</h2><p>NPS score</p></div>
      <div class="stat"><h2>{promoters}</h2><p>Promoters (9-10)</p></div>
      <div class="stat"><h2>{passives}</h2><p>Passive (7-8)</p></div>
      <div class="stat"><h2>{detractors}</h2><p>Detractors (0-6)</p></div>
    </div>
    <div class="section">
      <h2>⚠️ Priority issues (needs manager attention)</h2>
      <ul>{issues_html}</ul>
    </div>
    <div class="section">
      <h2>✅ Staff highlights (share with team)</h2>
      <ul>{praise_html}</ul>
    </div>
    <p><a href="/">← Guest feedback link</a></p>
    </body></html>
    """


if __name__ == "__main__":
    print(f"\n  {RESTAURANT_NAME} — Feedback System")
    print(f"  Guest link:    http://localhost:5000/")
    print(f"  Manager view:  http://localhost:5000/dashboard\n")
    app.run(host="0.0.0.0", port=5000, debug=False)