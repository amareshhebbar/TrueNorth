"""
Legal Aid CLI — Terminal Legal Intake for NGO Workers
=======================================================

WHAT THIS IS
------------
A beautiful terminal app for legal aid workers at NGOs.
Worker sits with the client, runs this tool, and it guides
them through a structured intake conversation.

Output: a formatted legal case brief they can hand to an advocate.
Also exports to JSON and plain text for record-keeping.

NO SERVER. NO DATABASE. NO DEPLOYMENT.
Just Python + your API key. Runs anywhere.

PROJECT STRUCTURE
-----------------
    legal-aid-cli/
    ├── app.py           ← this file
    ├── goal.yaml        ← intake questionnaire config
    ├── requirements.txt
    └── output/          ← created automatically, case files saved here

requirements.txt contents:
---------------------------
    anthropic>=0.49.0
    # (truenorth installed from monorepo — see INSTALL below)

INSTALL
-------
    # From the TrueNorth monorepo root:
    cd packages/core
    pip install -e .
    pip install anthropic

    cd ../../sample-projects/legal-aid-cli

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-ant-...

    # Normal run
    python app.py

    # Hindi mode
    LANG_HINT=hindi python app.py

    # Dry run (no API calls — for testing)
    DRY_RUN=1 python app.py

    # Resume interrupted session
    SESSION_ID=abc123 python app.py

WHAT YOU SEE
------------
    ╔══════════════════════════════════════════════════════════╗
    ║         LEGAL AID INTAKE — TrueNorth                    ║
    ║         Free Legal Assistance Programme                  ║
    ╚══════════════════════════════════════════════════════════╝

    Worker ID  : LW-042
    Date       : 15 June 2025
    Session    : la_20250615_142301

    ──────────────────────────────────────────────────────────
    Assistant  : Namaste! I am here to help prepare your legal
                 matter. Take your time — there are no wrong answers.
    ──────────────────────────────────────────────────────────
    Client     : मेरा नाम रमेश है

    [PROGRESS: ████████░░░░░░░░ 45%]  Turn 6/~12

    ...

    ╔══════════════════════════════════════════════════════════╗
    ║  CASE BRIEF GENERATED                                    ║
    ╚══════════════════════════════════════════════════════════╝

    Client     : Ramesh Kumar
    Case Type  : Wage Theft
    Strength   : STRONG
    Limitation : File before 14 June 2027 (2 years from incident)

    Applicable laws:
      • Payment of Wages Act 1936, Section 15
      • Minimum Wages Act 1948
      • Labour Court jurisdiction

    Immediate steps:
      1. File a complaint with the Labour Commissioner this week
      2. Collect wage slips and employment letter
      3. Record any witnesses who saw non-payment

    Saved to: output/la_20250615_142301_ramesh-kumar.json
              output/la_20250615_142301_ramesh-kumar.txt
"""

import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

from truenorth.core.engine      import TrueNorthEngine
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.llm.router       import LLMRouter

# ── Config ────────────────────────────────────────────────────────────────────

GOAL_YAML   = os.environ.get("GOAL_YAML",   "goal.yaml")
DRY_RUN     = os.environ.get("DRY_RUN",     "0") == "1"
LANG_HINT   = os.environ.get("LANG_HINT",   "")
WORKER_ID   = os.environ.get("WORKER_ID",   "LW-001")
SESSION_ID  = os.environ.get("SESSION_ID",  "")
OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Terminal colours ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
WHITE  = "\033[97m"
BG_DARK= "\033[40m"

def clr(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

def banner():
    width = 62
    print()
    print(clr("╔" + "═" * (width-2) + "╗", BOLD, CYAN))
    print(clr("║" + " LEGAL AID INTAKE  —  TrueNorth".center(width-2) + "║", BOLD, CYAN))
    print(clr("║" + " Free Legal Assistance Programme".center(width-2) + "║", CYAN))
    print(clr("╚" + "═" * (width-2) + "╝", BOLD, CYAN))
    print()
    print(clr(f"  Worker ID  : {WORKER_ID}", DIM))
    print(clr(f"  Date       : {datetime.now().strftime('%d %B %Y')}", DIM))
    if DRY_RUN:
        print(clr("  Mode       : DRY RUN (mock LLM)", YELLOW))
    print()

def progress_bar(pct: float, width: int = 30) -> str:
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    colour = GREEN if pct >= 80 else CYAN if pct >= 40 else YELLOW
    return clr(f"[{bar}]", colour) + clr(f" {pct:.0f}%", BOLD)

def print_separator():
    print(clr("  " + "─" * 58, DIM))

def print_agent(text: str):
    print()
    print(clr("  Assistant  : ", CYAN, BOLD) + text)
    print()

def print_client(text: str):
    print(clr("  Client     : ", WHITE, BOLD) + clr(text, WHITE))

def print_progress(pct: float, turn: int):
    print()
    print(f"  {progress_bar(pct)}  " + clr(f" Turn {turn}", DIM))
    print()

def print_field_extracted(field: str, value: str, confidence: float):
    colour = GREEN if confidence >= 0.85 else YELLOW if confidence >= 0.60 else RED
    print(clr(f"  ✓ Extracted  {field}: {str(value)[:60]}  ({confidence:.0%})", colour, DIM))

# ── Case brief display ────────────────────────────────────────────────────────

def print_case_brief(content: dict):
    width = 62
    print()
    print(clr("╔" + "═" * (width-2) + "╗", BOLD, GREEN))
    print(clr("║" + " CASE BRIEF GENERATED".center(width-2) + "║", BOLD, GREEN))
    print(clr("╚" + "═" * (width-2) + "╝", BOLD, GREEN))
    print()

    def field(label: str, value, colour=WHITE):
        val = str(value) if not isinstance(value, list) else ", ".join(str(v) for v in value)
        print(f"  {clr(label + ':', BOLD, CYAN):<35} {clr(val, colour)}")

    strength = str(content.get("case_strength", "UNKNOWN")).upper()
    strength_colour = GREEN if "STRONG" in strength else YELLOW if "MODERATE" in strength else RED

    field("Client",          content.get("incident_summary", "")[:60])
    field("Case type",       content.get("case_type_formal", ""))
    field("Strength",        strength, strength_colour)
    field("Limitation",      content.get("limitation_period", ""))
    field("Free aid eligible", "✅ YES" if content.get("eligible_for_free_aid") else "Check criteria")
    field("Est. compensation", content.get("estimated_compensation_range", ""))

    print()
    print(clr("  Applicable laws:", BOLD, CYAN))
    for law in content.get("applicable_laws", [])[:5]:
        print(f"    {clr('•', YELLOW)} {law}")

    print()
    print(clr("  Immediate steps:", BOLD, CYAN))
    for i, step in enumerate(content.get("immediate_steps", []), 1):
        wrapped = textwrap.fill(step, width=54, subsequent_indent="       ")
        print(f"    {clr(str(i)+'.', YELLOW)} {wrapped}")

    print()
    print(clr("  Documents needed:", BOLD, CYAN))
    for doc in content.get("documents_needed", [])[:6]:
        print(f"    {clr('→', GREEN)} {doc}")

    missing = content.get("documents_missing", [])
    if missing:
        print()
        print(clr("  Documents missing (collect before advocate meeting):", BOLD, RED))
        for doc in missing[:4]:
            print(f"    {clr('✗', RED)} {doc}")

    print()
    print_separator()
    print(clr("  Advocate brief (for the lawyer):", BOLD, CYAN))
    brief = content.get("advocate_brief", "")
    for line in textwrap.wrap(brief, width=58):
        print(f"    {line}")

    print()

# ── Save output ───────────────────────────────────────────────────────────────

def save_output(session_id: str, content: dict, collected: dict):
    name = collected.get("client_name", "unknown").lower().replace(" ", "-")
    stem = f"{session_id}_{name}"

    # JSON — full structured data
    json_path = OUTPUT_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_id":      session_id,
            "worker_id":       WORKER_ID,
            "date":            datetime.now().isoformat(),
            "collected_fields": collected,
            "case_brief":      content,
        }, f, indent=2, ensure_ascii=False)

    # Plain text — for printing / sharing
    txt_path = OUTPUT_DIR / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"LEGAL AID INTAKE\n")
        f.write(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}\n")
        f.write(f"Worker: {WORKER_ID}\n")
        f.write(f"Session: {session_id}\n\n")
        f.write("─" * 50 + "\n")
        f.write("CLIENT INFORMATION\n")
        f.write("─" * 50 + "\n")
        for k, v in collected.items():
            f.write(f"{k.replace('_',' ').title()}: {v}\n")
        f.write("\n" + "─" * 50 + "\n")
        f.write("CASE BRIEF\n")
        f.write("─" * 50 + "\n")
        f.write(f"Case type   : {content.get('case_type_formal', '')}\n")
        f.write(f"Strength    : {content.get('case_strength', '')}\n")
        f.write(f"Limitation  : {content.get('limitation_period', '')}\n\n")
        f.write("Applicable laws:\n")
        for law in content.get("applicable_laws", []):
            f.write(f"  • {law}\n")
        f.write("\nImmediate steps:\n")
        for i, step in enumerate(content.get("immediate_steps", []), 1):
            f.write(f"  {i}. {step}\n")
        f.write("\nAdvocate brief:\n")
        f.write(content.get("advocate_brief", "") + "\n")

    return json_path, txt_path

# ── Main conversation loop ────────────────────────────────────────────────────

async def run():
    banner()

    # Load goal
    try:
        config = YAMLLoader.load(GOAL_YAML)
    except FileNotFoundError:
        print(clr(f"\n  ❌  {GOAL_YAML} not found\n", RED))
        sys.exit(1)

    session_id = SESSION_ID or f"la_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(clr(f"  Session    : {session_id}", DIM))
    print()
    print_separator()

    # Build engine
    router = LLMRouter()
    if DRY_RUN:
        from truenorth.testing.mock_llm import MockLLMClient
        router.register_client("mock", MockLLMClient(default="Test response for dry run"))

    engine = TrueNorthEngine(
        goal_config = config,
        session_id  = session_id,
        router      = router,
    )

    # Start
    resp = await engine.start()
    print_agent(resp.text)

    # Conversation loop
    while not engine.state.is_complete:
        print_progress(engine.state.completion_pct, engine.state.current_turn)

        try:
            raw = input(clr("  Client     : ", BOLD, WHITE))
        except (EOFError, KeyboardInterrupt):
            print(clr("\n\n  Interrupted — session data not saved.\n", YELLOW))
            return

        if not raw.strip():
            continue
        if raw.strip().lower() in ("quit", "q", "exit", "/exit"):
            print(clr("\n  Exiting. Partial session not saved.\n", YELLOW))
            return

        print_client(raw.strip())  # echo styled

        resp = await engine.process_message(raw.strip())

        # Show what was extracted this turn
        for field_data in resp.fields_extracted:
            if isinstance(field_data, dict):
                print_field_extracted(
                    field_data.get("field", "?"),
                    field_data.get("value", "?"),
                    field_data.get("confidence", 0),
                )

        print_agent(resp.text)

        if resp.final_output:
            break

    # Done — display and save
    if engine.state.is_complete:
        output = engine.state.final_output
        content = {}

        if output and hasattr(output, "content"):
            content = output.content if isinstance(output.content, dict) else {}
        elif resp.final_output:
            content = resp.final_output.content if isinstance(resp.final_output.content, dict) else {}

        print_case_brief(content)

        json_path, txt_path = save_output(
            session_id, content, engine.state.collected_fields
        )

        print(clr("  Files saved:", BOLD, GREEN))
        print(f"    📄 {json_path}")
        print(f"    📝 {txt_path}")
        print()
        print(clr("  Share the .txt file with the advocate.", DIM))
        print()

if __name__ == "__main__":
    asyncio.run(run())