"""
TrueNorth Example: Multilingual Auto-Detection Demo
=====================================================

WHAT THIS DOES
--------------
Shows TrueNorth handling Hindi, Kannada, Telugu, Tamil, Marathi, and English
in the same session — automatically, with no configuration.

The user types in any Indian language.
TrueNorth detects it, responds in the same language, extracts fields correctly,
and generates structured output regardless of which language was used.

This is one of TrueNorth's core differentiators for India:
→ No language selection screen
→ No separate app for each language
→ Hinglish (mixed Hindi-English) also handled
→ Regional vocabulary and units understood

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install anthropic

FILE STRUCTURE
--------------
    my-multilingual-app/
    ├── 03-multilingual-demo.py   ← this file
    └── (no goal.yaml needed — inline goal used)

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-ant-...
    python 03-multilingual-demo.py

    # Run a specific language:
    DEMO_LANG=hindi  python 03-multilingual-demo.py
    DEMO_LANG=kannada python 03-multilingual-demo.py
    DEMO_LANG=tamil  python 03-multilingual-demo.py

    # Run all languages back to back:
    DEMO_LANG=all python 03-multilingual-demo.py

WHAT YOU WILL SEE
-----------------
    ── HINDI DEMO ───────────────────────────────────
    Agent (auto-detected Hindi):
      नमस्ते! मैं यहाँ आपकी मदद के लिए हूँ।
      आपका नाम क्या है?

    User types: राहुल शर्मा
    User types: मेरी उम्र 28 साल है
    User types: मुझे वजन कम करना है

    Output:
    { "name": "Rahul Sharma", "age": 28, "goal": "weight loss",
      "detected_language": "hindi", ... }

    ── KANNADA DEMO ──────────────────────────────────
    Agent: ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಹೆಸರು ಏನು?
    User:  ರವಿ ಕುಮಾರ್
    ...

HOW LANGUAGE DETECTION WORKS
------------------------------
    Stage 1 of the TrueNorth pipeline runs language detection.
    It uses a lightweight model (no API call needed for detection).
    Detected language is stored in session state.
    Every subsequent response generated in that language.

    For Hinglish (mixed):
      User: "mera naam Priya hai and I want to lose weight"
      → Detected as Hinglish
      → Responses in Hinglish: "Great Priya! Aap kitne kg hain abhi?"

LANGUAGE CODES SUPPORTED
--------------------------
    hi  → Hindi        kn  → Kannada      te  → Telugu
    ta  → Tamil        mr  → Marathi      gu  → Gujarati
    pa  → Punjabi      bn  → Bengali      ml  → Malayalam
    en  → English      hi-en → Hinglish   or  → Odia
"""

import asyncio
import json
import os
from typing import Optional

from truenorth.core.engine           import TrueNorthEngine
from truenorth.core.yaml_loader      import YAMLLoader
from truenorth.intelligence.language_detector import LanguageDetector

# ── Inline goal config (no YAML file needed for this demo) ─────────────────

INLINE_GOAL = {
    "id": "multilingual_demo",
    "name": "Multilingual Health Assessment",
    "persona": {
        "name": "Health Assistant",
        "tone": "warm",
        "empathy_level": "medium",
        "language": "auto",          # KEY: auto-detects from user input
        "greeting": "Hello! / नमस्ते! / ನಮಸ್ಕಾರ! / வணக்கம்!\nI can speak your language. Please reply in Hindi, Kannada, Tamil, Telugu, or English — I will understand.",
    },
    "fields": [
        {"name": "name",         "type": "text",    "required": True,
         "question": "What is your name? / आपका नाम क्या है?"},
        {"name": "age",          "type": "integer",  "required": True, "min": 1, "max": 120,
         "question": "How old are you?"},
        {"name": "weight_kg",    "type": "number",   "required": True,
         "question": "What is your weight?"},
        {"name": "height_cm",    "type": "number",   "required": True,
         "question": "What is your height?"},
        {"name": "health_goal",  "type": "text",     "required": True,
         "allowed_values": ["lose weight", "build strength", "improve fitness", "manage condition", "general wellness"],
         "question": "What is your main health goal?"},
        {"name": "city",         "type": "text",     "required": False,
         "question": "Which city are you from?"},
    ],
    "output": {
        "format": "json",
        "template": (
            "Summarise the health assessment for {name}, age {age}, "
            "{weight_kg}kg, {height_cm}cm, from {city}. Goal: {health_goal}. "
            "Return JSON with: bmi, bmi_category, language_detected, "
            "personalised_message (in the user's language), and next_steps (3 items)."
        ),
    },
}

# ── Pre-written demo conversations per language ─────────────────────────────

DEMO_SCRIPTS = {
    "hindi": {
        "label": "HINDI (हिंदी)",
        "turns": [
            "नमस्ते",                                          # Hello
            "मेरा नाम राहुल शर्मा है",                        # My name is Rahul Sharma
            "28 साल का हूँ",                                  # I am 28 years old
            "वजन 72 किलो है",                                 # Weight is 72 kg
            "लम्बाई 175 सेंटीमीटर है",                        # Height is 175 cm
            "मुझे वजन कम करना है",                             # I want to lose weight
            "मुंबई से हूँ",                                   # I am from Mumbai
        ],
    },
    "kannada": {
        "label": "KANNADA (ಕನ್ನಡ)",
        "turns": [
            "ನಮಸ್ಕಾರ",                                         # Hello
            "ನನ್ನ ಹೆಸರು ರವಿ ಕುಮಾರ್",                          # My name is Ravi Kumar
            "ನನಗೆ 32 ವರ್ಷ",                                    # I am 32 years old
            "ತೂಕ 80 ಕಿಲೋ",                                    # Weight 80 kg
            "ಎತ್ತರ 170 ಸೆಂ.ಮೀ",                               # Height 170 cm
            "ದೇಹವನ್ನು ಫಿಟ್ ಮಾಡಿಕೊಳ್ಳಬೇಕು",                     # I want to get fit
            "ಬೆಂಗಳೂರಿನಿಂದ",                                   # From Bengaluru
        ],
    },
    "tamil": {
        "label": "TAMIL (தமிழ்)",
        "turns": [
            "வணக்கம்",                                          # Hello
            "என் பெயர் ப்ரியா",                                 # My name is Priya
            "என் வயது 25",                                      # My age is 25
            "என் எடை 58 கிலோ",                                  # My weight is 58 kg
            "உயரம் 162 செமீ",                                   # Height 162 cm
            "உடல் எடையை குறைக்கணும்",                           # Want to reduce weight
            "சென்னையில் இருக்கேன்",                              # I am in Chennai
        ],
    },
    "hinglish": {
        "label": "HINGLISH (हिंदी + English Mixed)",
        "turns": [
            "hello bhai",
            "Mera naam Arjun hai",
            "I am 30 years old",
            "Weight 85 kg hai mera",
            "Height around 180 cm",
            "Muscle build karna chahta hoon",
            "Pune mein rehta hoon",
        ],
    },
    "english": {
        "label": "ENGLISH",
        "turns": [
            "Hello",
            "My name is Sarah",
            "I'm 26 years old",
            "I weigh 55 kilograms",
            "My height is 165 centimetres",
            "I want to improve my overall fitness",
            "I live in Hyderabad",
        ],
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def divider(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


def show_detection_stats(state):
    """Show what language was detected and the engine's state."""
    lang = getattr(state, "detected_language", "unknown") or "auto"
    print(f"\n  📊 Language detected : {lang}")
    print(f"  📊 Fields collected  : {len(state.collected_fields)}")
    print(f"  📊 Completion        : {state.completion_pct:.0f}%")
    print(f"  📊 Current turn      : {state.current_turn}\n")


# ── Run a single language demo ──────────────────────────────────────────────

async def run_language_demo(
    language: str,
    script:   dict,
    interactive: bool = False,
) -> Optional[dict]:
    """
    Run a complete demo for one language.
    If interactive=True, user types instead of using the script.
    """
    divider(f"DEMO — {script['label']}")

    engine = TrueNorthEngine(
        goal_config = INLINE_GOAL,
        session_id  = f"multilang_{language}",
    )

    # Start
    first = await engine.start()
    print(f"Agent: {first.text}\n")

    turns = script["turns"] if not interactive else []

    if interactive:
        print("(Type your messages. Press Enter after each. Ctrl+C to skip.)\n")

    turn_index = 0
    while not engine.state.is_complete:

        if interactive:
            try:
                user_input = input(f"You [{language}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(Skipped)")
                break
        else:
            if turn_index >= len(turns):
                break
            user_input = turns[turn_index]
            print(f"User: {user_input}")
            turn_index += 1

        if not user_input:
            continue

        response = await engine.process_message(user_input)
        print(f"Agent: {response.text}\n")

        if response.final_output:
            print(f"✅ Session complete!\n")
            show_detection_stats(engine.state)
            content = response.final_output.content

            print("── STRUCTURED OUTPUT ──────────────────────────────────")
            print(json.dumps(content, indent=2, ensure_ascii=False))

            # Highlight language detection result
            detected = content.get("language_detected") or engine.state.detected_language
            if detected:
                print(f"\n✅ Language correctly detected as: {detected.upper()}")

            # Show personalised message in native language
            if isinstance(content, dict):
                msg = content.get("personalised_message")
                if msg:
                    print(f"\n💬 Personalised message ({language}):\n   {msg}")

            return content

    if not engine.state.is_complete:
        print(f"(Demo script ran out of turns at {engine.state.completion_pct:.0f}% completion)")
        show_detection_stats(engine.state)

    return None


# ── Show live language detector ──────────────────────────────────────────────

async def demo_language_detector():
    """Show the language detector working on sample phrases."""
    divider("LANGUAGE DETECTION — How it works")

    detector = LanguageDetector()
    test_phrases = [
        ("नमस्ते, मेरा नाम राहुल है",         "Hindi"),
        ("ನನ್ನ ಹೆಸರು ರವಿ",                  "Kannada"),
        ("என் பெயர் ப்ரியா",                 "Tamil"),
        ("నా పేరు రాహుల్",                   "Telugu"),
        ("Hello, my name is Sarah",          "English"),
        ("Mera weight 65 kg hai",            "Hinglish"),
        ("माझे नाव सुनील आहे",               "Marathi"),
        ("ਮੇਰਾ ਨਾਮ ਹਰਪ੍ਰੀਤ ਹੈ",             "Punjabi"),
        ("আমার নাম সুমিত",                   "Bengali"),
    ]

    print(f"{'Phrase':<45} {'Expected':<12} {'Detected'}")
    print("─" * 75)
    for phrase, expected in test_phrases:
        detected = detector.detect(phrase)
        match    = "✅" if expected.lower() in (detected or "").lower() else "⚠️ "
        print(f"{phrase[:43]:<45} {expected:<12} {detected or 'unknown'} {match}")
    print()


# ── Interactive mode ─────────────────────────────────────────────────────────

async def interactive_session():
    """Let the user type in any language and see TrueNorth respond."""
    divider("INTERACTIVE — Type in any Indian language")
    print("You can type in Hindi, Kannada, Tamil, Telugu, Marathi, Gujarati, or English.")
    print("TrueNorth will detect your language and respond in it.\n")

    engine = TrueNorthEngine(goal_config=INLINE_GOAL, session_id="interactive_ml")
    first  = await engine.start()
    print(f"Agent: {first.text}\n")

    while not engine.state.is_complete:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(Exiting)")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "q", "exit"):
            break

        resp = await engine.process_message(user_input)
        print(f"\nAgent [{resp.detected_language or 'auto'}]: {resp.text}\n")

        if resp.final_output:
            print("── OUTPUT ──")
            print(json.dumps(resp.final_output.content, indent=2, ensure_ascii=False))
            break


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    demo_lang = os.environ.get("DEMO_LANG", "all").lower()

    print("=" * 60)
    print("  TrueNorth Multilingual Demo")
    print(f"  Mode: {demo_lang}")
    print("=" * 60)

    # Always show the language detector first
    await demo_language_detector()

    if demo_lang == "interactive":
        await interactive_session()
        return

    if demo_lang == "all":
        # Run all languages
        results = {}
        for lang, script in DEMO_SCRIPTS.items():
            output = await run_language_demo(lang, script)
            results[lang] = "✅ PASS" if output else "⚠️ INCOMPLETE"

        divider("SUMMARY")
        for lang, status in results.items():
            print(f"  {DEMO_SCRIPTS[lang]['label']:<40} {status}")
    elif demo_lang in DEMO_SCRIPTS:
        await run_language_demo(demo_lang, DEMO_SCRIPTS[demo_lang])
    else:
        print(f"Unknown language: {demo_lang}")
        print(f"Available: {', '.join(DEMO_SCRIPTS.keys())} | all | interactive")


if __name__ == "__main__":
    asyncio.run(main())