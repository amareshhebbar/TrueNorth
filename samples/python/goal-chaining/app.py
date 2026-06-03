"""
TrueNorth Example: Goal Chaining
==================================

WHAT THIS DOES
--------------
Shows how TrueNorth automatically chains goals together.

Example: Fitness intake → Nutrition plan → Weekly check-in

The user answers fitness questions once.
When fitness intake completes, TrueNorth automatically:
  1. Detects the next goal from the chain config
  2. Carries relevant fields (no re-asking age, weight, etc.)
  3. Starts the nutrition session seamlessly
  4. User never knows there were two separate goals

This creates multi-step AI journeys from simple YAML chains.

Real-world use cases:
  Medical intake → Lab test form (carry name, DOB, allergies)
  Job application → Technical assessment (carry name, role, experience)
  Loan inquiry → KYC form (carry name, income, property details)
  Fitness intake → Nutrition plan → Supplement plan

INSTALL
-------
    cd packages/core
    pip install -e .
    pip install anthropic

FILE STRUCTURE
--------------
    my-app/
    ├── 06-goal-chaining.py   ← this file (self-contained)
    └── (no YAML files needed — inline goals used)

HOW TO RUN
----------
    export ANTHROPIC_API_KEY=sk-ant-...
    python 06-goal-chaining.py

    # Run only the chaining demo:
    DEMO=chain python 06-goal-chaining.py

    # Run with a specific goal path:
    CHAIN=fitness-nutrition python 06-goal-chaining.py
    CHAIN=medical-lab       python 06-goal-chaining.py

WHAT YOU WILL SEE
-----------------
    ── GOAL 1: Fitness Intake ────────────────────────────────
    Agent: What is your name?
    User:  Priya
    Agent: How old are you?
    User:  28
    Agent: What is your weight?
    User:  65 kg
    ...
    Agent: ✅ Perfect! Your fitness profile is complete.
           Let me now build your personalised nutrition plan...

    ── AUTO-TRANSITION (zero user input needed) ──────────────
    Chain detected: primary_goal="lose weight" → nutrition_plan
    Carrying fields: name=Priya, age=28, weight_kg=65.0, ...
    Starting next goal: nutrition_plan

    ── GOAL 2: Nutrition Plan (fields pre-filled) ────────────
    Agent: Great, Priya! Based on your fitness goal to lose weight
           and your profile, I need just a few more details for
           your nutrition plan...
    Agent: Do you have any food allergies or intolerances?
    User:  Lactose intolerant
    Agent: How many meals do you prefer per day?
    ...

    FINAL OUTPUT: Combined fitness + nutrition JSON

HOW CHAINING WORKS IN YAML
----------------------------
    # In goal_1.yaml:
    chain:
      on_complete:
        - if:   { primary_goal: "lose weight" }
          then: nutrition_plan
          carry_fields:
            - name
            - age
            - weight_kg
            - { activity_level: user_activity_level }  # rename on carry
        - if:   { primary_goal: "build muscle" }
          then: strength_plan
          carry_fields: [name, age, weight_kg, current_lifts_kg]
        - else: general_plan   # fallback

    carry_fields supports:
      - "field_name"                  → carry as-is
      - {"old_name": "new_name"}     → carry with rename
      - {"field": "new", "min": 0.8} → carry only if confidence >= 0.8

STATE TRANSFER API
-------------------
    from truenorth.agents.state_transfer import StateTransfer, FieldMap

    transfer = StateTransfer(auto_infer=True)  # auto-detect common fields
    result   = transfer.extract(
        source_state           = completed_engine.state,
        source_goal_id         = "fitness_plan",
        target_goal_id         = "nutrition_plan",
        target_required_fields = ["age", "weight_kg", "activity_level"],
    )
    print(result.carried_fields)   # {"age": 28, "weight_kg": 65.0, ...}
    print(result.coverage_pct)     # 85.0 (85% of target fields pre-filled)
    print(result.missing_fields)   # ["calorie_goal", "meal_preference"]
"""

import asyncio
import json
import os
from typing import Optional

from truenorth.core.engine             import TrueNorthEngine
from truenorth.core.yaml_loader        import YAMLLoader
from truenorth.agents.state_transfer   import StateTransfer, FieldMap, GoalChain, ChainStep
from truenorth.llm.router              import LLMRouter

# ── Goal definitions ─────────────────────────────────────────────────────────

FITNESS_GOAL = {
    "id": "fitness_plan",
    "name": "Fitness Intake",
    "persona": {
        "name": "Alex",
        "tone": "energetic and motivating",
        "language": "en",
        "greeting": "Hey! I am your fitness coach. Let us get your profile set up — takes about 3 minutes.",
    },
    "fields": [
        {"name": "name",           "type": "text",    "required": True,
         "question": "What is your name?"},
        {"name": "age",            "type": "integer",  "required": True, "min": 16, "max": 100,
         "question": "How old are you?"},
        {"name": "weight_kg",      "type": "number",   "required": True,
         "question": "Current weight in kg?"},
        {"name": "height_cm",      "type": "number",   "required": True,
         "question": "Height in cm?"},
        {"name": "primary_goal",   "type": "text",    "required": True,
         "allowed_values": ["lose weight", "build muscle", "general fitness"],
         "question": "Main goal? (lose weight / build muscle / general fitness)"},
        {"name": "activity_level", "type": "text",    "required": True,
         "allowed_values": ["sedentary", "light", "moderate", "active"],
         "question": "Activity level? (sedentary / light / moderate / active)"},
        {"name": "days_per_week",  "type": "integer",  "required": True, "min": 1, "max": 7,
         "question": "Training days per week?"},
    ],
    "output": {
        "format": "json",
        "template": (
            "Create a fitness profile for {name}, {age}y, {weight_kg}kg, {height_cm}cm. "
            "Goal: {primary_goal}. Activity: {activity_level}. "
            "Training: {days_per_week} days/week. "
            "Return JSON with bmi, bmi_category, fitness_profile_summary."
        ),
    },
    # Goal chain configuration
    "chain": {
        "on_complete": [
            {
                "if":           {"primary_goal": "lose weight"},
                "then":         "nutrition_plan",
                "carry_fields": ["name", "age", "weight_kg", "height_cm",
                                 "activity_level", "days_per_week",
                                 {"primary_goal": "fitness_goal"}],
            },
            {
                "if":           {"primary_goal": "build muscle"},
                "then":         "strength_nutrition",
                "carry_fields": ["name", "age", "weight_kg", "height_cm",
                                 "activity_level",
                                 {"primary_goal": "fitness_goal"}],
            },
            {
                "else":         "general_wellness",
                "carry_fields": ["name", "age", "weight_kg"],
            },
        ],
    },
}

NUTRITION_GOAL = {
    "id": "nutrition_plan",
    "name": "Nutrition Plan",
    "persona": {
        "name": "Alex",
        "tone": "helpful and encouraging",
        "language": "en",
        "greeting": (
            "Great, {name}! Based on your profile, let me build your nutrition plan. "
            "Just a few more questions..."
        ),
    },
    "fields": [
        # name, age, weight_kg, height_cm, activity_level are carried from fitness
        # Only ask NEW fields the nutrition plan needs
        {"name": "food_allergies",      "type": "text",    "required": True,
         "question": "Any food allergies or intolerances? (dairy / gluten / nuts / none)"},
        {"name": "diet_preference",     "type": "text",    "required": True,
         "allowed_values": ["vegetarian", "vegan", "non-vegetarian", "no preference"],
         "question": "Diet preference? (vegetarian / vegan / non-vegetarian / no preference)"},
        {"name": "meals_per_day",       "type": "integer",  "required": True, "min": 2, "max": 8,
         "question": "How many meals per day do you prefer?"},
        {"name": "cooking_time_mins",   "type": "integer",  "required": True, "min": 0, "max": 120,
         "question": "How many minutes can you spend on meal prep per day?"},
        {"name": "budget_per_day_inr",  "type": "integer",  "required": False, "min": 0, "max": 5000,
         "question": "Approximate daily food budget in ₹? (say 'flexible' if no limit)"},
    ],
    "output": {
        "format": "json",
        "template": (
            "Create a personalised nutrition plan for {name}.\n"
            "Profile: {age}y, {weight_kg}kg, {height_cm}cm. "
            "Activity: {activity_level}. Goal: {fitness_goal}.\n"
            "Diet: {diet_preference}. Allergies: {food_allergies}. "
            "Meals: {meals_per_day}/day. Prep time: {cooking_time_mins} min/day. "
            "Budget: ₹{budget_per_day_inr}/day.\n"
            "Return JSON with: daily_calories, macros (protein/carb/fat in grams), "
            "meal_plan (3 days), foods_to_emphasise, foods_to_avoid, "
            "sample_indian_meal_plan (breakfast/lunch/dinner/snack for 1 day)."
        ),
    },
}

# ── Demo turns ────────────────────────────────────────────────────────────────

FITNESS_TURNS = [
    "Priya Sharma",   # name
    "28",             # age
    "65",             # weight_kg
    "162",            # height_cm
    "lose weight",    # primary_goal → triggers nutrition chain
    "moderate",       # activity_level
    "4",              # days_per_week
]

NUTRITION_TURNS = [
    "lactose intolerant",   # food_allergies
    "vegetarian",           # diet_preference
    "3",                    # meals_per_day
    "30",                   # cooking_time_mins
    "300",                  # budget_per_day_inr
]


def banner(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


# ── DEMO 1: Automatic goal chaining ──────────────────────────────────────────

async def demo_auto_chaining():
    banner("AUTO CHAINING — Fitness → Nutrition (seamless)")

    router          = LLMRouter()
    fitness_output  = None
    nutrition_output = None

    # ── Part 1: Run fitness goal ───────────────────────────────────────────
    print("── GOAL 1: Fitness Intake ──────────────────────────────\n")

    fitness_engine = TrueNorthEngine(
        goal_config = FITNESS_GOAL,
        session_id  = "chain_demo_fitness",
        router      = router,
    )
    resp = await fitness_engine.start()
    print(f"Agent: {resp.text}\n")

    for turn in FITNESS_TURNS:
        if fitness_engine.state.is_complete:
            break
        print(f"User:  {turn}")
        resp = await fitness_engine.process_message(turn)
        print(f"Agent: {resp.text}\n")
        if resp.output:
            fitness_output = resp.output.content
            break

    if not fitness_output:
        print("(Fitness session did not complete — run with real API keys)")
        return

    print("✅ Fitness intake complete!\n")

    # ── Part 2: Detect next goal from chain config ─────────────────────────
    print("── CHAIN DETECTION ─────────────────────────────────────\n")

    collected = fitness_engine.state.collected_fields
    chain     = GoalChain.from_yaml(FITNESS_GOAL.get("chain", {}))
    next_step = chain.next("fitness_plan", collected)

    if next_step:
        print(f"  Primary goal detected: '{collected.get('primary_goal', '?')}'")
        print(f"  Chain routes to:       '{next_step.goal_id}'")
        print(f"  Fields to carry:       {[m.source_field for m in chain.field_map_for(next_step).mappings]}")
    else:
        print("  No chain defined — session would end here")
        return

    # ── Part 3: State transfer ─────────────────────────────────────────────
    print("\n── STATE TRANSFER ──────────────────────────────────────\n")

    field_map = chain.field_map_for(next_step)
    transfer  = StateTransfer(field_map=field_map, auto_infer=True)
    result    = transfer.extract(
        source_state           = {"collected_fields": collected,
                                  "field_confidences": {k: 0.9 for k in collected}},
        source_goal_id         = "fitness_plan",
        target_goal_id         = next_step.goal_id,
        target_required_fields = ["name", "age", "weight_kg", "height_cm",
                                   "activity_level", "fitness_goal"],
    )

    print(f"  Fields carried:   {list(result.carried_fields.keys())}")
    print(f"  Fields skipped:   {result.skipped_fields}")
    print(f"  Fields missing:   {result.missing_fields}")
    print(f"  Coverage:         {result.coverage_pct:.0f}%\n")

    # ── Part 4: Run nutrition goal with pre-filled fields ──────────────────
    print("── GOAL 2: Nutrition Plan (pre-filled from fitness) ────\n")

    nutrition_engine = TrueNorthEngine(
        goal_config = NUTRITION_GOAL,
        session_id  = "chain_demo_nutrition",
        router      = router,
        seed_fields = result.carried_fields,   # ← pre-filled from fitness
    )

    # Show what was pre-filled
    print(f"  Pre-filled (not asked again):")
    for k, v in result.carried_fields.items():
        print(f"    {k}: {v}")
    print()

    resp = await nutrition_engine.start()
    print(f"Agent: {resp.text}\n")

    for turn in NUTRITION_TURNS:
        if nutrition_engine.state.is_complete:
            break
        print(f"User:  {turn}")
        resp = await nutrition_engine.process_message(turn)
        print(f"Agent: {resp.text}\n")
        if resp.output:
            nutrition_output = resp.output.content
            break

    if nutrition_output:
        print("✅ Nutrition plan complete!\n")


# ── DEMO 2: Manual state transfer ────────────────────────────────────────────

async def demo_manual_transfer():
    banner("MANUAL STATE TRANSFER — Code-level control")

    print("Scenario: Medical intake → Lab test form")
    print("  Carry: name, DOB, blood_group, allergies")
    print("  Rename: chief_complaint → reason_for_test\n")

    # Simulate a completed medical session state
    medical_state = {
        "collected_fields": {
            "patient_name":   "Rahul Kumar",
            "date_of_birth":  "10 June 1985",
            "chief_complaint": "routine check-up, diabetes monitoring",
            "blood_group":    "B+",
            "known_allergies": "penicillin",
            "medications":    "metformin 500mg",
        },
        "field_confidences": {k: 0.92 for k in
                              ["patient_name", "date_of_birth", "chief_complaint",
                               "blood_group", "known_allergies", "medications"]},
    }

    # Build explicit field map
    fm = (
        FieldMap()
        .add("patient_name",    "patient_name")
        .add("date_of_birth",   "dob")
        .add("blood_group",     "blood_group")
        .add("known_allergies", "allergies")
        .add("chief_complaint", "reason_for_test")   # rename on carry
    )

    transfer = StateTransfer(field_map=fm, confidence_threshold=0.80)
    result   = transfer.extract(
        source_state           = medical_state,
        source_goal_id         = "medical_intake",
        target_goal_id         = "lab_test_form",
        target_required_fields = ["patient_name", "dob", "blood_group",
                                   "allergies", "reason_for_test"],
    )

    print("Transfer result:")
    print(f"  Carried {result.carried_count} fields ({result.coverage_pct:.0f}% coverage)")
    print()
    for src, tgt in [("patient_name", "patient_name"), ("date_of_birth", "dob"),
                     ("chief_complaint", "reason_for_test"), ("blood_group", "blood_group"),
                     ("known_allergies", "allergies")]:
        val = result.carried_fields.get(tgt, "NOT CARRIED")
        print(f"  {src:<25} → {tgt:<25} = {val}")

    d = result.to_dict()
    print(f"\n  to_dict() keys: {list(d.keys())}")


# ── DEMO 3: GoalChain YAML config ────────────────────────────────────────────

async def demo_chain_config():
    banner("GOAL CHAIN CONFIG — Reading from YAML")

    config = {
        "on_complete": [
            {
                "if":           {"primary_goal": "lose weight"},
                "then":         "nutrition_plan",
                "carry_fields": ["age", "weight_kg", {"activity_level": "current_activity"}],
            },
            {
                "if":           {"primary_goal": "build muscle"},
                "then":         "strength_plan",
                "carry_fields": ["age", "weight_kg"],
            },
            {
                "else":         "general_wellness",
                "carry_fields": ["age"],
            },
        ],
    }

    chain = GoalChain.from_yaml(config)
    print(f"  Goals in chain: {chain.all_goals()}\n")

    test_cases = [
        {"primary_goal": "lose weight",    "age": 28, "weight_kg": 65},
        {"primary_goal": "build muscle",   "age": 32, "weight_kg": 80},
        {"primary_goal": "run a marathon", "age": 25, "weight_kg": 60},
    ]

    print(f"  {'Collected goal':<30} {'Routes to':<20} {'Fields carried'}")
    print("  " + "─" * 65)

    for collected in test_cases:
        step = chain.next("fitness_plan", collected)
        if step:
            fm     = chain.field_map_for(step)
            fields = [f"{m.source_field}→{m.target_field}" for m in fm.mappings]
            print(f"  {collected['primary_goal']:<30} {step.goal_id:<20} {fields}")
        else:
            print(f"  {collected['primary_goal']:<30} (no match)")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    demo  = os.environ.get("DEMO",  "all").lower()
    chain = os.environ.get("CHAIN", "fitness-nutrition").lower()

    print("=" * 60)
    print("  TrueNorth Goal Chaining Demo")
    print("=" * 60)

    if demo in ("all", "chain"):
        await demo_auto_chaining()

    if demo in ("all", "transfer"):
        await demo_manual_transfer()

    if demo in ("all", "config"):
        await demo_chain_config()

    print("\n" + "=" * 60)
    print("  Goal chaining turns single sessions into journeys.")
    print("  Each goal carries state — users never repeat themselves.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())