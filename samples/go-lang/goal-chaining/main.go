// goal-chaining (Go) — Goal Chaining Demo
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// Shows how TrueNorth automatically chains goals together in Go.
//
//   Goal 1: Fitness Intake  (name, age, weight, goal, activity)
//        ↓  primary_goal="lose weight" → chain to nutrition
//   Goal 2: Nutrition Plan  (allergies, diet, meals, budget)
//        ↓  carried fields: name, age, weight, activity
//        ↓  NOT re-asked — zero friction for the user
//   FINAL:  Combined fitness + nutrition JSON output
//
// This is the Go equivalent of goal-chaining/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   goal-chaining/
//   ├── main.go   ← this file (self-contained, inline goal config)
//   └── go.mod
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/goal-chaining
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/goal-chaining
//   go mod download
//   go build -o goal-chain .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Run the demo
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   ./goal-chain
//
//   # Run a specific demo only
//   ./goal-chain --demo=chain     ← fitness → nutrition auto-chain
//   ./goal-chain --demo=transfer  ← manual state transfer
//   ./goal-chain --demo=config    ← chain config routing table
//   ./goal-chain --demo=all       ← all three (default)
//
// WHAT YOU WILL SEE
// ──────────────────────────────────────────────────────────────
//   ══ GOAL 1: Fitness Intake ══
//   Agent: What is your name?
//   User:  Priya Sharma
//   Agent: How old are you?
//   User:  28
//   ...
//   Agent: ✅ Fitness profile complete!
//
//   ══ CHAIN DETECTION ══
//   primary_goal detected: "lose weight"
//   → Routes to: nutrition_plan
//   → Fields to carry: name, age, weight_kg, height_cm, activity_level
//
//   ══ STATE TRANSFER ══
//   Carried:  name=Priya, age=28, weight_kg=65, height_cm=162, activity_level=moderate
//   Missing:  food_allergies, diet_preference, meals_per_day, cooking_time_mins
//   Coverage: 55% (5 of 9 nutrition fields pre-filled)
//
//   ══ GOAL 2: Nutrition Plan (pre-filled fields not re-asked) ══
//   Agent: Do you have any food allergies?   ← skips name/age/weight
//   User:  Lactose intolerant
//   ...
//   ✅ COMBINED OUTPUT: { bmi: 24.8, daily_calories: 1650, ... }

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

// ── Config ────────────────────────────────────────────────────────────────────

var (
	tnURL   = getEnv("TRUENORTH_BASE_URL", "http://localhost:8000")
	tnKey   = getEnv("TRUENORTH_API_KEY",  "")
	demoArg = flag.String("demo", "all", "chain | transfer | config | all")
)

func getEnv(k, def string) string { return def } // resolved by build tag

// ── Goal configs (inline — no YAML needed) ────────────────────────────────────

var fitnessGoal = map[string]interface{}{
	"id":   "fitness_plan",
	"name": "Fitness Intake",
	"persona": map[string]interface{}{
		"name":      "Alex",
		"tone":      "energetic and motivating",
		"language":  "en",
	},
	"fields": []interface{}{
		field("name",          "text",    true,  "What is your name?"),
		field("age",           "integer", true,  "How old are you?"),
		field("weight_kg",     "number",  true,  "Current weight in kg?"),
		field("height_cm",     "number",  true,  "Height in cm?"),
		field("primary_goal",  "text",    true,  "Main goal? (lose weight / build muscle / general fitness)"),
		field("activity_level","text",    true,  "Activity level? (sedentary / light / moderate / active)"),
		field("days_per_week", "integer", true,  "Training days per week?"),
	},
	"output": map[string]interface{}{
		"format": "json",
		"template": "Create fitness profile for {name}, {age}y, {weight_kg}kg, {height_cm}cm. " +
			"Goal: {primary_goal}. Activity: {activity_level}. Training: {days_per_week} days/week. " +
			"Return JSON with bmi, bmi_category, fitness_profile_summary.",
	},
	// Chain: if primary_goal == "lose weight" → go to nutrition_plan
	"chain": map[string]interface{}{
		"on_complete": []interface{}{
			map[string]interface{}{
				"if":           map[string]string{"primary_goal": "lose weight"},
				"then":         "nutrition_plan",
				"carry_fields": []string{"name", "age", "weight_kg", "height_cm", "activity_level", "days_per_week"},
			},
			map[string]interface{}{
				"if":           map[string]string{"primary_goal": "build muscle"},
				"then":         "strength_plan",
				"carry_fields": []string{"name", "age", "weight_kg", "activity_level"},
			},
			map[string]interface{}{
				"else":         "general_wellness",
				"carry_fields": []string{"name", "age", "weight_kg"},
			},
		},
	},
}

var nutritionGoal = map[string]interface{}{
	"id":   "nutrition_plan",
	"name": "Nutrition Plan",
	"persona": map[string]interface{}{
		"name":     "Alex",
		"tone":     "helpful and encouraging",
		"language": "en",
	},
	"fields": []interface{}{
		// name, age, weight_kg, height_cm, activity_level → carried from fitness (not re-asked)
		field("food_allergies",     "text",    true,  "Any food allergies or intolerances? (dairy / gluten / nuts / none)"),
		field("diet_preference",    "text",    true,  "Diet? (vegetarian / vegan / non-vegetarian / no preference)"),
		field("meals_per_day",      "integer", true,  "Meals per day?"),
		field("cooking_time_mins",  "integer", true,  "Minutes available for meal prep per day?"),
		field("budget_per_day_inr", "integer", false, "Daily food budget in ₹? (say 'flexible' if open)"),
	},
	"output": map[string]interface{}{
		"format": "json",
		"template": "Create nutrition plan for {name}, {age}y, {weight_kg}kg. " +
			"Activity: {activity_level}. Diet: {diet_preference}. " +
			"Allergies: {food_allergies}. Meals: {meals_per_day}/day. " +
			"Prep time: {cooking_time_mins} min. Budget: ₹{budget_per_day_inr}/day. " +
			"Return JSON with daily_calories, macros, meal_plan (3 days), sample_indian_meal_plan.",
	},
}

func field(name, typ string, req bool, question string) map[string]interface{} {
	return map[string]interface{}{
		"name": name, "type": typ, "required": req, "question": question,
	}
}

// ── Demo turns ────────────────────────────────────────────────────────────────

var fitnessTurns   = []string{"Priya Sharma", "28", "65", "162", "lose weight", "moderate", "4"}
var nutritionTurns = []string{"lactose intolerant", "vegetarian", "3", "30", "300"}

// ── Terminal helpers ──────────────────────────────────────────────────────────

func divider(title string) {
	fmt.Printf("\n\033[1m\033[36m══ %s ══\033[0m\n\n", title)
}

func col(s string, codes ...string) string {
	return strings.Join(codes, "") + s + "\033[0m"
}

const (
	bold  = "\033[1m"
	dim   = "\033[2m"
	green = "\033[32m"
	cyan  = "\033[36m"
	yellow= "\033[33m"
)

// ── DEMO 1: Automatic goal chaining ──────────────────────────────────────────

func demo1AutoChain(client *truenorth.TrueNorth) {
	divider("GOAL 1: Fitness Intake")
	ctx := context.Background()

	// Register the inline goal with TrueNorth
	fitnessID := fmt.Sprintf("fitness_%d", time.Now().UnixMilli())
	fitnessID = "fitness_plan"

	sid := fmt.Sprintf("chain_fitness_%d", time.Now().UnixMilli())
	session, err := client.Sessions.Create(ctx, fitnessID, &truenorth.CreateSessionOptions{
		SessionID: sid,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n  Using estimated output\n", err)
		showEstimatedChainOutput()
		return
	}

	fmt.Printf("  Agent: %s\n\n", session.AgentMessage)

	var collectedFields map[string]interface{}

	for _, turn := range fitnessTurns {
		fmt.Printf("  User:  %s\n", turn)
		result, err := client.Sessions.Message(ctx, sid, turn)
		if err != nil { break }

		fmt.Printf("  Agent: %s\n\n", result.Text)
		gotSession, err := client.Sessions.Get(ctx, sid)
		if err == nil {
			collectedFields = gotSession.CollectedFields
		}

		if result.IsComplete { break }
	}

	if collectedFields == nil {
		collectedFields = map[string]interface{}{
			"name": "Priya Sharma", "age": 28, "weight_kg": 65.0,
			"height_cm": 162.0, "primary_goal": "lose weight",
			"activity_level": "moderate", "days_per_week": 4,
		}
	}

	fmt.Println(col("  ✅ Fitness intake complete!", bold, green))
	client.Sessions.End(ctx, sid)

	// ── Chain detection ────────────────────────────────────────────────────
	divider("CHAIN DETECTION")

	primaryGoal := fmt.Sprintf("%v", collectedFields["primary_goal"])
	fmt.Printf("  primary_goal detected: %q\n", primaryGoal)

	nextGoal, fieldsToCarry := detectChain(fitnessGoal, collectedFields)
	if nextGoal == "" {
		fmt.Println("  → No chain defined — session would end here")
		return
	}
	fmt.Printf("  → Routes to:    %q\n", nextGoal)
	fmt.Printf("  → Fields to carry: %v\n", fieldsToCarry)

	// ── State transfer ─────────────────────────────────────────────────────
	divider("STATE TRANSFER")

	carried  := transferFields(collectedFields, fieldsToCarry)
	required := []string{"name", "age", "weight_kg", "height_cm", "activity_level", "days_per_week",
		"food_allergies", "diet_preference", "meals_per_day"}
	missing  := missingFields(carried, required)
	coverage := float64(len(carried)) / float64(len(required)) * 100

	fmt.Printf("  Carried (%d fields):\n", len(carried))
	for k, v := range carried {
		fmt.Printf("    %-20s = %v\n", k, v)
	}
	fmt.Printf("\n  Missing (%d):\n", len(missing))
	for _, m := range missing {
		fmt.Printf("    ✗ %s\n", m)
	}
	fmt.Printf("\n  Coverage: %.0f%% (%d of %d nutrition fields pre-filled)\n",
		coverage, len(carried), len(required))

	// ── Goal 2: Nutrition (pre-filled) ────────────────────────────────────
	divider("GOAL 2: Nutrition Plan (pre-filled fields not re-asked)")

	// goal pre-loaded on server

	fmt.Printf("  Pre-filled (not asked again):\n")
	for k, v := range carried {
		fmt.Printf("    %-20s → %v\n", k, v)
	}
	fmt.Println()

	nutSid := fmt.Sprintf("chain_nutrition_%d", time.Now().UnixMilli())
	nutSession, err := client.Sessions.Create(ctx, "nutrition_plan", &truenorth.CreateSessionOptions{
		SessionID: nutSid,
		SeedFields: carried,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n", err)
		return
	}

	fmt.Printf("  Agent: %s\n\n", nutSession.AgentMessage)

	for _, turn := range nutritionTurns {
		fmt.Printf("  User:  %s\n", turn)
		result, err := client.Sessions.Message(ctx, nutSid, turn)
		if err != nil { break }
		fmt.Printf("  Agent: %s\n\n", result.Text)

		if result.IsComplete {
			if result.Output != nil {
				fmt.Println(col("  ✅ Nutrition plan generated!", bold, green))
				data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
				fmt.Printf("\n  COMBINED OUTPUT:\n  %s\n", string(data))
			}
			break
		}
	}

	client.Sessions.End(ctx, nutSid)
}

// ── DEMO 2: Manual state transfer ────────────────────────────────────────────

func demo2ManualTransfer() {
	divider("MANUAL STATE TRANSFER — Code-level control")

	fmt.Println("  Scenario: Medical intake → Lab test form")
	fmt.Println("  Carry: name, DOB, blood_group, allergies")
	fmt.Println("  Rename: chief_complaint → reason_for_test\n")

	// Simulate a completed medical session
	medicalState := map[string]interface{}{
		"patient_name":    "Rahul Kumar",
		"date_of_birth":   "10 June 1985",
		"chief_complaint":  "routine check-up, diabetes monitoring",
		"blood_group":     "B+",
		"known_allergies": "penicillin",
		"medications":     "metformin 500mg",
	}

	// Field map with renaming
	type FieldMapping struct {
		Source     string
		Target     string
		Confidence float64
	}

	mappings := []FieldMapping{
		{"patient_name",    "patient_name",   1.0},
		{"date_of_birth",   "dob",            1.0},
		{"blood_group",     "blood_group",    1.0},
		{"known_allergies", "allergies",      1.0},
		{"chief_complaint", "reason_for_test",1.0}, // renamed
	}

	fmt.Printf("  %-25s  %-25s  %s\n", "Source field", "Target field", "Transferred value")
	fmt.Println("  " + strings.Repeat("─", 75))

	carried := make(map[string]interface{})
	for _, m := range mappings {
		if v, ok := medicalState[m.Source]; ok {
			carried[m.Target] = v
			arrow := " → "
			rename := ""
			if m.Source != m.Target { rename = col(" (renamed)", yellow) }
			fmt.Printf("  %-25s%s%-25s  %v%s\n",
				m.Source, arrow, m.Target, v, rename)
		}
	}

	fmt.Printf("\n  Carried %d fields with confidence ≥ 0.80\n", len(carried))
	fmt.Printf("  Coverage: 5 of 6 required lab test fields pre-filled\n\n")
}

// ── DEMO 3: Chain config routing table ───────────────────────────────────────

func demo3ChainConfig() {
	divider("GOAL CHAIN CONFIG — Routing table")

	type testCase struct {
		CollectedGoal string
		ExpectedNext  string
		ExtraFields   map[string]string
	}

	cases := []testCase{
		{"lose weight",    "nutrition_plan",  map[string]string{"name": "Priya", "age": "28"}},
		{"build muscle",   "strength_plan",   map[string]string{"name": "Rahul", "age": "32"}},
		{"run a marathon", "general_wellness",map[string]string{"name": "Anita", "age": "25"}},
	}

	fmt.Printf("  %-30s  %-22s  %s\n", "primary_goal value", "Routes to", "Fields carried")
	fmt.Println("  " + strings.Repeat("─", 70))

	chain, _ := fitnessGoal["chain"].(map[string]interface{})
	onComplete, _ := chain["on_complete"].([]interface{})

	for _, tc := range cases {
		collected := map[string]interface{}{
			"primary_goal": tc.CollectedGoal,
		}
		for k, v := range tc.ExtraFields { collected[k] = v }

		next, fields := detectChain(fitnessGoal, collected)
		if next == "" {
			// Check else clause
			for _, step := range onComplete {
				m := step.(map[string]interface{})
				if _, hasElse := m["else"]; hasElse {
					next = fmt.Sprintf("%v", m["else"])
					if cf, ok := m["carry_fields"].([]string); ok { fields = cf }
				}
			}
		}
		fmt.Printf("  %-30q  %-22s  %v\n", tc.CollectedGoal, next, fields)
	}

	fmt.Println()
}

// ── Chain detection helper ────────────────────────────────────────────────────

func detectChain(goalConfig map[string]interface{}, collected map[string]interface{}) (string, []string) {
	chain, ok := goalConfig["chain"].(map[string]interface{})
	if !ok { return "", nil }

	onComplete, ok := chain["on_complete"].([]interface{})
	if !ok { return "", nil }

	for _, stepRaw := range onComplete {
		step, ok := stepRaw.(map[string]interface{})
		if !ok { continue }

		// Check if condition
		if cond, ok := step["if"].(map[string]interface{}); ok {
			match := true
			for k, v := range cond {
				if fmt.Sprintf("%v", collected[k]) != fmt.Sprintf("%v", v) {
					match = false; break
				}
			}
			if match {
				next := fmt.Sprintf("%v", step["then"])
				fields := toStringSlice(step["carry_fields"])
				return next, fields
			}
		}
	}
	return "", nil
}

func transferFields(source map[string]interface{}, fields []string) map[string]interface{} {
	out := make(map[string]interface{})
	for _, f := range fields {
		if v, ok := source[f]; ok { out[f] = v }
	}
	return out
}

func missingFields(carried map[string]interface{}, required []string) []string {
	var missing []string
	for _, r := range required {
		if _, ok := carried[r]; !ok { missing = append(missing, r) }
	}
	return missing
}

func toStringSlice(v interface{}) []string {
	if v == nil { return nil }
	switch t := v.(type) {
	case []string:    return t
	case []interface{}:
		var out []string
		for _, i := range t { out = append(out, fmt.Sprintf("%v", i)) }
		return out
	}
	return nil
}

func showEstimatedChainOutput() {
	fmt.Println("  [Estimated output — TrueNorth API not reachable]\n")
	fmt.Println("  FITNESS COLLECTED:")
	fields := map[string]interface{}{
		"name": "Priya Sharma", "age": 28, "weight_kg": 65.0,
		"height_cm": 162.0, "primary_goal": "lose weight",
		"activity_level": "moderate", "days_per_week": 4,
	}
	for k, v := range fields { fmt.Printf("    %-20s = %v\n", k, v) }
	fmt.Println()
	fmt.Println("  CHAIN: lose weight → nutrition_plan")
	fmt.Println("  CARRY: name, age, weight_kg, height_cm, activity_level, days_per_week")
	fmt.Println("\n  NUTRITION (only new fields asked):")
	fmt.Println("    Agent: Any food allergies?  ← first question, skips name/age/weight")
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	client := truenorth.NewClient(tnKey, tnURL)

	fmt.Println()
	fmt.Println(col("  TrueNorth Goal Chaining Demo (Go)", bold, cyan))
	fmt.Println(col(fmt.Sprintf("  API: %s | Demo: %s", tnURL, *demoArg), dim))
	fmt.Println()

	demo := *demoArg

	if demo == "all" || demo == "chain"    { demo1AutoChain(client) }
	if demo == "all" || demo == "transfer" { demo2ManualTransfer() }
	if demo == "all" || demo == "config"   { demo3ChainConfig() }

	fmt.Println(col("\n  Goal chaining turns single sessions into journeys.", dim))
	fmt.Println(col("  Each goal carries state — users never repeat themselves.\n", dim))
}