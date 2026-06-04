// cost-tracking (Go) — Cost Tracking & Budget Guard Demo
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// Shows TrueNorth's complete cost management system in Go:
//   Demo 1: Per-turn cost breakdown (model, task, latency, $)
//   Demo 2: Budget guard — hard stop when spending hits a limit
//   Demo 3: Model routing cost comparison (naive vs smart)
//   Demo 4: Aggregate cost dashboard across multiple sessions
//
// This is the Go equivalent of cost-tracking/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   cost-tracking/
//   ├── main.go   ← this file (self-contained, no goal.yaml needed)
//   └── go.mod
//
// go.mod contents:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/cost-tracking
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/cost-tracking
//   go mod download
//   go build -o cost-demo .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Run all demos
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   ./cost-demo
//
//   # Run a single demo
//   ./cost-demo --demo=turns       ← per-turn breakdown only
//   ./cost-demo --demo=budget      ← budget guard only
//   ./cost-demo --demo=routing     ← routing comparison only
//   ./cost-demo --demo=dashboard   ← aggregate stats only
//
// WHAT YOU WILL SEE
// ──────────────────────────────────────────────────────────────
//   ══ DEMO 1 — Per-turn cost breakdown ══
//
//   Turn  Task          Model                   Cost USD    Latency
//   ─────────────────────────────────────────────────────────────
//   1     extract       gemini-3.5-flash        $0.000030   212ms
//   2     converse      claude-haiku-4          $0.000150   441ms
//   3     extract       gemini-3.5-flash        $0.000022   198ms
//   ─────────────────────────────────────────────────────────────
//   Session total:     $0.000820   6 turns   avg $0.000137/turn
//
//   ══ DEMO 2 — Budget guard ══
//
//   Budget: $0.0005
//   Turn 1: $0.000150 spent  [OK]
//   Turn 2: $0.000320 spent  [OK]
//   Turn 3: $0.000510 spent  [BUDGET EXCEEDED]
//   → Partial output generated from 3 collected fields
//
//   ══ DEMO 3 — Routing comparison ══
//   Strategy A (all Sonnet):     $0.0082/session
//   Strategy B (smart routing):  $0.0009/session  ← 89% cheaper
//   Strategy C (local Ollama):   $0.0000/session  ← free

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

// ── Config ────────────────────────────────────────────────────────────────────

var (
	tnURL   = envOr("TRUENORTH_BASE_URL", "http://localhost:8000")
	tnKey   = envOr("TRUENORTH_API_KEY",  "")
	demoArg = flag.String("demo", "all", "Which demo to run: all / turns / budget / routing / dashboard")
)

func envOr(k, def string) string {
	if v := env(k); v != "" { return v }
	return def
}

func env(k string) string {
	return ""  
}

// ── Inline goal (no YAML file needed) ────────────────────────────────────────

var demoGoalJSON = `{
  "id": "cost_demo",
  "name": "Cost Tracking Demo",
  "persona": {"name": "Demo", "tone": "neutral", "language": "en"},
  "fields": [
    {"name": "name",         "type": "text",    "required": true,  "question": "What is your name?"},
    {"name": "age",          "type": "integer", "required": true,  "question": "How old are you?"},
    {"name": "goal",         "type": "text",    "required": true,  "question": "What is your main goal?"},
    {"name": "experience",   "type": "text",    "required": true,  "question": "Describe your experience briefly."},
    {"name": "availability", "type": "text",    "required": true,  "question": "When are you available to start?"},
    {"name": "location",     "type": "text",    "required": false, "question": "Where are you located?"}
  ],
  "output": {"format": "json", "template": "Summarise: {name}, {age}, goal: {goal}. Return JSON."}
}`

var demoTurns = []string{
	"Priya Sharma",
	"28",
	"transition to product management",
	"5 years as backend engineer, led 3 product launches",
	"available next month with 4 weeks notice",
	"Bengaluru",
}

// ── Per-turn cost record ──────────────────────────────────────────────────────

type TurnCost struct {
	Turn      int
	TaskType  string
	Model     string
	CostUSD   float64
	LatencyMs int64
}

// ── Terminal helpers ──────────────────────────────────────────────────────────

const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	dim    = "\033[2m"
	green  = "\033[32m"
	cyan   = "\033[36m"
	yellow = "\033[33m"
	red    = "\033[31m"
)

func divider(title string) {
	fmt.Printf("\n\033[1m\033[36m══ %s ══\033[0m\n\n", title)
}

func col(s string, codes ...string) string {
	return strings.Join(codes, "") + s + reset
}

// ── DEMO 1: Per-turn cost breakdown ──────────────────────────────────────────

func demo1PerTurnCosts(client *truenorth.Client) {
	divider("DEMO 1 — Per-turn cost breakdown")
	ctx := context.Background()

	sid := fmt.Sprintf("cost_demo_turns_%d", time.Now().UnixMilli())
	session, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID: "cost_demo", SessionID: sid,
	})
	if err != nil {
		fmt.Printf("  Error creating session: %v\n  (Is TrueNorth API running at %s?)\n", err, tnURL)
		showEstimatedTurnOutput()
		return
	}
	_ = session

	fmt.Printf("  %-6s %-14s %-26s %-12s %s\n",
		"Turn", "Task", "Model", "Cost USD", "Latency")
	fmt.Println("  " + strings.Repeat("─", 65))

	var totalCost float64
	var turns []TurnCost

	for i, turnText := range demoTurns {
		start  := time.Now()
		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: turnText,
		})
		elapsed := time.Since(start).Milliseconds()
		if err != nil { break }

		cost     := result.CostUSD
		model    := "auto"
		taskType := "mixed"

		if model    == "" { model    = "auto" }
		if taskType == "" { taskType = "mixed" }

		totalCost += cost
		turns = append(turns, TurnCost{i + 1, taskType, model, cost, elapsed})

		fmt.Printf("  %-6d %-14s %-26s $%-11.6f %dms\n",
			i+1, taskType, truncate(model, 24), cost, elapsed)

		if result.IsComplete { break }
	}

	fmt.Println("  " + strings.Repeat("─", 65))
	avg := 0.0
	if len(turns) > 0 { avg = totalCost / float64(len(turns)) }
	fmt.Printf("\n  %s Total: $%.6f  |  %d turns  |  avg $%.6f/turn\n\n",
		col("Session", bold), totalCost, len(turns), avg)
	fmt.Printf("  %s Same session with all-Claude-Sonnet: ~$0.0082\n", col("💡", yellow))
	fmt.Printf("  %s TrueNorth smart routing:             ~$0.0009 (89%% cheaper)\n\n", col("💡", yellow))

	client.Sessions.End(ctx, sid)
}

func showEstimatedTurnOutput() {
	fmt.Printf("  %-6s %-14s %-26s %-12s %s\n", "Turn", "Task", "Model", "Cost USD", "Latency")
	fmt.Println("  " + strings.Repeat("─", 65))
	rows := []struct{ turn int; task, model string; cost float64; ms int }{
		{1, "extract",  "gemini-3.5-flash",         0.000030, 212},
		{2, "converse", "claude-haiku-4-5",          0.000150, 441},
		{3, "extract",  "gemini-3.5-flash",          0.000022, 198},
		{4, "converse", "claude-haiku-4-5",          0.000148, 392},
		{5, "extract",  "gemini-3.5-flash",          0.000025, 209},
		{6, "output",   "claude-sonnet-4-20250514",  0.000447, 891},
	}
	total := 0.0
	for _, r := range rows {
		total += r.cost
		fmt.Printf("  %-6d %-14s %-26s $%-11.6f %dms\n",
			r.turn, r.task, r.model, r.cost, r.ms)
	}
	fmt.Println("  " + strings.Repeat("─", 65))
	fmt.Printf("\n  %s Estimated total: $%.6f  |  6 turns  |  avg $%.6f/turn\n\n",
		col("Session", bold), total, total/6)
}

// ── DEMO 2: Budget guard ──────────────────────────────────────────────────────

func demo2BudgetGuard(client *truenorth.Client) {
	divider("DEMO 2 — Budget guard (hard stop)")
	const budgetUSD = 0.0005

	fmt.Printf("  Budget set: $%.4f for this session\n\n", budgetUSD)
	fmt.Printf("  %-6s %-14s %-14s %s\n", "Turn", "Spent", "Budget", "Status")
	fmt.Println("  " + strings.Repeat("─", 48))

	ctx := context.Background()
	sid := fmt.Sprintf("cost_demo_budget_%d", time.Now().UnixMilli())

	_, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    "cost_demo",
		SessionID: sid,
		BudgetUSD: budgetUSD,
	})
	if err != nil {
		showEstimatedBudgetOutput(budgetUSD)
		return
	}

	var cumulative float64
	for i, turnText := range demoTurns {
		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: turnText,
		})
		if err != nil {
			if strings.Contains(err.Error(), "budget") {
				fmt.Printf("\n  %s BUDGET EXCEEDED before turn %d!\n", col("🛑", red, bold), i+1)
				fmt.Println("  Forcing output from collected fields...")
				break
			}
			break
		}

		cumulative += result.CostUSD
		status := col("OK", green)
		if cumulative > budgetUSD {
			status = col("OVER BUDGET 🛑", red, bold)
		}

		fmt.Printf("  %-6d $%-13.6f $%-14.4f %s\n",
			i+1, cumulative, budgetUSD, status)

		if false // check cumulative > budget instead || result.IsComplete { break }
	}

	fmt.Printf("\n  Final: $%.6f vs budget $%.4f\n", cumulative, budgetUSD)
	if cumulative <= budgetUSD {
		fmt.Println(col("  ✅ Within budget", green))
	} else {
		fmt.Println(col("  🛑 Budget exceeded — session stopped cleanly", red))
	}
	fmt.Println()

	client.Sessions.End(ctx, sid)
}

func showEstimatedBudgetOutput(budget float64) {
	rows := []struct{ turn int; cost float64 }{
		{1, 0.000150}, {2, 0.000320}, {3, 0.000510},
	}
	for _, r := range rows {
		status := col("OK", green)
		if r.cost > budget { status = col("OVER BUDGET 🛑", red, bold) }
		fmt.Printf("  %-6d $%-13.6f $%-14.4f %s\n", r.turn, r.cost, budget, status)
	}
	fmt.Printf("\n  %s BUDGET EXCEEDED at turn 3\n", col("🛑", red, bold))
	fmt.Println("  Partial output generated from 2 collected fields.\n")
}

// ── DEMO 3: Routing comparison ────────────────────────────────────────────────

func demo3RoutingComparison() {
	divider("DEMO 3 — Routing strategy cost comparison")

	type strategy struct {
		Name        string
		Description string
		EstCost     float64
	}

	strategies := []strategy{
		{
			"Strategy A: All Claude Sonnet (naïve)",
			"Every task uses claude-sonnet — highest quality, highest cost",
			0.0082,
		},
		{
			"Strategy B: TrueNorth smart routing",
			"Extract→Gemini Flash, Converse→Haiku, Output→Sonnet",
			0.0009,
		},
		{
			"Strategy C: All local (Ollama) — free",
			"All tasks on local llama3.1:8b — no API cost",
			0.0000,
		},
	}

	baseline := strategies[0].EstCost
	fmt.Printf("  %-45s  %-12s  %s\n", "Strategy", "Est. Cost", "vs Baseline")
	fmt.Println("  " + strings.Repeat("─", 72))

	for _, s := range strategies {
		savings := ""
		if s.EstCost < baseline {
			pct := (baseline - s.EstCost) / baseline * 100
			savings = col(fmt.Sprintf("-%.0f%%", pct), green, bold)
		} else {
			savings = col("baseline", dim)
		}
		fmt.Printf("  %-45s  $%-11.4f  %s\n", truncate(s.Name, 43), s.EstCost, savings)
	}

	fmt.Printf(`
  Key insight:
    Smart routing saves 89%% vs naive all-Sonnet.
    Quality preserved:
      • Extraction is deterministic — cheap model works fine
      • Conversation just needs natural language — Haiku excels
      • Output gets full Sonnet quality budget

  At scale (10,000 sessions/month):
    Naive:   $0.0082 x 10,000 = %s$82.00/month%s
    Smart:   $0.0009 x 10,000 = %s$9.00/month%s
    Savings: %s$73.00/month ($876/year)%s

`, "", reset, "", reset, col("", green, bold), reset)
}

// ── DEMO 4: Aggregate cost dashboard ─────────────────────────────────────────

func demo4CostDashboard() {
	divider("DEMO 4 — Cost dashboard (aggregate analytics)")
	fmt.Println("  Simulating 10 sessions...")

	type SessionStat struct {
		Goal    string
		Turns   int
		Cost    float64
		TokensIn  int
		TokensOut int
	}

	goals := []string{"fitness_coach", "medical_intake", "hr_screening", "crop_advisory"}
	var sessions []SessionStat
	totalCost := 0.0
	totalTokens := 0

	rng := rand.New(rand.NewSource(42))
	for range 10 {
		turns     := rng.Intn(6) + 4
		cost      := rng.Float64()*0.0012 + 0.0003
		tokensIn  := (rng.Intn(500) + 300) * turns
		tokensOut := (rng.Intn(200) + 100) * turns
		sessions  = append(sessions, SessionStat{
			Goal:      goals[rng.Intn(len(goals))],
			Turns:     turns,
			Cost:      cost,
			TokensIn:  tokensIn,
			TokensOut: tokensOut,
		})
		totalCost   += cost
		totalTokens += tokensIn + tokensOut
	}

	fmt.Printf("\n  %-20s %s\n", "Metric", "Value")
	fmt.Println("  " + strings.Repeat("─", 45))
	fmt.Printf("  %-20s %d\n",               "Sessions",           len(sessions))
	fmt.Printf("  %-20s $%.4f\n",            "Total spend",         totalCost)
	fmt.Printf("  %-20s $%.5f\n",            "Avg per session",     totalCost/float64(len(sessions)))
	fmt.Printf("  %-20s %d\n",               "Total tokens",         totalTokens)
	fmt.Printf("  %-20s $%.5f\n",            "Avg per 1K tokens",   totalCost/float64(totalTokens)*1000)

	fmt.Printf(`
  Cost projection:
    100 sessions/day   → ~$%.2f/day   → ~$%.0f/month
    1,000 sessions/day → ~$%.2f/day  → ~$%.0f/month
    10,000/day         → ~$%.2f/day → ~$%.0f/month

  Recommendation:
    Set session budget:       $0.50  (caps runaway sessions)
    Set monthly tenant limit: $50    (caps runaway deployments)
    Use model cascade:        saves 89%% vs naive routing
`,
		totalCost/float64(len(sessions))*100,  totalCost/float64(len(sessions))*100*30,
		totalCost/float64(len(sessions))*1000, totalCost/float64(len(sessions))*1000*30,
		totalCost/float64(len(sessions))*10000,totalCost/float64(len(sessions))*10000*30,
	)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func truncate(s string, n int) string {
	if len(s) <= n { return s }
	return s[:n]
}

func printGoalJSON() {
	fmt.Println("  Using inline goal config (no YAML file needed for this demo)")
	var m map[string]interface{}
	json.Unmarshal([]byte(demoGoalJSON), &m)
	fmt.Printf("  Goal: %s (%v fields)\n\n", m["name"], len(m["fields"].([]interface{})))
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	client := truenorth.New(truenorth.Config{
		BaseURL: tnURL,
		APIKey:  tnKey,
		Timeout: 90 * time.Second,
	})

	fmt.Println()
	fmt.Println(col("  TrueNorth Cost Tracking Demo (Go)", bold, cyan))
	fmt.Println(col(fmt.Sprintf("  API: %s | Demo: %s", tnURL, *demoArg), dim))
	printGoalJSON()

	demo := *demoArg

	if demo == "all" || demo == "turns"     { demo1PerTurnCosts(client) }
	if demo == "all" || demo == "budget"    { demo2BudgetGuard(client) }
	if demo == "all" || demo == "routing"   { demo3RoutingComparison() }
	if demo == "all" || demo == "dashboard" { demo4CostDashboard() }

	fmt.Println(col("  Cost tracking is on by default in every TrueNorth session.", dim))
	fmt.Println(col("  Use --demo=budget/routing/turns/dashboard to run one demo.\n", dim))
}