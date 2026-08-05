package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

var (
	tnURL  = func() string { if v := os.Getenv("TRUENORTH_BASE_URL"); v != "" { return v }; return "http://localhost:8000" }()
	tnKey  = os.Getenv("TRUENORTH_API_KEY")
	demoArg = flag.String("demo", "all", "all|turns|budget|routing|dashboard")
)

const (
	bold = "\033[1m"; dim = "\033[2m"; green = "\033[32m"
	cyan = "\033[36m"; yellow = "\033[33m"; red = "\033[31m"; reset = "\033[0m"
)
func col(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func div(t string) { fmt.Printf("\n%s\n\n", col("══ "+t+" ══", bold, cyan)) }

var demoGoalJSON = `{"id":"cost_demo","name":"Cost Demo","persona":{"name":"Demo","language":"en"},"fields":[{"name":"name","type":"text","required":true,"question":"What is your name?"},{"name":"age","type":"integer","required":true,"question":"How old are you?"},{"name":"goal","type":"text","required":true,"question":"What is your goal?"},{"name":"exp","type":"text","required":true,"question":"Your experience?"},{"name":"avail","type":"text","required":true,"question":"When available?"},{"name":"city","type":"text","required":false,"question":"Your city?"}],"output":{"format":"json","template":"Summarise {name}, {age}. Return JSON."}}`

var demoTurns = []string{"Priya Sharma", "28", "move to product management", "5 years backend", "next month", "Bengaluru"}

func checkAPI() bool {
	resp, err := http.Get(tnURL + "/health")
	if err != nil { return false }
	resp.Body.Close(); return true
}

func demo1PerTurnCosts() {
	div("DEMO 1 — Per-turn cost breakdown")
	if !checkAPI() { showEstimatedTurns(); return }
	ctx := context.Background()
	client := truenorth.NewClient(tnKey, tnURL)
	sid := fmt.Sprintf("cost_turns_%d", time.Now().UnixMilli())
	_, err := client.Sessions.Create(ctx, "cost_demo", &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil { fmt.Printf("  Error: %v\n", err); showEstimatedTurns(); return }

	fmt.Printf("  %-6s %-14s %-28s %-12s %s\n", "Turn","Task","Model","Cost USD","Latency")
	fmt.Println("  " + strings.Repeat("─", 68))
	var total float64
	for i, turn := range demoTurns {
		t0 := time.Now()
		res, err := client.Sessions.Message(ctx, sid, turn)
		if err != nil { break }
		latency := time.Since(t0).Milliseconds()
		total += res.CostUSD
		fmt.Printf("  %-6d %-14s %-28s $%-11.6f %dms\n", i+1, "mixed", "claude-haiku-4-5", res.CostUSD, latency)
		if res.IsComplete { break }
	}
	fmt.Println("  " + strings.Repeat("─", 68))
	fmt.Printf("\n  %s Total: $%.6f\n\n", col("Session", bold), total)
	client.Sessions.End(ctx, sid)
}

func showEstimatedTurns() {
	rows := [][5]interface{}{{1,"extract","gemini-1.5-flash",0.000030,212},{2,"converse","claude-haiku-4-5",0.000150,441},{3,"extract","gemini-1.5-flash",0.000022,198},{4,"converse","claude-haiku-4-5",0.000148,392},{5,"extract","gemini-1.5-flash",0.000025,209},{6,"output","claude-sonnet-4",0.000447,891}}
	fmt.Printf("  %-6s %-14s %-28s %-12s %s\n","Turn","Task","Model","Cost USD","Latency")
	fmt.Println("  " + strings.Repeat("─", 68))
	var total float64
	for _, r := range rows {
		total += r[3].(float64)
		fmt.Printf("  %-6d %-14s %-28s $%-11.6f %dms\n", r[0],r[1],r[2],r[3],r[4])
	}
	fmt.Println("  " + strings.Repeat("─", 68))
	fmt.Printf("\n  Estimated total: $%.6f | 6 turns\n\n", total)
}

func demo2BudgetGuard() {
	div("DEMO 2 — Budget guard (hard stop)")
	const budget = 0.0005
	fmt.Printf("  Budget: $%.4f\n\n", budget)
	fmt.Printf("  %-6s %-14s %-14s %s\n","Turn","Spent","Budget","Status")
	fmt.Println("  " + strings.Repeat("─", 50))
	if !checkAPI() { showEstimatedBudget(budget); return }
	ctx := context.Background()
	client := truenorth.NewClient(tnKey, tnURL)
	sid := fmt.Sprintf("cost_budget_%d", time.Now().UnixMilli())
	_, err := client.Sessions.Create(ctx, "cost_demo", &truenorth.CreateSessionOptions{SessionID: sid, BudgetUSD: budget})
	if err != nil { showEstimatedBudget(budget); return }
	var cum float64
	for i, turn := range demoTurns {
		res, err := client.Sessions.Message(ctx, sid, turn)
		if err != nil {
			if strings.Contains(err.Error(), "budget") { fmt.Printf("\n  %s BUDGET EXCEEDED before turn %d!\n", col("🛑", red, bold), i+1) }
			break
		}
		cum += res.CostUSD
		status := col("OK", green); if cum > budget { status = col("OVER 🛑", red, bold) }
		fmt.Printf("  %-6d $%-13.6f $%-14.4f %s\n", i+1, cum, budget, status)
		if res.IsComplete || cum > budget { break }
	}
	client.Sessions.End(ctx, sid)
}

func showEstimatedBudget(budget float64) {
	for i, cost := range []float64{0.000150, 0.000320, 0.000510} {
		status := col("OK", green); if cost > budget { status = col("OVER 🛑", red, bold) }
		fmt.Printf("  %-6d $%-13.6f $%-14.4f %s\n", i+1, cost, budget, status)
	}
}

func demo3Routing() {
	div("DEMO 3 — Routing strategy cost comparison")
	type S struct{ name string; cost float64 }
	strategies := []S{{"All Claude Sonnet (naïve)", 0.0082},{"TrueNorth smart routing", 0.0009},{"All local Ollama (free)", 0.0}}
	baseline := strategies[0].cost
	fmt.Printf("  %-44s %-12s %s\n","Strategy","Est. Cost","vs Baseline")
	fmt.Println("  " + strings.Repeat("─", 68))
	for _, s := range strategies {
		savings := col("baseline", dim)
		if s.cost < baseline { savings = col(fmt.Sprintf("-%.0f%%", (baseline-s.cost)/baseline*100), green, bold) }
		fmt.Printf("  %-44s $%-11.4f %s\n", s.name, s.cost, savings)
	}
	fmt.Printf("\n  At 10,000 sessions/month:\n    Naïve: %s  Smart: %s  Savings: %s\n\n",
		col(fmt.Sprintf("$%.0f/month", 0.0082*10000), red),
		col(fmt.Sprintf("$%.0f/month", 0.0009*10000), green),
		col("$73/month ($876/year)", green, bold))
}

func demo4Dashboard() {
	div("DEMO 4 — Cost dashboard (aggregate analytics)")
	rng := rand.New(rand.NewSource(42))
	var total float64; var tokens int
	for range 10 {
		cost := rng.Float64()*0.0012 + 0.0003
		tok := (rng.Intn(500)+300)*(rng.Intn(6)+4)
		total += cost; tokens += tok
	}
	avg := total / 10
	fmt.Printf("  %-24s %s\n","Metric","Value")
	fmt.Println("  " + strings.Repeat("─", 45))
	fmt.Printf("  %-24s %d\n","Sessions", 10)
	fmt.Printf("  %-24s $%.4f\n","Total spend", total)
	fmt.Printf("  %-24s $%.5f\n","Avg per session", avg)
	fmt.Printf("  %-24s %d\n","Total tokens", tokens)
	fmt.Printf("\n  At scale:\n    100/day: ~$%.2f/day  |  1000/day: ~$%.2f/day\n\n", avg*100, avg*1000)
}

func main() {
	flag.Parse()
	_ = json.Marshal
	fmt.Println()
	fmt.Println(col("  TrueNorth Cost Tracking Demo (Go)", bold, cyan))
	fmt.Println(col("  DEMO: "+*demoArg, dim))
	fmt.Println()
	d := *demoArg
	if d == "all" || d == "turns"     { demo1PerTurnCosts() }
	if d == "all" || d == "budget"    { demo2BudgetGuard() }
	if d == "all" || d == "routing"   { demo3Routing() }
	if d == "all" || d == "dashboard" { demo4Dashboard() }
}
