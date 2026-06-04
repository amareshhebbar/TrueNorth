package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

var (
	flagLang   = flag.String("lang",   "auto",    "Language hint: auto/hindi/english/kannada")
	flagOutput = flag.String("output", "",         "Save JSON output to file")
	flagGoal   = flag.String("goal",   "farm_advisory", "Goal ID to run")
)

const (
	reset = "\033[0m"; bold = "\033[1m"; dim = "\033[2m"
	green = "\033[32m"; cyan = "\033[36m"; yellow = "\033[33m"; red = "\033[31m"
)

func col(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func progressBar(pct float64, width int) string {
	filled := int(float64(width) * pct / 100.0)
	if filled > width { filled = width }
	return col("["+strings.Repeat("█", filled)+strings.Repeat("░", width-filled)+"]", green) +
		col(fmt.Sprintf(" %.0f%%", pct), bold)
}

func printBanner() {
	fmt.Println()
	fmt.Println(col("  ┌─────────────────────────────────────────────┐", bold, cyan))
	fmt.Println(col("  │  KisanMitra Farm Advisor  (Go CLI)          │", bold, cyan))
	fmt.Println(col("  │  Powered by TrueNorth · Apache 2.0          │", cyan))
	fmt.Println(col("  └─────────────────────────────────────────────┘", bold, cyan))
	fmt.Println()
	fmt.Println(col(fmt.Sprintf("  Date : %s  |  Goal : %s", time.Now().Format("02 Jan 2006"), *flagGoal), dim))
	fmt.Println()
}

func main() {
	flag.Parse()
	baseURL := os.Getenv("TRUENORTH_BASE_URL"); if baseURL == "" { baseURL = "http://localhost:8000" }
	apiKey  := os.Getenv("TRUENORTH_API_KEY")

	printBanner()

	client := truenorth.NewClient(apiKey, baseURL)
	ctx    := context.Background()

	if resp, err := http.Get(baseURL + "/health"); err != nil {
		fmt.Println(col("  ⚠  TrueNorth not reachable at "+baseURL, yellow))
		fmt.Println(col("  Run: uvicorn truenorth.api.main:app --port 8000\n", dim))
	} else {
		resp.Body.Close()
		fmt.Println(col("  ✓ Connected: "+baseURL, green))
		fmt.Println()
	}

	sid := fmt.Sprintf("fa_go_%d", time.Now().UnixMilli())
	session, err := client.Sessions.Create(ctx, *flagGoal, &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil {
		fmt.Println(col("  Error: "+err.Error(), red))
		os.Exit(1)
	}

	fmt.Println(col("  Advisor: ", bold, cyan) + session.AgentMessage)
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	turn    := 0

	for {
		turn++
		fmt.Printf("  %s  Turn %d\n\n", progressBar(session.CompletionPct, 28), turn)
		fmt.Print(col("  Farmer:  ", bold))

		if !scanner.Scan() { break }
		text := strings.TrimSpace(scanner.Text())
		if text == "" { continue }
		if text == "quit" || text == "q" {
			fmt.Println(col("\n  Session ended.\n", yellow))
			return
		}

		result, err := client.Sessions.Message(ctx, sid, text)
		if err != nil { fmt.Println(col("  Error: "+err.Error(), red)); continue }

		fmt.Println()
		fmt.Println(col("  Advisor: ", bold, cyan) + result.Text)
		fmt.Println()

		session.CompletionPct = result.CompletionPct

		if result.IsComplete && result.Output != nil {
			data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
			fmt.Println(col("  ── ADVISORY ─────────────────────────────────────", bold, green))
			fmt.Println(string(data))
			if *flagOutput != "" {
				out := map[string]interface{}{"session_id": sid, "advisory": result.Output.Content}
				data, _ = json.MarshalIndent(out, "", "  ")
				os.WriteFile(*flagOutput, data, 0644)
				fmt.Println(col("  Saved: "+*flagOutput, green))
			}
			break
		}
	}
	client.Sessions.End(ctx, sid)
}
