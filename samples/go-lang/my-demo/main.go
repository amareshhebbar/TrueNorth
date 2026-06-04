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
	flagGoal     = flag.String("goal", "patient_intake", "Goal ID to run")
	flagScripted = flag.Bool("scripted", false, "Use scripted answers")
	flagOut      = flag.String("out", "", "Save output JSON to file")
)

var scriptedAnswers = []string{
	"Priya Nair", "28", "chest pain and fever for 3 days",
	"7", "since this morning", "gets worse climbing stairs",
	"paracetamol helps", "paracetamol 500mg", "no allergies",
	"hypertension 2 years ago", "husband Rajan 9876543210",
}

const (
	bold = "\033[1m"; dim = "\033[2m"; green = "\033[32m"
	cyan = "\033[36m"; yellow = "\033[33m"; reset = "\033[0m"
)
func col(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func pct(p float64) string {
	lbl := fmt.Sprintf("[%.0f%%]", p)
	if p >= 100 { return col(lbl, green, bold) }
	if p >= 50  { return col(lbl, cyan, bold) }
	return col(lbl, yellow, bold)
}

func main() {
	flag.Parse()
	tnURL := os.Getenv("TRUENORTH_BASE_URL"); if tnURL == "" { tnURL = "http://localhost:8000" }
	tnKey := os.Getenv("TRUENORTH_API_KEY")
	client := truenorth.NewClient(tnKey, tnURL)
	ctx := context.Background()

	fmt.Println()
	fmt.Printf("  %s\n", col("TrueNorth Demo (Go) — "+*flagGoal, bold, cyan))
	fmt.Println("  " + strings.Repeat("─", 45))

	if resp, err := http.Get(tnURL + "/health"); err != nil {
		fmt.Println(col("  ⚠  TrueNorth not reachable at "+tnURL, yellow))
		fmt.Println(col("  Run: uvicorn truenorth.api.main:app --port 8000", dim))
	} else {
		resp.Body.Close()
		fmt.Println(col("  ✓ Connected: "+tnURL, green))
	}
	if *flagScripted { fmt.Println(col("  Mode: SCRIPTED", cyan)) }
	fmt.Println()

	sessionID := "demo_" + time.Now().Format("20060102_150405")
	session, err := client.Sessions.Create(ctx, *flagGoal, &truenorth.CreateSessionOptions{SessionID: sessionID})
	if err != nil { fmt.Printf("  Error: %v\n", err); os.Exit(1) }

	startTime := time.Now()
	scanner := bufio.NewScanner(os.Stdin)
	scriptIdx := 0

	fmt.Printf("  %s Agent: %s\n\n", pct(0), session.AgentMessage)

	for {
		var userInput string
		if *flagScripted && scriptIdx < len(scriptedAnswers) {
			userInput = scriptedAnswers[scriptIdx]; scriptIdx++
			fmt.Printf("  You: %s\n", col(userInput, bold))
			time.Sleep(300 * time.Millisecond)
		} else {
			fmt.Print("  You: ")
			if !scanner.Scan() { break }
			userInput = strings.TrimSpace(scanner.Text())
		}
		if userInput == "" { continue }
		if userInput == "quit" || userInput == "q" { fmt.Println(col("\n  Exited.\n", yellow)); break }

		result, err := client.Sessions.Message(ctx, sessionID, userInput)
		if err != nil { fmt.Printf("  Error: %v\n", err); continue }

		fmt.Printf("\n  %s Agent: %s\n\n", pct(result.CompletionPct), result.Text)

		if result.IsComplete && result.Output != nil {
			elapsed := time.Since(startTime).Seconds()
			fmt.Println("  " + strings.Repeat("─", 55))
			fmt.Println(col("  OUTPUT", bold, cyan))
			fmt.Println("  " + strings.Repeat("─", 55))
			data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
			fmt.Println(string(data))
			fmt.Println("  " + strings.Repeat("─", 55))
			fmt.Printf("  Session : %s\n  Time    : %.1fs\n", col(sessionID, dim), elapsed)
			if *flagOut != "" {
				out := map[string]interface{}{"session_id": sessionID, "output": result.Output.Content}
				data, _ = json.MarshalIndent(out, "", "  ")
				os.WriteFile(*flagOut, data, 0644)
				fmt.Printf("  Saved   : %s\n", *flagOut)
			}
			break
		}
	}
	client.Sessions.End(ctx, sessionID)
}
