// my-demo (Go) — Basic TrueNorth Starter Demo
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// The simplest possible TrueNorth demo in Go.
// Loads a goal YAML, runs a conversation in the terminal,
// prints the structured output at the end.
//
// Use this as your starting point for any new TrueNorth project.
// Swap in a different goal.yaml and it works for any domain.
//
// This is the Go equivalent of my-demo/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   my-demo/
//   ├── main.go             ← this file
//   ├── medical-intake.yaml ← your goal config (swap this out)
//   └── go.mod
//
//   Copy any goal from samples/go-lang/*/goal.yaml and run it here.
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/my-demo
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/my-demo
//   go mod download
//   go build -o demo .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Run the demo
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   export ANTHROPIC_API_KEY=sk-ant-...
//   ./demo
//
//   # Use a different goal file
//   ./demo --goal=../farm_advisory/goal.yaml
//   ./demo --goal=../hr-screener/goal.yaml
//
//   # Scripted mode (no typing — useful for CI testing)
//   ./demo --scripted
//
// WHAT YOU SEE (normal mode)
// ──────────────────────────────────────────────────────────────
//   TrueNorth Demo (Go) — medical-intake.yaml
//   ─────────────────────────────────────────
//   ✓ Connected: http://localhost:8000
//   Goal: Patient Intake  (12 fields)
//
//   [0%] Agent: Hello! I am here to help with your intake.
//              What is your full name?
//
//   You: Priya Nair
//
//   [15%] Agent: Nice to meet you Priya! How old are you?
//
//   You: 28
//   ...
//   [100%] Agent: Thank you! Your intake is complete.
//
//   ── OUTPUT ───────────────────────────────────────────────
//   {
//     "patient_name": "Priya Nair",
//     "age": 28,
//     "chief_complaint": "headache and fever",
//     "clinical_summary": "28-year-old female presenting with..."
//   }
//   ─────────────────────────────────────────────────────────
//   Session: demo_20250615_143022
//   Turns  : 8
//   Time   : 45.3 seconds

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

// ── Flags ─────────────────────────────────────────────────────────────────────

var (
	flagGoal     = flag.String("goal", "medical-intake.yaml", "Path to goal YAML file")
	flagScripted = flag.Bool("scripted", false,               "Use scripted answers (no typing)")
	flagOutput   = flag.String("out", "",                     "Save output JSON to this path")
)

// ── Scripted answers (for testing without typing) ─────────────────────────────

var scriptedAnswers = []string{
	"Priya Nair",
	"28",
	"Severe headache and fever for 3 days",
	"7",
	"3 days, since Monday",
	"Gets worse when I try to sit up",
	"Rest and paracetamol help a little",
	"Paracetamol 500mg when needed",
	"No known allergies",
	"Nothing significant",
	"My husband Rajan, 9876543210",
}

// ── ANSI colours ──────────────────────────────────────────────────────────────

const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	dim    = "\033[2m"
	green  = "\033[32m"
	cyan   = "\033[36m"
	yellow = "\033[33m"
)

func c(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }

// ── Progress prefix ───────────────────────────────────────────────────────────

func pct(p float64) string {
	if p >= 100 { return c("[100%]", green, bold) }
	if p >= 50  { return c(fmt.Sprintf("[%.0f%%]", p), cyan, bold) }
	return c(fmt.Sprintf("[%.0f%%]", p), yellow, bold)
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	tnURL := os.Getenv("TRUENORTH_BASE_URL")
	if tnURL == "" { tnURL = "http://localhost:8000" }
	tnKey := os.Getenv("TRUENORTH_API_KEY")

	// ── Header ────────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Printf("  %s\n", c("TrueNorth Demo (Go) — "+*flagGoal, bold, cyan))
	fmt.Println("  " + strings.Repeat("─", 45))

	// ── Connect ───────────────────────────────────────────────────────────
	client := truenorth.New(truenorth.Config{
		BaseURL: tnURL, APIKey: tnKey, Timeout: 90 * time.Second,
	})
	ctx := context.Background()

	if h, err := client.Health(ctx); err != nil {
		fmt.Printf("  %s %s\n", c("✗", yellow), c("TrueNorth API not reachable at "+tnURL, yellow))
		fmt.Println(c("  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000\n", dim))
	} else {
		fmt.Printf("  %s Connected: %s (%s)\n", c("✓", green), tnURL, h.Status)
	}

	// ── Load goal ─────────────────────────────────────────────────────────
	goalData, err := client.Goals.Load(ctx, *flagGoal)
	if err != nil {
		fmt.Printf("  %s Could not load %s: %v\n", c("✗", yellow), *flagGoal, err)
		fmt.Println(c("  Using default goal ID: patient_intake", dim))
		goalData = &truenorth.GoalInfo{ID: "patient_intake", Name: "Patient Intake", FieldCount: 12}
	}
	fmt.Printf("  Goal: %s  (%d fields)\n\n", c(goalData.Name, bold), goalData.FieldCount)

	if *flagScripted {
		fmt.Println(c("  Mode: SCRIPTED (no typing needed)", cyan))
		fmt.Println()
	}

	// ── Create session ────────────────────────────────────────────────────
	sessionID := "demo_" + time.Now().Format("20060102_150405")
	session, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    goalData.ID,
		SessionID: sessionID,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n", err)
		os.Exit(1)
	}

	startTime := time.Now()
	turn      := 0
	scanner   := bufio.NewScanner(os.Stdin)

	fmt.Printf("  %s Agent: %s\n\n", pct(0), session.AgentMessage)

	// ── Conversation loop ─────────────────────────────────────────────────
	for !session.IsComplete {
		var userInput string

		if *flagScripted {
			if turn >= len(scriptedAnswers) {
				fmt.Println(c("  (script exhausted — switching to interactive)", yellow))
				*flagScripted = false
			} else {
				userInput = scriptedAnswers[turn]
				fmt.Printf("  You: %s\n", c(userInput, bold))
				time.Sleep(300 * time.Millisecond) // simulate typing
			}
		}

		if !*flagScripted && userInput == "" {
			fmt.Print("  You: ")
			if !scanner.Scan() { break }
			userInput = strings.TrimSpace(scanner.Text())
		}

		if userInput == "" { continue }
		if userInput == "quit" || userInput == "q" {
			fmt.Println(c("\n  Exited early.\n", yellow))
			return
		}

		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sessionID, Text: userInput,
		})
		if err != nil {
			fmt.Printf("  %s Error: %v\n", c("✗", yellow), err)
			continue
		}

		turn++
		fmt.Printf("\n  %s Agent: %s\n\n",
			pct(result.CompletionPct), result.Text)

		session.IsComplete    = result.IsComplete
		session.CompletionPct = result.CompletionPct

		if result.IsComplete && result.Output != nil {
			// ── Print output ───────────────────────────────────────────
			elapsed := time.Since(startTime).Seconds()

			fmt.Println("  " + strings.Repeat("─", 55))
			fmt.Println(c("  OUTPUT", bold, cyan))
			fmt.Println("  " + strings.Repeat("─", 55))

			data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
			fmt.Println(string(data))

			fmt.Println("  " + strings.Repeat("─", 55))
			fmt.Printf("  Session : %s\n", c(sessionID, dim))
			fmt.Printf("  Turns   : %d\n", turn)
			fmt.Printf("  Time    : %.1f seconds\n", elapsed)
			fmt.Printf("  Cost    : ~$%.6f (estimated)\n\n", float64(turn)*0.00015)

			// ── Save output ────────────────────────────────────────────
			if *flagOutput != "" {
				out := map[string]interface{}{
					"session_id":  sessionID,
					"goal":        goalData.Name,
					"turns":       turn,
					"elapsed_sec": elapsed,
					"output":      result.Output.Content,
				}
				data, _ := json.MarshalIndent(out, "", "  ")
				if err := os.WriteFile(*flagOutput, data, 0644); err != nil {
					fmt.Println(c("  Could not save: "+err.Error(), yellow))
				} else {
					fmt.Printf("  %s Saved to %s\n\n", c("✓", green), *flagOutput)
				}
			}
			break
		}
	}

	client.Sessions.End(ctx, sessionID)
	fmt.Println(c("  Done.\n", dim))
}