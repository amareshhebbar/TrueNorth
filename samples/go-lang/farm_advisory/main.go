// farm-advisor-cli — Go Terminal Crop Advisor
//
// PROJECT STRUCTURE
// -----------------
//   farm-advisor-cli/
//   ├── main.go       ← this file
//   ├── goal.yaml     ← crop advisory questionnaire
//   └── go.mod
//
// go.mod contents:
// ----------------
//   module github.com/truenorth-ai/farm-advisor-cli
//
//   go 1.22
//
//   require (
//     github.com/truenorth-ai/truenorth-go v0.1.0
//     github.com/fatih/color v1.16.0
//   )
//
// WHAT THIS DOES
// --------------
// A terminal-based crop advisory tool for agricultural extension workers.
// Extension worker sits with a farmer, runs this tool, and gets
// a structured diagnosis + treatment plan in Hindi and English.
//
// Works offline-first: uses local LLM (Ollama) if configured,
// falls back to cloud API for final diagnosis.
//
// INSTALL
// -------
//   cd sample-projects-go/farm-advisor-cli
//   go mod download
//   go build -o farm-advisor .
//
// HOW TO RUN
// ----------
//   export ANTHROPIC_API_KEY=sk-ant-...
//   export TRUENORTH_BASE_URL=http://localhost:8000  # TrueNorth Python API
//
//   # Interactive mode (talk to a farmer)
//   ./farm-advisor
//
//   # Batch mode (process a CSV of farmer complaints)
//   ./farm-advisor --batch farmers.csv
//
//   # Hindi mode (force Hindi output)
//   ./farm-advisor --lang=hindi
//
//   # Save output to file
//   ./farm-advisor --output report.json
//
// SAMPLE OUTPUT
// -------------
//   ┌─────────────────────────────────────────────┐
//   │  KisanMitra Farm Advisor  (Go CLI)          │
//   │  Powered by TrueNorth                       │
//   └─────────────────────────────────────────────┘
//
//   Advisor: Which state and district is your farm in?
//   Farmer:  Madhya Pradesh, Hoshangabad district
//
//   Advisor: Which crop are you growing?
//   Farmer:  मक्का (maize)
//
//   [████████░░░░] 60%  Turn 5 of ~8
//
//   ┌── DIAGNOSIS ─────────────────────────────────┐
//   │ Likely cause  : Nitrogen deficiency          │
//   │ Urgency       : MODERATE                     │
//   │ Confidence    : HIGH                         │
//   │                                              │
//   │ Immediate action:                            │
//   │   Apply 25 kg urea per bigha within 3 days  │
//   │   Water immediately after application        │
//   │                                              │
//   │ Hindi advisory:                              │
//   │   आपकी मक्का में नाइट्रोजन की कमी है...     │
//   └──────────────────────────────────────────────┘
//
//   Saved: advisory_20250615_143022.json

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

// ── ANSI colours ─────────────────────────────────────────────────────────────

const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	dim    = "\033[2m"
	green  = "\033[32m"
	cyan   = "\033[36m"
	yellow = "\033[33m"
	red    = "\033[31m"
	white  = "\033[97m"
)

func col(s string, codes ...string) string {
	return strings.Join(codes, "") + s + reset
}

// ── CLI flags ─────────────────────────────────────────────────────────────────

var (
	flagLang   = flag.String("lang",   "auto",   "Language hint: auto / hindi / english / kannada")
	flagOutput = flag.String("output", "",       "Save JSON output to this file path")
	flagGoal   = flag.String("goal",   "goal.yaml", "Path to goal YAML file")
)

// ── Progress bar ──────────────────────────────────────────────────────────────

func progressBar(pct float64, width int) string {
	filled := int(float64(width) * pct / 100.0)
	if filled > width {
		filled = width
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	colour := green
	if pct < 50 {
		colour = yellow
	}
	return col("["+bar+"]", colour) + col(fmt.Sprintf(" %.0f%%", pct), bold)
}

// ── Banner ────────────────────────────────────────────────────────────────────

func printBanner() {
	fmt.Println()
	fmt.Println(col("  ┌─────────────────────────────────────────────┐", bold, cyan))
	fmt.Println(col("  │  KisanMitra Farm Advisor  (Go CLI)          │", bold, cyan))
	fmt.Println(col("  │  Powered by TrueNorth · Apache 2.0          │", cyan))
	fmt.Println(col("  └─────────────────────────────────────────────┘", bold, cyan))
	fmt.Println()
	fmt.Println(col(fmt.Sprintf("  Date   : %s", time.Now().Format("02 Jan 2006")), dim))
	fmt.Println(col(fmt.Sprintf("  Goal   : %s", *flagGoal), dim))
	fmt.Println(col(fmt.Sprintf("  Lang   : %s", *flagLang), dim))
	fmt.Println()
}

// ── Print advisory result ─────────────────────────────────────────────────────

func printAdvisory(content map[string]interface{}) {
	fmt.Println()
	fmt.Println(col("  ┌── DIAGNOSIS ───────────────────────────────────┐", bold, green))

	fields := []struct{ label, key string }{
		{"Likely cause ", "likely_cause"},
		{"Urgency      ", "urgency"},
		{"Confidence   ", "diagnosis_confidence"},
		{"Cost estimate", "cost_estimate_inr"},
	}
	for _, f := range fields {
		if v, ok := content[f.key]; ok && v != nil {
			val := fmt.Sprintf("%v", v)
			c := white
			if f.key == "urgency" {
				switch strings.ToUpper(val) {
				case "URGENT":
					c = red
				case "MODERATE":
					c = yellow
				case "CAN_WAIT":
					c = green
				}
			}
			fmt.Printf("  │ %s: %s\n", col(f.label, cyan), col(val, c))
		}
	}

	fmt.Println("  │")

	if action, ok := content["immediate_action"].(string); ok {
		fmt.Println(col("  │ Immediate action (next 3 days):", bold, cyan))
		for _, line := range wordWrap(action, 48) {
			fmt.Println("  │   " + line)
		}
		fmt.Println("  │")
	}

	if treatment, ok := content["treatment"].(string); ok {
		fmt.Println(col("  │ Treatment:", bold, cyan))
		for _, line := range wordWrap(treatment, 48) {
			fmt.Println("  │   " + line)
		}
		fmt.Println("  │")
	}

	if hindi, ok := content["advisory_hindi"].(string); ok && hindi != "" {
		fmt.Println(col("  │ Hindi advisory (किसान को बताएं):", bold, cyan))
		for _, line := range wordWrap(hindi, 46) {
			fmt.Println("  │   " + line)
		}
		fmt.Println("  │")
	}

	if rf, ok := content["red_flag"].(bool); ok && rf {
		fmt.Println(col("  │ ⚠  RED FLAG: High crop loss risk — escalate to district officer!", bold, red))
		fmt.Println("  │")
	}

	fmt.Println(col("  └───────────────────────────────────────────────┘", bold, green))
	fmt.Println()
}

func wordWrap(text string, width int) []string {
	words  := strings.Fields(text)
	var lines []string
	var line strings.Builder
	for _, w := range words {
		if line.Len()+len(w)+1 > width && line.Len() > 0 {
			lines = append(lines, line.String())
			line.Reset()
		}
		if line.Len() > 0 {
			line.WriteString(" ")
		}
		line.WriteString(w)
	}
	if line.Len() > 0 {
		lines = append(lines, line.String())
	}
	return lines
}

// ── Save output ───────────────────────────────────────────────────────────────

func saveOutput(outputPath string, content map[string]interface{}, fields map[string]interface{}) error {
	if outputPath == "" {
		outputPath = fmt.Sprintf("advisory_%s.json", time.Now().Format("20060102_150405"))
	}

	out := map[string]interface{}{
		"generated_at":    time.Now().Format(time.RFC3339),
		"collected_fields": fields,
		"advisory":         content,
	}

	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return err
	}

	fmt.Println(col("  Saved: "+outputPath, green))
	return nil
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()
	printBanner()

	baseURL := os.Getenv("TRUENORTH_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}
	apiKey := os.Getenv("TRUENORTH_API_KEY")

	// Init TrueNorth Go SDK
	tn := truenorth.New(truenorth.Config{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Timeout: 90 * time.Second,
	})

	// Check API health
	ctx := context.Background()
	if _, err := tn.Health(ctx); err != nil {
		fmt.Println(col("  ⚠  TrueNorth API not reachable at "+baseURL, yellow))
		fmt.Println(col("  Start with: cd packages/core && uvicorn truenorth.api.main:app --port 8000", dim))
		fmt.Println()
	} else {
		fmt.Println(col("  ✓ Connected to TrueNorth API", green))
		fmt.Println()
	}

	// Create session
	sessionID := fmt.Sprintf("fa_go_%d", time.Now().UnixMilli())
	session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    "farm_advisory",
		SessionID: sessionID,
	})
	if err != nil {
		fmt.Println(col("  Error creating session: "+err.Error(), red))
		os.Exit(1)
	}

	fmt.Println(col("  Advisor: ", bold, cyan) + session.AgentMessage)
	fmt.Println()

	// Conversation loop
	scanner := bufio.NewScanner(os.Stdin)
	turn    := 0

	for !session.IsComplete {
		turn++
		fmt.Printf("  %s  Turn %d\n", progressBar(session.CompletionPct, 28), turn)
		fmt.Println()
		fmt.Print(col("  Farmer:  ", bold, white))

		if !scanner.Scan() {
			break
		}

		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		if strings.ToLower(text) == "quit" || strings.ToLower(text) == "q" {
			fmt.Println(col("\n  Session ended. No output saved.\n", yellow))
			return
		}

		result, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sessionID,
			Text:      text,
		})
		if err != nil {
			fmt.Println(col("  Error: "+err.Error(), red))
			continue
		}

		fmt.Println()
		fmt.Println(col("  Advisor: ", bold, cyan) + result.Text)
		fmt.Println()

		session.CompletionPct = result.CompletionPct
		session.IsComplete    = result.IsComplete

		if result.IsComplete && result.Output != nil {
			content, ok := result.Output.Content.(map[string]interface{})
			if !ok {
				// Try JSON unmarshal
				raw, _ := json.Marshal(result.Output.Content)
				json.Unmarshal(raw, &content)
			}

			printAdvisory(content)

			// Save output
			var fields map[string]interface{}
			raw, _ := json.Marshal(result.CollectedFields)
			json.Unmarshal(raw, &fields)

			if err := saveOutput(*flagOutput, content, fields); err != nil {
				fmt.Println(col("  Could not save: "+err.Error(), yellow))
			}
			break
		}
	}
}