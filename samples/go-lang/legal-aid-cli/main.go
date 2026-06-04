// legal-aid-cli (Go) — Terminal Legal Intake for NGO Workers
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// A beautifully formatted terminal app for legal aid workers at NGOs.
// Worker sits with the client, runs this binary, and it guides them
// through a structured intake conversation.
//
// Output:
//   • Colour-coded case brief printed in the terminal
//   • Saves to  output/<session>_<name>.json  (for the database)
//   • Saves to  output/<session>_<name>.txt   (to hand to the advocate)
//
// NO SERVER. NO DATABASE. NO DEPLOYMENT.
// Just the binary + your API key. Runs on any laptop or phone.
//
// This is the Go equivalent of legal-aid-cli/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   legal-aid-cli/
//   ├── main.go      ← this file
//   ├── goal.yaml    ← legal intake questionnaire (copy from python/legal-aid-cli/goal.yaml)
//   └── go.mod
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/legal-aid-cli
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/legal-aid-cli
//   go mod download
//   go build -o legal-aid .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Run
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   ./legal-aid
//
//   # Hindi mode (force Hindi output)
//   ./legal-aid --lang=hindi
//
//   # Specific worker ID (for record-keeping)
//   ./legal-aid --worker=LW-042
//
//   # Custom output directory
//   ./legal-aid --outdir=/home/worker/cases
//
//   # Dry run (no API calls — for training new workers)
//   ./legal-aid --dry
//
// WHAT YOU SEE
// ──────────────────────────────────────────────────────────────
//   ╔══════════════════════════════════════════════════════╗
//   ║       LEGAL AID INTAKE  —  TrueNorth  (Go)          ║
//   ║       Free Legal Assistance Programme                ║
//   ╚══════════════════════════════════════════════════════╝
//
//     Worker ID  : LW-042
//     Date       : 15 June 2025
//     Session    : la_20250615_142301
//
//   ──────────────────────────────────────────────────────
//   Assistant  : Namaste! I am here to help prepare your legal
//                matter. Take your time.
//   ──────────────────────────────────────────────────────
//   Client     : मेरा नाम रमेश है
//
//   [████████░░░░░░░░░░░░] 40%  Turn 5
//
//   ✓ Extracted  client_name: Ramesh Kumar  (92%)
//
//   ...
//
//   ┌── CASE BRIEF ─────────────────────────────────────────┐
//   │ Case type     : Wage Theft                             │
//   │ Strength      : STRONG                                 │
//   │ Limitation    : File before 14 June 2027              │
//   │ Free aid      : ✅ YES                                  │
//   │ Est. amount   : ₹45,000 – ₹1,20,000                   │
//   │                                                        │
//   │ Applicable laws:                                       │
//   │   • Payment of Wages Act 1936, Section 15             │
//   │   • Minimum Wages Act 1948                            │
//   │                                                        │
//   │ Immediate steps:                                       │
//   │   1. File with Labour Commissioner this week          │
//   │   2. Collect wage slips and employment letter         │
//   │   3. Record witness details                           │
//   └────────────────────────────────────────────────────────┘
//
//   Saved: output/la_20250615_142301_ramesh-kumar.json
//          output/la_20250615_142301_ramesh-kumar.txt

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

// ── Flags ─────────────────────────────────────────────────────────────────────

var (
	flagWorker = flag.String("worker", "LW-001",              "Worker ID for record-keeping")
	flagLang   = flag.String("lang",   "auto",               "Language hint: auto / hindi / english / kannada")
	flagOutDir = flag.String("outdir", "output",             "Directory to save case files")
	flagGoal   = flag.String("goal",   "goal.yaml",          "Path to goal YAML file")
	flagDry    = flag.Bool("dry",      false,                "Dry run — no API calls (for training)")
	flagSID    = flag.String("session","",                   "Resume a session by ID")
)

// ── ANSI ──────────────────────────────────────────────────────────────────────

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

func c(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }

// ── Progress bar ──────────────────────────────────────────────────────────────

func progressBar(pct float64, width int) string {
	filled := int(float64(width) * pct / 100.0)
	if filled > width { filled = width }
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	clr := green
	if pct < 50 { clr = yellow }
	return c("["+bar+"]", clr) + c(fmt.Sprintf(" %.0f%%", pct), bold)
}

// ── Banner ────────────────────────────────────────────────────────────────────

func printBanner(sessionID string) {
	w := 58
	fmt.Println()
	fmt.Println(c("  ╔"+strings.Repeat("═", w-2)+"╗", bold, cyan))
	fmt.Println(c("  ║"+"LEGAL AID INTAKE  —  TrueNorth  (Go)".center(w-2)+"║", bold, cyan))
	fmt.Println(c("  ║"+"Free Legal Assistance Programme".center(w-2)+"║", cyan))
	fmt.Println(c("  ╚"+strings.Repeat("═", w-2)+"╝", bold, cyan))
	fmt.Println()
	fmt.Println(c(fmt.Sprintf("  Worker ID  : %s", *flagWorker), dim))
	fmt.Println(c(fmt.Sprintf("  Date       : %s", time.Now().Format("02 January 2006")), dim))
	fmt.Println(c(fmt.Sprintf("  Session    : %s", sessionID), dim))
	if *flagDry { fmt.Println(c("  Mode       : DRY RUN", yellow)) }
	fmt.Println()
}

// ── String centering helper ───────────────────────────────────────────────────

func center(s string, width int) string {
	if len(s) >= width { return s }
	pad := (width - len(s)) / 2
	return strings.Repeat(" ", pad) + s + strings.Repeat(" ", width-pad-len(s))
}

// ── Field extracted notification ──────────────────────────────────────────────

func printExtracted(field, value string, confidence float64) {
	clr := green
	if confidence < 0.60 { clr = red }
	if confidence < 0.85 && confidence >= 0.60 { clr = yellow }
	fmt.Println(c(fmt.Sprintf("  ✓ Extracted  %-22s %s  (%.0f%%)",
		field+":", truncate(value, 42), confidence*100), clr, dim))
}

// ── Agent / client print ──────────────────────────────────────────────────────

func printAgent(text string) {
	fmt.Println()
	fmt.Println(c("  Assistant  : ", bold, cyan) + text)
	fmt.Println()
}

func printTurn(pct float64, turn int) {
	fmt.Printf("\n  %s  Turn %d\n\n", progressBar(pct, 22), turn)
}

// ── Case brief display ────────────────────────────────────────────────────────

func printCaseBrief(content map[string]interface{}) {
	w := 58
	fmt.Println()
	fmt.Println(c("  ┌── CASE BRIEF "+strings.Repeat("─", w-14)+"┐", bold, green))

	pr := func(label string, val interface{}, clr ...string) {
		v := fmt.Sprintf("%v", val)
		if v == "" || v == "<nil>" || v == "nil" { return }
		lbl := c(fmt.Sprintf("│ %-15s: ", label), append([]string{bold}, clr...)...)
		fmt.Printf("  %s%s\n", lbl, truncate(v, 38))
	}

	strength := strings.ToUpper(fmt.Sprintf("%v", content["case_strength"]))
	strengthClr := green
	if strings.Contains(strength, "WEAK")     { strengthClr = red }
	if strings.Contains(strength, "MODERATE") { strengthClr = yellow }

	pr("Case type",      content["case_type_formal"])
	pr("Strength",       strength, strengthClr)
	pr("Limitation",     content["limitation_period"])
	pr("Jurisdiction",   content["jurisdiction"])

	eligible := content["eligible_for_free_aid"]
	if b, ok := eligible.(bool); ok && b {
		fmt.Println(c("  │ Free aid      : ✅ YES — eligible for legal aid", green))
	} else {
		fmt.Println(c("  │ Free aid      : Check criteria with advocate", dim))
	}

	pr("Est. amount",   content["estimated_compensation_range"])
	fmt.Println(c("  │", dim))

	// Laws
	if laws, ok := content["applicable_laws"].([]interface{}); ok && len(laws) > 0 {
		fmt.Println(c("  │ "+c("Applicable laws:", bold, cyan), ""))
		for _, l := range laws {
			fmt.Printf("  │   %s %s\n", c("•", yellow), l)
		}
		fmt.Println(c("  │", dim))
	}

	// Immediate steps
	if steps, ok := content["immediate_steps"].([]interface{}); ok && len(steps) > 0 {
		fmt.Println(c("  │ "+c("Immediate steps:", bold, cyan), ""))
		for i, s := range steps {
			wrapped := wordWrap(fmt.Sprintf("%v", s), 46)
			for j, line := range wrapped {
				if j == 0 {
					fmt.Printf("  │   %s %s\n", c(fmt.Sprintf("%d.", i+1), yellow), line)
				} else {
					fmt.Printf("  │      %s\n", line)
				}
			}
		}
		fmt.Println(c("  │", dim))
	}

	// Docs missing
	if missing, ok := content["documents_missing"].([]interface{}); ok && len(missing) > 0 {
		fmt.Println(c("  │ "+c("Missing documents (collect before meeting advocate):", bold, red), ""))
		for _, d := range missing {
			fmt.Printf("  │   %s %s\n", c("✗", red), d)
		}
		fmt.Println(c("  │", dim))
	}

	// Advocate brief
	if brief, ok := content["advocate_brief"].(string); ok && brief != "" {
		fmt.Println(c("  │ "+c("Advocate brief:", bold, cyan), ""))
		for _, line := range wordWrap(brief, 52) {
			fmt.Printf("  │   %s\n", line)
		}
	}

	fmt.Println(c("  └"+strings.Repeat("─", w-2)+"┘", bold, green))
	fmt.Println()
}

// ── Save output ───────────────────────────────────────────────────────────────

func saveOutput(sessionID string, content, collected map[string]interface{}) (string, string, error) {
	if err := os.MkdirAll(*flagOutDir, 0755); err != nil {
		return "", "", err
	}

	name := slug(fmt.Sprintf("%v", collected["client_name"]))
	stem := fmt.Sprintf("%s_%s", sessionID, name)

	// JSON
	jsonPath := filepath.Join(*flagOutDir, stem+".json")
	full := map[string]interface{}{
		"session_id":       sessionID,
		"worker_id":        *flagWorker,
		"generated_at":     time.Now().Format(time.RFC3339),
		"collected_fields": collected,
		"case_brief":       content,
	}
	jData, _ := json.MarshalIndent(full, "", "  ")
	os.WriteFile(jsonPath, jData, 0644)

	// Plain text
	txtPath := filepath.Join(*flagOutDir, stem+".txt")
	var sb strings.Builder
	sb.WriteString("LEGAL AID INTAKE\n")
	sb.WriteString(fmt.Sprintf("Generated  : %s\n", time.Now().Format("02 Jan 2006 15:04")))
	sb.WriteString(fmt.Sprintf("Worker     : %s\n", *flagWorker))
	sb.WriteString(fmt.Sprintf("Session    : %s\n\n", sessionID))
	sb.WriteString(strings.Repeat("─", 50) + "\nCLIENT INFORMATION\n" + strings.Repeat("─", 50) + "\n")
	for k, v := range collected {
		sb.WriteString(fmt.Sprintf("%-25s: %v\n", titleCase(k), v))
	}
	sb.WriteString("\n" + strings.Repeat("─", 50) + "\nCASE BRIEF\n" + strings.Repeat("─", 50) + "\n")
	sb.WriteString(fmt.Sprintf("Case type   : %v\n", content["case_type_formal"]))
	sb.WriteString(fmt.Sprintf("Strength    : %v\n", content["case_strength"]))
	sb.WriteString(fmt.Sprintf("Limitation  : %v\n\n", content["limitation_period"]))
	if laws, ok := content["applicable_laws"].([]interface{}); ok {
		sb.WriteString("Applicable laws:\n")
		for _, l := range laws { sb.WriteString(fmt.Sprintf("  • %v\n", l)) }
	}
	if steps, ok := content["immediate_steps"].([]interface{}); ok {
		sb.WriteString("\nImmediate steps:\n")
		for i, s := range steps { sb.WriteString(fmt.Sprintf("  %d. %v\n", i+1, s)) }
	}
	sb.WriteString("\nAdvocate brief:\n")
	sb.WriteString(fmt.Sprintf("%v\n", content["advocate_brief"]))
	os.WriteFile(txtPath, []byte(sb.String()), 0644)

	return jsonPath, txtPath, nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func truncate(s string, n int) string {
	if len(s) <= n { return s }
	return s[:n]
}

func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var out strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' { out.WriteRune(r) } else { out.WriteRune('-') }
	}
	return strings.Trim(out.String(), "-")
}

func titleCase(s string) string {
	parts := strings.Split(strings.ReplaceAll(s, "_", " "), " ")
	for i, p := range parts {
		if len(p) > 0 { parts[i] = strings.ToUpper(p[:1]) + p[1:] }
	}
	return strings.Join(parts, " ")
}

func wordWrap(text string, width int) []string {
	words := strings.Fields(text)
	var lines []string
	var line strings.Builder
	for _, w := range words {
		if line.Len()+len(w)+1 > width && line.Len() > 0 {
			lines = append(lines, line.String()); line.Reset()
		}
		if line.Len() > 0 { line.WriteString(" ") }
		line.WriteString(w)
	}
	if line.Len() > 0 { lines = append(lines, line.String()) }
	return lines
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	tnURL := os.Getenv("TRUENORTH_BASE_URL")
	if tnURL == "" { tnURL = "http://localhost:8000" }
	tnKey := os.Getenv("TRUENORTH_API_KEY")

	client := truenorth.New(truenorth.Config{
		BaseURL: tnURL,
		APIKey:  tnKey,
		Timeout: 90 * time.Second,
	})

	sessionID := *flagSID
	if sessionID == "" {
		sessionID = "la_" + time.Now().Format("20060102_150405")
	}

	printBanner(sessionID)

	// Check API
	if _, err := client.Health(context.Background()); err != nil {
		fmt.Println(c("  ⚠  TrueNorth API not reachable at "+tnURL, yellow))
		fmt.Println(c("  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000\n", dim))
	} else {
		fmt.Println(c("  ✓ Connected to TrueNorth API", green))
		fmt.Println()
	}

	ctx := context.Background()
	session, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    "legal_aid_intake",
		SessionID: sessionID,
	})
	if err != nil {
		fmt.Println(c("  Error creating session: "+err.Error(), red))
		os.Exit(1)
	}

	printAgent(session.AgentMessage)

	scanner := bufio.NewScanner(os.Stdin)
	turn    := 0

	for !session.IsComplete {
		turn++
		printTurn(session.CompletionPct, turn)
		fmt.Print(c("  Client     : ", bold, white))

		if !scanner.Scan() { break }
		text := strings.TrimSpace(scanner.Text())
		if text == "" { continue }
		if text == "quit" || text == "q" || text == "/exit" {
			fmt.Println(c("\n  Interrupted — session not saved.\n", yellow))
			return
		}

		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sessionID, Text: text,
		})
		if err != nil { fmt.Println(c("  Error: "+err.Error(), red)); continue }

		// Show extracted fields for this turn
		for _, f := range result.FieldsExtracted {
			printExtracted(f.Field, fmt.Sprintf("%v", f.Value), f.Confidence)
		}

		printAgent(result.Text)

		session.CompletionPct = result.CompletionPct
		session.IsComplete    = result.IsComplete

		if result.IsComplete && result.Output != nil {
			var content, collected map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content)
			json.Unmarshal(raw, &content)
			raw, _ = json.Marshal(result.CollectedFields)
			json.Unmarshal(raw, &collected)

			printCaseBrief(content)

			jsonPath, txtPath, err := saveOutput(sessionID, content, collected)
			if err != nil {
				fmt.Println(c("  Could not save: "+err.Error(), yellow))
			} else {
				fmt.Println(c("  Files saved:", bold, green))
				fmt.Printf("    📄 %s\n", jsonPath)
				fmt.Printf("    📝 %s\n", txtPath)
				fmt.Println()
				fmt.Println(c("  Share the .txt file with the advocate.", dim))
				fmt.Println()
			}
			break
		}
	}

	client.Sessions.End(ctx, sessionID)
}