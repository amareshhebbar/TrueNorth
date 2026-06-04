package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

var (
	flagWorker = flag.String("worker", "LW-001",          "Worker ID")
	flagOutDir = flag.String("outdir", "output",          "Output directory")
	flagGoal   = flag.String("goal",   "legal_aid_intake","Goal ID")
)

const (
	reset = "\033[0m"; bold = "\033[1m"; dim = "\033[2m"
	green = "\033[32m"; cyan = "\033[36m"; yellow = "\033[33m"; red = "\033[31m"; white = "\033[97m"
)
func col(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func progressBar(pct float64, width int) string {
	filled := int(float64(width)*pct/100); if filled > width { filled = width }
	clr := green; if pct < 50 { clr = yellow }
	return col("["+strings.Repeat("█",filled)+strings.Repeat("░",width-filled)+"]", clr) + col(fmt.Sprintf(" %.0f%%", pct), bold)
}
func center(s string, width int) string {
	if len(s) >= width { return s }
	pad := (width - len(s)) / 2
	return strings.Repeat(" ", pad) + s + strings.Repeat(" ", width-pad-len(s))
}
func wordWrap(text string, width int) []string {
	words := strings.Fields(text); var lines []string; var line strings.Builder
	for _, w := range words {
		if line.Len()+len(w)+1 > width && line.Len() > 0 { lines = append(lines, line.String()); line.Reset() }
		if line.Len() > 0 { line.WriteString(" ") }; line.WriteString(w)
	}
	if line.Len() > 0 { lines = append(lines, line.String()) }; return lines
}
func truncate(s string, n int) string { if len(s) <= n { return s }; return s[:n] }

func printBanner(sessionID string) {
	W := 58
	fmt.Println(); fmt.Println(col("  ╔"+strings.Repeat("═",W-2)+"╗", bold, cyan))
	fmt.Println(col("  ║"+center("LEGAL AID INTAKE  —  TrueNorth  (Go)", W-2)+"║", bold, cyan))
	fmt.Println(col("  ║"+center("Free Legal Assistance Programme", W-2)+"║", cyan))
	fmt.Println(col("  ╚"+strings.Repeat("═",W-2)+"╝", bold, cyan))
	fmt.Println()
	fmt.Println(col(fmt.Sprintf("  Worker  : %s   Date : %s", *flagWorker, time.Now().Format("02 January 2006")), dim))
	fmt.Println(col("  Session : "+sessionID, dim)); fmt.Println()
}

func printExtracted(field, value string, confidence float64) {
	clr := green; if confidence < 0.60 { clr = red } else if confidence < 0.85 { clr = yellow }
	fmt.Println(col(fmt.Sprintf("  ✓ %-22s %-44s (%.0f%%)", field+":", truncate(value,42), confidence*100), clr, dim))
}

func printCaseBrief(content map[string]interface{}) {
	W := 58; fmt.Println(); fmt.Println(col("  ┌── CASE BRIEF "+strings.Repeat("─",W-14)+"┐", bold, green))
	pr := func(label string, val interface{}) {
		v := fmt.Sprintf("%v", val); if v == "" || v == "<nil>" { return }
		fmt.Printf("  │ %s: %s\n", col(fmt.Sprintf("%-14s", label), bold, cyan), truncate(v, 37))
	}
	strength := strings.ToUpper(fmt.Sprintf("%v", content["case_strength"]))
	pr("Case type", content["case_type_formal"]); pr("Strength", strength)
	pr("Limitation", content["limitation_period"]); pr("Est. amount", content["estimated_compensation_range"])
	if b, ok := content["eligible_for_free_aid"].(bool); ok && b { fmt.Println(col("  │ Free aid      : ✅ YES", green)) }
	fmt.Println(col("  │", dim))
	if laws, ok := content["applicable_laws"].([]interface{}); ok && len(laws) > 0 {
		fmt.Println("  │ " + col("Applicable laws:", bold, cyan))
		for _, l := range laws { fmt.Printf("  │   %s %s\n", col("•", yellow), l) }
		fmt.Println(col("  │", dim))
	}
	if steps, ok := content["immediate_steps"].([]interface{}); ok && len(steps) > 0 {
		fmt.Println("  │ " + col("Immediate steps:", bold, cyan))
		for i, s := range steps {
			lines := wordWrap(fmt.Sprintf("%v", s), 46)
			for j, line := range lines {
				if j == 0 { fmt.Printf("  │   %s %s\n", col(fmt.Sprintf("%d.", i+1), yellow), line) } else { fmt.Printf("  │      %s\n", line) }
			}
		}
	}
	fmt.Println(col("  └"+strings.Repeat("─",W-2)+"┘", bold, green)); fmt.Println()
}

func saveOutput(sessionID string, content, collected map[string]interface{}) (string, string, error) {
	os.MkdirAll(*flagOutDir, 0755)
	name := strings.ToLower(strings.ReplaceAll(fmt.Sprintf("%v", collected["client_name"]), " ", "-"))
	stem := sessionID + "_" + name
	jsonPath := filepath.Join(*flagOutDir, stem+".json")
	json.NewEncoder(func() *os.File { f, _ := os.Create(jsonPath); return f }()).Encode(map[string]interface{}{"session_id":sessionID,"worker":*flagWorker,"collected":collected,"brief":content})
	txtPath := filepath.Join(*flagOutDir, stem+".txt")
	os.WriteFile(txtPath, []byte(fmt.Sprintf("LEGAL AID INTAKE\nWorker: %s\nSession: %s\nCase: %v\nStrength: %v\n", *flagWorker, sessionID, content["case_type_formal"], content["case_strength"])), 0644)
	return jsonPath, txtPath, nil
}

func main() {
	flag.Parse()
	tnURL := os.Getenv("TRUENORTH_BASE_URL"); if tnURL == "" { tnURL = "http://localhost:8000" }
	tnKey := os.Getenv("TRUENORTH_API_KEY")
	sessionID := "la_" + time.Now().Format("20060102_150405")
	printBanner(sessionID)
	client := truenorth.NewClient(tnKey, tnURL)
	ctx := context.Background()
	if resp, err := http.Get(tnURL+"/health"); err != nil {
		fmt.Println(col("  ⚠  TrueNorth not reachable at "+tnURL, yellow))
	} else { resp.Body.Close(); fmt.Println(col("  ✓ Connected: "+tnURL, green)); fmt.Println() }
	session, err := client.Sessions.Create(ctx, *flagGoal, &truenorth.CreateSessionOptions{SessionID: sessionID})
	if err != nil { fmt.Println(col("  Error: "+err.Error(), red)); os.Exit(1) }
	fmt.Println(col("  Assistant : ", bold, cyan) + session.AgentMessage); fmt.Println()
	scanner := bufio.NewScanner(os.Stdin); turn := 0
	for {
		turn++
		fmt.Printf("\n  %s  Turn %d\n\n", progressBar(session.CompletionPct, 22), turn)
		fmt.Print(col("  Client    : ", bold, white))
		if !scanner.Scan() { break }
		text := strings.TrimSpace(scanner.Text())
		if text == "" { continue }
		if text == "quit" || text == "q" { fmt.Println(col("\n  Interrupted.\n", yellow)); return }
		result, err := client.Sessions.Message(ctx, sessionID, text)
		if err != nil { fmt.Println(col("  Error: "+err.Error(), red)); continue }
		for _, f := range result.FieldsExtracted {
			field := fmt.Sprintf("%v", f["field"])
			value := fmt.Sprintf("%v", f["value"])
			conf  := 0.0; if v, ok := f["confidence"].(float64); ok { conf = v }
			printExtracted(field, value, conf)
		}
		fmt.Println(); fmt.Println(col("  Assistant : ", bold, cyan) + result.Text); fmt.Println()
		session.CompletionPct = result.CompletionPct
		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content); json.Unmarshal(raw, &content)
			var collected map[string]interface{}
			if fs, err := client.Sessions.Get(ctx, sessionID); err == nil && fs != nil { collected = fs.CollectedFields }
			printCaseBrief(content)
			jsonPath, txtPath, _ := saveOutput(sessionID, content, collected)
			fmt.Println(col("  Files saved:", bold, green))
			fmt.Printf("    📄 %s\n    📝 %s\n\n", jsonPath, txtPath)
			break
		}
	}
	client.Sessions.End(ctx, sessionID)
}
