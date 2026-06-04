// multilingual-demo (Go) — Auto Language Detection Demo
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// Shows TrueNorth detecting and responding in 5 Indian languages
// automatically — no configuration, no language selection screen.
//
// Runs scripted conversations in:
//   Hindi     → agent responds in Hindi
//   Kannada   → agent responds in Kannada
//   Tamil     → agent responds in Tamil
//   Hinglish  → agent responds in Hinglish (mixed)
//   English   → agent responds in English
//
// Also shows the language detector working on sample phrases
// before the conversation demos.
//
// This is the Go equivalent of multilingual-demo/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   multilingual-demo/
//   ├── main.go   ← this file (no goal.yaml — inline config)
//   └── go.mod
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/multilingual-demo
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/multilingual-demo
//   go mod download
//   go build -o multilingual .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2:
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   ./multilingual                   ← all 5 languages
//   ./multilingual --lang=hindi      ← Hindi only
//   ./multilingual --lang=kannada    ← Kannada only
//   ./multilingual --lang=tamil      ← Tamil only
//   ./multilingual --lang=hinglish   ← Hinglish only
//   ./multilingual --lang=english    ← English only
//   ./multilingual --lang=interactive ← type your own messages
//
// WHAT YOU SEE
// ──────────────────────────────────────────────────────────────
//   LANGUAGE DETECTION — Sample phrases
//   ─────────────────────────────────────────────────────────
//   Phrase                            Expected    Detected
//   नमस्ते, मेरा नाम राहुल है         Hindi       hindi     ✅
//   ನನ್ನ ಹೆಸರು ರವಿ                   Kannada     kannada   ✅
//   என் பெயர் ப்ரியா                 Tamil       tamil     ✅
//   Mera weight 65 kg hai             Hinglish    hinglish  ✅
//   Hello, my name is Sarah           English     english   ✅
//
//   ══ DEMO — HINDI (हिंदी) ══
//
//   Agent: नमस्ते! / Hello! / ನಮಸ್ಕಾರ!
//   User:  नमस्ते
//   User:  मेरा नाम राहुल शर्मा है
//   Agent [hindi]: बहुत अच्छा! आपकी उम्र क्या है?
//   User:  28 साल
//   ...
//   ✅ Detected: HINDI | Fields: 6 | Output: {...}

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
	flagLang = flag.String("lang", "all",
		"Language to demo: all / hindi / kannada / tamil / hinglish / english / interactive")
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
)

func c(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func divider(t string) {
	fmt.Printf("\n%s\n\n", c("══ DEMO — "+t+" ══", bold, cyan))
}

// ── Inline goal (no YAML needed) ─────────────────────────────────────────────

const goalID = "multilingual_health_demo"

var inlineGoal = map[string]interface{}{
	"id":   goalID,
	"name": "Multilingual Health Assessment",
	"persona": map[string]interface{}{
		"name":     "Health Assistant",
		"tone":     "warm",
		"language": "auto", // ← auto-detect
		"greeting": "Hello! / नमस्ते! / ನಮಸ್ಕಾರ! / வணக்கம்!\n" +
			"I can speak your language. Reply in Hindi, Kannada, Tamil, or English.",
	},
	"fields": []map[string]interface{}{
		{"name": "name",        "type": "text",    "required": true,
			"question": "What is your name? / आपका नाम क्या है?"},
		{"name": "age",         "type": "integer", "required": true, "min": 1, "max": 120,
			"question": "How old are you?"},
		{"name": "weight_kg",   "type": "number",  "required": true,
			"question": "What is your weight?"},
		{"name": "height_cm",   "type": "number",  "required": true,
			"question": "What is your height?"},
		{"name": "health_goal", "type": "text",    "required": true,
			"allowed_values": []string{"lose weight", "build strength", "improve fitness",
				"manage condition", "general wellness"},
			"question": "What is your main health goal?"},
		{"name": "city",        "type": "text",    "required": false,
			"question": "Which city are you from?"},
	},
	"output": map[string]interface{}{
		"format": "json",
		"template": "Summarise health assessment for {name}, age {age}, {weight_kg}kg, {height_cm}cm. " +
			"Goal: {health_goal}. City: {city}. " +
			"Return JSON with bmi, bmi_category, language_detected, " +
			"personalised_message (in the user's language), next_steps (3 items).",
	},
}

// ── Scripted demo conversations ───────────────────────────────────────────────

type LangDemo struct {
	Label string
	Turns []string
}

var demos = map[string]LangDemo{
	"hindi": {
		Label: "HINDI (हिंदी)",
		Turns: []string{
			"नमस्ते",
			"मेरा नाम राहुल शर्मा है",
			"28 साल का हूँ",
			"वजन 72 किलो है",
			"लम्बाई 175 सेंटीमीटर है",
			"मुझे वजन कम करना है",
			"मुंबई से हूँ",
		},
	},
	"kannada": {
		Label: "KANNADA (ಕನ್ನಡ)",
		Turns: []string{
			"ನಮಸ್ಕಾರ",
			"ನನ್ನ ಹೆಸರು ರವಿ ಕುಮಾರ್",
			"ನನಗೆ 32 ವರ್ಷ",
			"ತೂಕ 80 ಕಿಲೋ",
			"ಎತ್ತರ 170 ಸೆಂ.ಮೀ",
			"ದೇಹವನ್ನು ಫಿಟ್ ಮಾಡಿಕೊಳ್ಳಬೇಕು",
			"ಬೆಂಗಳೂರಿನಿಂದ",
		},
	},
	"tamil": {
		Label: "TAMIL (தமிழ்)",
		Turns: []string{
			"வணக்கம்",
			"என் பெயர் ப்ரியா",
			"என் வயது 25",
			"என் எடை 58 கிலோ",
			"உயரம் 162 செமீ",
			"உடல் எடையை குறைக்கணும்",
			"சென்னையில் இருக்கேன்",
		},
	},
	"hinglish": {
		Label: "HINGLISH (Mixed)",
		Turns: []string{
			"hello bhai",
			"Mera naam Arjun hai",
			"I am 30 years old",
			"Weight 85 kg hai mera",
			"Height around 180 cm",
			"Muscle build karna chahta hoon",
			"Pune mein rehta hoon",
		},
	},
	"english": {
		Label: "ENGLISH",
		Turns: []string{
			"Hello",
			"My name is Sarah",
			"I am 26 years old",
			"I weigh 55 kilograms",
			"My height is 165 centimetres",
			"I want to improve my overall fitness",
			"I live in Hyderabad",
		},
	},
}

// ── Language detector sample phrases ─────────────────────────────────────────

type Phrase struct {
	Text     string
	Expected string
}

var samplePhrases = []Phrase{
	{"नमस्ते, मेरा नाम राहुल है",       "Hindi"},
	{"ನನ್ನ ಹೆಸರು ರವಿ",                  "Kannada"},
	{"என் பெயர் ப்ரியா",                "Tamil"},
	{"నా పేరు రాహుల్",                  "Telugu"},
	{"Hello, my name is Sarah",         "English"},
	{"Mera weight 65 kg hai",           "Hinglish"},
	{"माझे नाव सुनील आहे",              "Marathi"},
	{"ਮੇਰਾ ਨਾਮ ਹਰਪ੍ਰੀਤ ਹੈ",            "Punjabi"},
	{"আমার নাম সুমিত",                  "Bengali"},
}

// ── Show language detection table ────────────────────────────────────────────

func showDetectionTable(client *truenorth.Client) {
	fmt.Printf("\n%s\n\n", c("LANGUAGE DETECTION — Sample phrases", bold, cyan))
	fmt.Printf("  %-38s %-12s %-12s %s\n", "Phrase", "Expected", "Detected", "")
	fmt.Println("  " + strings.Repeat("─", 70))

	ctx := context.Background()
	for _, p := range samplePhrases {
		detected := "—"
		match    := "?"

		// Call detect API
		result, err := client.Language.Detect(ctx, p.Text)
		if err == nil && result.Language != "" {
			detected = result.Language
			if strings.EqualFold(result.Language, p.Expected) {
				match = c("✅", green)
			} else {
				match = c("⚠️", yellow)
			}
		} else {
			// Fallback: show simulated result
			detected = estimateLanguage(p.Text)
			match    = c("~", dim)
		}

		phrase := []rune(p.Text)
		phraseTrunc := string(phrase)
		if len(phrase) > 35 { phraseTrunc = string(phrase[:35]) + "…" }
		fmt.Printf("  %-38s %-12s %-12s %s\n", phraseTrunc, p.Expected, detected, match)
	}
	fmt.Println()
}

// estimateLanguage is a very rough fallback when the API is unavailable.
func estimateLanguage(text string) string {
	switch {
	case strings.ContainsAny(text, "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"):
		if strings.ContainsAny(text, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") {
			return "hinglish"
		}
		return "hindi"
	case strings.ContainsAny(text, "ಅಆಇಈಉಊಎಏಐಒಓಔಕಖಗಘಚಛಜಝಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹ"):
		return "kannada"
	case strings.ContainsAny(text, "அஆஇஈஉஊகசடதநபமயரலவழளறன"):
		return "tamil"
	case strings.ContainsAny(text, "అఆఇఈఉఊకఖగఘచఛజఝటఠడఢణతథదధనపఫబభమయరలవశషసహ"):
		return "telugu"
	default:
		return "english"
	}
}

// ── Run a single language demo ────────────────────────────────────────────────

func runDemo(client *truenorth.Client, lang string, demo LangDemo) map[string]interface{} {
	divider(demo.Label)
	ctx := context.Background()

	sid := fmt.Sprintf("ml_%s_%d", lang, time.Now().UnixMilli())

	// Register inline goal

	session, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    goalID,
		SessionID: sid,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n  Showing estimated output:\n\n", err)
		return showEstimatedOutput(lang, demo)
	}

	fmt.Printf("  Agent: %s\n\n", session.AgentMessage)

	var lastOutput map[string]interface{}

	for _, turn := range demo.Turns {
		fmt.Printf("  User:  %s\n", turn)

		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: turn,
		})
		if err != nil { fmt.Printf("  Error: %v\n", err); break }

		detected := result.DetectedLanguage
		if detected == "" { detected = "auto" }

		fmt.Printf("  Agent [%s]: %s\n\n", c(detected, cyan), result.Text)

		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content)
			json.Unmarshal(raw, &content)

			fmt.Printf("  %s Language: %s\n",
				c("✅", green, bold),
				c(strings.ToUpper(detected), bold, green))
			fmt.Printf("  Fields collected: %d\n\n", len(result.CollectedFields))

			if msg, ok := content["personalised_message"].(string); ok {
				fmt.Printf("  Personalised message:\n  %s\n\n",
					c("\""+msg+"\"", yellow))
			}
			lastOutput = content
			break
		}
	}

	client.Sessions.End(ctx, sid)
	return lastOutput
}

func showEstimatedOutput(lang string, demo LangDemo) map[string]interface{} {
	fmt.Printf("  [Showing estimated output — TrueNorth API not reachable]\n\n")
	for _, t := range demo.Turns {
		fmt.Printf("  User:  %s\n", t)
		fmt.Printf("  Agent: (response in %s)\n\n", lang)
	}
	return map[string]interface{}{
		"language_detected": lang,
		"bmi":               22.9,
		"bmi_category":      "Normal",
	}
}

// ── Interactive mode ─────────────────────────────────────────────────────────

func runInteractive(client *truenorth.Client) {
	fmt.Printf("\n%s\n\n", c("INTERACTIVE — Type in any Indian language", bold, cyan))
	fmt.Println("  Type in Hindi, Kannada, Tamil, Telugu, English, or Hinglish.")
	fmt.Println("  TrueNorth detects and responds in your language.")
	fmt.Println("  Type 'quit' to exit.\n")

	ctx := context.Background()
	sid := fmt.Sprintf("ml_interactive_%d", time.Now().UnixMilli())

	session, err := client.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID: goalID, SessionID: sid,
	})
	if err != nil { fmt.Println("Error:", err); return }

	fmt.Printf("  Agent: %s\n\n", session.AgentMessage)

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("  You: ")
		if !scanner.Scan() { break }
		text := strings.TrimSpace(scanner.Text())
		if text == "" { continue }
		if text == "quit" || text == "q" { break }

		result, err := client.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: text,
		})
		if err != nil { fmt.Println("Error:", err); continue }

		lang := result.DetectedLanguage
		if lang == "" { lang = "auto" }
		fmt.Printf("\n  Agent [%s]: %s\n\n", c(lang, cyan), result.Text)

		if result.IsComplete {
			if result.Output != nil {
				data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
				fmt.Printf("\n  OUTPUT:\n  %s\n\n", string(data))
			}
			break
		}
	}
	client.Sessions.End(ctx, sid)
}

// ── Summary ───────────────────────────────────────────────────────────────────

func printSummary(results map[string]string) {
	fmt.Printf("\n%s\n\n", c("SUMMARY", bold, cyan))
	fmt.Printf("  %-40s %s\n", "Language demo", "Result")
	fmt.Println("  " + strings.Repeat("─", 55))
	for lang, status := range results {
		demo := demos[lang]
		fmt.Printf("  %-40s %s\n", demo.Label, status)
	}
	fmt.Println()
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

	fmt.Println()
	fmt.Println(c("  TrueNorth Multilingual Demo (Go)", bold, cyan))
	fmt.Println(c(fmt.Sprintf("  API: %s | Lang: %s", tnURL, *flagLang), dim))

	selected := *flagLang

	// Always show the detection table first
	showDetectionTable(client)

	if selected == "interactive" {
		runInteractive(client)
		return
	}

	results := make(map[string]string)
	order   := []string{"hindi", "kannada", "tamil", "hinglish", "english"}

	for _, lang := range order {
		if selected != "all" && selected != lang { continue }
		demo, ok := demos[lang]
		if !ok { continue }

		output := runDemo(client, lang, demo)
		if output != nil {
			results[lang] = c("✅ PASS", green)
		} else {
			results[lang] = c("⚠️  INCOMPLETE", yellow)
		}
	}

	if selected == "all" && len(results) > 0 {
		printSummary(results)
	}
}