package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

var (
	tnURL   = func() string { if v := os.Getenv("TRUENORTH_BASE_URL"); v != "" { return v }; return "http://localhost:8000" }()
	tnKey   = os.Getenv("TRUENORTH_API_KEY")
	flagLang = flag.String("lang", "all", "all|hindi|kannada|tamil|hinglish|english|interactive")
)

const (
	bold = "\033[1m"; dim = "\033[2m"; green = "\033[32m"
	cyan = "\033[36m"; yellow = "\033[33m"; reset = "\033[0m"
)
func col(s string, codes ...string) string { return strings.Join(codes, "") + s + reset }
func div(t string) { fmt.Printf("\n%s\n\n", col("══ DEMO — "+t+" ══", bold, cyan)) }

const goalID = "multilingual_health_demo"

func estimateLang(text string) string {
	for _, r := range text {
		switch {
		case r >= 0x0900 && r <= 0x097F:
			for _, ch := range text { if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') { return "hinglish" } }
			return "hindi"
		case r >= 0x0C80 && r <= 0x0CFF: return "kannada"
		case r >= 0x0B80 && r <= 0x0BFF: return "tamil"
		case r >= 0x0C00 && r <= 0x0C7F: return "telugu"
		case r >= 0x0980 && r <= 0x09FF: return "bengali"
		}
	}
	return "english"
}

type samplePhrase struct{ text, expected string }
var samplePhrases = []samplePhrase{
	{"नमस्ते, मेरा नाम राहुल है", "Hindi"},
	{"ನನ್ನ ಹೆಸರು ರವಿ", "Kannada"},
	{"என் பெயர் ப்ரியா", "Tamil"},
	{"Hello, my name is Sarah", "English"},
	{"Mera weight 65 kg hai", "Hinglish"},
}

func showDetectionTable() {
	fmt.Printf("\n%s\n\n", col("LANGUAGE DETECTION — Sample phrases", bold, cyan))
	fmt.Printf("  %-40s %-12s %-12s\n", "Phrase", "Expected", "Detected")
	fmt.Println("  " + strings.Repeat("─", 66))
	for _, p := range samplePhrases {
		detected := estimateLang(p.text)
		tick := col("✅", green)
		if !strings.EqualFold(detected, p.expected) { tick = col("⚠️", yellow) }
		runes := []rune(p.text)
		phrase := string(runes)
		if len(runes) > 36 { phrase = string(runes[:36]) + "…" }
		fmt.Printf("  %-40s %-12s %-12s %s\n", phrase, p.expected, detected, tick)
	}
	fmt.Println()
}

type langDemo struct{ label string; turns []string }
var demos = map[string]langDemo{
	"hindi":    {"HINDI (हिंदी)", []string{"नमस्ते","मेरा नाम राहुल शर्मा है","28 साल","72 किलो","175 सेंटीमीटर","वजन कम करना है","मुंबई"}},
	"kannada":  {"KANNADA (ಕನ್ನಡ)", []string{"ನಮಸ್ಕಾರ","ರವಿ ಕುಮಾರ್","32 ವರ್ಷ","80 ಕಿಲೋ","170 ಸೆಂ.ಮೀ","ಫಿಟ್ ಆಗಬೇಕು","ಬೆಂಗಳೂರು"}},
	"tamil":    {"TAMIL (தமிழ்)", []string{"வணக்கம்","ப்ரியா","25","58 கிலோ","162 செமீ","எடை குறைக்கணும்","சென்னை"}},
	"hinglish": {"HINGLISH (Mixed)", []string{"hello bhai","Mera naam Arjun hai","30 years old","85 kg hai","180 cm","muscle build karna","Pune"}},
	"english":  {"ENGLISH", []string{"Hello","Sarah","26","55 kilograms","165 centimetres","improve fitness","Hyderabad"}},
}

func runDemo(client *truenorth.TrueNorth, lang string, demo langDemo) {
	div(demo.label)
	ctx := context.Background()
	sid := fmt.Sprintf("ml_%s_%d", lang, time.Now().UnixMilli())
	session, err := client.Sessions.Create(ctx, goalID, &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil { fmt.Printf("  Error: %v\n  (showing estimated output)\n", err); for _, t := range demo.turns { fmt.Printf("  User:  %s\n  Agent: (response in %s)\n\n", t, lang) }; return }
	fmt.Printf("  Agent: %s\n\n", session.AgentMessage)
	for _, turn := range demo.turns {
		fmt.Printf("  User:  %s\n", turn)
		res, err := client.Sessions.Message(ctx, sid, turn)
		if err != nil { fmt.Printf("  Error: %v\n", err); break }
		detected := estimateLang(turn)
		fmt.Printf("  Agent [%s]: %s\n\n", col(detected, cyan), res.Text)
		if res.IsComplete { fmt.Println(col("  ✅ Language: "+strings.ToUpper(detected), green, bold)); break }
	}
	client.Sessions.End(ctx, sid)
}

func runInteractive(client *truenorth.TrueNorth) {
	fmt.Printf("\n%s\n\n", col("INTERACTIVE — Type in any Indian language", bold, cyan))
	fmt.Println("  Type in Hindi, Kannada, Tamil, or English. Type 'quit' to exit.\n")
	ctx := context.Background()
	sid := fmt.Sprintf("ml_interactive_%d", time.Now().UnixMilli())
	session, err := client.Sessions.Create(ctx, goalID, &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil { fmt.Printf("  Error: %v\n", err); return }
	fmt.Printf("  Agent: %s\n\n", session.AgentMessage)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("  You: ")
		if !scanner.Scan() { break }
		text := strings.TrimSpace(scanner.Text())
		if text == "" { continue }
		if text == "quit" || text == "q" { break }
		res, err := client.Sessions.Message(ctx, sid, text)
		if err != nil { fmt.Printf("  Error: %v\n", err); continue }
		lang := estimateLang(text)
		fmt.Printf("\n  Agent [%s]: %s\n\n", col(lang, cyan), res.Text)
		if res.IsComplete { break }
	}
	client.Sessions.End(ctx, sid)
}

func checkAPI() bool {
	resp, err := http.Get(tnURL + "/health")
	if err != nil { return false }
	resp.Body.Close(); return true
}

func main() {
	flag.Parse()
	fmt.Println()
	fmt.Println(col("  TrueNorth Multilingual Demo (Go)", bold, cyan))
	fmt.Println(col("  Lang: "+*flagLang, dim))
	showDetectionTable()

	client := truenorth.NewClient(tnKey, tnURL)
	if !checkAPI() { fmt.Println(col("  ⚠  TrueNorth not reachable — showing estimated output", yellow)) }

	if *flagLang == "interactive" { runInteractive(client); return }
	order := []string{"hindi","kannada","tamil","hinglish","english"}
	for _, lang := range order {
		if *flagLang != "all" && *flagLang != lang { continue }
		if demo, ok := demos[lang]; ok { runDemo(client, lang, demo) }
	}
}
