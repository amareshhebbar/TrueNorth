package main

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
	"github.com/gin-gonic/gin"
)

var (
	clinicName  = envOr("CLINIC_NAME",         "Our Clinic")
	goalID      = envOr("GOAL_ID",             "patient_intake")
	verifyToken = envOr("WA_VERIFY_TOKEN",      "cliniqflow-token")
	accessToken = envOr("WA_ACCESS_TOKEN",      "")
	phoneID     = envOr("WA_PHONE_NUMBER_ID",   "")
	tnURL       = envOr("TRUENORTH_BASE_URL",   "http://localhost:8000")
	tnKey       = envOr("TRUENORTH_API_KEY",    "")
	port        = envOr("PORT",                "8080")
	consoleMode bool
)

func envOr(k, def string) string { if v := os.Getenv(k); v != "" { return v }; return def }

var tn *truenorth.TrueNorth

type PatientSession struct {
	SessionID  string
	Phone      string
	StartedAt  time.Time
	LastActive time.Time
	Consented  bool
	TurnCount  int
	Pct        float64
}

type CompletedIntake struct {
	SessionID   string                 `json:"session_id"`
	Phone       string                 `json:"phone"`
	CompletedAt string                 `json:"completed_at"`
	IntakeData  map[string]interface{} `json:"intake"`
}

var (
	sessionMu sync.RWMutex
	sessions  = map[string]*PatientSession{}
	completed []CompletedIntake
)

func sid(phone string) string {
	h := md5.Sum([]byte(phone)); return fmt.Sprintf("cf_%x", h[:6])
}

func sendWA(phone, text string) {
	if accessToken == "" || phoneID == "" {
		fmt.Printf("\n📤 [%s]: %s\n\n", phone, text[:min(200, len(text))])
		return
	}
	url     := fmt.Sprintf("https://graph.facebook.com/v19.0/%s/messages", phoneID)
	body, _ := json.Marshal(map[string]interface{}{"messaging_product":"whatsapp","to":phone,"type":"text","text":map[string]string{"body":text[:min(4000,len(text))]}})
	req, _  := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil { log.Printf("WA error: %v", err); return }
	defer resp.Body.Close()
}

func min(a, b int) int { if a < b { return a }; return b }

func processMessage(phone, text string) {
	sessionID := sid(phone)
	sessionMu.RLock(); meta := sessions[sessionID]; sessionMu.RUnlock()
	ctx := context.Background()

	if meta == nil {
		sendWA(phone, fmt.Sprintf("*%s Intake*\n\nType *AGREE* to continue or *STOP* to cancel.", clinicName))
		session, err := tn.Sessions.Create(ctx, goalID, &truenorth.CreateSessionOptions{SessionID: sessionID})
		if err != nil { log.Printf("Create error: %v", err); sendWA(phone, "System starting up — try again."); return }
		sessionMu.Lock()
		sessions[sessionID] = &PatientSession{SessionID: sessionID, Phone: phone, StartedAt: time.Now(), LastActive: time.Now()}
		sessionMu.Unlock()
		_ = session
		return
	}

	if !meta.Consented {
		lower := strings.ToLower(text)
		if lower == "stop" || lower == "no" { sendWA(phone, "No problem. Data not stored."); sessionMu.Lock(); delete(sessions, sessionID); sessionMu.Unlock(); return }
		sessionMu.Lock(); meta.Consented = true; sessionMu.Unlock()
		session, err := tn.Sessions.Create(ctx, goalID, &truenorth.CreateSessionOptions{SessionID: sessionID + "_c"})
		if err != nil { return }
		sendWA(phone, session.AgentMessage)
		return
	}

	result, err := tn.Sessions.Message(ctx, sessionID+"_c", text)
	if err != nil { sendWA(phone, "Error — please try again."); return }
	sessionMu.Lock(); meta.LastActive = time.Now(); meta.TurnCount++; meta.Pct = result.CompletionPct; sessionMu.Unlock()
	sendWA(phone, result.Text)

	if result.IsComplete && result.Output != nil {
		var content map[string]interface{}
		raw, _ := json.Marshal(result.Output.Content)
		json.Unmarshal(raw, &content)
		sessionMu.Lock()
		completed = append(completed, CompletedIntake{SessionID: sessionID, Phone: phone, CompletedAt: time.Now().Format(time.RFC3339), IntakeData: content})
		delete(sessions, sessionID)
		sessionMu.Unlock()
		sendWA(phone, fmt.Sprintf("✅ *%s* — Intake complete! Doctor will review it. Thank you! 🙏", clinicName))
		tn.Sessions.End(ctx, sessionID+"_c")
	}
}

func setupRouter() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		sessionMu.RLock(); active := len(sessions); done := len(completed); sessionMu.RUnlock()
		c.JSON(200, gin.H{"status":"ok","clinic":clinicName,"active":active,"completed":done})
	})
	r.GET("/completed", func(c *gin.Context) {
		sessionMu.RLock(); defer sessionMu.RUnlock()
		c.JSON(200, gin.H{"intakes":completed,"total":len(completed)})
	})
	r.GET("/webhook", func(c *gin.Context) {
		if c.Query("hub.mode") == "subscribe" && c.Query("hub.verify_token") == verifyToken {
			c.String(200, c.Query("hub.challenge")); return
		}
		c.String(403, "Bad token")
	})
	r.POST("/webhook", func(c *gin.Context) {
		c.JSON(200, gin.H{"status":"ok"})
		var body map[string]interface{}
		if err := c.ShouldBindJSON(&body); err != nil { return }
		go func() {
			defer func() { recover() }()
			entry := body["entry"].([]interface{})[0].(map[string]interface{})
			changes := entry["changes"].([]interface{})[0].(map[string]interface{})
			value := changes["value"].(map[string]interface{})
			msgs, ok := value["messages"].([]interface{})
			if !ok || len(msgs) == 0 { return }
			msg   := msgs[0].(map[string]interface{})
			phone := fmt.Sprintf("%v", msg["from"])
			textM := msg["text"].(map[string]interface{})
			text  := strings.TrimSpace(fmt.Sprintf("%v", textM["body"]))
			if phone != "" && text != "" { processMessage(phone, text) }
		}()
	})
	return r
}

func runConsole() {
	fmt.Printf("\n  CliniqFlow (Go) — %s\n  CONSOLE MODE\n\n", clinicName)
	phone := "919999999999"
	sessionID := sid(phone) + "_c"
	ctx := context.Background()
	session, err := tn.Sessions.Create(ctx, goalID, &truenorth.CreateSessionOptions{SessionID: sessionID})
	if err != nil { fmt.Printf("  Error: %v\n", err); return }
	fmt.Printf("Bot: %s\n\n", session.AgentMessage)
	var input string
	for {
		fmt.Print("Patient: ")
		fmt.Scanln(&input); input = strings.TrimSpace(input)
		if input == "" || input == "quit" { break }
		result, err := tn.Sessions.Message(ctx, sessionID, input)
		if err != nil { fmt.Printf("Error: %v\n", err); continue }
		fmt.Printf("\nBot: %s\n\n", result.Text)
		if result.IsComplete && result.Output != nil {
			data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
			fmt.Println("── INTAKE COMPLETE ─────────────────────────────")
			fmt.Println(string(data)); break
		}
	}
	tn.Sessions.End(ctx, sessionID)
}

func main() {
	flag.BoolVar(&consoleMode, "console", false, "Console demo mode")
	flag.Parse()
	tn = truenorth.NewClient(tnKey, tnURL)
	fmt.Printf("\n  CliniqFlow (Go) — %s\n", clinicName)
	if resp, err := http.Get(tnURL+"/health"); err != nil {
		log.Printf("⚠  TrueNorth not reachable at %s", tnURL)
	} else { resp.Body.Close(); log.Printf("✓ TrueNorth @ %s", tnURL) }
	if consoleMode || accessToken == "" { runConsole(); return }
	r := setupRouter()
	log.Printf("CliniqFlow running at http://localhost:%s", port)
	if err := r.Run(":"+port); err != nil { log.Fatal(err) }
}

// suppress unused io import
var _ = io.Discard
