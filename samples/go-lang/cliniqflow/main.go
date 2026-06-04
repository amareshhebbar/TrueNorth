// CliniqFlow (Go) — Clinic Intake via WhatsApp
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// Production-ready clinic intake system in Go.
// Patient messages the clinic's WhatsApp number before their appointment.
// TrueNorth collects: chief complaint, pain, medications, allergies, history.
// Doctor sees a 30-second summary before the patient walks in.
// Replaces the paper clipboard at every clinic.
//
// This is the Go equivalent of cliniqflow/app.py.
// Uses the TrueNorth Go SDK which calls the Python REST API.
//
// HOW IT WORKS
// ──────────────────────────────────────────────────────────────
//  Patient → WhatsApp → Meta Webhook → Gin (this file)
//                                           │
//                                   TrueNorth Go SDK
//                                           │
//                               TrueNorth Python API (port 8000)
//                                           │
//                                    Anthropic / Gemini
//                                           │
//                               Response → WhatsApp → Patient
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   cliniqflow/
//   ├── main.go      ← this file
//   ├── goal.yaml    ← patient intake questionnaire (copy from python/cliniqflow/goal.yaml)
//   └── go.mod
//
// go.mod contents:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/cliniqflow
//   go 1.22
//   require (
//     github.com/truenorth-ai/truenorth-go v0.1.0
//     github.com/gin-gonic/gin v1.9.1
//   )
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   # Step 1: Start TrueNorth Python API
//   cd packages/core
//   pip install -e . && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Build this
//   cd samples/go-lang/cliniqflow
//   go mod download
//   go build -o cliniqflow .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   export ANTHROPIC_API_KEY=sk-ant-...
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   export CLINIC_NAME="Dr. Sharma Clinic"
//   export WA_VERIFY_TOKEN=cliniqflow-secret
//   export WA_ACCESS_TOKEN=your-whatsapp-token     ← optional
//   export WA_PHONE_NUMBER_ID=your-phone-id         ← optional
//   ./cliniqflow
//
//   # With ngrok for WhatsApp testing
//   ngrok http 8080
//   # Set webhook: https://your-ngrok-url/webhook
//
// LOCAL TEST (no WhatsApp credentials needed)
// ──────────────────────────────────────────────────────────────
//   # Just run — uses console mode when WA_ACCESS_TOKEN not set
//   ./cliniqflow --console
//
// API ENDPOINTS
// ──────────────────────────────────────────────────────────────
//   GET  /              → Dashboard (active intakes + completed today)
//   GET  /health        → Health check (JSON)
//   GET  /completed     → All completed intakes (JSON)
//   DELETE /sessions/:id → DPDP erasure endpoint
//   GET  /webhook       → WhatsApp verification
//   POST /webhook       → Incoming WhatsApp messages
//
// SAMPLE CONVERSATION
// ──────────────────────────────────────────────────────────────
//   Patient: Hi
//   CliniqFlow: Hello! I am here to help with your intake today.
//               Everything you share is private and goes to your doctor.
//               What is your full name?
//   Patient: Ravi Kumar
//   CliniqFlow: Nice to meet you Ravi! What brings you in today?
//   ...
//   CliniqFlow: Your intake is complete! The doctor will review
//               it before seeing you. Thank you! 

package main

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
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

// ── Config ────────────────────────────────────────────────────────────────────

var (
	clinicName     = env("CLINIC_NAME",          "Our Clinic")
	goalID         = env("GOAL_ID",              "patient_intake")
	verifyToken    = env("WA_VERIFY_TOKEN",       "cliniqflow-token")
	accessToken    = env("WA_ACCESS_TOKEN",       "")
	phoneNumberID  = env("WA_PHONE_NUMBER_ID",    "")
	truenorthURL   = env("TRUENORTH_BASE_URL",    "http://localhost:8000")
	truenorthKey   = env("TRUENORTH_API_KEY",     "")
	port           = env("PORT",                  "8080")
	consoleMode    bool
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" { return v }
	return def
}

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── Store ─────────────────────────────────────────────────────────────────────

type Store struct {
	mu        sync.RWMutex
	sessions  map[string]*PatientSession
	completed []CompletedIntake
}

func newStore() *Store {
	return &Store{sessions: make(map[string]*PatientSession)}
}

func (s *Store) get(id string) (*PatientSession, bool) {
	s.mu.RLock(); defer s.mu.RUnlock()
	v, ok := s.sessions[id]
	return v, ok
}

func (s *Store) set(id string, p *PatientSession) {
	s.mu.Lock(); defer s.mu.Unlock()
	s.sessions[id] = p
}

func (s *Store) delete(id string) {
	s.mu.Lock(); defer s.mu.Unlock()
	delete(s.sessions, id)
}

func (s *Store) addCompleted(c CompletedIntake) {
	s.mu.Lock(); defer s.mu.Unlock()
	s.completed = append(s.completed, c)
}

func (s *Store) stats() (active, done int) {
	s.mu.RLock(); defer s.mu.RUnlock()
	return len(s.sessions), len(s.completed)
}

// ── TrueNorth client ──────────────────────────────────────────────────────────

var tn *truenorth.Client

func sessionID(phone string) string {
	h := md5.Sum([]byte(phone))
	return fmt.Sprintf("cf_%x", h[:6])
}

// ── WhatsApp sender ───────────────────────────────────────────────────────────

func sendWA(phone, text string) {
	if accessToken == "" || phoneNumberID == "" {
		fmt.Printf("\n📤 [%s]: %s\n\n", phone, truncate(text, 200))
		return
	}
	url := fmt.Sprintf("https://graph.facebook.com/v19.0/%s/messages", phoneNumberID)
	payload := map[string]interface{}{
		"messaging_product": "whatsapp",
		"to":                phone,
		"type":              "text",
		"text":              map[string]string{"body": truncate(text, 4000)},
	}
	body, _ := json.Marshal(payload)
	req, _  := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("WA send error: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		log.Printf("WA error %d: %s", resp.StatusCode, string(b)[:min(200, len(b))])
	}
}

func truncate(s string, n int) string {
	if len(s) <= n { return s }
	return s[:n]
}
func min(a, b int) int {
	if a < b { return a }
	return b
}

// ── Process a patient message ─────────────────────────────────────────────────

func processMessage(store *Store, phone, text string) {
	sid  := sessionID(phone)
	meta, exists := store.get(sid)

	if !exists {
		consent := fmt.Sprintf(
			"*%s Intake*\n\n"+
				"We will collect your health information for your appointment today. "+
				"This is confidential and shared only with your doctor.\n\n"+
				"Type *AGREE* to continue or *STOP* to cancel.",
			clinicName,
		)
		sendWA(phone, consent)

		ctx := context.Background()
		session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
			GoalID:    goalID,
			SessionID: sid,
		})
		if err != nil {
			log.Printf("Create session error: %v", err)
			sendWA(phone, "Sorry, the intake system is temporarily unavailable. Please try again.")
			return
		}

		store.set(sid, &PatientSession{
			SessionID:  sid,
			Phone:      phone,
			StartedAt:  time.Now(),
			LastActive: time.Now(),
			Consented:  false,
		})
		_ = session
		return
	}

	if !meta.Consented {
		lower := strings.ToLower(text)
		if lower == "stop" || lower == "no" || lower == "cancel" {
			sendWA(phone, "No problem. Your data has not been stored. See you at the clinic.")
			store.delete(sid)
			return
		}
		meta.Consented = true
		store.set(sid, meta)

		ctx := context.Background()
		session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
			GoalID:    goalID,
			SessionID: sid + "_conv",
		})
		if err != nil {
			log.Printf("Start conv error: %v", err)
			return
		}
		sendWA(phone, session.AgentMessage)

		if !isAcknowledgement(lower) {
			go func() {
				r, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
					SessionID: sid + "_conv",
					Text:      text,
				})
				if err == nil {
					sendWA(phone, r.Text)
					handleCompletion(store, phone, sid, r)
				}
			}()
		}
		return
	}

	ctx := context.Background()
	result, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
		SessionID: sid + "_conv",
		Text:      text,
	})
	if err != nil {
		log.Printf("Message error: %v", err)
		sendWA(phone, "I had trouble processing that. Could you try again?")
		return
	}

	meta.LastActive = time.Now()
	meta.TurnCount++
	meta.Pct = result.CompletionPct
	store.set(sid, meta)

	sendWA(phone, result.Text)
	handleCompletion(store, phone, sid, result)
}

func isAcknowledgement(s string) bool {
	for _, w := range []string{"agree", "yes", "ok", "start", "begin", "sure", "हाँ", "ہاں"} {
		if s == w { return true }
	}
	return false
}

func handleCompletion(store *Store, phone, sid string, result truenorth.MessageResult) {
	if !result.IsComplete { return }

	var content map[string]interface{}
	if result.Output != nil {
		raw, _ := json.Marshal(result.Output.Content)
		json.Unmarshal(raw, &content)
	}

	if content != nil {
		store.addCompleted(CompletedIntake{
			SessionID:   sid,
			Phone:       phone,
			CompletedAt: time.Now().Format(time.RFC3339),
			IntakeData:  content,
		})
	}

	sendWA(phone, fmt.Sprintf(
		"*%s* — Your intake is complete!\n\n"+
			"Your doctor will review it before seeing you today. Thank you! ",
		clinicName,
	))

	store.delete(sid)
	tn.Sessions.End(context.Background(), sid+"_conv")
	log.Printf("Intake complete for %s", phone)
}

// ── Dashboard HTML ────────────────────────────────────────────────────────────

const dashTmpl = `<!DOCTYPE html><html><head><title>CliniqFlow — {{.Clinic}}</title>
<style>body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f0f9ff;}
h1{color:#0369a1;}.s{display:inline-block;background:white;border-radius:10px;
padding:16px 22px;margin:6px;border:1px solid #bae6fd;text-align:center;}
.s h2{margin:0;color:#0369a1;font-size:1.8rem;}.s p{margin:3px 0 0;color:#64748b;font-size:.82rem;}
table{width:100%;border-collapse:collapse;background:white;border-radius:10px;
margin-top:20px;overflow:hidden;}th{background:#0369a1;color:white;padding:10px;text-align:left;}
td{padding:10px;border-bottom:1px solid #e0f2fe;}</style></head><body>
<h1>CliniqFlow (Go) — {{.Clinic}}</h1>
<div>
  <div class="s"><h2>{{.Active}}</h2><p>Active intakes</p></div>
  <div class="s"><h2>{{.Done}}</h2><p>Completed today</p></div>
  <div class="s"><h2>{{.WA}}</h2><p>WhatsApp</p></div>
</div>
<table><thead><tr><th>Session</th><th>Progress</th><th>Turns</th><th>Last active</th></tr></thead>
<tbody>{{range .Sessions}}
<tr><td>{{.SessionID}}</td><td>{{printf "%.0f" .Pct}}%</td><td>{{.TurnCount}}</td>
<td>{{.LastActive.Format "15:04"}}</td></tr>
{{else}}<tr><td colspan=4 style="text-align:center;color:#94a3b8;padding:20px">No active intakes</td></tr>
{{end}}</tbody></table>
<p><small><a href="/health">Health</a> | <a href="/completed">Completed (JSON)</a></small></p>
</body></html>`

// ── Gin server ────────────────────────────────────────────────────────────────

func setupRouter(store *Store) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	tmpl := template.Must(template.New("dash").Parse(dashTmpl))

	r.GET("/", func(c *gin.Context) {
		active, done := store.stats()
		store.mu.RLock()
		var sessions []*PatientSession
		for _, v := range store.sessions { sessions = append(sessions, v) }
		store.mu.RUnlock()

		waStatus := "✅"
		if accessToken == "" { waStatus = "⚠️ console" }

		data := map[string]interface{}{
			"Clinic": clinicName, "Active": active, "Done": done,
			"WA": waStatus, "Sessions": sessions,
		}
		c.Header("Content-Type", "text/html")
		tmpl.Execute(c.Writer, data)
	})

	r.GET("/health", func(c *gin.Context) {
		h, err := tn.Health(context.Background())
		if err != nil {
			c.JSON(503, gin.H{"status": "truenorth_unreachable", "url": truenorthURL})
			return
		}
		active, done := store.stats()
		c.JSON(200, gin.H{
			"status": "ok", "clinic": clinicName, "goal": goalID,
			"truenorth": h, "active": active, "completed": done,
			"wa_connected": accessToken != "" && phoneNumberID != "",
		})
	})

	r.GET("/completed", func(c *gin.Context) {
		store.mu.RLock(); defer store.mu.RUnlock()
		c.JSON(200, gin.H{"intakes": store.completed, "total": len(store.completed)})
	})

	r.DELETE("/sessions/:id", func(c *gin.Context) {
		id := c.Param("id")
		_, ok := store.get(id)
		if !ok {
			c.JSON(404, gin.H{"error": "not found"})
			return
		}
		store.delete(id)
		tn.Sessions.End(context.Background(), id+"_conv")
		c.JSON(200, gin.H{"deleted": id, "status": "erased"})
	})

	// WhatsApp webhook verification
	r.GET("/webhook", func(c *gin.Context) {
		if c.Query("hub.mode") == "subscribe" && c.Query("hub.verify_token") == verifyToken {
			c.String(200, c.Query("hub.challenge"))
			return
		}
		c.String(403, "Bad token")
	})

	// Incoming WhatsApp messages
	r.POST("/webhook", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"}) // acknowledge immediately

		var body map[string]interface{}
		if err := c.ShouldBindJSON(&body); err != nil { return }

		go func() {
			defer func() { recover() }()
			entry   := jsonGet[[]interface{}](body, "entry")
			if len(entry) == 0 { return }
			changes := jsonGet[[]interface{}](entry[0].(map[string]interface{}), "changes")
			if len(changes) == 0 { return }
			value   := jsonGet[map[string]interface{}](changes[0].(map[string]interface{}), "value")
			msgs    := jsonGet[[]interface{}](value, "messages")
			if len(msgs) == 0 { return }

			msg   := msgs[0].(map[string]interface{})
			phone := str(msg["from"])
			textM := jsonGet[map[string]interface{}](msg, "text")
			text  := str(textM["body"])

			if phone != "" && text != "" {
				log.Printf("📨 %s: %s", phone, truncate(text, 60))
				processMessage(store, phone, text)
			}
		}()
	})

	return r
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

func jsonGet[T any](m map[string]interface{}, k string) T {
	var zero T
	if v, ok := m[k]; ok {
		if t, ok := v.(T); ok { return t }
	}
	return zero
}
func str(v interface{}) string {
	if v == nil { return "" }
	return fmt.Sprintf("%v", v)
}

// ── Console demo ──────────────────────────────────────────────────────────────

func runConsoleDemo() {
	fmt.Printf("\n  CliniqFlow (Go) — %s\n", clinicName)
	fmt.Println("  CONSOLE MODE — simulating a patient conversation")
	fmt.Println("  Type 'quit' to exit\n")

	phone   := "919999999999"
	sid     := sessionID(phone) + "_conv"
	ctx     := context.Background()

	session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID: goalID, SessionID: sid,
	})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Printf("Bot: %s\n\n", session.AgentMessage)

	var input string
	for {
		fmt.Print("Patient: ")
		fmt.Scanln(&input)
		input = strings.TrimSpace(input)
		if input == "" || input == "quit" { break }

		result, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: input,
		})
		if err != nil { fmt.Println("Error:", err); continue }

		fmt.Printf("\nBot: %s\n\n", result.Text)

		if result.IsComplete {
			if result.Output != nil {
				data, _ := json.MarshalIndent(result.Output.Content, "  ", "  ")
				fmt.Println("\n── INTAKE COMPLETE ──────────────────────────────")
				fmt.Println(string(data))
			}
			break
		}
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.BoolVar(&consoleMode, "console", false, "Run in console demo mode")
	flag.Parse()

	tn = truenorth.New(truenorth.Config{
		BaseURL: truenorthURL,
		APIKey:  truenorthKey,
		Timeout: 90 * time.Second,
	})

	if h, err := tn.Health(context.Background()); err != nil {
		log.Printf("TrueNorth not reachable at %s — run: uvicorn truenorth.api.main:app --port 8000", truenorthURL)
	} else {
		log.Printf("TrueNorth: %s @ %s", h.Status, truenorthURL)
	}

	if consoleMode || accessToken == "" {
		runConsoleDemo()
		return
	}

	store := newStore()
	r     := setupRouter(store)
	log.Printf("CliniqFlow (Go) — %s", clinicName)
	log.Printf("Dashboard: http://localhost:%s/", port)
	log.Printf("Webhook:   http://localhost:%s/webhook", port)
	r.Run(":" + port)
}