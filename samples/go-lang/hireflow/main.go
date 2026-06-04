// HireFlow (Go) — AI Candidate Screening REST API + Dashboard
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// Go version of HireFlow. Candidate gets a link, completes
// an 8-minute AI screening, recruiter gets a ranked scorecard.
//
// Exposes a clean REST API + HTML dashboard.
// Role fully configurable via environment variables — change
// ROLE_TITLE, ROLE_SKILLS, ROLE_MIN_EXP without touching code.
//
// This is the Go equivalent of hireflow/app.py.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   hireflow/
//   ├── main.go      ← this file
//   ├── goal.yaml    ← screening questionnaire (copy from python/hireflow/goal.yaml)
//   └── go.mod
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/hireflow
//   go 1.22
//   require (
//     github.com/truenorth-ai/truenorth-go v0.1.0
//     github.com/gin-gonic/gin v1.9.1
//   )
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/hireflow
//   go mod download
//   go build -o hireflow .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: Build and run
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   export COMPANY_NAME="TechCorp"
//   export ROLE_TITLE="Senior Backend Engineer"
//   export ROLE_MIN_EXP=4
//   export ROLE_SKILLS="Python, distributed systems, PostgreSQL"
//   export ROLE_BUDGET_LPA=30
//   ./hireflow
//
//   # Dashboard
//   open http://localhost:8080/dashboard
//
// LOCAL TEST (console mode)
// ──────────────────────────────────────────────────────────────
//   ./hireflow --console
//
// REST API
// ──────────────────────────────────────────────────────────────
//   POST /api/screen/start     → start a screening session
//     Body: {"candidate_id": "c001"}
//     Response: {"session_id": "...", "agent_message": "..."}
//
//   POST /api/screen/message   → continue a session
//     Body: {"session_id": "...", "text": "My name is Priya"}
//     Response: {"text": "...", "is_complete": false, "completion_pct": 45}
//
//   GET  /api/candidates        → all candidates sorted by score
//   GET  /api/shortlist         → SHORTLIST + STRONG_HIRE only
//   GET  /api/candidates/:id    → single candidate scorecard
//   GET  /dashboard             → HTML recruiter view
//   GET  /health                → health check

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
	"github.com/gin-gonic/gin"
)

// ── Config ────────────────────────────────────────────────────────────────────

var (
	companyName    = cfg("COMPANY_NAME",       "TechCorp")
	roleTitle      = cfg("ROLE_TITLE",         "Senior Backend Engineer")
	roleMinExp     = cfgInt("ROLE_MIN_EXP",    4)
	roleSkills     = cfg("ROLE_SKILLS",        "Python, system design, PostgreSQL")
	roleBudgetLPA  = cfgInt("ROLE_BUDGET_LPA", 30)
	roleMaxNotice  = cfgInt("ROLE_MAX_NOTICE", 60)
	goalID         = cfg("GOAL_ID",            "hr_screening")
	tnURL          = cfg("TRUENORTH_BASE_URL",  "http://localhost:8000")
	tnKey          = cfg("TRUENORTH_API_KEY",   "")
	atsWebhook     = cfg("ATS_WEBHOOK_URL",     "")
	port           = cfg("PORT",               "8080")
	consoleMode    bool
)

func cfg(k, def string) string {
	if v := os.Getenv(k); v != "" { return v }
	return def
}
func cfgInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil { return n }
	}
	return def
}

// ── Types ─────────────────────────────────────────────────────────────────────

type ScreeningSession struct {
	SessionID   string
	CandidateID string
	StartedAt   time.Time
	LastActive  time.Time
	TurnCount   int
	Pct         float64
}

type Candidate struct {
	SessionID      string                 `json:"session_id"`
	CandidateID    string                 `json:"candidate_id"`
	ScreenedAt     string                 `json:"screened_at"`
	Name           string                 `json:"name"`
	OverallScore   int                    `json:"overall_score"`
	Recommendation string                 `json:"recommendation"`
	TechScore      int                    `json:"technical_score"`
	CultureScore   int                    `json:"culture_score"`
	Strengths      []string               `json:"strengths"`
	Concerns       []string               `json:"concerns"`
	RedFlags       []string               `json:"red_flags"`
	NextRoundQs    []string               `json:"next_round_questions"`
	SalaryFit      string                 `json:"salary_fit"`
	SummaryText    string                 `json:"summary"`
	Full           map[string]interface{} `json:"full_scorecard,omitempty"`
}

// ── Store ─────────────────────────────────────────────────────────────────────

type Store struct {
	mu         sync.RWMutex
	sessions   map[string]*ScreeningSession
	candidates []*Candidate
}

func newStore() *Store { return &Store{sessions: make(map[string]*ScreeningSession)} }

func (s *Store) getSession(id string) (*ScreeningSession, bool) {
	s.mu.RLock(); defer s.mu.RUnlock()
	v, ok := s.sessions[id]; return v, ok
}
func (s *Store) setSession(id string, v *ScreeningSession) {
	s.mu.Lock(); defer s.mu.Unlock(); s.sessions[id] = v
}
func (s *Store) deleteSession(id string) {
	s.mu.Lock(); defer s.mu.Unlock(); delete(s.sessions, id)
}
func (s *Store) addCandidate(c *Candidate) {
	s.mu.Lock(); defer s.mu.Unlock()
	s.candidates = append(s.candidates, c)
}
func (s *Store) allCandidates() []*Candidate {
	s.mu.RLock(); defer s.mu.RUnlock()
	out := make([]*Candidate, len(s.candidates))
	copy(out, s.candidates)
	sort.Slice(out, func(i, j int) bool { return out[i].OverallScore > out[j].OverallScore })
	return out
}
func (s *Store) shortlist() []*Candidate {
	all := s.allCandidates()
	var out []*Candidate
	for _, c := range all {
		if c.Recommendation == "STRONG_HIRE" || c.Recommendation == "SHORTLIST" {
			out = append(out, c)
		}
	}
	return out
}

// ── TrueNorth client ──────────────────────────────────────────────────────────

var tn *truenorth.Client

// ── ATS webhook push ─────────────────────────────────────────────────────────

func pushToATS(c *Candidate) {
	if atsWebhook == "" { return }
	go func() {
		body, _ := json.Marshal(c)
		req, _  := http.NewRequest("POST", atsWebhook, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil { log.Printf("ATS webhook error: %v", err); return }
		defer resp.Body.Close()
		log.Printf("ATS webhook: %d", resp.StatusCode)
	}()
}

// ── Parse LLM output → Candidate ─────────────────────────────────────────────

func parseCandidate(sid, cid string, content map[string]interface{}) *Candidate {
	name := strVal(content["candidate_name"])
	if name == "" { name = "Unknown" }
	rec  := strings.ToUpper(strVal(content["recommendation"]))
	if !strings.Contains("STRONG_HIRE SHORTLIST HOLD REJECT", rec) { rec = "UNKNOWN" }

	return &Candidate{
		SessionID:      sid,
		CandidateID:    cid,
		ScreenedAt:     time.Now().Format(time.RFC3339),
		Name:           name,
		OverallScore:   intVal(content["overall_score"]),
		Recommendation: rec,
		TechScore:      intVal(content["technical_score"]),
		CultureScore:   intVal(content["culture_score"]),
		Strengths:      sliceVal(content["strengths"]),
		Concerns:       sliceVal(content["concerns"]),
		RedFlags:       sliceVal(content["red_flags"]),
		NextRoundQs:    sliceVal(content["next_round_questions"]),
		SalaryFit:      strVal(content["salary_fit"]),
		SummaryText:    strVal(content["summary"]),
		Full:           content,
	}
}

func strVal(v interface{}) string {
	if v == nil { return "" }
	return fmt.Sprintf("%v", v)
}
func intVal(v interface{}) int {
	if v == nil { return 0 }
	switch t := v.(type) {
	case float64: return int(t)
	case int:     return t
	}
	return 0
}
func sliceVal(v interface{}) []string {
	if v == nil { return nil }
	if sl, ok := v.([]interface{}); ok {
		var out []string
		for _, i := range sl { out = append(out, fmt.Sprintf("%v", i)) }
		return out
	}
	return nil
}

// ── Dashboard HTML ────────────────────────────────────────────────────────────

const dashHTML = `<!DOCTYPE html>
<html><head><title>HireFlow — {{.Company}}</title><meta charset="UTF-8">
<style>
body{font-family:-apple-system,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;background:#f8fafc;}
h1{color:#1e293b;margin-bottom:4px;}h2{color:#475569;font-size:1rem;font-weight:400;margin-bottom:20px;}
.s{display:inline-block;background:white;border-radius:10px;padding:15px 20px;margin:5px;
   border:1px solid #e2e8f0;text-align:center;min-width:110px;}
.s h2{margin:0;font-size:1.8rem;font-weight:700;}
.s p{margin:3px 0 0;color:#64748b;font-size:.78rem;}
table{width:100%;border-collapse:collapse;background:white;border-radius:10px;
      overflow:hidden;margin-top:20px;box-shadow:0 1px 3px rgba(0,0,0,.06);}
th{background:#1e293b;color:white;padding:10px 12px;text-align:left;font-size:.82rem;}
td{padding:10px 12px;border-bottom:1px solid #f1f5f9;font-size:.88rem;}
tr:last-child td{border:none;} tr:hover td{background:#f8fafc;}
.SH{color:#16a34a;font-weight:700;} .SL{color:#2563eb;font-weight:700;}
.HO{color:#d97706;font-weight:700;} .RE{color:#dc2626;font-weight:700;}
.UN{color:#64748b;font-weight:700;}
small{color:#94a3b8;font-size:.78rem;}
.no{text-align:center;color:#94a3b8;padding:24px!important;}
</style></head><body>
<h1>💼 HireFlow</h1>
<h2>{{.Company}} — {{.Role}}</h2>
<div>
  <div class="s"><h2>{{.Active}}</h2><p>Screening now</p></div>
  <div class="s"><h2>{{.Total}}</h2><p>Completed</p></div>
  <div class="s" style="border-color:#bbf7d0"><h2 style="color:#16a34a">{{.Strong}}</h2><p>Strong hire</p></div>
  <div class="s" style="border-color:#bfdbfe"><h2 style="color:#2563eb">{{.Shortlist}}</h2><p>Shortlist</p></div>
  <div class="s" style="border-color:#fecaca"><h2 style="color:#dc2626">{{.Rejected}}</h2><p>Rejected</p></div>
</div>
<table>
<thead><tr>
  <th>Candidate</th><th>Score</th><th>Recommendation</th>
  <th>Technical</th><th>Culture</th><th>Salary fit</th><th>Screened</th>
</tr></thead>
<tbody>
{{range .Candidates}}
<tr>
  <td><strong>{{.Name}}</strong><br><small>{{.CandidateID}}</small></td>
  <td><strong>{{.OverallScore}}/100</strong></td>
  <td class="{{rec .Recommendation}}">{{.Recommendation}}</td>
  <td>{{.TechScore}}</td>
  <td>{{.CultureScore}}</td>
  <td>{{.SalaryFit}}</td>
  <td>{{slice .ScreenedAt 0 16}}</td>
</tr>
{{else}}
<tr><td class="no" colspan="7">No candidates screened yet. Share the API link!</td></tr>
{{end}}
</tbody>
</table>
<p style="margin-top:16px;font-size:.82rem;color:#94a3b8">
  <a href="/api/shortlist">Shortlist JSON</a> ·
  <a href="/api/candidates">All candidates</a> ·
  <a href="/health">Health</a>
</p>
</body></html>`

// ── Gin router ────────────────────────────────────────────────────────────────

func setupRouter(store *Store) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	funcMap := template.FuncMap{
		"rec": func(s string) string {
			return string([]rune(s)[:min(2, len([]rune(s)))])
		},
		"slice": func(s string, i, j int) string {
			if j > len(s) { j = len(s) }
			if i > len(s) { return "" }
			return s[i:j]
		},
	}
	tmpl := template.Must(template.New("dash").Funcs(funcMap).Parse(dashHTML))

	// Dashboard
	r.GET("/dashboard", func(c *gin.Context) {
		all := store.allCandidates()
		sl  := store.shortlist()
		strong, rejected := 0, 0
		for _, ca := range all {
			if ca.Recommendation == "STRONG_HIRE" { strong++ }
			if ca.Recommendation == "REJECT"       { rejected++ }
		}
		store.mu.RLock()
		active := len(store.sessions)
		store.mu.RUnlock()

		data := map[string]interface{}{
			"Company":    companyName,  "Role":      roleTitle,
			"Active":     active,       "Total":     len(all),
			"Strong":     strong,       "Shortlist": len(sl),
			"Rejected":   rejected,     "Candidates": all,
		}
		c.Header("Content-Type", "text/html")
		tmpl.Execute(c.Writer, data)
	})

	// Health
	r.GET("/health", func(c *gin.Context) {
		h, err := tn.Health(context.Background())
		if err != nil {
			c.JSON(503, gin.H{"status": "truenorth_unreachable"})
			return
		}
		store.mu.RLock(); active := len(store.sessions); store.mu.RUnlock()
		c.JSON(200, gin.H{
			"status": "ok", "company": companyName, "role": roleTitle,
			"min_exp": roleMinExp, "budget_lpa": roleBudgetLPA,
			"truenorth": h, "active": active, "screened": len(store.allCandidates()),
		})
	})

	// Candidates
	r.GET("/api/candidates", func(c *gin.Context) {
		all := store.allCandidates()
		c.JSON(200, gin.H{"candidates": all, "total": len(all)})
	})
	r.GET("/api/shortlist", func(c *gin.Context) {
		c.JSON(200, gin.H{"candidates": store.shortlist()})
	})
	r.GET("/api/candidates/:id", func(c *gin.Context) {
		for _, ca := range store.allCandidates() {
			if ca.SessionID == c.Param("id") || ca.CandidateID == c.Param("id") {
				c.JSON(200, ca); return
			}
		}
		c.JSON(404, gin.H{"error": "not found"})
	})

	// Start screening
	r.POST("/api/screen/start", func(c *gin.Context) {
		var req struct{ CandidateID string `json:"candidate_id"` }
		c.ShouldBindJSON(&req)
		if req.CandidateID == "" { req.CandidateID = fmt.Sprintf("candidate_%d", time.Now().UnixMilli()) }

		sid := fmt.Sprintf("hf_%s_%d", req.CandidateID, time.Now().UnixMilli())
		session, err := tn.Sessions.Create(context.Background(), truenorth.CreateSessionRequest{
			GoalID: goalID, SessionID: sid,
		})
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()}); return
		}

		store.setSession(sid, &ScreeningSession{
			SessionID:   sid,
			CandidateID: req.CandidateID,
			StartedAt:   time.Now(),
			LastActive:  time.Now(),
		})

		c.JSON(200, gin.H{
			"session_id":    sid,
			"agent_message": session.AgentMessage,
			"completion_pct": 0,
		})
	})

	// Continue screening
	r.POST("/api/screen/message", func(c *gin.Context) {
		var req struct {
			SessionID string `json:"session_id"`
			Text      string `json:"text"`
		}
		c.ShouldBindJSON(&req)
		if req.SessionID == "" || req.Text == "" {
			c.JSON(400, gin.H{"error": "session_id and text required"}); return
		}

		meta, ok := store.getSession(req.SessionID)
		if !ok {
			c.JSON(404, gin.H{"error": "session not found"}); return
		}

		result, err := tn.Sessions.Message(context.Background(), truenorth.MessageRequest{
			SessionID: req.SessionID, Text: req.Text,
		})
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()}); return
		}

		meta.LastActive = time.Now()
		meta.TurnCount++
		meta.Pct = result.CompletionPct
		store.setSession(req.SessionID, meta)

		resp := gin.H{
			"text":           result.Text,
			"is_complete":    result.IsComplete,
			"completion_pct": result.CompletionPct,
		}

		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content)
			json.Unmarshal(raw, &content)

			ca := parseCandidate(req.SessionID, meta.CandidateID, content)
			store.addCandidate(ca)
			store.deleteSession(req.SessionID)
			tn.Sessions.End(context.Background(), req.SessionID)
			pushToATS(ca)

			log.Printf("✅ %s → %s (%d/100)", ca.Name, ca.Recommendation, ca.OverallScore)
			resp["scorecard"] = ca
		}

		c.JSON(200, resp)
	})

	return r
}

// ── Console demo ──────────────────────────────────────────────────────────────

func runConsoleDemo() {
	fmt.Printf("\n  HireFlow (Go) — %s / %s\n", companyName, roleTitle)
	fmt.Printf("  Min exp: %d years | Budget: %d LPA | Notice: %d days\n\n",
		roleMinExp, roleBudgetLPA, roleMaxNotice)

	ctx := context.Background()
	sid := fmt.Sprintf("hf_demo_%d", time.Now().UnixMilli())

	session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID: goalID, SessionID: sid,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n  Is TrueNorth API running at %s?\n", err, tnURL)
		return
	}
	fmt.Printf("  HireFlow: %s\n\n", session.AgentMessage)

	var input string
	for {
		fmt.Print("  Candidate: ")
		fmt.Scanln(&input)
		input = strings.TrimSpace(input)
		if input == "" || input == "quit" { break }

		result, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
			SessionID: sid, Text: input,
		})
		if err != nil { fmt.Println("  Error:", err); continue }

		fmt.Printf("\n  HireFlow: %s\n\n", result.Text)

		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content)
			json.Unmarshal(raw, &content)

			ca := parseCandidate(sid, "demo", content)
			fmt.Println("\n  ── SCORECARD ────────────────────────────────────────")
			fmt.Printf("  Candidate:      %s\n",   ca.Name)
			fmt.Printf("  Overall score:  %d/100\n", ca.OverallScore)
			fmt.Printf("  Recommendation: %s\n",   ca.Recommendation)
			fmt.Printf("  Technical:      %d/100\n", ca.TechScore)
			fmt.Printf("  Culture:        %d/100\n", ca.CultureScore)
			fmt.Printf("  Salary fit:     %s\n",   ca.SalaryFit)
			if len(ca.Strengths) > 0 {
				fmt.Println("\n  Strengths:")
				for _, s := range ca.Strengths { fmt.Printf("    + %s\n", s) }
			}
			if len(ca.Concerns) > 0 {
				fmt.Println("\n  Concerns:")
				for _, s := range ca.Concerns { fmt.Printf("    - %s\n", s) }
			}
			if len(ca.RedFlags) > 0 {
				fmt.Println("\n  ⚠ Red flags:")
				for _, s := range ca.RedFlags { fmt.Printf("    ⚠ %s\n", s) }
			}
			break
		}
	}

	tn.Sessions.End(ctx, sid)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func min(a, b int) int { if a < b { return a }; return b }

func truncateStr(s string, n int) string {
	if len(s) <= n { return s }
	return s[:n]
}

// read from env at runtime
func init() {
	// re-read env at startup (the top-level var inits use os.Getenv via cfg)
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.BoolVar(&consoleMode, "console", false, "Run in console demo mode")
	flag.Parse()

	tn = truenorth.New(truenorth.Config{
		BaseURL: tnURL,
		APIKey:  tnKey,
		Timeout: 90 * time.Second,
	})

	// Print config
	fmt.Printf("\n  HireFlow (Go) — %s\n", companyName)
	fmt.Printf("  Role: %s | Min exp: %d years | Budget: %d LPA\n\n",
		roleTitle, roleMinExp, roleBudgetLPA)

	// Health check
	if h, err := tn.Health(context.Background()); err != nil {
		log.Printf("⚠  TrueNorth not reachable at %s", tnURL)
		log.Printf("   Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000")
	} else {
		log.Printf("✓ TrueNorth: %s @ %s", h.Status, tnURL)
	}

	if consoleMode {
		runConsoleDemo()
		return
	}

	store := newStore()
	r     := setupRouter(store)

	log.Printf("Dashboard: http://localhost:%s/dashboard", port)
	log.Printf("Start:     POST http://localhost:%s/api/screen/start", port)
	log.Printf("Message:   POST http://localhost:%s/api/screen/message", port)
	r.Run(":" + port)
}