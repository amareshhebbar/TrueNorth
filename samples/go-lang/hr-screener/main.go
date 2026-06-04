// hr-screener-api — Go Gin REST API + HTML Dashboard
//
// PROJECT STRUCTURE
// -----------------
//   hr-screener-api/
//   ├── main.go          ← this file (Gin HTTP server)
//   ├── goal.yaml        ← screening questionnaire
//   └── go.mod
//
// go.mod contents:
// ----------------
//   module github.com/truenorth-ai/hr-screener-api
//
//   go 1.22
//
//   require (
//     github.com/truenorth-ai/truenorth-go v0.1.0
//     github.com/gin-gonic/gin v1.9.1
//   )
//
// WHAT THIS DOES
// --------------
// A REST API server for AI-powered candidate screening.
// Exposes a clean JSON API so any frontend (React, Vue, Next.js)
// can embed a screening chat flow.
//
// Includes:
//   - POST /api/sessions/start   → start a screening session
//   - POST /api/sessions/message → continue a session
//   - GET  /api/candidates       → all completed scorecards
//   - GET  /api/candidates/:id   → single scorecard
//   - GET  /dashboard            → HTML recruiter dashboard
//   - GET  /health               → health check
//
// INSTALL
// -------
//   cd sample-projects-go/hr-screener-api
//   go mod download
//   go build -o hr-screener .
//
// HOW TO RUN
// ----------
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: This Go API
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   export COMPANY_NAME="TechCorp"
//   export ROLE_TITLE="Senior Backend Engineer"
//   ./hr-screener
//
//   # Runs on port 8080 by default
//   open http://localhost:8080/dashboard
//
// REST API USAGE
// --------------
//   # Start a screening
//   curl -X POST http://localhost:8080/api/sessions/start \
//     -H "Content-Type: application/json" \
//     -d '{"candidate_id": "c001"}'
//
//   # Send a message
//   curl -X POST http://localhost:8080/api/sessions/message \
//     -H "Content-Type: application/json" \
//     -d '{"session_id": "...", "text": "My name is Priya"}'
//
//   # Get all candidates (sorted by score)
//   curl http://localhost:8080/api/candidates

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
	"github.com/gin-gonic/gin"
)

// ── Config ────────────────────────────────────────────────────────────────────

var (
	companyName = getEnv("COMPANY_NAME",    "TechCorp")
	roleTitle   = getEnv("ROLE_TITLE",      "Senior Backend Engineer")
	truenorthURL= getEnv("TRUENORTH_BASE_URL","http://localhost:8000")
	apiKey      = getEnv("TRUENORTH_API_KEY","")
	port        = getEnv("PORT",            "8080")
)

func getEnv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// ── Types ─────────────────────────────────────────────────────────────────────

type ActiveSession struct {
	SessionID   string
	CandidateID string
	StartedAt   time.Time
	LastTurn    int
	Pct         float64
}

type CandidateResult struct {
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
	Summary        string                 `json:"summary"`
	Full           map[string]interface{} `json:"full_scorecard"`
}

// ── Store ─────────────────────────────────────────────────────────────────────

type Store struct {
	mu       sync.RWMutex
	sessions map[string]*ActiveSession
	results  []*CandidateResult
}

func newStore() *Store {
	return &Store{sessions: make(map[string]*ActiveSession)}
}

func (s *Store) addSession(sess *ActiveSession) {
	s.mu.Lock(); defer s.mu.Unlock()
	s.sessions[sess.SessionID] = sess
}

func (s *Store) getSession(id string) (*ActiveSession, bool) {
	s.mu.RLock(); defer s.mu.RUnlock()
	v, ok := s.sessions[id]
	return v, ok
}

func (s *Store) removeSession(id string) {
	s.mu.Lock(); defer s.mu.Unlock()
	delete(s.sessions, id)
}

func (s *Store) addResult(r *CandidateResult) {
	s.mu.Lock(); defer s.mu.Unlock()
	s.results = append(s.results, r)
}

func (s *Store) allResults() []*CandidateResult {
	s.mu.RLock(); defer s.mu.RUnlock()
	out := make([]*CandidateResult, len(s.results))
	copy(out, s.results)
	sort.Slice(out, func(i, j int) bool {
		return out[i].OverallScore > out[j].OverallScore
	})
	return out
}

func (s *Store) shortlist() []*CandidateResult {
	all := s.allResults()
	var out []*CandidateResult
	for _, r := range all {
		if r.Recommendation == "STRONG_HIRE" || r.Recommendation == "SHORTLIST" {
			out = append(out, r)
		}
	}
	return out
}

// ── TrueNorth client ──────────────────────────────────────────────────────────

var tn *truenorth.Client

func initTN() {
	tn = truenorth.New(truenorth.Config{
		BaseURL: truenorthURL,
		APIKey:  apiKey,
		Timeout: 90 * time.Second,
	})
}

// ── Handlers ──────────────────────────────────────────────────────────────────

func startHandler(store *Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			CandidateID string `json:"candidate_id"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "candidate_id required"})
			return
		}

		sessID := fmt.Sprintf("hr_%s_%d", req.CandidateID, time.Now().UnixMilli())

		session, err := tn.Sessions.Create(context.Background(), truenorth.CreateSessionRequest{
			GoalID:    "hr_screener",
			SessionID: sessID,
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		store.addSession(&ActiveSession{
			SessionID:   sessID,
			CandidateID: req.CandidateID,
			StartedAt:   time.Now(),
		})

		c.JSON(http.StatusOK, gin.H{
			"session_id":    sessID,
			"agent_message": session.AgentMessage,
			"completion_pct": 0,
		})
	}
}

func messageHandler(store *Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			SessionID string `json:"session_id"`
			Text      string `json:"text"`
		}
		if err := c.ShouldBindJSON(&req); err != nil || req.SessionID == "" || req.Text == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "session_id and text required"})
			return
		}

		active, ok := store.getSession(req.SessionID)
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
			return
		}

		result, err := tn.Sessions.Message(context.Background(), truenorth.MessageRequest{
			SessionID: req.SessionID,
			Text:      req.Text,
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		active.LastTurn = result.Turn
		active.Pct      = result.CompletionPct

		resp := gin.H{
			"text":           result.Text,
			"is_complete":    result.IsComplete,
			"completion_pct": result.CompletionPct,
		}

		if result.IsComplete && result.Output != nil {
			content := toMap(result.Output.Content)
			cr := parseResult(req.SessionID, active.CandidateID, content)
			store.addResult(cr)
			store.removeSession(req.SessionID)
			tn.Sessions.End(context.Background(), req.SessionID)
			resp["scorecard"] = cr
		}

		c.JSON(http.StatusOK, resp)
	}
}

func candidatesHandler(store *Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"candidates": store.allResults(),
			"total":      len(store.allResults()),
		})
	}
}

func shortlistHandler(store *Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"candidates": store.shortlist()})
	}
}

func healthHandler(c *gin.Context) {
	ctx := context.Background()
	h, err := tn.Health(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "truenorth_unreachable", "url": truenorthURL,
		})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"status": "ok", "company": companyName, "role": roleTitle, "truenorth": h,
	})
}

// ── Dashboard HTML ────────────────────────────────────────────────────────────

const dashboardTmpl = `<!DOCTYPE html>
<html><head><title>HireFlow — {{.Company}}</title>
<meta charset="UTF-8">
<style>
body{font-family:-apple-system,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#f8fafc;}
h1{color:#1e293b;}
.stat{display:inline-block;background:white;border-radius:10px;padding:16px 22px;
      margin:6px;text-align:center;border:1px solid #e2e8f0;min-width:110px;}
.stat h2{margin:0;font-size:1.8rem;}
.stat p{margin:3px 0 0;color:#64748b;font-size:.8rem;}
table{width:100%;border-collapse:collapse;margin-top:24px;background:white;
      border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);}
th{background:#1e293b;color:white;padding:11px 14px;text-align:left;font-size:.85rem;}
td{padding:11px 14px;border-bottom:1px solid #f1f5f9;font-size:.9rem;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:#f8fafc;}
.SH{color:#16a34a;font-weight:700;}
.SL{color:#2563eb;font-weight:700;}
.HO{color:#d97706;font-weight:700;}
.RE{color:#dc2626;font-weight:700;}
.api{background:#1e293b;color:#94a3b8;border-radius:8px;padding:14px 18px;
     margin-top:24px;font-family:monospace;font-size:.82rem;line-height:1.8;}
</style></head><body>
<h1>💼 HireFlow — {{.Company}} / {{.Role}}</h1>
<div>
  <div class="stat"><h2>{{.Active}}</h2><p>Screening now</p></div>
  <div class="stat"><h2>{{.Total}}</h2><p>Completed</p></div>
  <div class="stat"><h2 style="color:#16a34a">{{.StrongHire}}</h2><p>Strong hire</p></div>
  <div class="stat"><h2 style="color:#2563eb">{{.Shortlist}}</h2><p>Shortlisted</p></div>
  <div class="stat"><h2 style="color:#dc2626">{{.Rejected}}</h2><p>Rejected</p></div>
</div>
<table>
<thead><tr><th>Candidate</th><th>Score</th><th>Recommendation</th>
<th>Tech</th><th>Culture</th><th>Screened</th></tr></thead>
<tbody>{{range .Results}}
<tr>
  <td><strong>{{.Name}}</strong><br><small style="color:#94a3b8">{{.CandidateID}}</small></td>
  <td><strong>{{.OverallScore}}/100</strong></td>
  <td class="{{slice .Recommendation 0 2}}">{{.Recommendation}}</td>
  <td>{{.TechScore}}</td>
  <td>{{.CultureScore}}</td>
  <td>{{slice .ScreenedAt 0 16}}</td>
</tr>
{{else}}<tr><td colspan="6" style="text-align:center;color:#94a3b8;padding:24px">No candidates screened yet</td></tr>
{{end}}</tbody></table>
<div class="api">
  POST /api/sessions/start    — start a new screening<br>
  POST /api/sessions/message  — send a message<br>
  GET  /api/candidates        — all candidates (JSON)<br>
  GET  /api/candidates/shortlist — shortlisted only
</div>
</body></html>`

func dashboardHandler(store *Store) gin.HandlerFunc {
	tmpl := template.Must(template.New("d").Funcs(template.FuncMap{
		"slice": func(s string, i, j int) string {
			if j > len(s) { j = len(s) }
			if i > len(s) { return "" }
			return s[i:j]
		},
	}).Parse(dashboardTmpl))

	return func(c *gin.Context) {
		all      := store.allResults()
		sl       := store.shortlist()
		strong   := 0; rejected := 0
		for _, r := range all {
			if r.Recommendation == "STRONG_HIRE"  { strong++ }
			if r.Recommendation == "REJECT"        { rejected++ }
		}

		store.mu.RLock()
		active := len(store.sessions)
		store.mu.RUnlock()

		data := map[string]interface{}{
			"Company":   companyName,
			"Role":      roleTitle,
			"Active":    active,
			"Total":     len(all),
			"StrongHire": strong,
			"Shortlist": len(sl),
			"Rejected":  rejected,
			"Results":   all,
		}

		c.Header("Content-Type", "text/html")
		tmpl.Execute(c.Writer, data)
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func toMap(v interface{}) map[string]interface{} {
	if m, ok := v.(map[string]interface{}); ok {
		return m
	}
	raw, _ := json.Marshal(v)
	var m map[string]interface{}
	json.Unmarshal(raw, &m)
	return m
}

func toStringSlice(v interface{}) []string {
	if v == nil { return nil }
	if sl, ok := v.([]interface{}); ok {
		var out []string
		for _, i := range sl { out = append(out, fmt.Sprintf("%v", i)) }
		return out
	}
	return nil
}

func parseInt(v interface{}) int {
	if v == nil { return 0 }
	switch t := v.(type) {
	case float64: return int(t)
	case int:     return t
	}
	return 0
}

func parseResult(sid, cid string, c map[string]interface{}) *CandidateResult {
	name := fmt.Sprintf("%v", c["candidate_name"])
	if name == "" || name == "<nil>" { name = "Unknown" }
	rec  := fmt.Sprintf("%v", c["recommendation"])
	if !strings.Contains("STRONG_HIRE SHORTLIST HOLD REJECT", rec) { rec = "UNKNOWN" }

	return &CandidateResult{
		SessionID:      sid,
		CandidateID:    cid,
		ScreenedAt:     time.Now().Format(time.RFC3339),
		Name:           name,
		OverallScore:   parseInt(c["overall_score"]),
		Recommendation: rec,
		TechScore:      parseInt(c["technical_score"]),
		CultureScore:   parseInt(c["culture_score"]),
		Strengths:      toStringSlice(c["strengths"]),
		Concerns:       toStringSlice(c["concerns"]),
		RedFlags:       toStringSlice(c["red_flags"]),
		NextRoundQs:    toStringSlice(c["next_round_questions"]),
		Summary:        fmt.Sprintf("%v", c["summary"]),
		Full:           c,
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	initTN()
	store := newStore()
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.GET("/health",                    healthHandler)
	r.GET("/dashboard",                 dashboardHandler(store))
	r.POST("/api/sessions/start",       startHandler(store))
	r.POST("/api/sessions/message",     messageHandler(store))
	r.GET("/api/candidates",            candidatesHandler(store))
	r.GET("/api/candidates/shortlist",  shortlistHandler(store))

	log.Printf("HireFlow API — %s / %s", companyName, roleTitle)
	log.Printf("Dashboard: http://localhost:%s/dashboard", port)
	log.Printf("TrueNorth: %s", truenorthURL)
	r.Run(":" + port)
}