package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
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

var (
	companyName = envOr("COMPANY_NAME",      "TechCorp")
	roleTitle   = envOr("ROLE_TITLE",        "Senior Backend Engineer")
	goalID      = envOr("GOAL_ID",           "hr_screening")
	tnURL       = envOr("TRUENORTH_BASE_URL", "http://localhost:8000")
	tnKey       = envOr("TRUENORTH_API_KEY",  "")
	port        = envOr("PORT",              "8080")
	consoleMode bool
)

func envOr(k, def string) string { if v := os.Getenv(k); v != "" { return v }; return def }

var tn *truenorth.TrueNorth

type Screening struct {
	SessionID   string; CandidateID string
	StartedAt   time.Time; Pct float64
}

type Candidate struct {
	SessionID      string                 `json:"session_id"`
	CandidateID    string                 `json:"candidate_id"`
	ScreenedAt     string                 `json:"screened_at"`
	Name           string                 `json:"name"`
	OverallScore   int                    `json:"overall_score"`
	Recommendation string                 `json:"recommendation"`
	Full           map[string]interface{} `json:"full,omitempty"`
}

var (
	mu         sync.RWMutex
	screenings = map[string]*Screening{}
	candidates []*Candidate
)

func parseCandidate(sid, cid string, content map[string]interface{}) *Candidate {
	name := fmt.Sprintf("%v", content["candidate_name"]); if name == "" || name == "<nil>" { name = "Unknown" }
	rec  := strings.ToUpper(fmt.Sprintf("%v", content["recommendation"]))
	if !strings.Contains("STRONG_HIRE SHORTLIST HOLD REJECT", rec) { rec = "UNKNOWN" }
	score := 0; if v, ok := content["overall_score"].(float64); ok { score = int(v) }
	return &Candidate{SessionID: sid, CandidateID: cid, ScreenedAt: time.Now().Format(time.RFC3339), Name: name, OverallScore: score, Recommendation: rec, Full: content}
}

func allCandidates() []*Candidate {
	mu.RLock(); defer mu.RUnlock()
	out := make([]*Candidate, len(candidates)); copy(out, candidates)
	sort.Slice(out, func(i, j int) bool { return out[i].OverallScore > out[j].OverallScore })
	return out
}

func setupRouter() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.GET("/dashboard", func(c *gin.Context) {
		all := allCandidates()
		mu.RLock(); active := len(screenings); mu.RUnlock()
		rows := ""
		for _, ca := range all {
			rows += fmt.Sprintf("<tr><td><b>%s</b></td><td>%d/100</td><td>%s</td></tr>", ca.Name, ca.OverallScore, ca.Recommendation)
		}
		if rows == "" { rows = "<tr><td colspan=3 style='text-align:center;color:#94a3b8'>No candidates yet</td></tr>" }
		c.Header("Content-Type", "text/html")
		c.String(200, `<!DOCTYPE html><html><head><title>HireFlow</title></head><body>`+
			`<h1>💼 HireFlow — `+companyName+` / `+roleTitle+`</h1>`+
			`<p>Active: `+fmt.Sprint(active)+` | Screened: `+fmt.Sprint(len(all))+`</p>`+
			`<table border=1><tr><th>Name</th><th>Score</th><th>Recommendation</th></tr>`+rows+`</table>`+
			`<p><a href="/api/candidates">All JSON</a></p></body></html>`)
	})
	r.GET("/health", func(c *gin.Context) {
		mu.RLock(); active := len(screenings); mu.RUnlock()
		c.JSON(200, gin.H{"status":"ok","role":roleTitle,"active":active,"screened":len(allCandidates())})
	})
	r.GET("/api/candidates", func(c *gin.Context) {
		all := allCandidates(); c.JSON(200, gin.H{"candidates":all,"total":len(all)})
	})
	r.GET("/api/shortlist", func(c *gin.Context) {
		var sl []*Candidate
		for _, ca := range allCandidates() { if ca.Recommendation == "STRONG_HIRE" || ca.Recommendation == "SHORTLIST" { sl = append(sl, ca) } }
		c.JSON(200, gin.H{"candidates":sl})
	})
	r.POST("/api/screen/start", func(c *gin.Context) {
		var req struct{ CandidateID string `json:"candidate_id"` }
		c.ShouldBindJSON(&req)
		if req.CandidateID == "" { req.CandidateID = fmt.Sprintf("c_%d", time.Now().UnixMilli()) }
		sessID := fmt.Sprintf("hf_%s_%d", req.CandidateID, time.Now().UnixMilli())
		session, err := tn.Sessions.Create(context.Background(), goalID, &truenorth.CreateSessionOptions{SessionID: sessID})
		if err != nil { c.JSON(500, gin.H{"error":err.Error()}); return }
		mu.Lock(); screenings[sessID] = &Screening{SessionID:sessID,CandidateID:req.CandidateID,StartedAt:time.Now()}; mu.Unlock()
		c.JSON(200, gin.H{"session_id":sessID,"agent_message":session.AgentMessage,"completion_pct":0})
	})
	r.POST("/api/screen/message", func(c *gin.Context) {
		var req struct{ SessionID string `json:"session_id"`; Text string `json:"text"` }
		c.ShouldBindJSON(&req)
		if req.SessionID == "" || req.Text == "" { c.JSON(400, gin.H{"error":"session_id and text required"}); return }
		mu.RLock(); meta := screenings[req.SessionID]; mu.RUnlock()
		if meta == nil { c.JSON(404, gin.H{"error":"not found"}); return }
		result, err := tn.Sessions.Message(context.Background(), req.SessionID, req.Text)
		if err != nil { c.JSON(500, gin.H{"error":err.Error()}); return }
		mu.Lock(); meta.Pct = result.CompletionPct; mu.Unlock()
		resp := gin.H{"text":result.Text,"is_complete":result.IsComplete,"completion_pct":result.CompletionPct}
		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content); json.Unmarshal(raw, &content)
			ca := parseCandidate(req.SessionID, meta.CandidateID, content)
			mu.Lock(); candidates = append(candidates, ca); delete(screenings, req.SessionID); mu.Unlock()
			tn.Sessions.End(context.Background(), req.SessionID)
			log.Printf("✅ %s → %s (%d/100)", ca.Name, ca.Recommendation, ca.OverallScore)
			resp["scorecard"] = ca
		}
		c.JSON(200, resp)
	})
	return r
}

func runConsole() {
	fmt.Printf("\n  HireFlow (Go) — %s / %s\n\n", companyName, roleTitle)
	sid := fmt.Sprintf("hf_demo_%d", time.Now().UnixMilli())
	session, err := tn.Sessions.Create(context.Background(), goalID, &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil { fmt.Printf("  Error: %v\n", err); return }
	fmt.Printf("  HireFlow: %s\n\n", session.AgentMessage)
	var input string
	for {
		fmt.Print("  Candidate: "); fmt.Scanln(&input); input = strings.TrimSpace(input)
		if input == "" || input == "quit" { break }
		result, err := tn.Sessions.Message(context.Background(), sid, input)
		if err != nil { fmt.Printf("  Error: %v\n", err); continue }
		fmt.Printf("\n  HireFlow: %s\n\n", result.Text)
		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content); json.Unmarshal(raw, &content)
			ca := parseCandidate(sid, "demo", content)
			fmt.Printf("\n  SCORECARD\n  Name: %s\n  Score: %d/100\n  Rec: %s\n", ca.Name, ca.OverallScore, ca.Recommendation)
			break
		}
	}
	tn.Sessions.End(context.Background(), sid)
}

func main() {
	flag.BoolVar(&consoleMode, "console", false, "Console demo mode")
	flag.Parse()
	tn = truenorth.NewClient(tnKey, tnURL)
	fmt.Printf("\n  HireFlow (Go) — %s / %s\n", companyName, roleTitle)
	if resp, err := http.Get(tnURL+"/health"); err != nil {
		log.Printf("⚠  TrueNorth not reachable at %s", tnURL)
	} else { resp.Body.Close(); log.Printf("✓ TrueNorth @ %s", tnURL) }
	if consoleMode { runConsole(); return }
	r := setupRouter()
	log.Printf("Dashboard: http://localhost:%s/dashboard", port)
	log.Printf("Start:     POST http://localhost:%s/api/screen/start", port)
	r.Run(":"+port)
}
