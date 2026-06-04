package main

import (
	"context"
	"encoding/json"
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
	companyName = func() string { if v := os.Getenv("COMPANY_NAME"); v != "" { return v }; return "TechCorp" }()
	roleTitle   = func() string { if v := os.Getenv("ROLE_TITLE"); v != "" { return v }; return "Senior Backend Engineer" }()
	tnURL       = func() string { if v := os.Getenv("TRUENORTH_BASE_URL"); v != "" { return v }; return "http://localhost:8000" }()
	tnKey       = os.Getenv("TRUENORTH_API_KEY")
	port        = func() string { if v := os.Getenv("PORT"); v != "" { return v }; return "8081" }()
)

var tn *truenorth.TrueNorth

type Session struct{ SessionID, CandidateID string; StartedAt time.Time; Pct float64 }
type CandidateResult struct {
	SessionID, CandidateID, ScreenedAt, Name, Recommendation string
	OverallScore, TechScore, CultureScore                     int
	Strengths, Concerns, RedFlags                             []string
	Full                                                      map[string]interface{} `json:"full,omitempty"`
}

var (
	mu         sync.RWMutex
	sessions   = map[string]*Session{}
	results    []*CandidateResult
)

func toStrSlice(v interface{}) []string {
	if v == nil { return nil }
	if sl, ok := v.([]interface{}); ok { var out []string; for _, i := range sl { out = append(out, fmt.Sprintf("%v", i)) }; return out }
	return nil
}
func toInt(v interface{}) int {
	if v == nil { return 0 }; if f, ok := v.(float64); ok { return int(f) }; return 0
}

func parseResult(sid, cid string, c map[string]interface{}) *CandidateResult {
	name := fmt.Sprintf("%v", c["candidate_name"]); if name == "<nil>" { name = "Unknown" }
	rec  := strings.ToUpper(fmt.Sprintf("%v", c["recommendation"]))
	return &CandidateResult{
		SessionID: sid, CandidateID: cid, ScreenedAt: time.Now().Format(time.RFC3339),
		Name: name, Recommendation: rec, OverallScore: toInt(c["overall_score"]),
		TechScore: toInt(c["technical_score"]), CultureScore: toInt(c["culture_score"]),
		Strengths: toStrSlice(c["strengths"]), Concerns: toStrSlice(c["concerns"]),
		RedFlags: toStrSlice(c["red_flags"]), Full: c,
	}
}

func allResults() []*CandidateResult {
	mu.RLock(); defer mu.RUnlock()
	out := make([]*CandidateResult, len(results)); copy(out, results)
	sort.Slice(out, func(i, j int) bool { return out[i].OverallScore > out[j].OverallScore })
	return out
}

func initTN() {
	tn = truenorth.NewClient(tnKey, tnURL)
}

func startHandler(store interface{}) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct{ CandidateID string `json:"candidate_id"` }
		c.ShouldBindJSON(&req)
		if req.CandidateID == "" { req.CandidateID = fmt.Sprintf("c_%d", time.Now().UnixMilli()) }
		sessID := fmt.Sprintf("hr_%s_%d", req.CandidateID, time.Now().UnixMilli())
		session, err := tn.Sessions.Create(context.Background(), "hr_screener", &truenorth.CreateSessionOptions{SessionID: sessID})
		if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
		mu.Lock(); sessions[sessID] = &Session{SessionID: sessID, CandidateID: req.CandidateID, StartedAt: time.Now()}; mu.Unlock()
		c.JSON(200, gin.H{"session_id": sessID, "agent_message": session.AgentMessage})
	}
}

func messageHandler(store interface{}) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct{ SessionID string `json:"session_id"`; Text string `json:"text"` }
		c.ShouldBindJSON(&req)
		if req.SessionID == "" || req.Text == "" { c.JSON(400, gin.H{"error": "required"}); return }
		mu.RLock(); meta := sessions[req.SessionID]; mu.RUnlock()
		if meta == nil { c.JSON(404, gin.H{"error": "not found"}); return }
		result, err := tn.Sessions.Message(context.Background(), req.SessionID, req.Text)
		if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
		resp := gin.H{"text": result.Text, "is_complete": result.IsComplete, "completion_pct": result.CompletionPct}
		if result.IsComplete && result.Output != nil {
			var content map[string]interface{}
			raw, _ := json.Marshal(result.Output.Content); json.Unmarshal(raw, &content)
			cr := parseResult(req.SessionID, meta.CandidateID, content)
			mu.Lock(); results = append(results, cr); delete(sessions, req.SessionID); mu.Unlock()
			tn.Sessions.End(context.Background(), req.SessionID)
			log.Printf("✅ %s → %s (%d/100)", cr.Name, cr.Recommendation, cr.OverallScore)
			resp["scorecard"] = cr
		}
		c.JSON(200, resp)
	}
}

func main() {
	initTN()
	fmt.Printf("\n  HR Screener (Go) — %s / %s\n", companyName, roleTitle)
	if resp, err := http.Get(tnURL+"/health"); err != nil {
		log.Printf("⚠  TrueNorth not reachable at %s", tnURL)
	} else { resp.Body.Close(); log.Printf("✓ TrueNorth @ %s", tnURL) }
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/health", func(c *gin.Context) {
		mu.RLock(); active := len(sessions); mu.RUnlock()
		c.JSON(200, gin.H{"status":"ok","role":roleTitle,"active":active,"screened":len(allResults())})
	})
	r.GET("/api/candidates", func(c *gin.Context) { all := allResults(); c.JSON(200, gin.H{"candidates":all,"total":len(all)}) })
	r.POST("/api/sessions/start",   startHandler(nil))
	r.POST("/api/sessions/message", messageHandler(nil))
	r.GET("/dashboard", func(c *gin.Context) {
		c.Header("Content-Type","text/html")
		c.String(200, "<h1>HR Screener — "+companyName+"</h1><p>"+roleTitle+"</p><p><a href=/api/candidates>Candidates JSON</a></p>")
	})
	log.Printf("http://localhost:%s/dashboard", port)
	r.Run(":"+port)
}
