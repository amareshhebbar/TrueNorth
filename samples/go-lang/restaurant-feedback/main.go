// restaurant-feedback (Go) — Web Chat Feedback + Manager Dashboard
//
// WHAT THIS DOES
// ──────────────────────────────────────────────────────────────
// A web-based conversational feedback system for restaurants.
// Guest scans a QR code at the table → opens a chat in their phone browser
// → answers questions naturally → manager sees live dashboard.
//
// Features:
//   • Mobile-first chat UI — works on any smartphone browser
//   • Manager dashboard at /dashboard with NPS, scores, issues, praise
//   • No app download — pure HTML/CSS/JS served from Go
//   • Concurrent sessions — 100 guests can chat simultaneously
//   • Auto-calculates NPS from scores
//
// This is the Go equivalent of restaurant-feedback/app.py.
// Python used Flask; this uses Go's stdlib net/http — no framework needed.
//
// FILE STRUCTURE
// ──────────────────────────────────────────────────────────────
//   restaurant-feedback/
//   ├── main.go      ← this file (server + embedded HTML)
//   ├── goal.yaml    ← feedback questionnaire (copy from python version)
//   └── go.mod
//
// go.mod:
// ──────────────────────────────────────────────────────────────
//   module github.com/truenorth-ai/restaurant-feedback
//   go 1.22
//   require github.com/truenorth-ai/truenorth-go v0.1.0
//
// INSTALL
// ──────────────────────────────────────────────────────────────
//   cd samples/go-lang/restaurant-feedback
//   go mod download
//   go build -o feedback .
//
// HOW TO RUN
// ──────────────────────────────────────────────────────────────
//   # Step 1: TrueNorth Python API
//   cd packages/core && uvicorn truenorth.api.main:app --port 8000
//
//   # Step 2: This server
//   export TRUENORTH_BASE_URL=http://localhost:8000
//   export RESTAURANT_NAME="The Spice Garden"
//   ./feedback
//
//   # Guest link (share or print as QR code)
//   open http://localhost:5001/
//
//   # Manager dashboard
//   open http://localhost:5001/dashboard
//
// QR CODE FOR TABLES
// ──────────────────────────────────────────────────────────────
//   Install qrencode: sudo dnf install qrencode   (Fedora)
//                     brew install qrencode        (Mac)
//
//   qrencode -o table-qr.png "http://YOUR_IP:5001/"
//   Print and laminate. One per table.
//
// API ENDPOINTS
// ──────────────────────────────────────────────────────────────
//   GET  /              → Guest chat UI (mobile-first HTML)
//   GET  /dashboard     → Manager dashboard (NPS, issues, highlights)
//   GET  /health        → Health check
//   POST /api/start     → Start a feedback session
//   POST /api/message   → Continue a feedback session
//   GET  /api/stats     → Raw stats (JSON)

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

// ── Config ────────────────────────────────────────────────────────────────────

var (
	restaurantName = cfg("RESTAURANT_NAME", "The Spice Garden")
	goalID         = cfg("GOAL_ID",         "restaurant_feedback")
	tnURL          = cfg("TRUENORTH_BASE_URL","http://localhost:8000")
	tnKey          = cfg("TRUENORTH_API_KEY","")
	port           = cfg("PORT",            "5001")
)

func cfg(k, def string) string {
	if v := os.Getenv(k); v != "" { return v }
	return def
}

// ── Types ─────────────────────────────────────────────────────────────────────

type GuestSession struct {
	SessionID  string
	StartedAt  time.Time
	LastActive time.Time
	Pct        float64
}

type FeedbackResult struct {
	SessionID    string                 `json:"session_id"`
	Timestamp    string                 `json:"timestamp"`
	OverallScore float64                `json:"overall_score"`
	NPS          int                    `json:"nps_score"`
	NPSCategory  string                 `json:"nps_category"`
	Sentiment    string                 `json:"sentiment"`
	PriorityIssue string               `json:"priority_issue"`
	Compliment   string                 `json:"compliment_to_share"`
	ActionNeeded bool                   `json:"manager_action_required"`
	Full         map[string]interface{} `json:"full"`
}

// ── Store ─────────────────────────────────────────────────────────────────────

type Store struct {
	mu       sync.RWMutex
	sessions map[string]*GuestSession
	results  []*FeedbackResult
}

func newStore() *Store { return &Store{sessions: make(map[string]*GuestSession)} }

func (s *Store) getSession(id string) (*GuestSession, bool) {
	s.mu.RLock(); defer s.mu.RUnlock()
	v, ok := s.sessions[id]; return v, ok
}
func (s *Store) setSession(id string, v *GuestSession) {
	s.mu.Lock(); defer s.mu.Unlock(); s.sessions[id] = v
}
func (s *Store) deleteSession(id string) {
	s.mu.Lock(); defer s.mu.Unlock(); delete(s.sessions, id)
}
func (s *Store) addResult(r *FeedbackResult) {
	s.mu.Lock(); defer s.mu.Unlock(); s.results = append(s.results, r)
}
func (s *Store) getResults() []*FeedbackResult {
	s.mu.RLock(); defer s.mu.RUnlock()
	out := make([]*FeedbackResult, len(s.results)); copy(out, s.results); return out
}
func (s *Store) sessionCount() int {
	s.mu.RLock(); defer s.mu.RUnlock(); return len(s.sessions)
}

// ── TrueNorth client ──────────────────────────────────────────────────────────

var tn *truenorth.Client
var store *Store

// ── Random session ID ─────────────────────────────────────────────────────────

func newSID() string {
	b := make([]byte, 6); rand.Read(b)
	return "rf_" + hex.EncodeToString(b)
}

// ── Parse feedback result ─────────────────────────────────────────────────────

func parseResult(sid string, content map[string]interface{}) *FeedbackResult {
	nps := 0
	if v, ok := content["nps_score"].(float64); ok { nps = int(v) }

	npsCategory := "Passive"
	switch {
	case nps >= 9: npsCategory = "Promoter"
	case nps <= 6: npsCategory = "Detractor"
	}

	overall := 0.0
	if v, ok := content["overall_score"].(float64); ok { overall = v }

	return &FeedbackResult{
		SessionID:     sid,
		Timestamp:     time.Now().Format(time.RFC3339),
		OverallScore:  overall,
		NPS:           nps,
		NPSCategory:   npsCategory,
		Sentiment:     strVal(content["sentiment"]),
		PriorityIssue: strVal(content["priority_issue"]),
		Compliment:    strVal(content["compliment_to_share"]),
		ActionNeeded:  boolVal(content["manager_action_required"]),
		Full:          content,
	}
}

func strVal(v interface{}) string  { if v == nil { return "" }; return fmt.Sprintf("%v", v) }
func boolVal(v interface{}) bool   {
	if b, ok := v.(bool); ok { return b }
	return strings.ToLower(fmt.Sprintf("%v", v)) == "true"
}

// ── Guest chat HTML (embedded — no separate file needed) ─────────────────────

const guestHTML = `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>{{.Restaurant}} — Feedback</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     min-height:100vh;background:linear-gradient(135deg,#78350f,#92400e,#b45309);
     display:flex;flex-direction:column;align-items:center;padding:16px;}
.hero{color:white;text-align:center;padding:16px 0 12px;}
.hero h1{font-size:1.5rem;font-weight:800;}
.hero p{font-size:.85rem;opacity:.8;margin-top:4px;}
.card{background:white;border-radius:20px;width:100%;max-width:420px;
      display:flex;flex-direction:column;height:72vh;min-height:380px;
      box-shadow:0 20px 60px rgba(0,0,0,.35);}
.progress-bar{height:3px;background:#fef3c7;}
.progress-fill{height:100%;background:#f59e0b;transition:width .5s ease;width:0%;}
.messages{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px;}
.bubble{padding:10px 13px;border-radius:15px;max-width:87%;font-size:.88rem;
        line-height:1.55;word-wrap:break-word;}
.bot{background:#fef3c7;color:#92400e;align-self:flex-start;border-bottom-left-radius:4px;}
.user{background:#f59e0b;color:white;align-self:flex-end;border-bottom-right-radius:4px;}
.typing{display:flex;gap:4px;align-items:center;padding:10px 13px;
        background:#fef3c7;border-radius:15px;border-bottom-left-radius:4px;align-self:flex-start;}
.dot{width:6px;height:6px;background:#d97706;border-radius:50%;animation:b .9s infinite;}
.dot:nth-child(2){animation-delay:.15s;}.dot:nth-child(3){animation-delay:.3s;}
@keyframes b{0%,80%,100%{transform:translateY(0);}40%{transform:translateY(-6px);}}
.input-row{padding:10px 12px;border-top:1px solid #fde68a;display:flex;gap:7px;}
input{flex:1;border:1.5px solid #fde68a;border-radius:11px;padding:9px 13px;
      font-size:.88rem;outline:none;background:#fffbf0;}
input:focus{border-color:#f59e0b;}
button{background:#f59e0b;color:white;border:none;border-radius:11px;
       padding:9px 16px;font-weight:600;font-size:.88rem;cursor:pointer;}
button:disabled{opacity:.4;cursor:default;}
.done{text-align:center;padding:30px 20px;}
.done .emoji{font-size:3.5rem;margin-bottom:12px;}
.done h2{color:#92400e;font-size:1.2rem;margin-bottom:8px;}
.done p{color:#a16207;font-size:.88rem;line-height:1.65;}
.stars{color:#f59e0b;font-size:1.6rem;margin:6px 0;}
</style></head><body>
<div class="hero"><h1>🍽️ {{.Restaurant}}</h1>
<p>Share your experience — just 2 minutes</p></div>
<div class="card" id="card">
  <div class="progress-bar"><div class="progress-fill" id="prog"></div></div>
  <div class="messages" id="msgs"></div>
  <div class="input-row">
    <input id="inp" placeholder="Type your reply…" autocomplete="off"/>
    <button id="btn" onclick="send()">Send</button>
  </div>
</div>
<script>
let sid=null,done=false;
async function start(){
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({})});
  const d=await r.json();
  sid=d.session_id; addBubble('bot',d.text);
  document.getElementById('inp').focus();
}
async function send(){
  const inp=document.getElementById('inp');
  const text=inp.value.trim();
  if(!text||done)return;
  inp.value=''; addBubble('user',text);
  showTyping(); document.getElementById('btn').disabled=true;
  const r=await fetch('/api/message',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:sid,text})});
  const d=await r.json(); hideTyping();
  document.getElementById('prog').style.width=(d.completion||0)+'%';
  if(d.is_complete){
    done=true; addBubble('bot',d.text);
    setTimeout(()=>showDone(d.thank_you),700);
  } else {
    addBubble('bot',d.text);
    document.getElementById('btn').disabled=false;
    document.getElementById('inp').focus();
  }
}
function showDone(msg){
  document.getElementById('card').innerHTML=
    '<div class="done"><div class="emoji">🙏</div>'+
    '<div class="stars">★★★★★</div><h2>Thank you!</h2>'+
    '<p>'+(msg||'Your feedback helps us improve every day. See you again!')+
    '</p></div>';
}
let tel=null;
function addBubble(role,text){
  const msgs=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='bubble '+role; d.textContent=text;
  msgs.appendChild(d); msgs.scrollTop=msgs.scrollHeight;
}
function showTyping(){
  const msgs=document.getElementById('msgs');
  tel=document.createElement('div'); tel.className='typing';
  tel.innerHTML='<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  msgs.appendChild(tel); msgs.scrollTop=msgs.scrollHeight;
}
function hideTyping(){if(tel){tel.remove();tel=null;}}
document.getElementById('inp').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
start();
</script></body></html>`

// ── Dashboard HTML ────────────────────────────────────────────────────────────

const dashTmpl = `<!DOCTYPE html>
<html><head><title>{{.Restaurant}} Dashboard</title><meta charset="UTF-8">
<style>
body{font-family:-apple-system,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#fffbeb;}
h1{color:#92400e;margin-bottom:4px;}h2{color:#a16207;font-size:1rem;font-weight:500;margin-bottom:20px;}
.s{display:inline-block;background:white;border-radius:10px;padding:15px 20px;margin:5px;
   text-align:center;border:2px solid #fde68a;min-width:120px;}
.s h2{margin:0;font-size:1.8rem;color:#d97706;font-weight:700;}
.s p{margin:3px 0 0;color:#64748b;font-size:.78rem;}
.box{background:white;border-radius:12px;padding:16px;margin:16px 0;border:1px solid #fde68a;}
.box h3{color:#92400e;font-size:.95rem;margin-bottom:10px;}
li{margin:6px 0;font-size:.88rem;color:#374151;line-height:1.5;}
.none{color:#94a3b8;font-style:italic;}
</style></head><body>
<h1>🍽️ {{.Restaurant}}</h1>
<h2>{{.Date}} · Manager Dashboard</h2>
<div>
  <div class="s"><h2>{{.Total}}</h2><p>Responses today</p></div>
  <div class="s"><h2>{{printf "%.1f" .AvgScore}}/10</h2><p>Avg score</p></div>
  <div class="s"><h2>{{printf "%+.0f" .NPS}}</h2><p>NPS score</p></div>
  <div class="s" style="border-color:#bbf7d0"><h2 style="color:#16a34a">{{.Promoters}}</h2><p>Promoters (9-10)</p></div>
  <div class="s" style="border-color:#fecaca"><h2 style="color:#dc2626">{{.Detractors}}</h2><p>Detractors (0-6)</p></div>
  <div class="s"><h2>{{.ActiveSessions}}</h2><p>Currently chatting</p></div>
</div>
<div class="box">
  <h3>⚠️ Priority issues — needs manager attention</h3>
  {{if .Issues}}<ul>{{range .Issues}}<li>{{.}}</li>{{end}}</ul>
  {{else}}<p class="none">No critical issues today ✅</p>{{end}}
</div>
<div class="box">
  <h3>✅ Compliments — share with your team</h3>
  {{if .Praise}}<ul>{{range .Praise}}<li>"{{.}}"</li>{{end}}</ul>
  {{else}}<p class="none">Keep collecting feedback — highlights will appear here</p>{{end}}
</div>
<p><small><a href="/">← Guest feedback link</a> | <a href="/api/stats">Raw stats (JSON)</a></small></p>
</body></html>`

// ── Handlers ──────────────────────────────────────────────────────────────────

func handleGuest(w http.ResponseWriter, r *http.Request) {
	tmpl, _ := template.New("g").Parse(guestHTML)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.Execute(w, map[string]string{"Restaurant": restaurantName})
}

func handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", 405); return }
	w.Header().Set("Content-Type", "application/json")

	sid := newSID()
	ctx := context.Background()

	session, err := tn.Sessions.Create(ctx, truenorth.CreateSessionRequest{
		GoalID:    goalID,
		SessionID: sid,
	})
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	store.setSession(sid, &GuestSession{
		SessionID: sid, StartedAt: time.Now(), LastActive: time.Now(),
	})

	json.NewEncoder(w).Encode(map[string]interface{}{
		"session_id": sid,
		"text":       session.AgentMessage,
		"completion": 0,
	})
}

func handleMessage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", 405); return }
	w.Header().Set("Content-Type", "application/json")

	var req struct {
		SessionID string `json:"session_id"`
		Text      string `json:"text"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	if req.SessionID == "" || req.Text == "" {
		json.NewEncoder(w).Encode(map[string]string{"error": "session_id and text required"})
		return
	}

	meta, ok := store.getSession(req.SessionID)
	if !ok {
		json.NewEncoder(w).Encode(map[string]string{"error": "session not found"})
		return
	}

	ctx := context.Background()
	result, err := tn.Sessions.Message(ctx, truenorth.MessageRequest{
		SessionID: req.SessionID, Text: req.Text,
	})
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	meta.LastActive = time.Now()
	meta.Pct        = result.CompletionPct
	store.setSession(req.SessionID, meta)

	resp := map[string]interface{}{
		"text":       result.Text,
		"is_complete": result.IsComplete,
		"completion": result.CompletionPct,
	}

	if result.IsComplete && result.Output != nil {
		var content map[string]interface{}
		raw, _ := json.Marshal(result.Output.Content)
		json.Unmarshal(raw, &content)

		fr := parseResult(req.SessionID, content)
		store.addResult(fr)
		store.deleteSession(req.SessionID)
		tn.Sessions.End(ctx, req.SessionID)

		thankYou := strVal(content["response_message"])
		if thankYou == "" {
			thankYou = fmt.Sprintf("Thank you! See you again at %s. 😊", restaurantName)
		}
		resp["thank_you"] = thankYou

		log.Printf("✅ Feedback: score=%.1f NPS=%d action=%v",
			fr.OverallScore, fr.NPS, fr.ActionNeeded)
	}

	json.NewEncoder(w).Encode(resp)
}

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	results := store.getResults()

	totalScore, promoters, passives, detractors := 0.0, 0, 0, 0
	var issues, praise []string

	for _, res := range results {
		totalScore += res.OverallScore
		switch res.NPSCategory {
		case "Promoter":   promoters++
		case "Passive":    passives++
		case "Detractor":  detractors++
		}
		if res.ActionNeeded && res.PriorityIssue != "" {
			issues = append(issues, res.PriorityIssue)
		}
		if res.Compliment != "" {
			praise = append(praise, res.Compliment)
		}
	}

	avg := 0.0
	if len(results) > 0 { avg = totalScore / float64(len(results)) }

	nps := 0.0
	if total := promoters + passives + detractors; total > 0 {
		nps = float64(promoters-detractors) / float64(total) * 100
	}

	if len(issues) > 5  { issues = issues[:5] }
	if len(praise) > 5  { praise = praise[:5] }

	tmpl, _ := template.New("d").Parse(dashTmpl)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.Execute(w, map[string]interface{}{
		"Restaurant":     restaurantName,
		"Date":           time.Now().Format("02 January 2006"),
		"Total":          len(results),
		"AvgScore":       avg,
		"NPS":            nps,
		"Promoters":      promoters,
		"Detractors":     detractors,
		"ActiveSessions": store.sessionCount(),
		"Issues":         issues,
		"Praise":         praise,
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	h, err := tn.Health(context.Background())
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]string{
			"status": "truenorth_unreachable", "url": tnURL,
		})
		return
	}
	results := store.getResults()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok", "restaurant": restaurantName,
		"goal": goalID, "truenorth": h,
		"active_sessions": store.sessionCount(),
		"feedback_today":  len(results),
	})
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"results": store.getResults(),
		"total":   len(store.getResults()),
	})
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	store = newStore()
	tn    = truenorth.New(truenorth.Config{
		BaseURL: tnURL, APIKey: tnKey, Timeout: 90 * time.Second,
	})

	fmt.Printf("\n  Restaurant Feedback (Go) — %s\n", restaurantName)
	fmt.Printf("  TrueNorth API: %s\n", tnURL)

	if h, err := tn.Health(context.Background()); err != nil {
		fmt.Printf("  ⚠  API not reachable — start it first\n")
	} else {
		fmt.Printf("  ✓ Connected: %s\n", h.Status)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/",            handleGuest)
	mux.HandleFunc("/dashboard",   handleDashboard)
	mux.HandleFunc("/health",      handleHealth)
	mux.HandleFunc("/api/start",   handleStart)
	mux.HandleFunc("/api/message", handleMessage)
	mux.HandleFunc("/api/stats",   handleStats)

	fmt.Printf("\n  Guest link : http://localhost:%s/\n", port)
	fmt.Printf("  Dashboard  : http://localhost:%s/dashboard\n\n", port)

	log.Fatal(http.ListenAndServe(":"+port, mux))
}