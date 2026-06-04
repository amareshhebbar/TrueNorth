package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	truenorth "github.com/truenorth-ai/truenorth-go"
)

var (
	restaurantName = func() string { if v := os.Getenv("RESTAURANT_NAME"); v != "" { return v }; return "The Spice Garden" }()
	goalID         = func() string { if v := os.Getenv("GOAL_ID"); v != "" { return v }; return "restaurant_feedback" }()
	tnURL          = func() string { if v := os.Getenv("TRUENORTH_BASE_URL"); v != "" { return v }; return "http://localhost:8000" }()
	tnKey          = os.Getenv("TRUENORTH_API_KEY")
	port           = func() string { if v := os.Getenv("PORT"); v != "" { return v }; return "5001" }()
)

var tn *truenorth.TrueNorth

type GuestSession struct{ SessionID string; StartedAt time.Time; Pct float64 }
type FeedbackResult struct {
	SessionID    string                 `json:"session_id"`
	Timestamp    string                 `json:"timestamp"`
	OverallScore float64                `json:"overall_score"`
	NPS          int                    `json:"nps"`
	NPSCategory  string                 `json:"nps_category"`
	PriorityIssue string               `json:"priority_issue"`
	Compliment   string                 `json:"compliment"`
	ActionNeeded bool                   `json:"action_needed"`
}

var (
	mu      sync.RWMutex
	sessions = map[string]*GuestSession{}
	results  []*FeedbackResult
)

func newSID() string { b := make([]byte, 6); rand.Read(b); return "rf_" + hex.EncodeToString(b) }

func parseResult(sid string, content map[string]interface{}) *FeedbackResult {
	nps := 0; if v, ok := content["nps_score"].(float64); ok { nps = int(v) }
	cat := "Passive"; if nps >= 9 { cat = "Promoter" } else if nps <= 6 { cat = "Detractor" }
	score := 0.0; if v, ok := content["overall_score"].(float64); ok { score = v }
	return &FeedbackResult{
		SessionID: sid, Timestamp: time.Now().Format(time.RFC3339),
		OverallScore: score, NPS: nps, NPSCategory: cat,
		PriorityIssue: fmt.Sprintf("%v", content["priority_issue"]),
		Compliment:    fmt.Sprintf("%v", content["compliment_to_share"]),
		ActionNeeded:  content["manager_action_required"] == true,
	}
}

const guestHTML = `<!DOCTYPE html><html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Restaurant Feedback</title>
<style>body{font-family:sans-serif;background:#78350f;display:flex;flex-direction:column;align-items:center;padding:16px;min-height:100vh}
.card{background:white;border-radius:16px;width:100%;max-width:400px;display:flex;flex-direction:column;height:70vh}
.msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.bot{background:#fef3c7;color:#92400e;align-self:flex-start;padding:10px;border-radius:12px;max-width:85%;font-size:.9rem}
.user{background:#f59e0b;color:white;align-self:flex-end;padding:10px;border-radius:12px;max-width:85%;font-size:.9rem}
.row{padding:10px;border-top:1px solid #fde68a;display:flex;gap:8px}
input{flex:1;border:1.5px solid #fde68a;border-radius:10px;padding:8px;font-size:.9rem;outline:none}
button{background:#f59e0b;color:white;border:none;border-radius:10px;padding:8px 16px;font-weight:600;cursor:pointer}
h1{color:white;text-align:center;margin-bottom:8px}</style></head>
<body><h1>🍽️ RESTAURANT_NAME_PLACEHOLDER</h1>
<div class="card"><div class="msgs" id="msgs"></div>
<div class="row"><input id="inp" placeholder="Type reply…" autocomplete="off"/>
<button onclick="send()">Send</button></div></div>
<script>
let sid=null,done=false;
async function start(){const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const d=await r.json();sid=d.session_id;add('bot',d.text);}
async function send(){const inp=document.getElementById('inp');const text=inp.value.trim();if(!text||done)return;inp.value='';add('user',text);
const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,text})});const d=await r.json();
if(d.is_complete){done=true;add('bot',d.text);setTimeout(()=>{document.querySelector('.card').innerHTML='<div style="text-align:center;padding:40px"><h2>🙏 Thank you!</h2><p>'+(d.thank_you||'See you again!')+'</p></div>'},600);}else{add('bot',d.text);}}
function add(role,text){const m=document.getElementById('msgs');const d=document.createElement('div');d.className=role;d.textContent=text;m.appendChild(d);m.scrollTop=m.scrollHeight;}
document.getElementById('inp').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
start();
</script></body></html>`

func handleGuest(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, strings.ReplaceAll(guestHTML, "RESTAURANT_NAME_PLACEHOLDER", restaurantName))
}

func handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", 405); return }
	w.Header().Set("Content-Type", "application/json")
	sid := newSID()
	session, err := tn.Sessions.Create(context.Background(), goalID, &truenorth.CreateSessionOptions{SessionID: sid})
	if err != nil { json.NewEncoder(w).Encode(map[string]string{"error": err.Error()}); return }
	mu.Lock(); sessions[sid] = &GuestSession{SessionID: sid, StartedAt: time.Now()}; mu.Unlock()
	json.NewEncoder(w).Encode(map[string]interface{}{"session_id": sid, "text": session.AgentMessage, "completion": 0})
}

func handleMessage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", 405); return }
	w.Header().Set("Content-Type", "application/json")
	var req struct{ SessionID string `json:"session_id"`; Text string `json:"text"` }
	json.NewDecoder(r.Body).Decode(&req)
	if req.SessionID == "" || req.Text == "" { json.NewEncoder(w).Encode(map[string]string{"error": "required"}); return }
	mu.RLock(); meta := sessions[req.SessionID]; mu.RUnlock()
	if meta == nil { json.NewEncoder(w).Encode(map[string]string{"error": "not found"}); return }
	result, err := tn.Sessions.Message(context.Background(), req.SessionID, req.Text)
	if err != nil { json.NewEncoder(w).Encode(map[string]string{"error": err.Error()}); return }
	resp := map[string]interface{}{"text": result.Text, "is_complete": result.IsComplete, "completion": result.CompletionPct}
	if result.IsComplete && result.Output != nil {
		var content map[string]interface{}
		raw, _ := json.Marshal(result.Output.Content); json.Unmarshal(raw, &content)
		fr := parseResult(req.SessionID, content)
		mu.Lock(); results = append(results, fr); delete(sessions, req.SessionID); mu.Unlock()
		tn.Sessions.End(context.Background(), req.SessionID)
		resp["thank_you"] = fmt.Sprintf("Thank you! See you again at %s 😊", restaurantName)
	}
	json.NewEncoder(w).Encode(resp)
}

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	mu.RLock(); total := len(results); active := len(sessions); mu.RUnlock()
	var avg float64; promoters, detractors := 0, 0
	mu.RLock()
	for _, res := range results {
		avg += res.OverallScore
		if res.NPSCategory == "Promoter" { promoters++ } else if res.NPSCategory == "Detractor" { detractors++ }
	}
	mu.RUnlock()
	if total > 0 { avg /= float64(total) }
	nps := 0.0; if total > 0 { nps = float64(promoters-detractors) / float64(total) * 100 }
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprintf(w, `<!DOCTYPE html><html><head><title>%s Dashboard</title></head><body>
<h1>🍽️ %s — Dashboard</h1>
<p>%s</p>
<div><b>Responses:</b> %d | <b>Avg score:</b> %.1f/10 | <b>NPS:</b> %+.0f | <b>Active:</b> %d</div>
<p><a href="/">Guest link</a> | <a href="/api/stats">Stats JSON</a></p>
</body></html>`, restaurantName, restaurantName, time.Now().Format("02 January 2006"), total, avg, nps, active)
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	mu.RLock(); defer mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"results": results, "total": len(results)})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	mu.RLock(); active := len(sessions); done := len(results); mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status":"ok","restaurant":restaurantName,"active":active,"feedback":done})
}

func main() {
	tn = truenorth.NewClient(tnKey, tnURL)
	fmt.Printf("\n  Restaurant Feedback (Go) — %s\n", restaurantName)
	if resp, err := http.Get(tnURL+"/health"); err != nil {
		log.Printf("⚠  TrueNorth not reachable at %s", tnURL)
	} else { resp.Body.Close(); log.Printf("✓ TrueNorth @ %s", tnURL) }
	mux := http.NewServeMux()
	mux.HandleFunc("/",            handleGuest)
	mux.HandleFunc("/dashboard",   handleDashboard)
	mux.HandleFunc("/health",      handleHealth)
	mux.HandleFunc("/api/start",   handleStart)
	mux.HandleFunc("/api/message", handleMessage)
	mux.HandleFunc("/api/stats",   handleStats)
	log.Printf("Guest:     http://localhost:%s/", port)
	log.Printf("Dashboard: http://localhost:%s/dashboard", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}

// keep strings import used
var _ = strings.Contains
