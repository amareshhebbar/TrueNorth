package truenorth

import (
	"fmt"
	"time"
)

type Session struct {
	ID               string                 `json:"session_id"`
	GoalID           string                 `json:"goal_id"`
	Status           string                 `json:"status"` 
	CurrentTurn      int                    `json:"current_turn"`
	CompletionPct    float64                `json:"completion_pct"`
	CollectedFields  map[string]interface{} `json:"collected_fields"`
	MissingRequired  []string               `json:"missing_required"`
	TotalCostUSD     float64                `json:"total_cost_usd"`
	IsComplete       bool                   `json:"is_complete"`
	DetectedLanguage string                 `json:"detected_language"`
	AgentMessage     string                 `json:"agent_message"`
	CreatedAt        float64                `json:"created_at"`
}

type MessageResult struct {
	SessionID       string                   `json:"session_id"`
	Turn            int                      `json:"turn"`
	Text            string                   `json:"text"`
	IsComplete      bool                     `json:"is_complete"`
	CompletionPct   float64                  `json:"completion_pct"`
	FieldsExtracted []map[string]interface{} `json:"fields_extracted"`
	CostUSD         float64                  `json:"cost_usd"`
	LatencyMs       int                      `json:"latency_ms"`
	EmotionDetected string                   `json:"emotion_detected"`
	Output          *Output                  `json:"output,omitempty"`
}

type Output struct {
	SessionID   string                 `json:"session_id"`
	GoalID      string                 `json:"goal_id"`
	Format      string                 `json:"format"` 
	Content     interface{}            `json:"content"`
	Fields      map[string]interface{} `json:"fields"`
	Metadata    map[string]interface{} `json:"metadata"`
	GeneratedAt float64                `json:"generated_at"`
}

type Goal struct {
	Name        string   `json:"name"`
	Version     string   `json:"version"`
	Description string   `json:"description"`
	Sector      string   `json:"sector"`
	Tags        []string `json:"tags"`
	Downloads   int      `json:"downloads"`
	Author      string   `json:"author"`
	License     string   `json:"license"`
}

type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
}

type CostSummary struct {
	GoalID             string                     `json:"goal_id"`
	PeriodDays         int                        `json:"period_days"`
	SessionCount       int                        `json:"session_count"`
	TotalCostUSD       float64                    `json:"total_cost_usd"`
	AvgCostPerSession  float64                    `json:"avg_cost_per_session"`
	ByModel            map[string]ModelCostDetail `json:"by_model"`
	ByTask             map[string]TaskCostDetail  `json:"by_task"`
}

type ModelCostDetail struct {
	CostUSD float64 `json:"cost_usd"`
	Tokens  int     `json:"tokens"`
	Calls   int     `json:"calls"`
}

type TaskCostDetail struct {
	CostUSD float64 `json:"cost_usd"`
	Calls   int     `json:"calls"`
	Pct     float64 `json:"pct"`
}

type GoalHealthReport struct {
	GoalID           string             `json:"goal_id"`
	WindowHours      int                `json:"window_hours"`
	SessionCount     int                `json:"session_count"`
	CompletedCount   int                `json:"completed_count"`
	CompletionRate   float64            `json:"completion_rate"`
	AvgTurns         float64            `json:"avg_turns"`
	AvgCostUSD       float64            `json:"avg_cost_usd"`
	P95LatencyMs     int                `json:"p95_latency_ms"`
	FieldSkipRates   map[string]float64 `json:"field_skip_rates"`
	AbandonmentMap   map[string]int     `json:"abandonment_map"`
	HallucinationRate float64           `json:"hallucination_rate"`
	GeneratedAt      time.Time          `json:"generated_at"`
}

type CreateSessionOptions struct {
	UserID     string                 `json:"user_id,omitempty"`
	SessionID  string                 `json:"session_id,omitempty"`
	BudgetUSD  float64                `json:"budget_usd,omitempty"`
	SeedFields map[string]interface{} `json:"seed_fields,omitempty"`
	Language   string                 `json:"language,omitempty"`
	TenantID   string                 `json:"tenant_id,omitempty"`
}

type APIError struct {
	StatusCode int    `json:"-"`
	ErrorCode  string `json:"error"`
	Message    string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("TrueNorthError(%d): %s — %s", e.StatusCode, e.ErrorCode, e.Message)
}