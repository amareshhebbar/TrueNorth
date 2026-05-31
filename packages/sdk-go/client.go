// Package truenorth provides the official Go client for the TrueNorth AI agent API.
//
// Install:
//   go get github.com/truenorth-ai/truenorth-go
//
// Usage:
//
//	client := truenorth.New(truenorth.Options{APIKey: "tn_live_..."})
//	session, err := client.Sessions.Create(ctx, "fitness-coach", nil)
//	result,  err := client.Sessions.Message(ctx, session.ID, "I am 28")
//	output,  err := client.Sessions.Output(ctx, session.ID)
//
// Same interface as Python SDK and Node SDK.
package truenorth

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"time"
)

const defaultBaseURL = "http://localhost:8000"
const defaultTimeout = 60 * time.Second
const sdkVersion     = "0.1.0"

// ─────────────────────────────────────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────────────────────────────────────

// Session represents an active or completed TrueNorth conversation.
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

// MessageResult is returned after sending a message to a session.
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
	Output          *Output                  `json:"output"`
}

// Output is the final structured output from a completed session.
type Output struct {
	SessionID   string                 `json:"session_id"`
	GoalID      string                 `json:"goal_id"`
	Format      string                 `json:"format"`
	Content     interface{}            `json:"content"`
	Fields      map[string]interface{} `json:"fields"`
	Metadata    map[string]interface{} `json:"metadata"`
	GeneratedAt float64                `json:"generated_at"`
}

// Goal represents a goal package from the registry.
type Goal struct {
	Name        string   `json:"name"`
	Version     string   `json:"version"`
	Description string   `json:"description"`
	Sector      string   `json:"sector"`
	Tags        []string `json:"tags"`
	Downloads   int      `json:"downloads"`
}

// APIError is returned when the server responds with 4xx or 5xx.
type APIError struct {
	StatusCode int
	ErrorCode  string `json:"error"`
	Message    string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("TrueNorthError(%d): %s — %s", e.StatusCode, e.ErrorCode, e.Message)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Transport
// ─────────────────────────────────────────────────────────────────────────────

type transport struct {
	baseURL string
	apiKey  string
	client  *http.Client
}

func newTransport(baseURL, apiKey string, timeout time.Duration) *transport {
	return &transport{
		baseURL: baseURL,
		apiKey:  apiKey,
		client:  &http.Client{Timeout: timeout},
	}
}

func (t *transport) headers() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	h.Set("Accept", "application/json")
	h.Set("User-Agent", "truenorth-go/"+sdkVersion)
	if t.apiKey != "" {
		h.Set("X-TrueNorth-Key", t.apiKey)
	}
	return h
}

func (t *transport) do(ctx context.Context, method, path string, body interface{}, params url.Values, out interface{}) error {
	rawURL := t.baseURL + path
	if len(params) > 0 {
		rawURL += "?" + params.Encode()
	}

	var bodyReader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("truenorth: marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, rawURL, bodyReader)
	if err != nil {
		return fmt.Errorf("truenorth: build request: %w", err)
	}
	req.Header = t.headers()

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("truenorth: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("truenorth: read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		var apiErr APIError
		apiErr.StatusCode = resp.StatusCode
		if err := json.Unmarshal(respBody, &apiErr); err != nil {
			apiErr.ErrorCode = "http_error"
			apiErr.Message   = string(respBody)
			if len(apiErr.Message) > 200 {
				apiErr.Message = apiErr.Message[:200]
			}
		}
		return &apiErr
	}

	if resp.StatusCode == 204 || out == nil {
		return nil
	}

	return json.Unmarshal(respBody, out)
}

func (t *transport) get(ctx context.Context, path string, params url.Values, out interface{}) error {
	return t.do(ctx, http.MethodGet, path, nil, params, out)
}

func (t *transport) post(ctx context.Context, path string, body interface{}, out interface{}) error {
	return t.do(ctx, http.MethodPost, path, body, nil, out)
}

func (t *transport) delete(ctx context.Context, path string) error {
	return t.do(ctx, http.MethodDelete, path, nil, nil, nil)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Sessions resource
// ─────────────────────────────────────────────────────────────────────────────

// SessionsClient provides session management operations.
type SessionsClient struct{ t *transport }

// CreateOptions are optional parameters for session creation.
type CreateOptions struct {
	UserID     string                 `json:"user_id,omitempty"`
	SessionID  string                 `json:"session_id,omitempty"`
	BudgetUSD  float64                `json:"budget_usd,omitempty"`
	SeedFields map[string]interface{} `json:"seed_fields,omitempty"`
	Language   string                 `json:"language,omitempty"`
}

// Create starts a new conversation session for the given goal.
func (s *SessionsClient) Create(ctx context.Context, goalID string, opts *CreateOptions) (*Session, error) {
	body := map[string]interface{}{"goal_id": goalID}
	if opts != nil {
		if opts.UserID    != "" { body["user_id"]    = opts.UserID    }
		if opts.SessionID != "" { body["session_id"] = opts.SessionID }
		if opts.BudgetUSD  > 0 { body["budget_usd"] = opts.BudgetUSD }
		if opts.Language  != "" { body["language"]   = opts.Language  }
		if opts.SeedFields != nil { body["seed_fields"] = opts.SeedFields }
	}
	var out Session
	return &out, s.t.post(ctx, "/v1/sessions", body, &out)
}

// Message sends a user message and returns the agent's response.
func (s *SessionsClient) Message(ctx context.Context, sessionID, text string) (*MessageResult, error) {
	var out MessageResult
	return &out, s.t.post(ctx, "/v1/sessions/"+sessionID+"/message", map[string]string{"text": text}, &out)
}

// Get retrieves the current state of a session.
func (s *SessionsClient) Get(ctx context.Context, sessionID string) (*Session, error) {
	var out Session
	return &out, s.t.get(ctx, "/v1/sessions/"+sessionID, nil, &out)
}

// Output returns the final structured output. Returns APIError(409) if not complete yet.
func (s *SessionsClient) Output(ctx context.Context, sessionID string) (*Output, error) {
	var wrapper struct {
		Output *Output `json:"output"`
	}
	if err := s.t.get(ctx, "/v1/sessions/"+sessionID+"/output", nil, &wrapper); err != nil {
		return nil, err
	}
	if wrapper.Output == nil {
		return nil, fmt.Errorf("truenorth: no output in response")
	}
	return wrapper.Output, nil
}

// ForceOutput generates output from whatever fields have been collected so far.
func (s *SessionsClient) ForceOutput(ctx context.Context, sessionID string) (*Output, error) {
	var wrapper struct {
		Output *Output `json:"output"`
		Text   string  `json:"text"`
	}
	if err := s.t.post(ctx, "/v1/sessions/"+sessionID+"/force-output", nil, &wrapper); err != nil {
		return nil, err
	}
	return wrapper.Output, nil
}

// End terminates and cleans up a session.
func (s *SessionsClient) End(ctx context.Context, sessionID string) error {
	return s.t.delete(ctx, "/v1/sessions/"+sessionID)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Goals resource
// ─────────────────────────────────────────────────────────────────────────────

// GoalsClient provides goal registry operations.
type GoalsClient struct{ t *transport }

// List returns goals from the registry, optionally filtered.
func (g *GoalsClient) List(ctx context.Context, query, sector string, limit int) ([]Goal, error) {
	params := url.Values{}
	if query  != "" { params.Set("q", query)           }
	if sector != "" { params.Set("sector", sector)     }
	if limit   > 0  { params.Set("limit", strconv.Itoa(limit)) }
	var out []Goal
	return out, g.t.get(ctx, "/v1/goals", params, &out)
}

// Get returns details for a specific goal version.
func (g *GoalsClient) Get(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" { version = "latest" }
	params := url.Values{"version": {version}}
	var out Goal
	return &out, g.t.get(ctx, "/v1/goals/"+name, params, &out)
}

// Install downloads and installs a goal from the registry.
func (g *GoalsClient) Install(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" { version = "latest" }
	var out Goal
	return &out, g.t.post(ctx, "/v1/goals/"+name+"/install", map[string]string{"version": version}, &out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Analytics resource
// ─────────────────────────────────────────────────────────────────────────────

// AnalyticsClient provides cost and health analytics.
type AnalyticsClient struct{ t *transport }

// Cost returns cost analytics for a goal over the given period.
func (a *AnalyticsClient) Cost(ctx context.Context, goalID string, periodDays int) (map[string]interface{}, error) {
	if periodDays == 0 { periodDays = 7 }
	params := url.Values{"goal": {goalID}, "period": {strconv.Itoa(periodDays)}}
	var out map[string]interface{}
	return out, a.t.get(ctx, "/v1/analytics/cost", params, &out)
}

// Health returns health metrics for a goal.
func (a *AnalyticsClient) Health(ctx context.Context, goalID string, windowHours int) (map[string]interface{}, error) {
	if windowHours == 0 { windowHours = 24 }
	params := url.Values{"goal": {goalID}, "window": {strconv.Itoa(windowHours)}}
	var out map[string]interface{}
	return out, a.t.get(ctx, "/v1/analytics/health", params, &out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Client
// ─────────────────────────────────────────────────────────────────────────────

// Options configure the TrueNorth client.
type Options struct {
	// APIKey is your tn_live_... or tn_test_... key.
	// Falls back to TRUENORTH_API_KEY environment variable.
	APIKey string

	// BaseURL is the TrueNorth API base URL.
	// Default: http://localhost:8000 or TRUENORTH_BASE_URL env var.
	BaseURL string

	// Timeout for each request. Default: 60s.
	Timeout time.Duration
}

// Client is the TrueNorth Go SDK client.
type Client struct {
	Sessions  *SessionsClient
	Goals     *GoalsClient
	Analytics *AnalyticsClient
	t         *transport
}

// New creates a new TrueNorth client.
//
//	client := truenorth.New(truenorth.Options{APIKey: os.Getenv("TRUENORTH_API_KEY")})
func New(opts Options) *Client {
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("TRUENORTH_API_KEY")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = os.Getenv("TRUENORTH_BASE_URL")
		if opts.BaseURL == "" {
			opts.BaseURL = defaultBaseURL
		}
	}
	if opts.Timeout == 0 {
		opts.Timeout = defaultTimeout
	}

	t := newTransport(opts.BaseURL, opts.APIKey, opts.Timeout)
	return &Client{
		Sessions:  &SessionsClient{t},
		Goals:     &GoalsClient{t},
		Analytics: &AnalyticsClient{t},
		t:         t,
	}
}

// Health checks if the API server is running.
func (c *Client) Health(ctx context.Context) (map[string]interface{}, error) {
	var out map[string]interface{}
	return out, c.t.get(ctx, "/health", nil, &out)
}

// RunSession is a convenience function that runs a full session with preset messages.
//
//	output, err := truenorth.RunSession(ctx, "fitness-coach", []string{
//	    "I am 28 years old",
//	    "65 kg",
//	    "lose weight",
//	}, truenorth.Options{APIKey: "tn_live_..."})
func RunSession(ctx context.Context, goalID string, messages []string, opts Options) (*Output, error) {
	client  := New(opts)
	session, err := client.Sessions.Create(ctx, goalID, nil)
	if err != nil { return nil, err }

	for _, msg := range messages {
		result, err := client.Sessions.Message(ctx, session.ID, msg)
		if err != nil { return nil, err }
		if result.IsComplete && result.Output != nil {
			return result.Output, nil
		}
	}
	return client.Sessions.ForceOutput(ctx, session.ID)
}