package truenorth

import (
	"context"
	"fmt"
	"net/url"
	"strconv"
)

type SessionsClient struct {
	t *transport
}

// Create starts a new conversation session for the given goalID
func (s *SessionsClient) Create(ctx context.Context, goalID string, opts *CreateSessionOptions) (*Session, error) {
	body := map[string]interface{}{"goal_id": goalID}
	if opts != nil {
		if opts.UserID    != "" { body["user_id"]    = opts.UserID    }
		if opts.SessionID != "" { body["session_id"] = opts.SessionID }
		if opts.TenantID  != "" { body["tenant_id"]  = opts.TenantID  }
		if opts.Language  != "" { body["language"]   = opts.Language  }
		if opts.BudgetUSD  > 0  { body["budget_usd"] = opts.BudgetUSD }
		if opts.SeedFields != nil { body["seed_fields"] = opts.SeedFields }
	}
	var out Session
	return &out, s.t.post(ctx, "/v1/sessions", body, &out)
}

// Message sends a user message to the session and returns the agent response
func (s *SessionsClient) Message(ctx context.Context, sessionID, text string) (*MessageResult, error) {
	var out MessageResult
	return &out, s.t.post(ctx, "/v1/sessions/"+sessionID+"/message",
		map[string]string{"text": text}, &out)
}

// Get retrieves the current state of a session
func (s *SessionsClient) Get(ctx context.Context, sessionID string) (*Session, error) {
	var out Session
	return &out, s.t.get(ctx, "/v1/sessions/"+sessionID, nil, &out)
}

// Output returns the final structured output
func (s *SessionsClient) Output(ctx context.Context, sessionID string) (*Output, error) {
	var wrapper struct {
		Output    *Output `json:"output"`
		SessionID string  `json:"session_id"`
	}
	if err := s.t.get(ctx, "/v1/sessions/"+sessionID+"/output", nil, &wrapper); err != nil {
		return nil, err
	}
	if wrapper.Output == nil {
		return nil, &APIError{
			StatusCode: 404,
			ErrorCode:  "no_output",
			Message:    fmt.Sprintf("no output in response for session %s", sessionID),
		}
	}
	return wrapper.Output, nil
}

// ForceOutput generates output from whatever fields have been collected so far
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

// End terminates and cleans up a session
func (s *SessionsClient) End(ctx context.Context, sessionID string) error {
	return s.t.delete(ctx, "/v1/sessions/"+sessionID)
}

type GoalsClient struct {
	t *transport
}

// List returns goals from the registry
func (g *GoalsClient) List(ctx context.Context, query, sector string, limit int) ([]Goal, error) {
	params := url.Values{}
	if query  != "" { params.Set("q", query)                       }
	if sector != "" { params.Set("sector", sector)                 }
	if limit   > 0  { params.Set("limit", strconv.Itoa(limit))     }
	var out []Goal
	return out, g.t.get(ctx, "/v1/goals", params, &out)
}

// Get returns full details for a specific goal version
func (g *GoalsClient) Get(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" {
		version = "latest"
	}
	params := url.Values{"version": {version}}
	var out Goal
	return &out, g.t.get(ctx, "/v1/goals/"+name, params, &out)
}

// Install downloads and installs a goal from the registry
func (g *GoalsClient) Install(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" {
		version = "latest"
	}
	var out Goal
	return &out, g.t.post(ctx, "/v1/goals/"+name+"/install",
		map[string]string{"version": version}, &out)
}

// AnalyticsClient provides cost and health analytics
type AnalyticsClient struct {
	t *transport
}

// Cost returns cost analytics for a goal over the given period
func (a *AnalyticsClient) Cost(ctx context.Context, goalID string, periodDays int) (*CostSummary, error) {
	if periodDays == 0 {
		periodDays = 7
	}
	params := url.Values{
		"goal":   {goalID},
		"period": {strconv.Itoa(periodDays)},
	}
	var out CostSummary
	return &out, a.t.get(ctx, "/v1/analytics/cost", params, &out)
}

// Health returns health metrics for a goal over the given window
func (a *AnalyticsClient) Health(ctx context.Context, goalID string, windowHours int) (*GoalHealthReport, error) {
	if windowHours == 0 {
		windowHours = 24
	}
	params := url.Values{
		"goal":   {goalID},
		"window": {strconv.Itoa(windowHours)},
	}
	var out GoalHealthReport
	return &out, a.t.get(ctx, "/v1/analytics/health", params, &out)
}