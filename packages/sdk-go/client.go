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
	return &out, s.t.post(ctx, "/sessions", body, &out)
}

func (s *SessionsClient) Message(ctx context.Context, sessionID, text string) (*MessageResult, error) {
	var out MessageResult
	return &out, s.t.post(ctx, "/sessions/"+sessionID+"/message",
		map[string]string{"text": text}, &out)
}

func (s *SessionsClient) Get(ctx context.Context, sessionID string) (*Session, error) {
	var out Session
	return &out, s.t.get(ctx, "/sessions/"+sessionID, nil, &out)
}

func (s *SessionsClient) Output(ctx context.Context, sessionID string) (*Output, error) {
	var wrapper struct {
		Output    *Output `json:"output"`
		SessionID string  `json:"session_id"`
	}
	if err := s.t.get(ctx, "/sessions/"+sessionID+"/output", nil, &wrapper); err != nil {
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

func (s *SessionsClient) ForceOutput(ctx context.Context, sessionID string) (*Output, error) {
	var wrapper struct {
		Output *Output `json:"output"`
		Text   string  `json:"text"`
	}
	if err := s.t.post(ctx, "/sessions/"+sessionID+"/force-output", nil, &wrapper); err != nil {
		return nil, err
	}
	return wrapper.Output, nil
}

func (s *SessionsClient) End(ctx context.Context, sessionID string) error {
	return s.t.delete(ctx, "/sessions/"+sessionID)
}

type GoalsClient struct {
	t *transport
}

func (g *GoalsClient) List(ctx context.Context, query, sector string, limit int) ([]Goal, error) {
	params := url.Values{}
	if query  != "" { params.Set("q", query)                       }
	if sector != "" { params.Set("sector", sector)                 }
	if limit   > 0  { params.Set("limit", strconv.Itoa(limit))     }
	var out []Goal
	return out, g.t.get(ctx, "/goals", params, &out)
}

func (g *GoalsClient) Get(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" {
		version = "latest"
	}
	params := url.Values{"version": {version}}
	var out Goal
	return &out, g.t.get(ctx, "/goals/"+name, params, &out)
}

func (g *GoalsClient) Install(ctx context.Context, name, version string) (*Goal, error) {
	if version == "" {
		version = "latest"
	}
	var out Goal
	return &out, g.t.post(ctx, "/goals/"+name+"/install",
		map[string]string{"version": version}, &out)
}

type AnalyticsClient struct {
	t *transport
}

func (a *AnalyticsClient) Cost(ctx context.Context, goalID string, periodDays int) (*CostSummary, error) {
	if periodDays == 0 {
		periodDays = 7
	}
	params := url.Values{
		"goal":   {goalID},
		"period": {strconv.Itoa(periodDays)},
	}
	var out CostSummary
	return &out, a.t.get(ctx, "/analytics/cost", params, &out)
}

func (a *AnalyticsClient) Health(ctx context.Context, goalID string, windowHours int) (*GoalHealthReport, error) {
	if windowHours == 0 {
		windowHours = 24
	}
	params := url.Values{
		"goal":   {goalID},
		"window": {strconv.Itoa(windowHours)},
	}
	var out GoalHealthReport
	return &out, a.t.get(ctx, "/analytics/health", params, &out)
}
