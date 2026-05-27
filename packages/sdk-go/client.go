// Package truenorth is the Go SDK for the TrueNorth AI agent framework.
package truenorth

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Client is the TrueNorth API client.
type Client struct {
	ServerURL  string
	APIKey     string
	httpClient *http.Client
}

// New creates a new TrueNorth client.
func New(serverURL, apiKey string) *Client {
	return &Client{
		ServerURL:  serverURL,
		APIKey:     apiKey,
		httpClient: &http.Client{},
	}
}

// StartSession creates a new agent session.
func (c *Client) StartSession(ctx context.Context, goalID string, userID string) (*Session, error) {
	payload := map[string]string{"goal_id": goalID}
	if userID != "" {
		payload["user_id"] = userID
	}
	var resp SessionResponse
	if err := c.post(ctx, "/sessions", payload, &resp); err != nil {
		return nil, err
	}
	return &Session{
		SessionID:      resp.SessionID,
		GoalID:         resp.GoalID,
		WelcomeMessage: resp.WelcomeMessage,
		IsResumed:      resp.IsResumed,
		client:         c,
	}, nil
}

// SendMessage sends a message to a session.
func (c *Client) SendMessage(ctx context.Context, sessionID, message string) (*MessageResponse, error) {
	payload := map[string]string{"message": message}
	var resp MessageResponse
	if err := c.post(ctx, fmt.Sprintf("/sessions/%s/messages", sessionID), payload, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Health checks the server health.
func (c *Client) Health(ctx context.Context) (string, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.ServerURL+"/health", nil)
	c.setHeaders(req)
	res, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	var result map[string]string
	json.NewDecoder(res.Body).Decode(&result)
	return result["status"], nil
}

func (c *Client) post(ctx context.Context, path string, payload any, result any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.ServerURL+path, bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	c.setHeaders(req)
	res, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode >= 400 {
		return fmt.Errorf("truenorth: HTTP %d", res.StatusCode)
	}
	return json.NewDecoder(res.Body).Decode(result)
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
}
