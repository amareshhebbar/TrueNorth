package truenorth

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

type transport struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

func (t *transport) request(ctx context.Context, method, path string, params url.Values, body interface{}, out interface{}) error {
	u, err := url.Parse(t.BaseURL + path)
	if err != nil {
		return err
	}
	if params != nil {
		u.RawQuery = params.Encode()
	}

	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, u.String(), reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	if t.APIKey != "" {
		req.Header.Set("X-TrueNorth-Key", t.APIKey)
	}

	resp, err := t.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		var apiErr APIError
		if err := json.NewDecoder(resp.Body).Decode(&apiErr); err == nil && apiErr.Message != "" {
			apiErr.StatusCode = resp.StatusCode
			return &apiErr
		}
		return &APIError{
			StatusCode: resp.StatusCode,
			ErrorCode:  "http_error",
			Message:    fmt.Sprintf("HTTP %d error", resp.StatusCode),
		}
	}

	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (t *transport) get(ctx context.Context, path string, params url.Values, out interface{}) error {
	return t.request(ctx, "GET", path, params, nil, out)
}

func (t *transport) post(ctx context.Context, path string, body interface{}, out interface{}) error {
	return t.request(ctx, "POST", path, nil, body, out)
}

func (t *transport) delete(ctx context.Context, path string) error {
	return t.request(ctx, "DELETE", path, nil, nil, nil)
}

type TrueNorth struct {
	Sessions  *SessionsClient
	Goals     *GoalsClient
	Analytics *AnalyticsClient
}

func NewClient(apiKey string, baseURL string) *TrueNorth {
	if baseURL == "" {
		baseURL = "https://api.truenorth.ai/v1"
	}
	t := &transport{
		BaseURL:    baseURL,
		APIKey:     apiKey,
		HTTPClient: http.DefaultClient,
	}
	return &TrueNorth{
		Sessions:  &SessionsClient{t: t},
		Goals:     &GoalsClient{t: t},
		Analytics: &AnalyticsClient{t: t},
	}
}
