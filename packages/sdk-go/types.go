package truenorth

// SessionResponse is returned when creating a session.
type SessionResponse struct {
	SessionID      string `json:"session_id"`
	GoalID         string `json:"goal_id"`
	WelcomeMessage string `json:"welcome_message"`
	IsResumed      bool   `json:"is_resumed"`
}

// MessageResponse is returned when sending a message.
type MessageResponse struct {
	SessionID    string                 `json:"session_id"`
	Response     string                 `json:"response"`
	EmotionState string                 `json:"emotion_state"`
	IsComplete   bool                   `json:"is_complete"`
	IsEscalated  bool                   `json:"is_escalated"`
	Profile      map[string]interface{} `json:"profile"`
	CostUSD      float64                `json:"cost_usd"`
}

// Session represents an active conversation session.
type Session struct {
	SessionID      string
	GoalID         string
	WelcomeMessage string
	IsResumed      bool
	client         *Client
}

// Send sends a message in this session.
func (s *Session) Send(ctx interface{ Done() <-chan struct{} }, message string) (*MessageResponse, error) {
	// ctx is context.Context
	import_ctx := ctx.(interface {
		Done() <-chan struct{}
		Err() error
	})
	_ = import_ctx
	return s.client.SendMessage(nil, s.SessionID, message)
}
