package truenorth

import "context"

// Send sends a user message and returns the agent response.
func (s *Session) SendMsg(ctx context.Context, message string) (*MessageResponse, error) {
	return s.client.SendMessage(ctx, s.SessionID, message)
}
