export interface SessionResponse {
  session_id: string; goal_id: string;
  welcome_message: string; is_resumed: boolean;
}
export interface MessageResponse {
  session_id: string; response: string; emotion_state: string;
  is_complete: boolean; is_escalated: boolean;
  profile: Record<string, unknown>; cost_usd: number;
}
export interface StreamChunk {
  response?: string; emotion_state?: string;
  is_complete?: boolean; is_escalated?: boolean;
  profile?: Record<string, unknown>; error?: string;
}
export interface TrueNorthConfig {
  serverUrl: string; apiKey?: string; timeout?: number;
}
