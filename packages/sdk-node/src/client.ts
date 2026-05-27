import WebSocket from "ws";
import type { SessionResponse, MessageResponse, StreamChunk, TrueNorthConfig } from "./types.js";

export class TrueNorthClient {
  private serverUrl: string;
  private apiKey?: string;

  constructor(config: TrueNorthConfig | string) {
    if (typeof config === "string") {
      this.serverUrl = config.replace(/\/$/, "");
    } else {
      this.serverUrl = config.serverUrl.replace(/\/$/, "");
      this.apiKey = config.apiKey;
    }
  }

  async startSession(goalId: string, userId?: string): Promise<Session> {
    const res = await this._fetch("/sessions", {
      method: "POST",
      body: JSON.stringify({ goal_id: goalId, user_id: userId }),
    });
    return new Session(await res.json() as SessionResponse, this);
  }

  async resumeSession(goalId: string, sessionId: string): Promise<Session> {
    const res = await this._fetch("/sessions", {
      method: "POST",
      body: JSON.stringify({ goal_id: goalId, resume_session_id: sessionId }),
    });
    return new Session(await res.json() as SessionResponse, this);
  }

  async sendMessage(sessionId: string, message: string): Promise<MessageResponse> {
    const res = await this._fetch(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({})) as any;
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    return res.json() as Promise<MessageResponse>;
  }

  streamSession(sessionId: string, onChunk: (c: StreamChunk) => void, onClose?: () => void): WebSocket {
    const wsUrl = this.serverUrl.replace(/^http/, "ws") + `/ws/${sessionId}`;
    const ws = new WebSocket(wsUrl, { headers: this.apiKey ? { "X-API-Key": this.apiKey } : {} });
    ws.on("message", (data) => {
      try { onChunk(JSON.parse(data.toString())); }
      catch { onChunk({ response: data.toString() }); }
    });
    ws.on("close", () => onClose?.());
    return ws;
  }

  async health(): Promise<{ status: string }> {
    return (await this._fetch("/health")).json();
  }

  private _fetch(path: string, init: RequestInit = {}): Promise<Response> {
    return fetch(this.serverUrl + path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { "X-API-Key": this.apiKey } : {}),
        ...(init.headers ?? {}),
      },
    });
  }
}

export class Session {
  readonly sessionId: string;
  readonly goalId: string;
  readonly welcomeMessage: string;
  readonly isResumed: boolean;
  private _client: TrueNorthClient;

  constructor(data: SessionResponse, client: TrueNorthClient) {
    this.sessionId = data.session_id;
    this.goalId = data.goal_id;
    this.welcomeMessage = data.welcome_message;
    this.isResumed = data.is_resumed;
    this._client = client;
  }

  send(message: string): Promise<MessageResponse> {
    return this._client.sendMessage(this.sessionId, message);
  }

  stream(onChunk: (c: StreamChunk) => void, onClose?: () => void): WebSocket {
    return this._client.streamSession(this.sessionId, onChunk, onClose);
  }
}
