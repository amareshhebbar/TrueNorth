const BASE = process.env.NEXT_PUBLIC_TRUENORTH_API_URL ?? "http://localhost:8000";
const KEY  = process.env.NEXT_PUBLIC_TRUENORTH_API_KEY ?? "";

const headers = () => ({
  "Content-Type": "application/json",
  ...(KEY ? { "X-API-Key": KEY } : {}),
});

export const api = {
  async startSession(goalId: string) {
    const r = await fetch(`${BASE}/sessions`, {
      method: "POST", headers: headers(),
      body: JSON.stringify({ goal_id: goalId }),
    });
    return r.json();
  },

  async sendMessage(sessionId: string, message: string) {
    const r = await fetch(`${BASE}/sessions/${sessionId}/messages`, {
      method: "POST", headers: headers(),
      body: JSON.stringify({ message }),
    });
    return r.json();
  },

  async getAgentHealth(goalId: string, days = 7) {
    const r = await fetch(`${BASE}/analytics/health/${goalId}?days=${days}`, { headers: headers() });
    return r.json();
  },
};
