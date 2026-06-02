import type {
  Session, MessageResult, Output, Goal,
  CostSummary, GoalHealthReport,
  CreateSessionOptions, TrueNorthClientOptions,
} from './types'
import { TrueNorthError } from './types'

export { TrueNorthError }

// ─── Raw API response types (snake_case from server) ─────────────────────────

interface RawSession {
  session_id: string; goal_id: string; status: string
  current_turn?: number; turn?: number
  completion_pct: number; collected_fields?: Record<string, unknown>
  missing_required?: string[]; missing_fields?: string[]
  total_cost_usd?: number; is_complete?: boolean
  detected_language?: string | null; agent_message?: string; created_at?: number
}

interface RawMessageResult {
  session_id: string; turn: number; text: string; is_complete: boolean
  completion_pct: number
  fields_extracted?: Array<{ field: string; value: unknown; confidence: number }>
  cost_usd: number; latency_ms: number; emotion_detected?: string | null
  output?: RawOutput | null
}

interface RawOutput {
  session_id: string; goal_id?: string; format?: string; content?: unknown
  fields?: Record<string, unknown>; metadata?: Record<string, unknown>
  generated_at?: number
}

// ─── Mappers ─────────────────────────────────────────────────────────────────

export function mapSession(r: RawSession): Session {
  return {
    id:               r.session_id,
    goalId:           r.goal_id,
    status:           r.status as Session['status'],
    currentTurn:      r.current_turn ?? r.turn ?? 0,
    completionPct:    r.completion_pct,
    collectedFields:  r.collected_fields ?? {},
    missingRequired:  r.missing_required ?? r.missing_fields ?? [],
    totalCostUsd:     r.total_cost_usd ?? 0,
    isComplete:       r.is_complete ?? false,
    detectedLanguage: r.detected_language ?? null,
    agentMessage:     r.agent_message ?? '',
    createdAt:        r.created_at ?? 0,
  }
}

export function mapMessage(r: RawMessageResult): MessageResult {
  return {
    sessionId:       r.session_id,
    turn:            r.turn,
    text:            r.text ?? '',
    isComplete:      r.is_complete,
    completionPct:   r.completion_pct,
    fieldsExtracted: r.fields_extracted ?? [],
    costUsd:         r.cost_usd,
    latencyMs:       r.latency_ms,
    emotionDetected: r.emotion_detected ?? null,
    output:          r.output ? mapOutput(r.output) : null,
  }
}

export function mapOutput(r: RawOutput): Output {
  return {
    sessionId:   r.session_id,
    goalId:      r.goal_id ?? '',
    format:      (r.format ?? 'json') as Output['format'],
    content:     r.content,
    fields:      r.fields ?? {},
    metadata:    r.metadata ?? {},
    generatedAt: r.generated_at ?? 0,
  }
}

// ─── HTTP Transport ───────────────────────────────────────────────────────────

export class Transport {
  readonly baseUrl: string
  // FIX: typed as HeadersInit-compatible object
  private readonly headers: Record<string, string>
  private readonly timeout: number

  constructor(opts: Required<TrueNorthClientOptions>) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '')
    this.timeout = opts.timeout
    this.headers = {
      'Content-Type': 'application/json',
      'Accept':       'application/json',
      ...(opts.apiKey ? { 'X-TrueNorth-Key': opts.apiKey } : {}),
    }
  }

  async request<T>(
    method:  string,
    path:    string,
    body?:   unknown,
    params?: Record<string, string | number | boolean | undefined>,
  ): Promise<T> {
    let url = this.baseUrl + path
    if (params) {
      const qs = Object.entries(params)
        .filter(([, v]) => v != null)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
      if (qs) url += '?' + qs
    }

    // FIX: AbortController is in DOM lib (tsconfig lib: ["ES2020", "DOM"])
    const ctrl  = new AbortController()
    // FIX: setTimeout / clearTimeout are in DOM lib or @types/node
    const timer = setTimeout(() => ctrl.abort(), this.timeout)

    try {
      // FIX: fetch is in DOM lib (tsconfig lib: ["ES2020", "DOM"])
      const resp = await fetch(url, {
        method,
        headers: this.headers,
        body:    body != null ? JSON.stringify(body) : undefined,
        signal:  ctrl.signal,
      })

      if (!resp.ok) {
        const text = await resp.text()
        let e: { error?: string; message?: string } = {}
        try { e = JSON.parse(text) } catch { /* ignore parse errors */ }
        throw new TrueNorthError(
          resp.status,
          e.error ?? 'http_error',
          e.message ?? text.slice(0, 200),
        )
      }

      if (resp.status === 204) return undefined as T
      return resp.json() as Promise<T>
    } finally {
      clearTimeout(timer)
    }
  }

  get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return this.request<T>('GET', path, undefined, params)
  }

  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>('POST', path, body)
  }

  delete<T>(path: string): Promise<T> {
    return this.request<T>('DELETE', path)
  }
}

// ─── Resource Clients ─────────────────────────────────────────────────────────

export class SessionsClient {
  constructor(private readonly t: Transport) {}

  async create(goalId: string, opts?: CreateSessionOptions): Promise<Session> {
    return mapSession(await this.t.post<RawSession>('/sessions', {
      goal_id:     goalId,
      user_id:     opts?.userId,
      session_id:  opts?.sessionId,
      budget_usd:  opts?.budgetUsd,
      seed_fields: opts?.seedFields,
      language:    opts?.language,
      tenant_id:   opts?.tenantId,
    }))
  }

  async message(sessionId: string, text: string): Promise<MessageResult> {
    return mapMessage(
      await this.t.post<RawMessageResult>(`/sessions/${sessionId}/message`, { text })
    )
  }

  async get(sessionId: string): Promise<Session> {
    return mapSession(await this.t.get<RawSession>(`/sessions/${sessionId}`))
  }

  async output(sessionId: string): Promise<Output> {
    const r = await this.t.get<{ output: RawOutput; session_id: string }>(
      `/sessions/${sessionId}/output`
    )
    return mapOutput({ ...r.output, session_id: r.session_id })
  }

  async forceOutput(sessionId: string): Promise<Output> {
    const r = await this.t.post<{ output: RawOutput; session_id: string; text: string }>(
      `/sessions/${sessionId}/force-output`
    )
    return mapOutput({ ...(r.output ?? {}), session_id: r.session_id })
  }

  async end(sessionId: string): Promise<void> {
    await this.t.delete(`/sessions/${sessionId}`)
  }
}

export class GoalsClient {
  constructor(private readonly t: Transport) {}

  async list(opts?: { q?: string; sector?: string; limit?: number }): Promise<Goal[]> {
    return this.t.get('/goals', opts as Record<string, string | number | undefined>)
  }

  async get(name: string, version = 'latest'): Promise<Goal> {
    return this.t.get(`/goals/${name}`, { version })
  }

  async install(name: string, version = 'latest'): Promise<Goal> {
    return this.t.post(`/goals/${name}/install`, { version })
  }
}

export class AnalyticsClient {
  constructor(private readonly t: Transport) {}

  async cost(goalId: string, periodDays = 7): Promise<CostSummary> {
    return this.t.get('/analytics/cost', { goal: goalId, period: periodDays })
  }

  async health(goalId: string, windowHours = 24): Promise<GoalHealthReport> {
    return this.t.get('/analytics/health', { goal: goalId, window: windowHours })
  }

  async costTrend(
    goalId:      string,
    periodDays   = 30,
    granularity: 'day' | 'hour' = 'day',
  ): Promise<Array<{ period: string; costUsd: number; sessions: number; tokens: number }>> {
    return this.t.get('/analytics/cost/trend', {
      goal: goalId, period: periodDays, granularity,
    })
  }
}