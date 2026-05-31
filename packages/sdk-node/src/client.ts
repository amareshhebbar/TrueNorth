/**
 * truenorth — official TypeScript / Node.js SDK
 *
 * Works in:
 *   Node.js 18+     (server-side, CLI, backend)
 *   Bun             (runtime compatible)
 *   Deno            (with npm: specifier)
 *   Next.js         (App Router + Pages Router)
 *   React           (via fetch in components or hooks)
 *
 * Install:
 *   npm install truenorth
 *   yarn add truenorth
 *   pnpm add truenorth
 *
 * Usage (Node / server):
 *   import { TrueNorth } from 'truenorth'
 *   const tn = new TrueNorth({ apiKey: 'tn_live_...' })
 *   const session = await tn.sessions.create('fitness-coach')
 *   const result  = await tn.sessions.message(session.id, 'I am 28')
 *   const output  = await tn.sessions.output(session.id)
 *
 * Usage (Next.js Server Action):
 *   'use server'
 *   import { TrueNorth } from 'truenorth'
 *   const tn = new TrueNorth({ apiKey: process.env.TRUENORTH_API_KEY! })
 *
 * Same interface as Python SDK and Go SDK — one mental model.
 */

// ─────────────────────────────────────────────────────────────────────────────
//  Types — mirrors Python SDK dataclasses exactly
// ─────────────────────────────────────────────────────────────────────────────

export interface Session {
  id:               string
  goalId:           string
  status:           'active' | 'complete' | 'error'
  currentTurn:      number
  completionPct:    number
  collectedFields:  Record<string, unknown>
  missingRequired:  string[]
  totalCostUsd:     number
  isComplete:       boolean
  detectedLanguage: string | null
  agentMessage:     string
  createdAt:        number
}

export interface MessageResult {
  sessionId:       string
  turn:            number
  text:            string
  isComplete:      boolean
  completionPct:   number
  fieldsExtracted: Array<{ field: string; value: unknown; confidence: number }>
  costUsd:         number
  latencyMs:       number
  emotionDetected: string | null
  output:          Output | null
}

export interface Output {
  sessionId:   string
  goalId:      string
  format:      'json' | 'text' | 'markdown'
  content:     unknown
  fields:      Record<string, unknown>
  metadata:    Record<string, unknown>
  generatedAt: number
}

export interface Goal {
  name:        string
  version:     string
  description: string
  sector:      string
  tags:        string[]
  downloads:   number
}

export interface CostSummary {
  goalId:              string
  periodDays:          number
  sessionCount:        number
  totalCostUsd:        number
  avgCostPerSession:   number
  byModel:             Record<string, { costUsd: number; calls: number }>
  byTask:              Record<string, { costUsd: number; calls: number; pct: number }>
}

export class TrueNorthError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly error:      string,
    message:                    string,
  ) {
    super(`TrueNorthError(${statusCode}): ${error} — ${message}`)
    this.name = 'TrueNorthError'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  HTTP transport
// ─────────────────────────────────────────────────────────────────────────────

interface TransportOptions {
  baseUrl: string
  apiKey:  string
  timeout: number
}

class Transport {
  private readonly baseUrl: string
  private readonly headers: Record<string, string>
  private readonly timeout: number

  constructor(opts: TransportOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '')
    this.timeout = opts.timeout
    this.headers = {
      'Content-Type':    'application/json',
      'Accept':          'application/json',
      ...(opts.apiKey ? { 'X-TrueNorth-Key': opts.apiKey } : {}),
    }
  }

  private async request<T>(
    method:  string,
    path:    string,
    body?:   unknown,
    params?: Record<string, string | number | undefined>,
  ): Promise<T> {
    let url = this.baseUrl + path
    if (params) {
      const qs = Object.entries(params)
        .filter(([, v]) => v != null)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
      if (qs) url += '?' + qs
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeout)

    try {
      const resp = await fetch(url, {
        method,
        headers: this.headers,
        body:    body != null ? JSON.stringify(body) : undefined,
        signal:  controller.signal,
      })
      clearTimeout(timer)

      if (!resp.ok) {
        const text = await resp.text()
        let errBody: { error?: string; message?: string } = {}
        try { errBody = JSON.parse(text) } catch {}
        throw new TrueNorthError(
          resp.status,
          errBody.error ?? 'http_error',
          errBody.message ?? text.slice(0, 200),
        )
      }
      if (resp.status === 204) return undefined as T
      return resp.json() as Promise<T>
    } finally {
      clearTimeout(timer)
    }
  }

  get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
    return this.request<T>('GET', path, undefined, params)
  }
  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>('POST', path, body)
  }
  delete<T>(path: string): Promise<T> {
    return this.request<T>('DELETE', path)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Resource clients
// ─────────────────────────────────────────────────────────────────────────────

// Raw API responses (snake_case)
interface RawSession {
  session_id: string; goal_id: string; status: string
  current_turn?: number; turn?: number
  completion_pct: number; collected_fields: Record<string, unknown>
  missing_required?: string[]; missing_fields?: string[]
  total_cost_usd: number; is_complete?: boolean
  detected_language?: string | null; agent_message?: string; created_at?: number
}
interface RawMessageResult {
  session_id: string; turn: number; text: string; is_complete: boolean
  completion_pct: number; fields_extracted?: Array<{ field: string; value: unknown; confidence: number }>
  cost_usd: number; latency_ms: number; emotion_detected?: string | null
  output?: RawOutput | null
}
interface RawOutput {
  session_id: string; goal_id?: string; format?: string; content?: unknown
  fields?: Record<string, unknown>; metadata?: Record<string, unknown>; generated_at?: number
}

function mapSession(r: RawSession): Session {
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
function mapMessage(r: RawMessageResult): MessageResult {
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
    output:          r.output ? mapOutput(r.output as RawOutput) : null,
  }
}
function mapOutput(r: RawOutput): Output {
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

export class SessionsClient {
  constructor(private readonly t: Transport) {}

  async create(
    goalId:      string,
    opts?: {
      userId?:     string
      sessionId?:  string
      budgetUsd?:  number
      seedFields?: Record<string, unknown>
      language?:   string
    },
  ): Promise<Session> {
    const r = await this.t.post<RawSession>('/v1/sessions', {
      goal_id:     goalId,
      user_id:     opts?.userId,
      session_id:  opts?.sessionId,
      budget_usd:  opts?.budgetUsd,
      seed_fields: opts?.seedFields,
      language:    opts?.language,
    })
    return mapSession(r)
  }

  async message(sessionId: string, text: string): Promise<MessageResult> {
    const r = await this.t.post<RawMessageResult>(`/v1/sessions/${sessionId}/message`, { text })
    return mapMessage(r)
  }

  async get(sessionId: string): Promise<Session> {
    return mapSession(await this.t.get<RawSession>(`/v1/sessions/${sessionId}`))
  }

  async output(sessionId: string): Promise<Output> {
    const r = await this.t.get<{ output: RawOutput; session_id: string }>(`/v1/sessions/${sessionId}/output`)
    return mapOutput({ ...r.output, session_id: r.session_id })
  }

  async forceOutput(sessionId: string): Promise<Output> {
    const r = await this.t.post<{ output: RawOutput; session_id: string; text: string }>(
      `/v1/sessions/${sessionId}/force-output`
    )
    return mapOutput({ ...r.output, session_id: r.session_id })
  }

  async end(sessionId: string): Promise<void> {
    await this.t.delete(`/v1/sessions/${sessionId}`)
  }
}

export class GoalsClient {
  constructor(private readonly t: Transport) {}

  async list(opts?: { q?: string; sector?: string; limit?: number }): Promise<Goal[]> {
    return this.t.get('/v1/goals', opts as Record<string, string | number | undefined>)
  }

  async get(name: string, version = 'latest'): Promise<Goal> {
    return this.t.get(`/v1/goals/${name}`, { version })
  }

  async install(name: string, version = 'latest'): Promise<Goal> {
    return this.t.post(`/v1/goals/${name}/install`, { version })
  }
}

export class AnalyticsClient {
  constructor(private readonly t: Transport) {}

  async cost(goal: string, periodDays = 7): Promise<CostSummary> {
    return this.t.get('/v1/analytics/cost', { goal, period: periodDays })
  }

  async health(goal: string, windowHours = 24): Promise<Record<string, unknown>> {
    return this.t.get('/v1/analytics/health', { goal, window: windowHours })
  }

  async trend(goal: string, periodDays = 30, granularity: 'day' | 'hour' = 'day') {
    return this.t.get('/v1/analytics/cost/trend', { goal, period: periodDays, granularity })
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TrueNorth client
// ─────────────────────────────────────────────────────────────────────────────

export interface TrueNorthOptions {
  /** API key (tn_live_... or tn_test_...). Reads TRUENORTH_API_KEY env if omitted. */
  apiKey?:  string
  /** API base URL. Default: http://localhost:8000 */
  baseUrl?: string
  /** Request timeout in milliseconds. Default: 60000 */
  timeout?: number
}

export class TrueNorth {
  readonly sessions:  SessionsClient
  readonly goals:     GoalsClient
  readonly analytics: AnalyticsClient
  private readonly t: Transport

  constructor(opts: TrueNorthOptions = {}) {
    const apiKey  = opts.apiKey  ?? (typeof process !== 'undefined' ? process.env.TRUENORTH_API_KEY ?? '' : '')
    const baseUrl = opts.baseUrl ?? (typeof process !== 'undefined' ? process.env.TRUENORTH_BASE_URL ?? 'http://localhost:8000' : 'http://localhost:8000')
    this.t         = new Transport({ baseUrl, apiKey, timeout: opts.timeout ?? 60_000 })
    this.sessions  = new SessionsClient(this.t)
    this.goals     = new GoalsClient(this.t)
    this.analytics = new AnalyticsClient(this.t)
  }

  health(): Promise<{ status: string; version: string }> {
    return this.t.get('/health')
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: runSession() — full conversation in one call
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run a complete session with a fixed list of user messages.
 * Returns the final output.
 *
 * @example
 * const output = await runSession('fitness-coach', [
 *   'I am 28 years old',
 *   '65 kg',
 *   'lose weight',
 *   '4 days per week',
 * ])
 * console.log(output.content)
 */
export async function runSession(
  goalId:   string,
  messages: string[],
  opts?:    TrueNorthOptions & { budgetUsd?: number },
): Promise<Output> {
  const tn      = new TrueNorth(opts)
  const session = await tn.sessions.create(goalId, { budgetUsd: opts?.budgetUsd })
  for (const msg of messages) {
    const result = await tn.sessions.message(session.id, msg)
    if (result.isComplete && result.output) return result.output
  }
  return tn.sessions.forceOutput(session.id)
}

export default TrueNorth