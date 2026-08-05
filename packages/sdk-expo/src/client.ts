import { useCallback, useEffect, useRef, useState } from 'react'

export interface Session {
  id: string; goalId: string; status: 'active' | 'complete' | 'error'
  currentTurn: number; completionPct: number
  collectedFields: Record<string, unknown>; missingRequired: string[]
  isComplete: boolean; agentMessage: string; totalCostUsd: number
}

export interface MessageResult {
  sessionId: string; turn: number; text: string; isComplete: boolean
  completionPct: number; costUsd: number; latencyMs: number
  output: Output | null
}

export interface Output {
  sessionId: string; goalId: string; format: string
  content: unknown; fields: Record<string, unknown>
  metadata: Record<string, unknown>; generatedAt: number
}

export interface Config {
  apiKey: string
  baseUrl?: string
  timeout?: number
}

export class TrueNorthError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly errorCode: string,
    message: string,
  ) {
    super(`TrueNorthError(${statusCode}): ${errorCode} — ${message}`)
    this.name = 'TrueNorthError'
  }
}

async function apiRequest<T>(
  config: Required<Config>,
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const ctrl  = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), config.timeout)
  try {
    const resp = await fetch(`${config.baseUrl.replace(/\/$/, '')}${path}`, {
      method,
      headers: {
        'Content-Type':    'application/json',
        'X-TrueNorth-Key': config.apiKey,
      },
      body:   body != null ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    })
    if (!resp.ok) {
      const text = await resp.text()
      let e: { error?: string; message?: string } = {}
      try { e = JSON.parse(text) } catch {}
      throw new TrueNorthError(resp.status, e.error ?? 'http_error', e.message ?? text.slice(0, 200))
    }
    return resp.json() as Promise<T>
  } finally {
    clearTimeout(timer)
  }
}

function resolved(c: Config): Required<Config> {
  return { baseUrl: 'http://localhost:8000', timeout: 60_000, ...c }
}

interface RawSession { session_id: string; goal_id: string; status: string; current_turn?: number; turn?: number; completion_pct: number; collected_fields?: Record<string, unknown>; missing_required?: string[]; total_cost_usd?: number; is_complete?: boolean; agent_message?: string }
interface RawMsg { session_id: string; turn: number; text: string; is_complete: boolean; completion_pct: number; cost_usd: number; latency_ms: number; output?: RawOutput | null }
interface RawOutput { session_id: string; goal_id?: string; format?: string; content?: unknown; fields?: Record<string, unknown>; metadata?: Record<string, unknown>; generated_at?: number }

function mapSess(r: RawSession): Session {
  return { id: r.session_id, goalId: r.goal_id, status: r.status as Session['status'], currentTurn: r.current_turn ?? r.turn ?? 0, completionPct: r.completion_pct, collectedFields: r.collected_fields ?? {}, missingRequired: r.missing_required ?? [], isComplete: r.is_complete ?? false, agentMessage: r.agent_message ?? '', totalCostUsd: r.total_cost_usd ?? 0 }
}
function mapMsg(r: RawMsg): MessageResult {
  return { sessionId: r.session_id, turn: r.turn, text: r.text ?? '', isComplete: r.is_complete, completionPct: r.completion_pct, costUsd: r.cost_usd, latencyMs: r.latency_ms, output: r.output ? { sessionId: r.session_id, goalId: r.output.goal_id ?? '', format: r.output.format ?? 'json', content: r.output.content, fields: r.output.fields ?? {}, metadata: r.output.metadata ?? {}, generatedAt: r.output.generated_at ?? 0 } : null }
}

export interface UseSessionReturn {
  agentText:       string
  send:            (text: string) => Promise<void>
  isComplete:      boolean
  output:          Output | null
  collectedFields: Record<string, unknown>
  completionPct:   number
  isLoading:       boolean
  error:           Error | null
  sessionId:       string | null
  totalCostUsd:    number
  restart:         () => void
}

export interface UseSessionOptions extends Config {
  userId?:      string
  seedFields?:  Record<string, unknown>
  budgetUsd?:   number
  onComplete?:  (output: Output) => void
  onError?:     (err: Error) => void
}

export function useTrueNorthSession(goalId: string, opts: UseSessionOptions): UseSessionReturn {
  const config = resolved(opts)
  const [sessionId,       setSessionId]       = useState<string | null>(null)
  const [agentText,       setAgentText]        = useState('')
  const [isComplete,      setIsComplete]       = useState(false)
  const [output,          setOutput]           = useState<Output | null>(null)
  const [collectedFields, setCollectedFields]  = useState<Record<string, unknown>>({})
  const [completionPct,   setCompletionPct]    = useState(0)
  const [isLoading,       setIsLoading]        = useState(true)
  const [error,           setError]            = useState<Error | null>(null)
  const [totalCostUsd,    setTotalCostUsd]     = useState(0)
  const [resetKey,        setResetKey]         = useState(0)
  const onCompleteRef = useRef(opts.onComplete)
  const onErrorRef    = useRef(opts.onError)
  onCompleteRef.current = opts.onComplete
  onErrorRef.current    = opts.onError

  useEffect(() => {
    let cancelled = false
    setIsLoading(true); setError(null)
    apiRequest<RawSession>(config, 'POST', '/sessions', {
      goal_id: goalId, user_id: opts.userId,
      budget_usd: opts.budgetUsd, seed_fields: opts.seedFields,
    }).then(r => {
      if (cancelled) return
      const s = mapSess(r)
      setSessionId(s.id); setAgentText(s.agentMessage)
      setCompletionPct(s.completionPct); setCollectedFields(s.collectedFields)
    }).catch(e => {
      if (!cancelled) { setError(e as Error); onErrorRef.current?.(e as Error) }
    }).finally(() => { if (!cancelled) setIsLoading(false) })
    return () => { cancelled = true }
  }, [goalId, resetKey])

  const send = useCallback(async (text: string) => {
    if (!sessionId || isLoading || isComplete) return
    setIsLoading(true); setError(null)
    try {
      const raw    = await apiRequest<RawMsg>(config, 'POST', `/sessions/${sessionId}/message`, { text })
      const result = mapMsg(raw)
      setAgentText(result.text)
      setCompletionPct(result.completionPct)
      setTotalCostUsd(p => p + result.costUsd)
      if (result.isComplete) {
        setIsComplete(true)
        const out = result.output ?? (await apiRequest<{ output: Output }>(config, 'GET', `/sessions/${sessionId}/output`)).output
        setOutput(out)
        onCompleteRef.current?.(out)
      }
    } catch (e) {
      const err = e as Error
      setError(err); onErrorRef.current?.(err)
    } finally {
      setIsLoading(false)
    }
  }, [sessionId, isLoading, isComplete, config])

  const restart = useCallback(() => {
    setSessionId(null); setAgentText(''); setIsComplete(false)
    setOutput(null); setCollectedFields({}); setCompletionPct(0)
    setIsLoading(true); setError(null); setTotalCostUsd(0)
    setResetKey(k => k + 1)
  }, [])

  return { agentText, send, isComplete, output, collectedFields, completionPct, isLoading, error, sessionId, totalCostUsd, restart }
}

export class TrueNorthClient {
  private readonly config: Required<Config>
  constructor(config: Config) { this.config = resolved(config) }

  async createSession(goalId: string, opts?: { userId?: string; budgetUsd?: number }): Promise<Session> {
    return mapSess(await apiRequest<RawSession>(this.config, 'POST', '/sessions', { goal_id: goalId, user_id: opts?.userId, budget_usd: opts?.budgetUsd }))
  }

  async sendMessage(sessionId: string, text: string): Promise<MessageResult> {
    return mapMsg(await apiRequest<RawMsg>(this.config, 'POST', `/sessions/${sessionId}/message`, { text }))
  }

  async getOutput(sessionId: string): Promise<Output> {
    const r = await apiRequest<{ output: Output }>(this.config, 'GET', `/sessions/${sessionId}/output`)
    return r.output
  }
}

export default useTrueNorthSession
