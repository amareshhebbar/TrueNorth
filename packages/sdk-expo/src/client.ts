/**
 * truenorth-rn — TrueNorth SDK for React Native & Expo
 *
 * Works in:
 *   Expo (SDK 49+, both managed and bare workflow)
 *   React Native (0.71+)
 *   Expo Go (for development)
 *   EAS Build (production)
 *
 * Install:
 *   npx expo install truenorth-rn
 *   # or
 *   npm install truenorth-rn
 *
 * No native modules — pure TypeScript over fetch().
 * Works on iOS, Android, and Expo Web.
 *
 * Usage:
 *   import { useTrueNorthSession } from 'truenorth-rn'
 *
 *   function IntakeScreen() {
 *     const { send, text, isComplete, output } = useTrueNorthSession('fitness-coach')
 *     return (
 *       <View>
 *         <Text>{text}</Text>
 *         <TextInput onSubmitEditing={e => send(e.nativeEvent.text)} />
 *       </View>
 *     )
 *   }
 */

import { useCallback, useEffect, useRef, useState } from 'react'

// ─────────────────────────────────────────────────────────────────────────────
//  Types — same shape as Node SDK
// ─────────────────────────────────────────────────────────────────────────────

export interface Session {
  id:               string
  goalId:           string
  status:           'active' | 'complete' | 'error'
  currentTurn:      number
  completionPct:    number
  collectedFields:  Record<string, unknown>
  missingRequired:  string[]
  isComplete:       boolean
  agentMessage:     string
  totalCostUsd:     number
}

export interface MessageResult {
  sessionId:     string
  turn:          number
  text:          string
  isComplete:    boolean
  completionPct: number
  costUsd:       number
  latencyMs:     number
  output:        Output | null
}

export interface Output {
  sessionId:   string
  goalId:      string
  format:      string
  content:     unknown
  fields:      Record<string, unknown>
  metadata:    Record<string, unknown>
  generatedAt: number
}

export class TrueNorthError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly errorCode:  string,
    message: string,
  ) {
    super(`TrueNorthError(${statusCode}): ${errorCode} — ${message}`)
    this.name = 'TrueNorthError'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core client (same as Node SDK, but with RN-aware config)
// ─────────────────────────────────────────────────────────────────────────────

export interface TrueNorthConfig {
  /** Your API key. Store in expo-constants or env. */
  apiKey:   string
  /** Base URL of your TrueNorth server. */
  baseUrl?: string
  /** Request timeout ms. Default 60000. */
  timeout?: number
}

async function request<T>(
  config:  Required<TrueNorthConfig>,
  method:  string,
  path:    string,
  body?:   unknown,
  params?: Record<string, string | number | undefined>,
): Promise<T> {
  let url = config.baseUrl.replace(/\/$/, '') + path
  if (params) {
    const qs = Object.entries(params)
      .filter(([, v]) => v != null)
      .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
      .join('&')
    if (qs) url += '?' + qs
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), config.timeout)

  try {
    const resp = await fetch(url, {
      method,
      headers: {
        'Content-Type':    'application/json',
        'Accept':          'application/json',
        'X-TrueNorth-Key': config.apiKey,
      },
      body:   body != null ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })

    if (!resp.ok) {
      const text = await resp.text()
      let err: { error?: string; message?: string } = {}
      try { err = JSON.parse(text) } catch {}
      throw new TrueNorthError(resp.status, err.error ?? 'http_error', err.message ?? text.slice(0, 200))
    }
    if (resp.status === 204) return undefined as T
    return resp.json() as Promise<T>
  } finally {
    clearTimeout(timer)
  }
}

// Raw API types
interface RawSession {
  session_id: string; goal_id: string; status: string; current_turn?: number; turn?: number
  completion_pct: number; collected_fields: Record<string, unknown>
  missing_required?: string[]; missing_fields?: string[]; total_cost_usd?: number
  is_complete?: boolean; agent_message?: string
}
interface RawMessageResult {
  session_id: string; turn: number; text: string; is_complete: boolean
  completion_pct: number; cost_usd: number; latency_ms: number
  output?: { session_id: string; goal_id?: string; format?: string; content?: unknown; fields?: Record<string, unknown>; metadata?: Record<string, unknown>; generated_at?: number } | null
}

function mapSession(r: RawSession): Session {
  return {
    id:              r.session_id,
    goalId:          r.goal_id,
    status:          r.status as Session['status'],
    currentTurn:     r.current_turn ?? r.turn ?? 0,
    completionPct:   r.completion_pct,
    collectedFields: r.collected_fields ?? {},
    missingRequired: r.missing_required ?? r.missing_fields ?? [],
    isComplete:      r.is_complete ?? false,
    agentMessage:    r.agent_message ?? '',
    totalCostUsd:    r.total_cost_usd ?? 0,
  }
}

function mapMessage(r: RawMessageResult): MessageResult {
  return {
    sessionId:     r.session_id,
    turn:          r.turn,
    text:          r.text ?? '',
    isComplete:    r.is_complete,
    completionPct: r.completion_pct,
    costUsd:       r.cost_usd,
    latencyMs:     r.latency_ms,
    output:        r.output
      ? { sessionId: r.session_id, goalId: r.output.goal_id ?? '', format: r.output.format ?? 'json',
          content: r.output.content, fields: r.output.fields ?? {}, metadata: r.output.metadata ?? {},
          generatedAt: r.output.generated_at ?? 0 }
      : null,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  React hooks — the main API for React Native / Expo
// ─────────────────────────────────────────────────────────────────────────────

export interface UseTrueNorthSessionOptions extends TrueNorthConfig {
  /** User ID for per-user rate limiting and tracking */
  userId?:    string
  /** Pre-fill known fields (e.g. from user profile) */
  seedFields?: Record<string, unknown>
  /** Per-session USD cost cap */
  budgetUsd?: number
  /** Called when the session completes */
  onComplete?: (output: Output) => void
  /** Called on any error */
  onError?:    (err: TrueNorthError | Error) => void
}

export interface UseTrueNorthSessionReturn {
  /** Current agent message waiting for user response */
  agentText:      string
  /** Send a user message */
  send:           (text: string) => Promise<void>
  /** True when the session is complete */
  isComplete:     boolean
  /** Final output (available when isComplete === true) */
  output:         Output | null
  /** Fields collected so far */
  collectedFields: Record<string, unknown>
  /** Completion percentage (0–100) */
  completionPct:  number
  /** True while waiting for a response */
  isLoading:      boolean
  /** Current error (cleared on next send) */
  error:          Error | null
  /** Session ID (for logging/debugging) */
  sessionId:      string | null
  /** Total cost in USD */
  totalCostUsd:   number
  /** Restart the session from scratch */
  restart:        () => void
}

/**
 * The primary React Native / Expo hook for TrueNorth.
 *
 * @example
 * function IntakeScreen() {
 *   const { agentText, send, isComplete, output, isLoading } =
 *     useTrueNorthSession('fitness-coach', {
 *       apiKey:  Constants.expoConfig.extra.truenorthApiKey,
 *       baseUrl: 'https://api.myapp.com',
 *     })
 *
 *   if (isComplete) return <OutputView output={output} />
 *   return (
 *     <View>
 *       {isLoading && <ActivityIndicator />}
 *       <Text style={styles.agentText}>{agentText}</Text>
 *       <MessageInput onSend={send} disabled={isLoading} />
 *     </View>
 *   )
 * }
 */
export function useTrueNorthSession(
  goalId: string,
  opts:   UseTrueNorthSessionOptions,
): UseTrueNorthSessionReturn {
  const config: Required<TrueNorthConfig> = {
    apiKey:  opts.apiKey,
    baseUrl: opts.baseUrl ?? 'http://localhost:8000',
    timeout: opts.timeout ?? 60_000,
  }

  const [sessionId,      setSessionId]      = useState<string | null>(null)
  const [agentText,      setAgentText]      = useState('')
  const [isComplete,     setIsComplete]     = useState(false)
  const [output,         setOutput]         = useState<Output | null>(null)
  const [collectedFields,setCollectedFields]= useState<Record<string, unknown>>({})
  const [completionPct,  setCompletionPct]  = useState(0)
  const [isLoading,      setIsLoading]      = useState(true)
  const [error,          setError]          = useState<Error | null>(null)
  const [totalCostUsd,   setTotalCostUsd]   = useState(0)
  const [resetKey,       setResetKey]       = useState(0)

  const onCompleteRef = useRef(opts.onComplete)
  const onErrorRef    = useRef(opts.onError)
  onCompleteRef.current = opts.onComplete
  onErrorRef.current    = opts.onError

  // Start session on mount (or restart)
  useEffect(() => {
    let cancelled = false

    async function init() {
      setIsLoading(true)
      setError(null)
      try {
        const body: Record<string, unknown> = {
          goal_id:     goalId,
          user_id:     opts.userId,
          budget_usd:  opts.budgetUsd,
          seed_fields: opts.seedFields,
        }
        const raw = await request<RawSession>(config, 'POST', '/v1/sessions', body)
        if (cancelled) return
        const sess = mapSession(raw)
        setSessionId(sess.id)
        setAgentText(sess.agentMessage)
        setCompletionPct(sess.completionPct)
        setCollectedFields(sess.collectedFields)
      } catch (err) {
        if (!cancelled) {
          const e = err instanceof Error ? err : new Error(String(err))
          setError(e)
          onErrorRef.current?.(e)
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    init()
    return () => { cancelled = true }
  }, [goalId, resetKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback(async (text: string) => {
    if (!sessionId || isLoading || isComplete) return
    setIsLoading(true)
    setError(null)

    try {
      const raw = await request<RawMessageResult>(
        config, 'POST', `/v1/sessions/${sessionId}/message`, { text }
      )
      const result = mapMessage(raw)
      setAgentText(result.text)
      setCompletionPct(result.completionPct)
      setTotalCostUsd(prev => prev + result.costUsd)

      if (result.isComplete) {
        setIsComplete(true)
        if (result.output) {
          setOutput(result.output)
          onCompleteRef.current?.(result.output)
        } else {
          // Fetch output separately
          const outWrapper = await request<{ output: ReturnType<typeof mapMessage>['output']; session_id: string }>(
            config, 'GET', `/v1/sessions/${sessionId}/output`
          )
          if (outWrapper.output) {
            setOutput(outWrapper.output)
            onCompleteRef.current?.(outWrapper.output)
          }
        }
      } else {
        setCollectedFields(prev => ({ ...prev })) // trigger re-render
      }
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err))
      setError(e)
      onErrorRef.current?.(e)
    } finally {
      setIsLoading(false)
    }
  }, [sessionId, isLoading, isComplete, config]) // eslint-disable-line react-hooks/exhaustive-deps

  const restart = useCallback(() => {
    setSessionId(null)
    setAgentText('')
    setIsComplete(false)
    setOutput(null)
    setCollectedFields({})
    setCompletionPct(0)
    setIsLoading(true)
    setError(null)
    setTotalCostUsd(0)
    setResetKey(k => k + 1)
  }, [])

  return {
    agentText, send, isComplete, output,
    collectedFields, completionPct,
    isLoading, error, sessionId, totalCostUsd, restart,
  }
}

/**
 * Lightweight hook for non-interactive use (batch processing, server calls).
 * Runs a session with preset messages and returns when complete.
 *
 * @example
 * const { output, isLoading } = useRunSession('fitness-coach', ['28', '65kg', 'lose weight'], config)
 */
export function useRunSession(
  goalId:   string,
  messages: string[],
  config:   TrueNorthConfig,
) {
  const [output,    setOutput]    = useState<Output | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error,     setError]     = useState<Error | null>(null)

  const resolvedConfig: Required<TrueNorthConfig> = {
    apiKey:  config.apiKey,
    baseUrl: config.baseUrl ?? 'http://localhost:8000',
    timeout: config.timeout ?? 60_000,
  }

  useEffect(() => {
    let cancelled = false
    async function run() {
      try {
        const sessRaw = await request<RawSession>(
          resolvedConfig, 'POST', '/v1/sessions', { goal_id: goalId }
        )
        const sessionId = sessRaw.session_id
        for (const msg of messages) {
          if (cancelled) return
          const msgRaw = await request<RawMessageResult>(
            resolvedConfig, 'POST', `/v1/sessions/${sessionId}/message`, { text: msg }
          )
          const result = mapMessage(msgRaw)
          if (result.isComplete && result.output) {
            if (!cancelled) setOutput(result.output)
            return
          }
        }
        // Force output with what we have
        const forceRaw = await request<{ output: Output }>(
          resolvedConfig, 'POST', `/v1/sessions/${sessRaw.session_id}/force-output`
        )
        if (!cancelled) setOutput(forceRaw.output)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err : new Error(String(err)))
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }
    run()
    return () => { cancelled = true }
  }, [goalId, resolvedConfig.apiKey, resolvedConfig.baseUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  return { output, isLoading, error }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bare fetch client (for use outside components)
// ─────────────────────────────────────────────────────────────────────────────

export class TrueNorthClient {
  private readonly config: Required<TrueNorthConfig>

  constructor(config: TrueNorthConfig) {
    this.config = { baseUrl: 'http://localhost:8000', timeout: 60_000, ...config }
  }

  async createSession(goalId: string, opts?: {
    userId?: string; budgetUsd?: number; seedFields?: Record<string, unknown>
  }): Promise<Session> {
    const r = await request<RawSession>(this.config, 'POST', '/v1/sessions', {
      goal_id: goalId, user_id: opts?.userId,
      budget_usd: opts?.budgetUsd, seed_fields: opts?.seedFields,
    })
    return mapSession(r)
  }

  async sendMessage(sessionId: string, text: string): Promise<MessageResult> {
    const r = await request<RawMessageResult>(this.config, 'POST', `/v1/sessions/${sessionId}/message`, { text })
    return mapMessage(r)
  }

  async getOutput(sessionId: string): Promise<Output> {
    const r = await request<{ output: Output; session_id: string }>(this.config, 'GET', `/v1/sessions/${sessionId}/output`)
    return r.output
  }
}