/**
 * truenorth — official TypeScript / Node.js SDK
 *
 * Works in:  Node.js 18+ · Bun · Deno · Next.js App Router · React · React Native
 *
 * Install:
 *   npm install truenorth
 *   yarn add truenorth
 *   pnpm add truenorth
 *
 * Usage:
 *   import { TrueNorth } from 'truenorth'
 *   const tn = new TrueNorth({ apiKey: 'tn_live_...' })
 *   const session = await tn.sessions.create('fitness-coach')
 *   const result  = await tn.sessions.message(session.id, 'I am 28')
 *   const output  = await tn.sessions.output(session.id)
 */

import { Transport, SessionsClient, GoalsClient, AnalyticsClient } from './client'
import type { TrueNorthClientOptions, Output } from './types'

export * from './types'
export * from './client'

// Add this at the top of your file
declare const process: any;

function getEnv(key: string): string {
  if (typeof process !== 'undefined' && process.env) {
    return process.env[key] ?? ''
  }
  return ''
}
export class TrueNorth {
  readonly sessions:  SessionsClient
  readonly goals:     GoalsClient
  readonly analytics: AnalyticsClient
  private readonly t: Transport

  constructor(opts: TrueNorthClientOptions = {}) {
    const apiKey  = opts.apiKey  ?? getEnv('TRUENORTH_API_KEY')
    const baseUrl = opts.baseUrl ?? (getEnv('TRUENORTH_BASE_URL') || 'http://localhost:8000')

    this.t = new Transport({ apiKey, baseUrl, timeout: opts.timeout ?? 60_000 })
    this.sessions  = new SessionsClient(this.t)
    this.goals     = new GoalsClient(this.t)
    this.analytics = new AnalyticsClient(this.t)
  }

  health(): Promise<{ status: string; version: string }> {
    return this.t.get('/health')
  }
}
export async function runSession(
  goalId:   string,
  messages: string[],
  opts?:    TrueNorthClientOptions & { budgetUsd?: number },
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