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
  fieldsExtracted: Array<{
    field:      string
    value:      unknown
    confidence: number
  }>
  costUsd:        number
  latencyMs:      number
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
  author:      string
  license:     string
}

export interface CostSummary {
  goalId:             string
  periodDays:         number
  sessionCount:       number
  totalCostUsd:       number
  avgCostPerSession:  number
  byModel:            Record<string, { costUsd: number; tokens: number; calls: number }>
  byTask:             Record<string, { costUsd: number; calls: number; pct: number }>
}

export interface GoalHealthReport {
  goalId:           string
  windowHours:      number
  sessionCount:     number
  completedCount:   Int16Array
  completionRate:   number
  avgTurns:         number
  avgCostUsd:       number
  p95LatencyMs:     number
  fieldSkipRates:   Record<string, number>
  abandonmentMap:   Record<string, number>
  hallucinationRate: number
}

export interface CreateSessionOptions {
  userId?:     string
  sessionId?:  string
  budgetUsd?:  number
  seedFields?: Record<string, unknown>
  language?:   string
  tenantId?:   string
}

export interface TrueNorthClientOptions {
  apiKey?:  string
  baseUrl?: string
  timeout?: number
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

function int(n: number): number { return n }
