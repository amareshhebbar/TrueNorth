import express, { Request, Response } from 'express'
import path   from 'path'
import crypto from 'crypto'
import { TrueNorth, Session } from '../../../packages/sdk-node'

const PORT  = parseInt(process.env.PORT              ?? '3003')
const TN_URL= process.env.TRUENORTH_BASE_URL         ?? 'http://localhost:8000'
const TN_KEY= process.env.TRUENORTH_API_KEY          ?? ''

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL })

interface ScholarSession {
  session:    Session
  startedAt:  number
}

interface ScholarResult {
  sessionId:    string
  name:         string
  state:        string
  matched:      number
  topScholarship: string
  totalAmount:  string
  screenedAt:   string
  full:         unknown
}

const sessions = new Map<string, ScholarSession>()
const results:  ScholarResult[] = []

const app = express()
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')))

app.post('/api/start', async (_req: Request, res: Response) => {
  const sessionId = `sw_${crypto.randomBytes(6).toString('hex')}`
  try {
    const session = await tn.sessions.create('scholarship_finder', { sessionId })
    sessions.set(sessionId, { session, startedAt: Date.now() })
    res.json({
      sessionId,
      text:          session.agentMessage,
      completionPct: 0,
    })
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

app.post('/api/message', async (req: Request, res: Response) => {
  const { sessionId, text } = req.body as { sessionId: string; text: string }

  if (!sessionId || !text) {
    res.status(400).json({ error: 'sessionId and text required' })
    return
  }

  const meta = sessions.get(sessionId)
  if (!meta) {
    res.status(404).json({ error: 'Session not found or expired' })
    return
  }

  try {
    const result = await tn.sessions.message(sessionId, text)

    if (result.isComplete && result.output) {
      const content = result.output.content as Record<string, unknown>
      const matched = Array.isArray(content.matched_scholarships)
        ? (content.matched_scholarships as unknown[]).length : 0
      const topPick = (content.top_recommendation as string) ?? '—'
      const amount  = (content.total_potential_amount as string) ?? '—'

      results.push({
        sessionId,
        name:          meta.session.agentMessage.slice(0, 30),
        state:         '—',
        matched,
        topScholarship: topPick,
        totalAmount:   amount,
        screenedAt:    new Date().toISOString(),
        full:          content,
      })

      sessions.delete(sessionId)
      await tn.sessions.end(sessionId).catch(() => {})

      res.json({
        text:          result.text,
        isComplete:    true,
        completionPct: 100,
        scholarships:  content.matched_scholarships,
        totalAmount:   amount,
        topPick,
        nextSteps:     content.next_steps,
        tip:           content.application_tip,
        message:       content.response_message,
      })
      return
    }

    res.json({
      text:          result.text,
      isComplete:    false,
      completionPct: result.completionPct,
    })
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

app.get('/api/stats', (_req: Request, res: Response) => {
  res.json({
    activeSessions: sessions.size,
    totalHelped:    results.length,
    avgMatched:     results.length
      ? results.reduce((s, r) => s + r.matched, 0) / results.length : 0,
  })
})

app.get('/health', async (_req: Request, res: Response) => {
  try {
    const h = await tn.health()
    res.json({ status: 'ok', truenorth: h, active: sessions.size })
  } catch {
    res.status(503).json({ status: 'truenorth_unreachable' })
  }
})

app.listen(PORT, () => {
  console.log(`\n  ScholarFinder Web`)
  console.log(`  Student link : http://localhost:${PORT}/`)
  console.log(`  Health       : http://localhost:${PORT}/health`)
  console.log(`  Stats        : http://localhost:${PORT}/api/stats\n`)
})
