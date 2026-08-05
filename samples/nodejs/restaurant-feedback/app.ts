import express, { Request, Response } from 'express'
import path    from 'path'
import crypto  from 'crypto'
import { TrueNorth, Session, MessageResult } from '../../../packages/sdk-node'

const RESTAURANT_NAME = process.env.RESTAURANT_NAME    ?? 'The Spice Garden'
const GOAL_ID         = process.env.GOAL_ID            ?? 'restaurant_feedback'
const TN_URL          = process.env.TRUENORTH_BASE_URL  ?? 'http://localhost:8000'
const TN_KEY          = process.env.TRUENORTH_API_KEY   ?? ''
const PORT            = parseInt(process.env.PORT       ?? '5002')

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

interface GuestSession {
  sessionId:  string
  startedAt:  number
  lastActive: number
  pct:        number
}

interface FeedbackResult {
  sessionId:    string
  timestamp:    string
  overallScore: number
  nps:          number
  npsCategory:  'Promoter' | 'Passive' | 'Detractor'
  sentiment:    string
  priorityIssue:string
  compliment:   string
  actionNeeded: boolean
  full:         Record<string, unknown>
}

const sessions = new Map<string, GuestSession>()
const results:  FeedbackResult[] = []

function parseResult(sid: string, content: Record<string, unknown>): FeedbackResult {
  const nps = Number(content.nps_score ?? 0)
  const npsCategory: FeedbackResult['npsCategory'] =
    nps >= 9 ? 'Promoter' : nps <= 6 ? 'Detractor' : 'Passive'

  return {
    sessionId:     sid,
    timestamp:     new Date().toISOString(),
    overallScore:  Number(content.overall_score ?? 0),
    nps,
    npsCategory,
    sentiment:     String(content.sentiment ?? ''),
    priorityIssue: String(content.priority_issue ?? ''),
    compliment:    String(content.compliment_to_share ?? ''),
    actionNeeded:  content.manager_action_required === true,
    full:          content,
  }
}

function buildDashboard(): string {
  const totalScore   = results.reduce((s, r) => s + r.overallScore, 0)
  const avg          = results.length ? totalScore / results.length : 0
  const promoters    = results.filter(r => r.npsCategory === 'Promoter').length
  const detractors   = results.filter(r => r.npsCategory === 'Detractor').length
  const total        = results.length
  const nps          = total ? ((promoters - detractors) / total * 100) : 0
  const issues       = results.filter(r => r.actionNeeded && r.priorityIssue).map(r => r.priorityIssue).slice(0, 5)
  const praise       = results.filter(r => r.compliment).map(r => r.compliment).slice(0, 5)

  const issueRows = issues.length
    ? issues.map(i => `<li>⚠️ ${i}</li>`).join('')
    : '<li class="none">No critical issues today ✅</li>'
  const praiseRows = praise.length
    ? praise.map(p => `<li>"${p}"</li>`).join('')
    : '<li class="none">Keep collecting feedback — highlights appear here</li>'

  return `<!DOCTYPE html><html><head>
<title>${RESTAURANT_NAME} — Dashboard</title><meta charset="UTF-8">
<style>
body{font-family:-apple-system,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#fffbeb;}
h1{color:#92400e;}h2{color:#a16207;font-size:1rem;font-weight:400;margin-bottom:20px;}
.s{display:inline-block;background:white;border-radius:10px;padding:15px 20px;margin:5px;
   text-align:center;border:2px solid #fde68a;min-width:120px;}
.s h2{margin:0;font-size:1.8rem;color:#d97706;font-weight:700;}
.s p{margin:3px 0 0;color:#64748b;font-size:.78rem;}
.box{background:white;border-radius:12px;padding:16px;margin:16px 0;border:1px solid #fde68a;}
.box h3{color:#92400e;font-size:.95rem;margin-bottom:10px;}
li{margin:6px 0;font-size:.88rem;color:#374151;line-height:1.5;}
.none{color:#94a3b8;font-style:italic;}
</style></head><body>
<h1>🍽️ ${RESTAURANT_NAME}</h1>
<h2>${new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'})} · Manager Dashboard</h2>
<div>
  <div class="s"><h2>${results.length}</h2><p>Responses today</p></div>
  <div class="s"><h2>${avg.toFixed(1)}/10</h2><p>Avg score</p></div>
  <div class="s"><h2>${nps > 0 ? '+' : ''}${nps.toFixed(0)}</h2><p>NPS score</p></div>
  <div class="s" style="border-color:#bbf7d0"><h2 style="color:#16a34a">${promoters}</h2><p>Promoters (9-10)</p></div>
  <div class="s" style="border-color:#fecaca"><h2 style="color:#dc2626">${detractors}</h2><p>Detractors (0-6)</p></div>
  <div class="s"><h2>${sessions.size}</h2><p>Currently chatting</p></div>
</div>
<div class="box">
  <h3>⚠️ Priority issues — needs manager attention</h3>
  <ul>${issueRows}</ul>
</div>
<div class="box">
  <h3>✅ Compliments — share with your team</h3>
  <ul>${praiseRows}</ul>
</div>
<p><small><a href="/">← Guest feedback link</a> · <a href="/api/stats">Raw stats (JSON)</a> · <a href="/health">Health</a></small></p>
</body></html>`
}

const app = express()
app.use(express.json())
app.use(express.static(path.join(__dirname, 'public')))

app.get('/', (_req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'))
})

app.get('/dashboard', (_req: Request, res: Response) => {
  res.type('html').send(buildDashboard())
})

app.get('/health', async (_req: Request, res: Response) => {
  try {
    const h = await tn.health()
    res.json({
      status:          'ok',
      restaurant:      RESTAURANT_NAME,
      truenorth:       h,
      active_sessions: sessions.size,
      feedback_today:  results.length,
    })
  } catch {
    res.status(503).json({ status: 'truenorth_unreachable', url: TN_URL })
  }
})

app.get('/api/stats', (_req: Request, res: Response) => {
  res.json({ results, total: results.length })
})

app.post('/api/start', async (_req: Request, res: Response) => {
  const sid = `rf_${crypto.randomBytes(6).toString('hex')}`
  try {
    const session: Session = await tn.sessions.create(GOAL_ID, { sessionId: sid })
    sessions.set(sid, {
      sessionId: sid, startedAt: Date.now(), lastActive: Date.now(), pct: 0,
    })
    res.json({ session_id: sid, text: session.agentMessage, completion: 0 })
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

app.post('/api/message', async (req: Request, res: Response) => {
  const { session_id: sid, text } = req.body as { session_id: string; text: string }

  if (!sid || !text) {
    res.status(400).json({ error: 'session_id and text required' })
    return
  }

  const meta = sessions.get(sid)
  if (!meta) {
    res.status(404).json({ error: 'Session not found' })
    return
  }

  try {
    const result: MessageResult = await tn.sessions.message(sid, text)

    meta.lastActive = Date.now()
    meta.pct        = result.completionPct
    sessions.set(sid, meta)

    const resp: Record<string, unknown> = {
      text:        result.text,
      is_complete: result.isComplete,
      completion:  result.completionPct,
    }

    if (result.isComplete && result.output) {
      const content = result.output.content as Record<string, unknown>
      const fr      = parseResult(sid, content)
      results.push(fr)
      sessions.delete(sid)
      await tn.sessions.end(sid).catch(() => {})

      const thankYou = String(content.response_message || '')
        || `Thank you! See you again at ${RESTAURANT_NAME}. 😊`

      resp.thank_you = thankYou
      console.log(`✅ Feedback: score=${fr.overallScore} NPS=${fr.nps} action=${fr.actionNeeded}`)
    }

    res.json(resp)
  } catch (err) {
    res.status(500).json({ error: String(err) })
  }
})

async function start(): Promise<void> {
  console.log(`\n  Restaurant Feedback (Node.js / TypeScript)`)
  console.log(`  ${RESTAURANT_NAME}`)
  console.log(`  TrueNorth API: ${TN_URL}`)

  try {
    const h = await tn.health()
    console.log(`  ✓ Connected: ${h.status}`)
  } catch {
    console.warn(`  ⚠  TrueNorth API not reachable at ${TN_URL}`)
    console.warn('  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000')
  }

  app.listen(PORT, () => {
    console.log(`\n  Guest link : http://localhost:${PORT}/`)
    console.log(`  Dashboard  : http://localhost:${PORT}/dashboard\n`)
  })
}

start().catch(console.error)
