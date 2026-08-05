import express, { Request, Response } from 'express'
import readline from 'readline'
import { TrueNorth, Session, MessageResult, Output } from '../../../packages/sdk-node'

const COMPANY_NAME = process.env.COMPANY_NAME       ?? 'TechCorp'
const ROLE_TITLE   = process.env.ROLE_TITLE         ?? 'Senior Backend Engineer'
const ROLE_MIN_EXP = parseInt(process.env.ROLE_MIN_EXP    ?? '4')
const ROLE_BUDGET  = parseInt(process.env.ROLE_BUDGET_LPA ?? '30')
const ROLE_NOTICE  = parseInt(process.env.ROLE_MAX_NOTICE ?? '60')
const GOAL_ID      = process.env.GOAL_ID            ?? 'hr_screening'
const TN_URL       = process.env.TRUENORTH_BASE_URL  ?? 'http://localhost:8000'
const TN_KEY       = process.env.TRUENORTH_API_KEY   ?? ''
const ATS_WEBHOOK  = process.env.ATS_WEBHOOK_URL    ?? ''
const PORT         = parseInt(process.env.PORT      ?? '3002')
const CONSOLE_MODE = process.env.CONSOLE            === '1'

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

interface ScreeningSession {
  sessionId: string; candidateId: string
  startedAt: number; lastActive: number; turnCount: number; pct: number
}

interface Candidate {
  sessionId: string; candidateId: string; screenedAt: string
  name: string; overallScore: number; recommendation: string
  techScore: number; cultureScore: number
  strengths: string[]; concerns: string[]; redFlags: string[]
  nextRoundQs: string[]; salaryFit: string; summary: string
  full?: Record<string, unknown>
}

const sessions  = new Map<string, ScreeningSession>()
const candidates: Candidate[] = []

function parseCandidate(sid: string, cid: string, c: Record<string, unknown>): Candidate {
  const toArr = (v: unknown) => Array.isArray(v) ? v.map(String) : []
  const rec   = String(c.recommendation ?? '').toUpperCase()
  return {
    sessionId:      sid, candidateId: cid,
    screenedAt:     new Date().toISOString(),
    name:           String(c.candidate_name ?? 'Unknown'),
    overallScore:   Number(c.overall_score  ?? 0),
    recommendation: ['STRONG_HIRE','SHORTLIST','HOLD','REJECT'].includes(rec) ? rec : 'UNKNOWN',
    techScore:      Number(c.technical_score ?? 0),
    cultureScore:   Number(c.culture_score  ?? 0),
    strengths:      toArr(c.strengths),
    concerns:       toArr(c.concerns),
    redFlags:       toArr(c.red_flags),
    nextRoundQs:    toArr(c.questions_for_next_round),
    salaryFit:      String(c.salary_fit ?? ''),
    summary:        String(c.summary    ?? ''),
    full:           c,
  }
}

function buildDashboard(): string {
  const all    = [...candidates].sort((a, b) => b.overallScore - a.overallScore)
  const strong = all.filter(c => c.recommendation === 'STRONG_HIRE').length
  const sl     = all.filter(c => ['STRONG_HIRE','SHORTLIST'].includes(c.recommendation)).length
  const rej    = all.filter(c => c.recommendation === 'REJECT').length
  const recClr = (r: string) => ({'STRONG_HIRE':'#16a34a','SHORTLIST':'#2563eb','HOLD':'#d97706','REJECT':'#dc2626'}[r] ?? '#64748b')
  const rows   = all.length
    ? all.map(c => `<tr>
        <td><strong>${c.name}</strong><br><small>${c.candidateId}</small></td>
        <td><strong>${c.overallScore}/100</strong></td>
        <td style="color:${recClr(c.recommendation)};font-weight:700">${c.recommendation}</td>
        <td>${c.techScore}</td><td>${c.cultureScore}</td><td>${c.salaryFit}</td>
        <td>${c.screenedAt.slice(0,16)}</td></tr>`).join('')
    : '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:24px">No candidates yet</td></tr>'
  return `<!DOCTYPE html><html><head><title>HireFlow</title><meta charset="UTF-8">
<style>body{font-family:-apple-system,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;background:#f8fafc;}
h1{color:#1e293b;}.s{display:inline-block;background:white;border-radius:10px;padding:14px 18px;margin:5px;
border:1px solid #e2e8f0;text-align:center;min-width:110px;}.s h2{margin:0;font-size:1.7rem;font-weight:700;}
.s p{margin:3px 0 0;color:#64748b;font-size:.78rem;}
table{width:100%;border-collapse:collapse;background:white;border-radius:10px;margin-top:20px;}
th{background:#1e293b;color:white;padding:10px 12px;text-align:left;font-size:.82rem;}
td{padding:10px 12px;border-bottom:1px solid #f1f5f9;font-size:.88rem;}small{color:#94a3b8;}</style></head><body>
<h1>💼 HireFlow (Node.js) — ${COMPANY_NAME} / ${ROLE_TITLE}</h1>
<div>
  <div class="s"><h2>${sessions.size}</h2><p>Active</p></div>
  <div class="s"><h2>${all.length}</h2><p>Screened</p></div>
  <div class="s" style="border-color:#bbf7d0"><h2 style="color:#16a34a">${strong}</h2><p>Strong hire</p></div>
  <div class="s" style="border-color:#bfdbfe"><h2 style="color:#2563eb">${sl}</h2><p>Shortlist</p></div>
  <div class="s" style="border-color:#fecaca"><h2 style="color:#dc2626">${rej}</h2><p>Rejected</p></div>
</div>
<table><thead><tr><th>Candidate</th><th>Score</th><th>Rec</th><th>Tech</th><th>Culture</th><th>Salary</th><th>When</th></tr></thead>
<tbody>${rows}</tbody></table>
<p style="margin-top:12px;font-size:.82rem;color:#94a3b8">
<a href="/api/shortlist">Shortlist</a> · <a href="/api/candidates">All</a> · <a href="/health">Health</a></p>
</body></html>`
}

const app = express()
app.use(express.json())

app.get('/dashboard', (_req: Request, res: Response) => res.type('html').send(buildDashboard()))

app.get('/health', async (_req: Request, res: Response) => {
  try {
    const h = await tn.health()
    res.json({ status:'ok', role:ROLE_TITLE, truenorth:h, active:sessions.size, screened:candidates.length })
  } catch { res.status(503).json({ status:'truenorth_unreachable' }) }
})

app.get('/api/candidates', (_req: Request, res: Response) => {
  const sorted = [...candidates].sort((a, b) => b.overallScore - a.overallScore)
  res.json({ candidates: sorted, total: sorted.length })
})

app.get('/api/shortlist', (_req: Request, res: Response) =>
  res.json({ candidates: candidates.filter(c => ['STRONG_HIRE','SHORTLIST'].includes(c.recommendation)) }))

app.get('/api/candidates/:id', (req: Request, res: Response) => {
  const c = candidates.find(c => c.sessionId === req.params.id || c.candidateId === req.params.id)
  c ? res.json(c) : res.status(404).json({ error: 'not_found' })
})

app.post('/api/screen/start', async (req: Request, res: Response) => {
  const { candidate_id } = req.body as { candidate_id?: string }
  const candidateId = candidate_id || `c_${Date.now()}`
  const sessionId   = `hf_${candidateId}_${Date.now()}`
  try {
    const session: Session = await tn.sessions.create(GOAL_ID, { sessionId })
    sessions.set(sessionId, { sessionId, candidateId, startedAt: Date.now(), lastActive: Date.now(), turnCount: 0, pct: 0 })
    res.json({ session_id: sessionId, agent_message: session.agentMessage, completion_pct: 0 })
  } catch (err) { res.status(500).json({ error: String(err) }) }
})

app.post('/api/screen/message', async (req: Request, res: Response) => {
  const { session_id, text } = req.body as { session_id: string; text: string }
  if (!session_id || !text) { res.status(400).json({ error: 'session_id and text required' }); return }
  const meta = sessions.get(session_id)
  if (!meta) { res.status(404).json({ error: 'session not found' }); return }
  try {
    const result: MessageResult = await tn.sessions.message(session_id, text)
    meta.lastActive = Date.now(); meta.turnCount++; meta.pct = result.completionPct
    sessions.set(session_id, meta)
    const resp: Record<string, unknown> = { text: result.text, is_complete: result.isComplete, completion_pct: result.completionPct }
    if (result.isComplete) {
      let output: Output | null = result.output ?? await tn.sessions.output(session_id).catch(() => null)
      if (output) {
        const candidate = parseCandidate(session_id, meta.candidateId, output.content as Record<string, unknown>)
        candidates.push(candidate)
        sessions.delete(session_id)
        await tn.sessions.end(session_id).catch(() => {})
        if (ATS_WEBHOOK) { fetch(ATS_WEBHOOK, { method:'POST', body:JSON.stringify(candidate), headers:{'Content-Type':'application/json'} }).catch(() => {}) }
        console.log(`✅ ${candidate.name} → ${candidate.recommendation} (${candidate.overallScore}/100)`)
        resp.scorecard = candidate
      }
    }
    res.json(resp)
  } catch (err) { res.status(500).json({ error: String(err) }) }
})

async function runConsoleDemo(): Promise<void> {
  console.log(`\n  HireFlow (TypeScript) — ${COMPANY_NAME} / ${ROLE_TITLE}\n`)
  const sessionId = `hf_demo_${Date.now()}`
  try {
    const session = await tn.sessions.create(GOAL_ID, { sessionId })
    console.log(`  HireFlow: ${session.agentMessage}\n`)
    const rl  = readline.createInterface({ input: process.stdin, output: process.stdout })
    const ask = () => new Promise<string>(r => rl.question('  Candidate: ', r))
    while (true) {
      const input = (await ask()).trim()
      if (!input || input === 'quit') break
      const result = await tn.sessions.message(sessionId, input)
      console.log(`\n  HireFlow: ${result.text}\n`)
      if (result.isComplete && result.output) {
        const c = parseCandidate(sessionId, 'demo', result.output.content as Record<string, unknown>)
        console.log(`\n  SCORECARD\n  Name: ${c.name}\n  Score: ${c.overallScore}/100\n  Recommendation: ${c.recommendation}`)
        console.log(`  Tech: ${c.techScore}  Culture: ${c.cultureScore}  Salary: ${c.salaryFit}`)
        if (c.strengths.length) { console.log('\n  Strengths:'); c.strengths.forEach(s => console.log(`    + ${s}`)) }
        if (c.concerns.length)  { console.log('\n  Concerns:');  c.concerns.forEach(s => console.log(`    - ${s}`)) }
        break
      }
    }
    rl.close()
    await tn.sessions.end(sessionId).catch(() => {})
  } catch (err) { console.error('Error:', err) }
}

async function main(): Promise<void> {
  console.log(`\n  HireFlow (TypeScript) — ${COMPANY_NAME} / ${ROLE_TITLE}`)
  try {
    const h = await tn.health()
    console.log(`  ✓ TrueNorth: ${h.status} @ ${TN_URL}`)
  } catch {
    console.warn(`  ⚠  TrueNorth not reachable at ${TN_URL}`)
    console.warn('  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000')
  }
  if (CONSOLE_MODE) { await runConsoleDemo(); return }
  app.listen(PORT, () => {
    console.log(`\n  ✓ http://localhost:${PORT}/dashboard`)
    console.log(`  POST http://localhost:${PORT}/api/screen/start`)
    console.log(`  POST http://localhost:${PORT}/api/screen/message\n`)
  })
}

main().catch(console.error)
