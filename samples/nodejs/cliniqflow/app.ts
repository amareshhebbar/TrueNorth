import express, { Request, Response } from 'express'
import axios   from 'axios'
import crypto  from 'crypto'
import readline from 'readline'

import { TrueNorth, Session, MessageResult, Output } from '../../../packages/sdk-node'

const CLINIC_NAME   = process.env.CLINIC_NAME          ?? 'Our Clinic'
const GOAL_ID       = process.env.GOAL_ID              ?? 'patient_intake'
const VERIFY_TOKEN  = process.env.WA_VERIFY_TOKEN      ?? 'cliniqflow-token'
const ACCESS_TOKEN  = process.env.WA_ACCESS_TOKEN      ?? ''
const PHONE_ID      = process.env.WA_PHONE_NUMBER_ID   ?? ''
const TN_URL        = process.env.TRUENORTH_BASE_URL   ?? 'http://localhost:8000'
const TN_KEY        = process.env.TRUENORTH_API_KEY    ?? ''
const PORT          = parseInt(process.env.PORT        ?? '3001')

const WA_API = `https://graph.facebook.com/v19.0/${PHONE_ID}/messages`

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

interface PatientSession {
  sessionId:   string
  phone:       string
  startedAt:   number
  lastActive:  number
  consented:   boolean
  turnCount:   number
  pct:         number
  session:     Session
}

interface CompletedIntake {
  sessionId:   string
  phone:       string
  completedAt: string
  intake:      unknown
}

const activeSessions = new Map<string, PatientSession>()
const completed: CompletedIntake[] = []

const sid = (phone: string) =>
  `cf_${crypto.createHash('md5').update(phone).digest('hex').slice(0, 12)}`

async function sendWA(phone: string, text: string): Promise<void> {
  if (!ACCESS_TOKEN || !PHONE_ID) {
    console.log(`\n📤 [${phone}]: ${text.slice(0, 200)}\n`)
    return
  }
  try {
    await axios.post(
      WA_API,
      { messaging_product: 'whatsapp', to: phone,
        type: 'text', text: { body: text.slice(0, 4000) } },
      { headers: { Authorization: `Bearer ${ACCESS_TOKEN}` }, timeout: 10_000 },
    )
  } catch (err: any) {
    console.error(`WA send error: ${err?.response?.status}`)
  }
}

async function handleMessage(phone: string, text: string): Promise<void> {
  const sessionId = sid(phone)
  const meta      = activeSessions.get(sessionId)

  if (!meta) {
    const consentMsg =
      `*${CLINIC_NAME} Intake*\n\n` +
      `We will collect your health information for your appointment today. ` +
      `This is confidential and goes only to your doctor.\n\n` +
      `Type *AGREE* to continue or *STOP* to cancel.`
    await sendWA(phone, consentMsg)

    try {
      const session = await tn.sessions.create(GOAL_ID, { sessionId })
      activeSessions.set(sessionId, {
        sessionId, phone, session,
        startedAt:  Date.now(),
        lastActive: Date.now(),
        consented:  false,
        turnCount:  0,
        pct:        0,
      })
    } catch (err) {
      console.error('Create session error:', err)
      await sendWA(phone, 'System is starting up — please try again in a moment.')
    }
    return
  }

  if (!meta.consented) {
    const lower = text.toLowerCase()
    if (['stop', 'no', 'cancel'].includes(lower)) {
      await sendWA(phone, 'No problem. Your data has not been stored. See you at the clinic.')
      activeSessions.delete(sessionId)
      return
    }
    meta.consented = true
    activeSessions.set(sessionId, meta)
    await sendWA(phone, meta.session.agentMessage)

    if (!['agree', 'yes', 'ok', 'start', 'begin'].includes(lower)) {

      const result = await tn.sessions.message(sessionId, text)
      meta.pct = result.completionPct
      meta.turnCount++
      activeSessions.set(sessionId, meta)
      await sendWA(phone, result.text)
      await checkCompletion(phone, sessionId, result)
    }
    return
  }

  try {
    const result = await tn.sessions.message(sessionId, text)
    meta.lastActive = Date.now()
    meta.turnCount++
    meta.pct        = result.completionPct
    activeSessions.set(sessionId, meta)
    await sendWA(phone, result.text)
    await checkCompletion(phone, sessionId, result)
  } catch (err) {
    console.error(`Message error ${sessionId}:`, err)
    await sendWA(phone, 'I had trouble processing that. Please try again.')
  }
}

async function checkCompletion(phone: string, sessionId: string, result: MessageResult): Promise<void> {
  if (!result.isComplete) return

  let output: Output | null = result.output
  if (!output) {
    output = await tn.sessions.output(sessionId).catch(() => null)
  }

  if (output) {
    completed.push({
      sessionId,
      phone,
      completedAt: new Date().toISOString(),
      intake:      output.content,
    })

    await sendWA(phone,
      `✅ *${CLINIC_NAME}* — Your intake is complete!\n\n` +
      `Your doctor will review it before seeing you today. Thank you! 🙏`
    )

    await tn.sessions.end(sessionId).catch(() => {})
    activeSessions.delete(sessionId)
    console.log(`✅ Intake complete for ${phone}`)
  }
}

const app = express()
app.use(express.json())

app.get('/', (_req: Request, res: Response) => {
  const rows = [...activeSessions.values()].map(m => `
    <tr><td>Patient</td><td>${m.pct.toFixed(0)}%</td>
    <td>${m.turnCount}</td>
    <td>${Math.round((Date.now() - m.lastActive) / 60000)} min ago</td></tr>`
  ).join('') || '<tr><td colspan=4 style="text-align:center;color:#94a3b8;padding:20px">No active intakes</td></tr>'

  res.type('html').send(`
    <!DOCTYPE html><html><head><title>CliniqFlow</title>
    <style>body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f0f9ff;}
    h1{color:#0369a1;}.s{display:inline-block;background:white;border-radius:10px;padding:16px 22px;
    margin:6px;border:1px solid #bae6fd;text-align:center;}
    .s h2{margin:0;color:#0369a1;font-size:1.8rem;}.s p{margin:3px 0 0;color:#64748b;font-size:.82rem;}
    table{width:100%;border-collapse:collapse;background:white;border-radius:10px;margin-top:20px;}
    th{background:#0369a1;color:white;padding:10px;text-align:left;}
    td{padding:10px;border-bottom:1px solid #e0f2fe;}</style></head><body>
    <h1>🏥 CliniqFlow (Node.js) — ${CLINIC_NAME}</h1>
    <div>
      <div class="s"><h2>${activeSessions.size}</h2><p>Active intakes</p></div>
      <div class="s"><h2>${completed.length}</h2><p>Completed today</p></div>
      <div class="s"><h2>${ACCESS_TOKEN ? '✅' : '⚠️'}</h2><p>WhatsApp</p></div>
    </div>
    <table><thead><tr><th>Patient</th><th>Progress</th><th>Turns</th><th>Last active</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p><small><a href="/health">Health</a> | <a href="/completed">Completed (JSON)</a></small></p>
    </body></html>`)
})

app.get('/health', async (_req: Request, res: Response) => {
  try {
    const h = await tn.health()
    res.json({ status: 'ok', clinic: CLINIC_NAME, goal: GOAL_ID,
               truenorth: h, active: activeSessions.size,
               completed: completed.length,
               wa_connected: !!(ACCESS_TOKEN && PHONE_ID) })
  } catch {
    res.status(503).json({ status: 'truenorth_unreachable' })
  }
})

app.get('/completed', (_req: Request, res: Response) => {
  res.json({ intakes: completed, total: completed.length })
})

app.delete('/sessions/:id', async (req: Request, res: Response) => {
  const { id } = req.params
  if (!activeSessions.has(id)) { res.status(404).json({ error: 'not_found' }); return }
  activeSessions.delete(id)
  await tn.sessions.end(id).catch(() => {})
  res.json({ deleted: id, status: 'erased' })
})

app.get('/webhook', (req: Request, res: Response) => {
  if (req.query['hub.mode'] === 'subscribe' &&
      req.query['hub.verify_token'] === VERIFY_TOKEN) {
    res.send(req.query['hub.challenge'] as string)
  } else {
    res.status(403).send('Bad token')
  }
})

app.post('/webhook', (req: Request, res: Response) => {
  res.status(200).json({ status: 'ok' })
  ;(async () => {
    try {
      const msgs = req.body?.entry?.[0]?.changes?.[0]?.value?.messages as any[]
      if (!msgs?.length) return
      const phone = msgs[0].from as string
      const text  = msgs[0].text?.body?.trim() as string
      if (phone && text) {
        console.log(`📨 ${phone}: ${text.slice(0, 60)}`)
        await handleMessage(phone, text)
      }
    } catch (err) { console.error('Webhook error:', err) }
  })()
})

async function runConsoleDemo(): Promise<void> {
  console.log(`\n  CliniqFlow (Node.js) — ${CLINIC_NAME}`)
  console.log('  CONSOLE MODE — simulating a patient conversation')
  console.log("  Type 'quit' to exit\n")

  const phone = '919999999999'
  const id    = sid(phone)

  try {
    const session = await tn.sessions.create(GOAL_ID, { sessionId: id })
    activeSessions.set(id, {
      sessionId: id, phone, session,
      startedAt: Date.now(), lastActive: Date.now(),
      consented: true, turnCount: 0, pct: 0,
    })
    console.log(`Bot: ${session.agentMessage}\n`)

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
    const ask = () => new Promise<string>(r => rl.question('Patient: ', r))

    while (true) {
      const input = (await ask()).trim()
      if (!input || input === 'quit') break

      const result = await tn.sessions.message(id, input)
      console.log(`\nBot: ${result.text}\n`)

      if (result.isComplete && result.output) {
        console.log('── INTAKE COMPLETE ───────────────────────────────────')
        console.log(JSON.stringify(result.output.content, null, 2))
        break
      }
    }
    rl.close()
  } catch (err) {
    console.error('Error:', err)
    console.error(`Is TrueNorth API running at ${TN_URL}?`)
  }
}

async function main(): Promise<void> {
  console.log(`\n  CliniqFlow (Node.js / TypeScript)`)
  console.log(`  ${CLINIC_NAME} | Goal: ${GOAL_ID}`)

  try {
    const h = await tn.health()
    console.log(`  ✓ TrueNorth API: ${h.status} @ ${TN_URL}`)
  } catch {
    console.warn(`  ⚠  TrueNorth not reachable at ${TN_URL}`)
    console.warn('  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000')
  }

  if (!ACCESS_TOKEN) {
    console.log('\n  ⚠  WA_ACCESS_TOKEN not set — CONSOLE MODE\n')
    await runConsoleDemo()
    return
  }

  app.listen(PORT, () => {
    console.log(`\n  ✓ Server running at http://localhost:${PORT}`)
    console.log(`  Dashboard: http://localhost:${PORT}/`)
    console.log(`  Webhook:   http://localhost:${PORT}/webhook\n`)
  })
}

main().catch(console.error)
