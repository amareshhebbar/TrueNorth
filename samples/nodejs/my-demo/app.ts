import * as fs       from 'fs'
import * as readline from 'readline'

import { TrueNorth, Session, MessageResult } from '../../../packages/sdk-node'

const GOAL_ID     = process.env.GOAL_ID           ?? 'patient_intake'
const TN_URL      = process.env.TRUENORTH_BASE_URL ?? 'http://localhost:8000'
const TN_KEY      = process.env.TRUENORTH_API_KEY  ?? ''
const SCRIPTED    = process.env.SCRIPTED           === '1'
const OUTPUT_FILE = process.env.OUTPUT_FILE        ?? ''

const R   = '\x1b[0m'
const B   = '\x1b[1m'
const DIM = '\x1b[2m'
const GRN = '\x1b[32m'
const CYN = '\x1b[36m'
const YLW = '\x1b[33m'

const col = (s: string, ...codes: string[]) => codes.join('') + s + R

const pctLabel = (p: number): string => {
  const label = `[${p.toFixed(0)}%]`.padEnd(6)
  if (p >= 100) return col(label, GRN, B)
  if (p >= 50)  return col(label, CYN, B)
  return col(label, YLW, B)
}

const SCRIPTED_ANSWERS = [
  'Priya Nair',
  '28',
  'Severe headache and fever for 3 days',
  '7',
  '3 days, started Monday morning',
  'Gets worse when I sit up quickly',
  'Paracetamol 500mg helps a bit',
  'Paracetamol 500mg, as needed',
  'No known allergies',
  'Nothing significant',
  'Husband Rajan Kumar, 9876543210',
]

function ask(rl: readline.Interface, prompt: string): Promise<string> {
  return new Promise(resolve => rl.question(prompt, resolve))
}

async function main(): Promise<void> {
  const sessionId = `demo_${new Date().toISOString().replace(/[-T:Z.]/g, '').slice(0, 15)}`
  const startTime = Date.now()

  console.log()
  console.log(col(`  TrueNorth Demo (TypeScript) — ${GOAL_ID}`, B, CYN))
  console.log('  ' + '─'.repeat(50))

  const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

  try {
    const h = await tn.health()
    console.log(`  ${col('✓', GRN)} Connected: ${TN_URL} (${h.status})`)
  } catch {
    console.log(`  ${col('⚠', YLW)} TrueNorth API not reachable at ${TN_URL}`)
    console.log(col('  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000\n', DIM))
  }

  if (SCRIPTED) {
    console.log(col('  Mode: SCRIPTED — pre-written answers will be used', CYN))
  }
  console.log()

  let session: Session
  try {
    session = await tn.sessions.create(GOAL_ID, { sessionId })
  } catch (err) {
    console.error(col(`  ✗ Could not create session: ${err}`, YLW))
    process.exit(1)
  }

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout })

  console.log(`  ${pctLabel(0)} Agent: ${session.agentMessage}\n`)

  let turn       = 0
  let complete   = false
  let scriptIdx  = 0

  while (!complete) {
    let userInput: string

    if (SCRIPTED) {
      if (scriptIdx >= SCRIPTED_ANSWERS.length) {
        console.log(col('  (script exhausted — switching to interactive)', YLW))
      } else {
        userInput = SCRIPTED_ANSWERS[scriptIdx++]
        console.log(`  You:   ${col(userInput, B)}`)
        await new Promise(r => setTimeout(r, 400))
      }
    } else {
      userInput = (await ask(rl, '  You:   ')).trim()
    }

    if (!userInput!) continue
    if (userInput === 'quit' || userInput === 'q') {
      console.log(col('\n  Exited early.\n', YLW))
      break
    }

    let result: MessageResult
    try {
      result = await tn.sessions.message(sessionId, userInput)
    } catch (err) {
      console.log(col(`  Error: ${err}`, YLW))
      continue
    }

    turn++
    console.log(`\n  ${pctLabel(result.completionPct)} Agent: ${result.text}\n`)
    complete = result.isComplete

    if (result.isComplete && result.output) {
      const elapsed = (Date.now() - startTime) / 1000

      console.log('  ' + '─'.repeat(60))
      console.log(col('  OUTPUT', B, CYN))
      console.log('  ' + '─'.repeat(60))
      console.log(JSON.stringify(result.output.content, null, 2)
        .split('\n').map(l => '  ' + l).join('\n'))
      console.log('  ' + '─'.repeat(60))
      console.log(`  Session : ${col(sessionId, DIM)}`)
      console.log(`  Turns   : ${turn}`)
      console.log(`  Time    : ${elapsed.toFixed(1)} seconds`)
      console.log(`  Cost    : ~$${(turn * 0.00015).toFixed(5)} (estimated)\n`)

      if (OUTPUT_FILE) {
        const out = {
          session_id:  sessionId,
          goal_id:     GOAL_ID,
          turns:       turn,
          elapsed_sec: elapsed,
          output:      result.output.content,
        }
        fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2), 'utf-8')
        console.log(`  ${col('✓', GRN)} Saved to ${OUTPUT_FILE}\n`)
      }
    }
  }

  rl.close()
  await tn.sessions.end(sessionId).catch(() => {})
  console.log(col('  Done.\n', DIM))
}

main().catch(err => { console.error(err); process.exit(1) })
