/**
 * legal-aid-cli (Node.js / TypeScript) — Terminal Legal Intake for NGO Workers
 * ==============================================================================
 *
 * WHAT THIS DOES
 * ──────────────────────────────────────────────────────────────────────────────
 * A colour-formatted terminal app for legal aid workers at NGOs.
 * Worker sits with the client, runs this tool, gets a structured case brief.
 *
 * Output:
 *   • Colour case brief printed in the terminal
 *   • Saves  output/<session>_<name>.json   (for the database)
 *   • Saves  output/<session>_<name>.txt    (to hand to the advocate)
 *
 * NO SERVER. NO DATABASE. NO DEPLOYMENT.
 * Just Node + your API key. Runs on any machine.
 *
 * This is the TypeScript equivalent of legal-aid-cli/app.py
 *
 * FILE STRUCTURE
 * ──────────────────────────────────────────────────────────────────────────────
 *   legal-aid-cli/
 *   ├── app.ts          ← this file
 *   ├── goal.yaml       ← copy from python/legal-aid-cli/goal.yaml
 *   ├── package.json
 *   └── tsconfig.json
 *
 * package.json:
 * ──────────────────────────────────────────────────────────────────────────────
 *   {
 *     "scripts": {
 *       "dev":   "ts-node app.ts",
 *       "build": "tsc",
 *       "start": "node dist/app.js"
 *     },
 *     "dependencies": {},
 *     "devDependencies": {
 *       "@types/node": "^20.0.0",
 *       "ts-node": "^10.9.0",
 *       "typescript": "^5.3.0"
 *     }
 *   }
 *
 * tsconfig.json:
 * ──────────────────────────────────────────────────────────────────────────────
 *   {
 *     "compilerOptions": {
 *       "target": "ES2020", "module": "commonjs",
 *       "lib": ["ES2020"], "outDir": "./dist",
 *       "strict": true, "esModuleInterop": true,
 *       "skipLibCheck": true, "types": ["node"]
 *     }
 *   }
 *
 * INSTALL
 * ──────────────────────────────────────────────────────────────────────────────
 *   # Step 1: TrueNorth Python API
 *   cd packages/core && uvicorn truenorth.api.main:app --port 8000
 *
 *   # Step 2:
 *   cd samples/nodejs/legal-aid-cli
 *   npm install
 *
 * HOW TO RUN
 * ──────────────────────────────────────────────────────────────────────────────
 *   export TRUENORTH_BASE_URL=http://localhost:8000
 *   npx ts-node app.ts
 *
 *   # With worker ID
 *   WORKER_ID=LW-042 npx ts-node app.ts
 *
 *   # Use a different goal
 *   GOAL_ID=farm_advisory npx ts-node app.ts
 *
 *   # Save output to custom directory
 *   OUTPUT_DIR=/home/worker/cases npx ts-node app.ts
 *
 * WHAT YOU SEE
 * ──────────────────────────────────────────────────────────────────────────────
 *   ╔══════════════════════════════════════════════════════╗
 *   ║      LEGAL AID INTAKE  —  TrueNorth  (Node.js)      ║
 *   ║      Free Legal Assistance Programme                 ║
 *   ╚══════════════════════════════════════════════════════╝
 *
 *     Worker : LW-042   Date : 15 June 2025   Session : la_20250615_143022
 *
 *   Assistant : Namaste! I am here to help prepare your legal matter.
 *   ──────────────────────────────────────────────────────────────────
 *   Client    :
 *
 *   [████████░░░░░░░░░░░░] 40%  Turn 5
 *
 *   ✓ client_name : Ramesh Kumar  (92%)
 *
 *   ┌── CASE BRIEF ─────────────────────────────────────────┐
 *   │ Case type    : Wage Theft                              │
 *   │ Strength     : STRONG                                  │
 *   │ Limitation   : File before 14 June 2027               │
 *   │ Free aid     : ✅ YES                                  │
 *   │ Est. amount  : ₹45,000 – ₹1,20,000                   │
 *   └────────────────────────────────────────────────────────┘
 *
 *   Saved: output/la_20250615_143022_ramesh-kumar.json
 *          output/la_20250615_143022_ramesh-kumar.txt
 */

import * as fs       from 'fs'
import * as path     from 'path'
import * as readline from 'readline'

// Import TrueNorth Node SDK
import { TrueNorth, Session, MessageResult } from '../../../packages/sdk-node'

// ── Config (from env) ─────────────────────────────────────────────────────────

const WORKER_ID  = process.env.WORKER_ID         ?? 'LW-001'
const GOAL_ID    = process.env.GOAL_ID           ?? 'legal_aid_intake'
const OUTPUT_DIR = process.env.OUTPUT_DIR        ?? 'output'
const TN_URL     = process.env.TRUENORTH_BASE_URL ?? 'http://localhost:8000'
const TN_KEY     = process.env.TRUENORTH_API_KEY  ?? ''

// ── ANSI ──────────────────────────────────────────────────────────────────────

const R   = '\x1b[0m'
const B   = '\x1b[1m'
const DIM = '\x1b[2m'
const GRN = '\x1b[32m'
const CYN = '\x1b[36m'
const YLW = '\x1b[33m'
const RED = '\x1b[31m'
const WHT = '\x1b[97m'

const col = (s: string, ...codes: string[]) => codes.join('') + s + R

// ── Helpers ───────────────────────────────────────────────────────────────────

const progressBar = (pct: number, width = 22): string => {
  const filled = Math.min(Math.round(width * pct / 100), width)
  const bar    = '█'.repeat(filled) + '░'.repeat(width - filled)
  const clr    = pct >= 80 ? GRN : pct >= 40 ? CYN : YLW
  return col(`[${bar}]`, clr) + col(` ${pct.toFixed(0)}%`, B)
}

const truncate = (s: string, n: number) => s.length <= n ? s : s.slice(0, n)

const wordWrap = (text: string, width: number): string[] => {
  const words = text.split(/\s+/)
  const lines: string[] = []
  let line = ''
  for (const word of words) {
    if (line.length + word.length + 1 > width && line.length > 0) {
      lines.push(line); line = ''
    }
    line += (line ? ' ' : '') + word
  }
  if (line) lines.push(line)
  return lines
}

const slug = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')

const titleCase = (s: string) =>
  s.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())

// ── Banner ────────────────────────────────────────────────────────────────────

function printBanner(sessionId: string): void {
  const W = 58
  const line = (s: string) => col(`  ║${s.padStart((W-2+s.length)/2).padEnd(W-2)}║`, B, CYN)
  console.log()
  console.log(col(`  ╔${'═'.repeat(W-2)}╗`, B, CYN))
  console.log(line('LEGAL AID INTAKE  —  TrueNorth  (Node.js)'))
  console.log(line('Free Legal Assistance Programme'))
  console.log(col(`  ╚${'═'.repeat(W-2)}╝`, B, CYN))
  console.log()
  console.log(col(`  Worker  : ${WORKER_ID}   Date : ${new Date().toLocaleDateString('en-GB', {day:'2-digit',month:'long',year:'numeric'})}`, DIM))
  console.log(col(`  Session : ${sessionId}`, DIM))
  console.log()
}

// ── Field extracted ───────────────────────────────────────────────────────────

function printExtracted(field: string, value: unknown, confidence: number): void {
  const clr = confidence >= 0.85 ? GRN : confidence >= 0.60 ? YLW : RED
  console.log(col(
    `  ✓ ${field.padEnd(22)} ${String(value).slice(0, 42).padEnd(42)}  (${(confidence*100).toFixed(0)}%)`,
    clr, DIM
  ))
}

// ── Case brief ────────────────────────────────────────────────────────────────

function printCaseBrief(content: Record<string, unknown>): void {
  const W = 58
  console.log()
  console.log(col(`  ┌── CASE BRIEF ${'─'.repeat(W-14)}┐`, B, GRN))

  const pr = (label: string, val: unknown, clr = WHT) => {
    const v = String(val ?? '')
    if (!v || v === 'undefined' || v === 'null') return
    console.log(`  │ ${col(label.padEnd(14) + ': ', B, CYN)}${col(truncate(v, 37), clr)}`)
  }

  const strength = String(content.case_strength ?? '').toUpperCase()
  const sClr     = strength.includes('STRONG') ? GRN : strength.includes('MODERATE') ? YLW : RED

  pr('Case type',    content.case_type_formal)
  pr('Strength',     strength, sClr)
  pr('Limitation',   content.limitation_period)
  pr('Jurisdiction', content.jurisdiction)

  const eligible = content.eligible_for_free_aid
  if (eligible === true || String(eligible).toLowerCase() === 'true') {
    console.log(col('  │ Free aid       : ✅ YES — eligible for legal aid', GRN))
  } else {
    console.log(col('  │ Free aid       : Check criteria with advocate', DIM))
  }

  pr('Est. amount',  content.estimated_compensation_range)
  console.log(col('  │', DIM))

  const laws = content.applicable_laws as unknown[]
  if (Array.isArray(laws) && laws.length) {
    console.log(`  │ ${col('Applicable laws:', B, CYN)}`)
    laws.forEach(l => console.log(`  │   ${col('•', YLW)} ${l}`))
    console.log(col('  │', DIM))
  }

  const steps = content.immediate_steps as unknown[]
  if (Array.isArray(steps) && steps.length) {
    console.log(`  │ ${col('Immediate steps:', B, CYN)}`)
    steps.forEach((s, i) => {
      const lines = wordWrap(String(s), 46)
      lines.forEach((line, j) => {
        if (j === 0) console.log(`  │   ${col(`${i+1}.`, YLW)} ${line}`)
        else         console.log(`  │      ${line}`)
      })
    })
    console.log(col('  │', DIM))
  }

  const missing = content.documents_missing as unknown[]
  if (Array.isArray(missing) && missing.length) {
    console.log(`  │ ${col('Missing documents:', B, RED)}`)
    missing.forEach(d => console.log(`  │   ${col('✗', RED)} ${d}`))
    console.log(col('  │', DIM))
  }

  const brief = content.advocate_brief as string
  if (brief) {
    console.log(`  │ ${col('Advocate brief:', B, CYN)}`)
    wordWrap(brief, 52).forEach(line => console.log(`  │   ${line}`))
  }

  console.log(col(`  └${'─'.repeat(W-2)}┘`, B, GRN))
  console.log()
}

// ── Save output ───────────────────────────────────────────────────────────────

function saveOutput(
  sessionId: string,
  content:   Record<string, unknown>,
  collected: Record<string, unknown>,
): { jsonPath: string; txtPath: string } {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  const name = slug(String(collected.client_name ?? 'unknown'))
  const stem = `${sessionId}_${name}`

  const jsonPath = path.join(OUTPUT_DIR, `${stem}.json`)
  fs.writeFileSync(jsonPath, JSON.stringify({
    session_id:        sessionId,
    worker_id:         WORKER_ID,
    generated_at:      new Date().toISOString(),
    collected_fields:  collected,
    case_brief:        content,
  }, null, 2), 'utf-8')

  const lines: string[] = [
    'LEGAL AID INTAKE',
    `Generated  : ${new Date().toLocaleString('en-GB')}`,
    `Worker     : ${WORKER_ID}`,
    `Session    : ${sessionId}`,
    '',
    '─'.repeat(50),
    'CLIENT INFORMATION',
    '─'.repeat(50),
    ...Object.entries(collected).map(([k, v]) => `${titleCase(k).padEnd(25)}: ${v}`),
    '',
    '─'.repeat(50),
    'CASE BRIEF',
    '─'.repeat(50),
    `Case type   : ${content.case_type_formal}`,
    `Strength    : ${content.case_strength}`,
    `Limitation  : ${content.limitation_period}`,
    '',
    'Applicable laws:',
    ...((content.applicable_laws as string[]) ?? []).map(l => `  • ${l}`),
    '',
    'Immediate steps:',
    ...((content.immediate_steps as string[]) ?? []).map((s, i) => `  ${i+1}. ${s}`),
    '',
    'Advocate brief:',
    String(content.advocate_brief ?? ''),
  ]

  const txtPath = path.join(OUTPUT_DIR, `${stem}.txt`)
  fs.writeFileSync(txtPath, lines.join('\n'), 'utf-8')

  return { jsonPath, txtPath }
}

// ── Readline helper ───────────────────────────────────────────────────────────

function createRL() {
  return readline.createInterface({ input: process.stdin, output: process.stdout })
}

const askLine = (rl: readline.Interface, prompt: string): Promise<string> =>
  new Promise(resolve => rl.question(prompt, resolve))

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const sessionId = `la_${new Date().toISOString().replace(/[-T:Z.]/g, '').slice(0, 15)}`
  printBanner(sessionId)

  const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

  try {
    const h = await tn.health()
    console.log(col(`  ✓ Connected to TrueNorth API (${h.status})`, GRN))
  } catch {
    console.log(col(`  ⚠  TrueNorth API not reachable at ${TN_URL}`, YLW))
    console.log(col('  Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000\n', DIM))
  }
  console.log()

  // ✅ track completionPct locally — Session object is read-only from API
  let completionPct = 0
  let complete      = false

  let session: Session
  try {
    session       = await tn.sessions.create(GOAL_ID, { sessionId })
    completionPct = session.completionPct
  } catch (err) {
    console.error(col(`  ✗ Could not create session: ${err}`, RED))
    process.exit(1)
  }

  console.log()
  console.log(col('  Assistant : ', B, CYN) + session.agentMessage)
  console.log()

  const rl   = createRL()
  let turn   = 0

  while (!complete) {
    turn++
    process.stdout.write(`\n  ${progressBar(completionPct)}  Turn ${turn}\n\n`)
    process.stdout.write(col('  Client    : ', B, WHT))

    const text = (await askLine(rl, '')).trim()
    if (!text) continue
    if (text === 'quit' || text === 'q' || text === '/exit') {
      console.log(col('\n  Interrupted — session not saved.\n', YLW))
      rl.close()
      return
    }

    let result: MessageResult
    try {
      result = await tn.sessions.message(sessionId, text)
    } catch (err) {
      console.log(col(`  Error: ${err}`, RED))
      continue
    }

    // Show extracted fields
    for (const f of result.fieldsExtracted ?? []) {
      printExtracted(f.field, f.value, f.confidence)
    }

    console.log()
    console.log(col('  Assistant : ', B, CYN) + result.text)
    console.log()

    // ✅ track locally — don't try to mutate Session
    completionPct = result.completionPct
    complete      = result.isComplete

    if (result.isComplete && result.output) {
      const content = result.output.content as Record<string, unknown>

      // ✅ FIX: collectedFields is on Session, not MessageResult
      // Fetch full session state to get collected fields
      let collected: Record<string, unknown> = {}
      try {
        const fullSession = await tn.sessions.get(sessionId)
        collected = fullSession.collectedFields as Record<string, unknown>
      } catch {
        // If fetch fails, build from output fields as fallback
        collected = (result.output.fields ?? {}) as Record<string, unknown>
      }

      printCaseBrief(content)

      try {
        const { jsonPath, txtPath } = saveOutput(sessionId, content, collected)
        console.log(col('  Files saved:', B, GRN))
        console.log(`    📄 ${jsonPath}`)
        console.log(`    📝 ${txtPath}`)
        console.log()
        console.log(col('  Share the .txt file with the advocate.', DIM))
        console.log()
      } catch (err) {
        console.log(col(`  Could not save files: ${err}`, YLW))
      }
    }
  }

  rl.close()
  await tn.sessions.end(sessionId).catch(() => {})
}

main().catch(err => { console.error(err); process.exit(1) })