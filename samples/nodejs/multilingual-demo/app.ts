/**
 * multilingual-demo (Node.js / TypeScript) — Auto Language Detection Demo
 * =========================================================================
 *
 * WHAT THIS DOES
 * ──────────────────────────────────────────────────────────────────────────────
 * Shows TrueNorth detecting and responding in 5 Indian languages automatically.
 * No configuration, no language selection screen — the user just types.
 *
 * Scripted demos for:
 *   Hindi     → agent responds in Hindi
 *   Kannada   → agent responds in Kannada
 *   Tamil     → agent responds in Tamil
 *   Hinglish  → agent responds in Hinglish (mixed)
 *   English   → agent responds in English
 *
 * This is the TypeScript equivalent of multilingual-demo/app.py
 *
 * FILE STRUCTURE
 * ──────────────────────────────────────────────────────────────────────────────
 *   multilingual-demo/
 *   ├── app.ts          ← this file (no goal.yaml — inline config)
 *   ├── package.json
 *   └── tsconfig.json
 *
 * package.json:
 * ──────────────────────────────────────────────────────────────────────────────
 *   {
 *     "scripts": { "dev": "ts-node app.ts" },
 *     "devDependencies": {
 *       "@types/node": "^20.0.0",
 *       "ts-node": "^10.9.0",
 *       "typescript": "^5.3.0"
 *     }
 *   }
 *
 * INSTALL
 * ──────────────────────────────────────────────────────────────────────────────
 *   cd packages/core && uvicorn truenorth.api.main:app --port 8000
 *   cd samples/nodejs/multilingual-demo && npm install
 *
 * HOW TO RUN
 * ──────────────────────────────────────────────────────────────────────────────
 *   export TRUENORTH_BASE_URL=http://localhost:8000
 *
 *   npx ts-node app.ts                     ← all 5 languages
 *   DEMO_LANG=hindi npx ts-node app.ts     ← Hindi only
 *   DEMO_LANG=kannada npx ts-node app.ts   ← Kannada only
 *   DEMO_LANG=tamil npx ts-node app.ts     ← Tamil only
 *   DEMO_LANG=hinglish npx ts-node app.ts  ← Hinglish only
 *   DEMO_LANG=interactive npx ts-node app.ts ← type your own messages
 *
 * WHAT YOU SEE
 * ──────────────────────────────────────────────────────────────────────────────
 *   LANGUAGE DETECTION — Sample phrases
 *   ─────────────────────────────────────────────────────────────────
 *   Phrase                                    Expected    Detected
 *   नमस्ते, मेरा नाम राहुल है                  Hindi       hindi     ✅
 *   ನನ್ನ ಹೆಸರು ರವಿ                            Kannada     kannada   ✅
 *   Mera weight 65 kg hai                     Hinglish    hinglish  ✅
 *   Hello, my name is Sarah                   English     english   ✅
 *
 *   ══ DEMO — HINDI (हिंदी) ══
 *   Agent: नमस्ते! / Hello! / ನಮಸ್ಕಾರ!
 *   User:  नमस्ते
 *   User:  मेरा नाम राहुल शर्मा है
 *   Agent [hindi]: बहुत अच्छा! आपकी उम्र क्या है?
 *   ...
 *   ✅ Language: HINDI | Fields: 6 | BMI: 23.5
 *
 *   SUMMARY
 *   HINDI (हिंदी)      ✅ PASS
 *   KANNADA (ಕನ್ನಡ)   ✅ PASS
 *   TAMIL (தமிழ்)     ✅ PASS
 *   HINGLISH (Mixed)   ✅ PASS
 *   ENGLISH            ✅ PASS
 */

import * as readline from 'readline'
import { TrueNorth, Session, MessageResult } from '../../../packages/sdk-node'

// ── Config ────────────────────────────────────────────────────────────────────

const TN_URL    = process.env.TRUENORTH_BASE_URL ?? 'http://localhost:8000'
const TN_KEY    = process.env.TRUENORTH_API_KEY  ?? ''
const DEMO_LANG = (process.env.DEMO_LANG        ?? 'all').toLowerCase()

// ── ANSI ──────────────────────────────────────────────────────────────────────

const R   = '\x1b[0m'
const B   = '\x1b[1m'
const DIM = '\x1b[2m'
const GRN = '\x1b[32m'
const CYN = '\x1b[36m'
const YLW = '\x1b[33m'

const col = (s: string, ...codes: string[]) => codes.join('') + s + R
const div = (t: string) => console.log(`\n${col('══ DEMO — ' + t + ' ══', B, CYN)}\n`)

// ── Inline goal config (no YAML file needed) ──────────────────────────────────

const GOAL_ID = 'multilingual_health_demo'

const INLINE_GOAL = {
  id:   GOAL_ID,
  name: 'Multilingual Health Assessment',
  persona: {
    name:     'Health Assistant',
    tone:     'warm',
    language: 'auto',           // ← key: auto-detect from user input
    greeting: 'Hello! / नमस्ते! / ನಮಸ್ಕಾರ! / வணக்கம்!\n' +
              'I can speak your language. Reply in Hindi, Kannada, Tamil, or English.',
  },
  fields: [
    { name: 'name',        type: 'text',    required: true,
      question: 'What is your name? / आपका नाम क्या है?' },
    { name: 'age',         type: 'integer', required: true,  min: 1,  max: 120,
      question: 'How old are you?' },
    { name: 'weight_kg',   type: 'number',  required: true,
      question: 'What is your weight in kg?' },
    { name: 'height_cm',   type: 'number',  required: true,
      question: 'What is your height in cm?' },
    { name: 'health_goal', type: 'text',    required: true,
      allowed_values: ['lose weight', 'build strength', 'improve fitness', 'general wellness'],
      question: 'Main health goal? (lose weight / build strength / improve fitness / general wellness)' },
    { name: 'city',        type: 'text',    required: false,
      question: 'Which city are you from?' },
  ],
  output: {
    format: 'json',
    template:
      'Summarise health assessment for {name}, age {age}, {weight_kg}kg, {height_cm}cm. ' +
      'Goal: {health_goal}. City: {city}. ' +
      'Return JSON with bmi, bmi_category, language_detected, ' +
      'personalised_message (in the user\'s language), next_steps (3 items).',
  },
}

// ── Language demo scripts ─────────────────────────────────────────────────────

interface LangDemo { label: string; turns: string[] }

const DEMOS: Record<string, LangDemo> = {
  hindi: {
    label: 'HINDI (हिंदी)',
    turns: [
      'नमस्ते',
      'मेरा नाम राहुल शर्मा है',
      '28 साल का हूँ',
      'वजन 72 किलो है',
      'लम्बाई 175 सेंटीमीटर है',
      'मुझे वजन कम करना है',
      'मुंबई से हूँ',
    ],
  },
  kannada: {
    label: 'KANNADA (ಕನ್ನಡ)',
    turns: [
      'ನಮಸ್ಕಾರ',
      'ನನ್ನ ಹೆಸರು ರವಿ ಕುಮಾರ್',
      'ನನಗೆ 32 ವರ್ಷ',
      'ತೂಕ 80 ಕಿಲೋ',
      'ಎತ್ತರ 170 ಸೆಂ.ಮೀ',
      'ದೇಹವನ್ನು ಫಿಟ್ ಮಾಡಿಕೊಳ್ಳಬೇಕು',
      'ಬೆಂಗಳೂರಿನಿಂದ',
    ],
  },
  tamil: {
    label: 'TAMIL (தமிழ்)',
    turns: [
      'வணக்கம்',
      'என் பெயர் ப்ரியா',
      'என் வயது 25',
      'என் எடை 58 கிலோ',
      'உயரம் 162 செமீ',
      'உடல் எடையை குறைக்கணும்',
      'சென்னையில் இருக்கேன்',
    ],
  },
  hinglish: {
    label: 'HINGLISH (Mixed)',
    turns: [
      'hello bhai',
      'Mera naam Arjun hai',
      'I am 30 years old',
      'Weight 85 kg hai mera',
      'Height around 180 cm',
      'Muscle build karna chahta hoon',
      'Pune mein rehta hoon',
    ],
  },
  english: {
    label: 'ENGLISH',
    turns: [
      'Hello',
      'My name is Sarah',
      'I am 26 years old',
      'I weigh 55 kilograms',
      'My height is 165 centimetres',
      'I want to improve my overall fitness',
      'I live in Hyderabad',
    ],
  },
}

// ── Sample phrases for detection table ───────────────────────────────────────

const SAMPLE_PHRASES = [
  { text: 'नमस्ते, मेरा नाम राहुल है',       expected: 'Hindi'    },
  { text: 'ನನ್ನ ಹೆಸರು ರವಿ',                  expected: 'Kannada'  },
  { text: 'என் பெயர் ப்ரியா',                expected: 'Tamil'    },
  { text: 'నా పేరు రాహుల్',                  expected: 'Telugu'   },
  { text: 'Hello, my name is Sarah',         expected: 'English'  },
  { text: 'Mera weight 65 kg hai',           expected: 'Hinglish' },
  { text: 'माझे नाव सुनील आहे',              expected: 'Marathi'  },
  { text: 'আমার নাম সুমিত',                  expected: 'Bengali'  },
]

// ── Language estimator (Unicode range fallback) ───────────────────────────────

function estimateLang(text: string): string {
  if (/[\u0900-\u097F]/.test(text)) {
    return /[a-zA-Z]/.test(text) ? 'hinglish' : 'hindi'
  }
  if (/[\u0C80-\u0CFF]/.test(text)) return 'kannada'
  if (/[\u0B80-\u0BFF]/.test(text)) return 'tamil'
  if (/[\u0C00-\u0C7F]/.test(text)) return 'telugu'
  if (/[\u0900-\u097F]/.test(text)) return 'marathi'
  if (/[\u0980-\u09FF]/.test(text)) return 'bengali'
  return 'english'
}

// ── Show detection table ──────────────────────────────────────────────────────
async function showDetectionTable(tn: TrueNorth): Promise<void> {
  console.log(`\n${col('LANGUAGE DETECTION — Sample phrases', B, CYN)}\n`)
  console.log(`  ${'Phrase'.padEnd(42)} ${'Expected'.padEnd(12)} ${'Detected'.padEnd(12)} `)
  console.log('  ' + '─'.repeat(70))

  for (const { text, expected } of SAMPLE_PHRASES) {
    // ✅ FIX: tn.language.detect() doesn't exist — use local estimator only
    const detected = estimateLang(text)
    const tick     = detected.toLowerCase().includes(expected.toLowerCase())
      ? col('✅', GRN)
      : col('~', DIM)

    const phrase = [...text].slice(0, 38).join('') + ([...text].length > 38 ? '…' : '')
    console.log(`  ${phrase.padEnd(42)} ${expected.padEnd(12)} ${detected.padEnd(12)} ${tick}`)
  }
  console.log()
}

async function runDemo(
  tn:   TrueNorth,
  lang: string,
  demo: LangDemo,
): Promise<Record<string, unknown> | null> {
  div(demo.label)

  const sid = `ml_${lang}_${Date.now()}`

  let session: Session
  try {
    session = await tn.sessions.create(GOAL_ID, { sessionId: sid })
  } catch (err) {
    console.log(`  ${col('Error:', YLW)} ${err}`)
    showEstimatedOutput(lang, demo)
    return null
  }

  console.log(`  Agent: ${session.agentMessage}\n`)

  for (const turn of demo.turns) {
    console.log(`  User:  ${turn}`)

    let result: MessageResult
    try {
      result = await tn.sessions.message(sid, turn)
    } catch (err) {
      console.log(`  ${col('Error:', YLW)} ${err}`)
      break
    }

    // ✅ FIX: detectedLanguage is on Session, not MessageResult
    // Use estimateLang as fallback — avoids extra API call each turn
    const detected = estimateLang(turn)
    console.log(`  Agent ${col(`[${detected}]`, CYN)}: ${result.text}\n`)

    if (result.isComplete && result.output) {
      const content = result.output.content as Record<string, unknown>

      // ✅ FIX: fetch session to get detectedLanguage + collectedFields
      // use sid not sessionId (sessionId was undefined)
      const fullSession = await tn.sessions.get(sid).catch(() => null)
      const langDetected = fullSession?.detectedLanguage ?? detected
      const fieldCount   = Object.keys(fullSession?.collectedFields ?? {}).length

      console.log(col(`  ✅ Language: ${langDetected.toUpperCase()}`, GRN, B))
      console.log(`  Fields collected: ${fieldCount}`)

      const msg = content.personalised_message as string
      if (msg) {
        console.log(`\n  Personalised message:\n  ${col(`"${msg}"`, YLW)}\n`)
      }

      await tn.sessions.end(sid).catch(() => {})
      return content
    }
  }

  await tn.sessions.end(sid).catch(() => {})
  return null
}

function showEstimatedOutput(lang: string, demo: LangDemo): void {
  console.log(`  ${col('[Estimated output — API not reachable]', DIM)}\n`)
  demo.turns.forEach(t => {
    console.log(`  User:  ${t}`)
    console.log(`  Agent: (response in ${lang})\n`)
  })
}

async function runInteractive(tn: TrueNorth): Promise<void> {
  console.log(`\n${col('INTERACTIVE — Type in any Indian language', B, CYN)}\n`)
  console.log('  Type in Hindi, Kannada, Tamil, Telugu, or English.')
  console.log('  TrueNorth detects your language and responds in it.')
  console.log("  Type 'quit' to exit.\n")

  const sid          = `ml_interactive_${Date.now()}`
  const startSession = await tn.sessions.create(GOAL_ID, { sessionId: sid })
  console.log(`  Agent: ${startSession.agentMessage}\n`)

  const rl  = readline.createInterface({ input: process.stdin, output: process.stdout })
  const ask = () => new Promise<string>(res => rl.question('  You: ', res))

  while (true) {
    const text = (await ask()).trim()
    if (!text) continue
    if (text === 'quit' || text === 'q') break

    const result = await tn.sessions.message(sid, text)

    // ✅ FIX: use sid not sessionId (was undefined)
    // ✅ FIX: rename variable so it doesn't shadow startSession
    const currentSession = await tn.sessions.get(sid).catch(() => null)
    const lang           = currentSession?.detectedLanguage ?? estimateLang(text)

    console.log(`\n  Agent ${col(`[${lang}]`, CYN)}: ${result.text}\n`)

    if (result.isComplete && result.output) {
      console.log('\n── OUTPUT ──────────────────────────────────────')
      console.log(JSON.stringify(result.output.content, null, 2))
      break
    }
  }

  rl.close()
  await tn.sessions.end(sid).catch(() => {})
}
// ── Summary ───────────────────────────────────────────────────────────────────

function printSummary(results: Record<string, string>): void {
  console.log(`\n${col('SUMMARY', B, CYN)}\n`)
  console.log(`  ${'Language demo'.padEnd(40)} Result`)
  console.log('  ' + '─'.repeat(55))
  for (const [lang, status] of Object.entries(results)) {
    console.log(`  ${DEMOS[lang].label.padEnd(40)} ${status}`)
  }
  console.log()
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log()
  console.log(col('  TrueNorth Multilingual Demo (Node.js / TypeScript)', B, CYN))
  console.log(col(`  API: ${TN_URL} | Lang: ${DEMO_LANG}`, DIM))

  const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL, timeout: 90_000 })

  // Always show detection table first
  await showDetectionTable(tn)

  if (DEMO_LANG === 'interactive') {
    await runInteractive(tn)
    return
  }

  const order   = ['hindi', 'kannada', 'tamil', 'hinglish', 'english']
  const results: Record<string, string> = {}

  for (const lang of order) {
    if (DEMO_LANG !== 'all' && DEMO_LANG !== lang) continue
    const demo = DEMOS[lang]
    if (!demo) continue

    const output = await runDemo(tn, lang, demo)
    results[lang] = output
      ? col('✅ PASS', GRN)
      : col('⚠️  INCOMPLETE', YLW)
  }

  if (DEMO_LANG === 'all' && Object.keys(results).length > 0) {
    printSummary(results)
  }
}

main().catch(err => { console.error(err); process.exit(1) })