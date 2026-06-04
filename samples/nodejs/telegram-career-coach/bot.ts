/**
 * Career Coach — Telegram Bot (TypeScript)
 * ==========================================
 *
 * WHAT THIS IS
 * ─────────────────────────────────────────
 * A Telegram bot that runs a personalised career coaching
 * session. User types /start in Telegram, the bot collects
 * their background and goals, and returns a concrete
 * 3-month roadmap to reach their target role.
 *
 * PROJECT STRUCTURE
 * ─────────────────────────────────────────
 *   telegram-career-coach/
 *   ├── bot.ts          ← this file
 *   ├── goal.yaml       ← career coaching questionnaire
 *   ├── package.json
 *   └── tsconfig.json
 *
 * package.json:
 * ─────────────────────────────────────────
 *   {
 *     "scripts": { "dev": "ts-node bot.ts" },
 *     "dependencies": { "node-telegram-bot-api": "^0.66.0" },
 *     "devDependencies": {
 *       "@types/node-telegram-bot-api": "^0.66.0",
 *       "@types/node": "^20.0.0",
 *       "ts-node": "^10.9.0",
 *       "typescript": "^5.3.0"
 *     }
 *   }
 *
 * INSTALL
 * ─────────────────────────────────────────
 *   cd sample-projects-node/telegram-career-coach
 *   npm install
 *
 * GET TELEGRAM BOT TOKEN
 * ─────────────────────────────────────────
 *   1. Open Telegram → search @BotFather
 *   2. Send /newbot
 *   3. Follow prompts — name your bot
 *   4. Copy the token (looks like 123456:ABCdef...)
 *
 * HOW TO RUN
 * ─────────────────────────────────────────
 *   # Terminal 1: TrueNorth Python API
 *   cd packages/core && uvicorn truenorth.api.main:app --port 8000
 *
 *   # Terminal 2: Telegram bot
 *   export TELEGRAM_BOT_TOKEN=123456:ABCdef...
 *   export TRUENORTH_BASE_URL=http://localhost:8000
 *   npx ts-node bot.ts
 *
 *   # Open Telegram, find your bot, send /start
 *
 * TELEGRAM COMMANDS
 * ─────────────────────────────────────────
 *   /start   — begin career coaching session
 *   /status  — show current session progress
 *   /reset   — clear session and start over
 *   /help    — show help message
 *
 * WHAT USERS SEE
 * ─────────────────────────────────────────
 *   User:  /start
 *   Bot:   Hey! I am your AI career coach...
 *          What's your name?
 *   User:  Priya
 *   Bot:   Nice to meet you, Priya! What's your
 *          current role and how long have you been in it?
 *   ...
 *   Bot:   🎯 Your Career Roadmap
 *
 *          *Readiness score: 62/100*
 *
 *          *Skill gaps to close:*
 *          • System design (your biggest gap)
 *          • SQL and data modelling
 *          • Product thinking
 *
 *          *Month 1-3: Foundation*
 *          • Complete a system design course
 *          • Build one real project with a database
 *          ...
 */

import TelegramBot from 'node-telegram-bot-api'
import crypto      from 'crypto'
import { TrueNorth, Session } from '../../../packages/sdk-node'

// ── Config ────────────────────────────────────────────────────────────────────

const BOT_TOKEN  = process.env.TELEGRAM_BOT_TOKEN  ?? ''
const TN_URL     = process.env.TRUENORTH_BASE_URL  ?? 'http://localhost:8000'
const TN_KEY     = process.env.TRUENORTH_API_KEY   ?? ''

if (!BOT_TOKEN) {
  console.error('❌  TELEGRAM_BOT_TOKEN not set')
  console.error('    Get one from @BotFather on Telegram')
  process.exit(1)
}

// ── TrueNorth SDK ──────────────────────────────────────────────────────────────

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL })

// ── Session store ─────────────────────────────────────────────────────────────

interface CoachSession {
  sessionId:  string
  session:    Session
  chatId:     number
  startedAt:  number
}

const sessions = new Map<number, CoachSession>()  // keyed by Telegram chat ID

// ── Telegram Bot ──────────────────────────────────────────────────────────────

const bot = new TelegramBot(BOT_TOKEN, { polling: true })

// Send with Markdown and no link previews
async function reply(chatId: number, text: string, md = true): Promise<void> {
  await bot.sendMessage(chatId, text, {
    parse_mode:               md ? 'Markdown' : undefined,
    disable_web_page_preview: true,
  })
}

// Typing indicator
async function typing(chatId: number): Promise<void> {
  await bot.sendChatAction(chatId, 'typing')
}

// ── /start command ────────────────────────────────────────────────────────────

bot.onText(/\/start/, async (msg) => {
  const chatId = msg.chat.id

  // Clear any existing session
  if (sessions.has(chatId)) {
    const old = sessions.get(chatId)!
    await tn.sessions.end(old.sessionId).catch(() => {})
    sessions.delete(chatId)
  }

  await typing(chatId)

  const sessionId = `tg_${chatId}_${crypto.randomBytes(4).toString('hex')}`

  try {
    const session = await tn.sessions.create('career_coach', {
      sessionId,
      userId: String(chatId),
    })

    sessions.set(chatId, {
      sessionId, session,
      chatId, startedAt: Date.now(),
    })

    await reply(chatId, session.agentMessage)
  } catch (err) {
    await reply(chatId,
      `⚠️ Could not connect to TrueNorth API at ${TN_URL}\n` +
      `Start it with:\n\`cd packages/core && uvicorn truenorth.api.main:app --port 8000\``,
      true
    )
  }
})

// ── /status command ───────────────────────────────────────────────────────────

bot.onText(/\/status/, async (msg) => {
  const chatId = msg.chat.id
  const meta   = sessions.get(chatId)

  if (!meta) {
    await reply(chatId, 'No active session. Send /start to begin.')
    return
  }

  const elapsed = Math.round((Date.now() - meta.startedAt) / 60000)
  await reply(chatId,
    `*Session status*\n\n` +
    `⏱ Started: ${elapsed} minutes ago\n` +
    `📊 Progress: ${meta.session.completionPct?.toFixed(0) ?? 0}%\n` +
    `↩ Continue typing to complete your session.`
  )
})

// ── /reset command ────────────────────────────────────────────────────────────

bot.onText(/\/reset/, async (msg) => {
  const chatId = msg.chat.id
  const meta   = sessions.get(chatId)

  if (meta) {
    await tn.sessions.end(meta.sessionId).catch(() => {})
    sessions.delete(chatId)
  }

  await reply(chatId, 'Session cleared. Send /start to begin a new session.')
})

// ── /help command ─────────────────────────────────────────────────────────────

bot.onText(/\/help/, async (msg) => {
  await reply(msg.chat.id,
    `*Career Coach Bot*\n\n` +
    `I will help you build a personalised career roadmap.\n\n` +
    `*Commands:*\n` +
    `/start — begin career coaching\n` +
    `/status — show session progress\n` +
    `/reset — clear and start over\n` +
    `/help — this message\n\n` +
    `_Powered by TrueNorth_`
  )
})

// ── All other messages ────────────────────────────────────────────────────────

bot.on('message', async (msg) => {
  const chatId = msg.chat.id
  const text   = msg.text?.trim()

  // Ignore commands (handled above)
  if (!text || text.startsWith('/')) return

  const meta = sessions.get(chatId)
  if (!meta) {
    await reply(chatId, 'Send /start to begin your career coaching session.')
    return
  }

  await typing(chatId)

  try {
    const result = await tn.sessions.message(meta.sessionId, text)

    // Update session state
    sessions.set(chatId, {
      ...meta,
      session: { ...meta.session, completionPct: result.completionPct,
                 isComplete: result.isComplete, currentTurn: result.turn },
    })

    if (result.isComplete && result.output) {
      // Send the coaching summary
      const content = result.output.content as Record<string, unknown>
      const summary = (content.telegram_summary as string)
        ?? buildFallbackSummary(content)

      await reply(chatId, summary)

      // Send a final confirmation
      await reply(chatId,
        `✅ Your roadmap is ready!\n\n` +
        `Send /start anytime to get a fresh assessment.\n` +
        `Good luck on your journey, ${
          (content.candidate_name as string)?.split(' ')[0] ?? 'there'
        }! 🚀`
      )

      sessions.delete(chatId)
      await tn.sessions.end(meta.sessionId).catch(() => {})
    } else {
      await reply(chatId, result.text)
    }
  } catch (err) {
    await reply(chatId,
      `Something went wrong. Try again or send /reset to start over.`,
      false
    )
  }
})

// ── Fallback summary formatter ─────────────────────────────────────────────────

function buildFallbackSummary(c: Record<string, unknown>): string {
  const lines: string[] = ['🎯 *Your Career Roadmap*\n']

  if (c.readiness_score) {
    lines.push(`*Readiness score: ${c.readiness_score}/100*\n`)
  }

  if (Array.isArray(c.skill_gaps) && c.skill_gaps.length) {
    lines.push(`*Skill gaps to close:*`)
    ;(c.skill_gaps as string[]).forEach(g => lines.push(`• ${g}`))
    lines.push('')
  }

  if (Array.isArray(c.quick_wins) && c.quick_wins.length) {
    lines.push(`*This week (quick wins):*`)
    ;(c.quick_wins as string[]).forEach(w => lines.push(`• ${w}`))
    lines.push('')
  }

  if (c.honest_assessment) {
    lines.push(`*Honest take:*\n${c.honest_assessment}`)
  }

  if (c.salary_projection) {
    lines.push(`\n*Target CTC: ${c.salary_projection}*`)
  }

  return lines.join('\n')
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('\n  Career Coach — Telegram Bot (TypeScript)')
  console.log('  ─────────────────────────────────────────')

  try {
    const h = await tn.health()
    console.log(`  ✓ TrueNorth API: ${h.status} @ ${TN_URL}`)
  } catch {
    console.warn(`  ⚠ TrueNorth not reachable at ${TN_URL}`)
    console.warn('    Run: cd packages/core && uvicorn truenorth.api.main:app --port 8000')
  }

  const me = await bot.getMe()
  console.log(`\n  ✓ Bot: @${me.username}`)
  console.log('  Open Telegram, find your bot, send /start\n')
}

main().catch(console.error)