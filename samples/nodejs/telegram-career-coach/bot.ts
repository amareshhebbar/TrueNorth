import TelegramBot from 'node-telegram-bot-api'
import crypto      from 'crypto'
import { TrueNorth, Session } from '../../../packages/sdk-node'

const BOT_TOKEN  = process.env.TELEGRAM_BOT_TOKEN  ?? ''
const TN_URL     = process.env.TRUENORTH_BASE_URL  ?? 'http://localhost:8000'
const TN_KEY     = process.env.TRUENORTH_API_KEY   ?? ''

if (!BOT_TOKEN) {
  console.error('❌  TELEGRAM_BOT_TOKEN not set')
  console.error('    Get one from @BotFather on Telegram')
  process.exit(1)
}

const tn = new TrueNorth({ apiKey: TN_KEY, baseUrl: TN_URL })

interface CoachSession {
  sessionId:  string
  session:    Session
  chatId:     number
  startedAt:  number
}

const sessions = new Map<number, CoachSession>()

const bot = new TelegramBot(BOT_TOKEN, { polling: true })

async function reply(chatId: number, text: string, md = true): Promise<void> {
  await bot.sendMessage(chatId, text, {
    parse_mode:               md ? 'Markdown' : undefined,
    disable_web_page_preview: true,
  })
}

async function typing(chatId: number): Promise<void> {
  await bot.sendChatAction(chatId, 'typing')
}

bot.onText(/\/start/, async (msg) => {
  const chatId = msg.chat.id

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

bot.onText(/\/reset/, async (msg) => {
  const chatId = msg.chat.id
  const meta   = sessions.get(chatId)

  if (meta) {
    await tn.sessions.end(meta.sessionId).catch(() => {})
    sessions.delete(chatId)
  }

  await reply(chatId, 'Session cleared. Send /start to begin a new session.')
})

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

bot.on('message', async (msg) => {
  const chatId = msg.chat.id
  const text   = msg.text?.trim()

  if (!text || text.startsWith('/')) return

  const meta = sessions.get(chatId)
  if (!meta) {
    await reply(chatId, 'Send /start to begin your career coaching session.')
    return
  }

  await typing(chatId)

  try {
    const result = await tn.sessions.message(meta.sessionId, text)

    sessions.set(chatId, {
      ...meta,
      session: { ...meta.session, completionPct: result.completionPct,
                 isComplete: result.isComplete, currentTurn: result.turn },
    })

    if (result.isComplete && result.output) {

      const content = result.output.content as Record<string, unknown>
      const summary = (content.telegram_summary as string)
        ?? buildFallbackSummary(content)

      await reply(chatId, summary)

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
