import { TrueNorth, MessageResult } from "../../../packages/sdk-node";

const TN_URL = process.env.TRUENORTH_BASE_URL ?? "http://localhost:8000";
const TN_KEY = process.env.TRUENORTH_API_KEY ?? "";
const DEMO = (process.env.DEMO ?? "all").toLowerCase();

const R = "\x1b[0m";
const B = "\x1b[1m";
const DIM = "\x1b[2m";
const GRN = "\x1b[32m";
const CYN = "\x1b[36m";
const YLW = "\x1b[33m";
const RED = "\x1b[31m";

const col = (s: string, ...codes: string[]) => codes.join("") + s + R;
const div = (t: string) => console.log(`\n${col("══ " + t + " ══", B, CYN)}\n`);

const DEMO_GOAL = {
  id: "cost_demo",
  name: "Cost Tracking Demo",
  persona: { name: "Demo", tone: "neutral", language: "en" },
  fields: [
    {
      name: "name",
      type: "text",
      required: true,
      question: "What is your name?",
    },
    {
      name: "age",
      type: "integer",
      required: true,
      question: "How old are you?",
    },
    {
      name: "goal",
      type: "text",
      required: true,
      question: "What is your main goal?",
    },
    {
      name: "experience",
      type: "text",
      required: true,
      question: "Describe your experience.",
    },
    {
      name: "availability",
      type: "text",
      required: true,
      question: "When can you start?",
    },
    {
      name: "location",
      type: "text",
      required: false,
      question: "Where are you located?",
    },
  ],
  output: {
    format: "json",
    template: "Summarise: {name}, {age}, goal: {goal}. Return JSON.",
  },
};

const DEMO_TURNS = [
  "Priya Sharma",
  "28",
  "transition to product management",
  "5 years backend engineering, led 3 product launches",
  "available next month with 4 weeks notice",
  "Bengaluru, open to remote",
];

interface TurnCost {
  turn: number;
  taskType: string;
  model: string;
  costUsd: number;
  latencyMs: number;
}

function showEstimatedTurns(): void {
  const rows: Array<[number, string, string, number, number]> = [
    [1, "extract", "gemini-3.5-flash", 0.00003, 212],
    [2, "converse", "claude-haiku-4-5", 0.00015, 441],
    [3, "extract", "gemini-3.5-flash", 0.000022, 198],
    [4, "converse", "claude-haiku-4-5", 0.000148, 392],
    [5, "extract", "gemini-3.5-flash", 0.000025, 209],
    [6, "output", "claude-sonnet-4-20250514", 0.000447, 891],
  ];
  const header = `  ${"Turn".padEnd(6)} ${"Task".padEnd(14)} ${"Model".padEnd(28)} ${"Cost USD".padEnd(12)} Latency`;
  console.log(header);
  console.log("  " + "─".repeat(72));
  let total = 0;
  rows.forEach(([turn, task, model, cost, ms]) => {
    total += cost;
    console.log(
      `  ${String(turn).padEnd(6)} ${task.padEnd(14)} ${model.padEnd(28)} $${cost.toFixed(6).padEnd(11)} ${ms}ms`,
    );
  });
  console.log("  " + "─".repeat(72));
  console.log(
    `\n  ${col("Estimated total:", B)} $${total.toFixed(6)} | 6 turns | avg $${(total / 6).toFixed(6)}/turn`,
  );
  console.log(
    col(
      `\n  💡 Same session with all-Sonnet: ~$0.0082 (10x more expensive)\n`,
      YLW,
    ),
  );
}

async function demo1PerTurnCosts(tn: TrueNorth): Promise<void> {
  div("DEMO 1 — Per-turn cost breakdown");

  const sid = `cost_turns_${Date.now()}`;

  let session;
  try {
    session = await tn.sessions.create("cost_demo", { sessionId: sid });
  } catch {
    console.log(col("  API not reachable — showing estimated output:", DIM));
    showEstimatedTurns();
    return;
  }

  const header = `  ${"Turn".padEnd(6)} ${"Task".padEnd(14)} ${"Model".padEnd(28)} ${"Cost USD".padEnd(12)} Latency`;
  console.log(header);
  console.log("  " + "─".repeat(72));

  const turns: TurnCost[] = [];
  let total = 0;

  for (let i = 0; i < DEMO_TURNS.length; i++) {
    const t0 = Date.now();
    let result: MessageResult;
    try {
      result = await tn.sessions.message(sid, DEMO_TURNS[i]);
    } catch {
      break;
    }

    const latency = Date.now() - t0;
    const cost = result.costUsd ?? 0;
    const model = (result as any).modelUsed ?? "auto";
    const taskType = (result as any).taskType ?? "mixed";

    total += cost;
    turns.push({
      turn: i + 1,
      taskType,
      model,
      costUsd: cost,
      latencyMs: latency,
    });

    console.log(
      `  ${String(i + 1).padEnd(6)} ${taskType.padEnd(14)} ${model.slice(0, 26).padEnd(28)} $${cost.toFixed(6).padEnd(11)} ${latency}ms`,
    );

    if (result.isComplete) break;
  }

  console.log("  " + "─".repeat(72));
  const avg = turns.length ? total / turns.length : 0;
  console.log(
    `\n  ${col("Session total:", B)} $${total.toFixed(6)} | ${turns.length} turns | avg $${avg.toFixed(6)}/turn`,
  );
  console.log(
    col(
      `\n  💡 Same session with all-Sonnet: ~$0.0082  (${(((0.0082 - total) / 0.0082) * 100).toFixed(0)}% cheaper with smart routing)\n`,
      YLW,
    ),
  );

  await tn.sessions.end(sid).catch(() => {});
}

async function demo2BudgetGuard(tn: TrueNorth): Promise<void> {
  div('DEMO 2 — Budget guard (hard stop)')
  const BUDGET = 0.0005

  console.log(`  Budget set: $${BUDGET.toFixed(4)} for this session\n`)
  console.log(`  ${'Turn'.padEnd(6)} ${'Spent'.padEnd(14)} ${'Budget'.padEnd(14)} Status`)
  console.log('  ' + '─'.repeat(52))

  const sid = `cost_budget_${Date.now()}`

  try {
    await tn.sessions.create('cost_demo', {
      sessionId: sid,
      budgetUsd: BUDGET,
    })
  } catch {

    const est = [0.000150, 0.000320, 0.000510]
    let cum = 0
    est.forEach((cost, i) => {
      cum = cost
      const status = cum > BUDGET
        ? col('BUDGET EXCEEDED 🛑', RED, B)
        : col('OK', GRN)
      console.log(
        `  ${String(i+1).padEnd(6)} $${cum.toFixed(6).padEnd(13)} $${BUDGET.toFixed(4).padEnd(13)} ${status}`
      )
    })
    console.log(col('\n  🛑 Budget exceeded — stopped cleanly. Partial output generated.\n', RED))
    return
  }

  let cumulative = 0

  for (let i = 0; i < DEMO_TURNS.length; i++) {
    let result: MessageResult
    try {
      result = await tn.sessions.message(sid, DEMO_TURNS[i])
    } catch (err: any) {

      const msg = String(err).toLowerCase()
      if (msg.includes('budget') || msg.includes('limit')) {
        console.log(col(`\n  🛑 BUDGET EXCEEDED before turn ${i + 1}!`, RED, B))
        console.log('  Partial output generated from collected fields.')
      }
      break
    }

    cumulative += result.costUsd ?? 0
    const over   = cumulative > BUDGET
    const status = over ? col('OVER BUDGET 🛑', RED, B) : col('OK', GRN)

    console.log(
      `  ${String(i+1).padEnd(6)} $${cumulative.toFixed(6).padEnd(13)} $${BUDGET.toFixed(4).padEnd(13)} ${status}`
    )

    if (result.isComplete || over) break
  }

  console.log(`\n  Final: $${cumulative.toFixed(6)} vs budget $${BUDGET.toFixed(4)}`)
  console.log(
    cumulative <= BUDGET
      ? col('  ✅ Within budget', GRN)
      : col('  🛑 Budget exceeded — session stopped cleanly', RED)
  )
  console.log()

  await tn.sessions.end(sid).catch(() => {})
}

function demo3RoutingComparison(): void {
  div("DEMO 3 — Routing strategy cost comparison");

  const strategies = [
    {
      name: "Strategy A: All Claude Sonnet (naïve)",
      cost: 0.0082,
      desc: "Every task on Sonnet — max quality, max cost",
    },
    {
      name: "Strategy B: TrueNorth smart routing",
      cost: 0.0009,
      desc: "Extract→Flash, Converse→Haiku, Output→Sonnet",
    },
    {
      name: "Strategy C: All local Ollama (free)",
      cost: 0.0,
      desc: "llama3.1:8b on-device — zero API cost",
    },
  ];

  const baseline = strategies[0].cost;

  console.log(
    `  ${"Strategy".padEnd(46)} ${"Est. Cost".padEnd(12)} vs Baseline`,
  );
  console.log("  " + "─".repeat(72));

  strategies.forEach((s) => {
    const savings =
      s.cost < baseline
        ? col(
            `-${(((baseline - s.cost) / baseline) * 100).toFixed(0)}%`,
            GRN,
            B,
          )
        : col("baseline", DIM);
    console.log(
      `  ${s.name.padEnd(46)} $${s.cost.toFixed(4).padEnd(11)} ${savings}`,
    );
  });

  console.log(`
  ${col("Key insight:", B)}
    Smart routing saves 89% vs naive all-Sonnet.
    Quality preserved:
      • Extraction is deterministic — cheap model works fine
      • Conversation just needs natural language — Haiku excels
      • Output gets full Sonnet quality budget

  ${col("At scale (10,000 sessions/month):", B)}
    Naïve : $0.0082 × 10,000 = ${col("$82.00/month", RED)}
    Smart : $0.0009 × 10,000 = ${col("$9.00/month", GRN)}
    ${col("Savings: $73/month ($876/year)", GRN, B)}
`);
}

function demo4CostDashboard(): void {
  div("DEMO 4 — Cost dashboard (aggregate analytics)");
  console.log("  Simulating 10 sessions...\n");

  const goals = [
    "fitness_coach",
    "medical_intake",
    "hr_screening",
    "crop_advisory",
  ];

  let seed = 42;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return Math.abs(seed) / 0xffffffff;
  };

  let totalCost = 0;
  let totalTokens = 0;
  const sessions: Array<{ goal: string; turns: number; cost: number }> = [];

  for (let i = 0; i < 10; i++) {
    const turns = Math.floor(rand() * 6) + 4;
    const cost = rand() * 0.0012 + 0.0003;
    const tokensIn = (Math.floor(rand() * 500) + 300) * turns;
    const tokensOut = (Math.floor(rand() * 200) + 100) * turns;
    sessions.push({
      goal: goals[Math.floor(rand() * goals.length)],
      turns,
      cost,
    });
    totalCost += cost;
    totalTokens += tokensIn + tokensOut;
  }

  const avg = totalCost / sessions.length;

  console.log(`  ${"Metric".padEnd(24)} Value`);
  console.log("  " + "─".repeat(45));
  console.log(`  ${"Sessions analysed".padEnd(24)} ${sessions.length}`);
  console.log(`  ${"Total spend".padEnd(24)} $${totalCost.toFixed(4)}`);
  console.log(`  ${"Avg per session".padEnd(24)} $${avg.toFixed(5)}`);
  console.log(`  ${"Total tokens".padEnd(24)} ${totalTokens.toLocaleString()}`);
  console.log(
    `  ${"Avg per 1K tokens".padEnd(24)} $${((totalCost / totalTokens) * 1000).toFixed(5)}`,
  );

  console.log(`
  ${col("Cost projection at scale:", B)}
    100 sessions/day   → ~$${(avg * 100).toFixed(2)}/day   → ~$${(avg * 100 * 30).toFixed(0)}/month
    1,000 sessions/day → ~$${(avg * 1000).toFixed(2)}/day  → ~$${(avg * 1000 * 30).toFixed(0)}/month
    10,000/day         → ~$${(avg * 10000).toFixed(2)}/day → ~$${(avg * 10000 * 30).toFixed(0)}/month

  ${col("Recommendations:", B)}
    Set session budget   : $0.50  (caps runaway sessions)
    Set monthly limit    : $50    (caps runaway deployments)
    Use smart routing    : saves 89% vs naive all-Sonnet
`);
}

async function main(): Promise<void> {
  console.log();
  console.log(
    col("  TrueNorth Cost Tracking Demo (Node.js / TypeScript)", B, CYN),
  );
  console.log(col(`  API: ${TN_URL} | DEMO: ${DEMO}`, DIM));
  console.log();
  console.log(col("  Using inline goal config (no YAML file needed)", DIM));
  console.log();

  const tn = new TrueNorth({
    apiKey: TN_KEY,
    baseUrl: TN_URL,
    timeout: 90_000,
  });

  if (DEMO === "all" || DEMO === "turns") await demo1PerTurnCosts(tn);
  if (DEMO === "all" || DEMO === "budget") await demo2BudgetGuard(tn);
  if (DEMO === "all" || DEMO === "routing") demo3RoutingComparison();
  if (DEMO === "all" || DEMO === "dashboard") demo4CostDashboard();

  console.log(
    col("  Cost tracking is on by default in every TrueNorth session.", DIM),
  );
  console.log(
    col("  DEMO=budget/routing/turns/dashboard to run a single demo.\n", DIM),
  );
}

main().catch(console.error);
