import { TrueNorth, Session, MessageResult } from "../../../packages/sdk-node";

const TN_URL = process.env.TRUENORTH_BASE_URL ?? "http://localhost:8000";
const TN_KEY = process.env.TRUENORTH_API_KEY ?? "";
const DEMO = (process.env.DEMO ?? "all").toLowerCase();

const R = "\x1b[0m";
const B = "\x1b[1m";
const DIM = "\x1b[2m";
const GRN = "\x1b[32m";
const CYN = "\x1b[36m";
const YLW = "\x1b[33m";

const col = (s: string, ...codes: string[]) => codes.join("") + s + R;
const div = (t: string) => console.log(`\n${col("══ " + t + " ══", B, CYN)}\n`);

interface ChainStep {
  if?: Record<string, string>;
  else?: string;
  then?: string;
  carry_fields: string[];
}

interface GoalConfig {
  id: string;
  name: string;
  persona: Record<string, unknown>;
  fields: Record<string, unknown>[];
  output: Record<string, unknown>;
  chain?: { on_complete: ChainStep[] };
}

const FITNESS_GOAL: GoalConfig = {
  id: "fitness_plan",
  name: "Fitness Intake",
  persona: { name: "Alex", tone: "energetic and motivating", language: "en" },
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
      name: "weight_kg",
      type: "number",
      required: true,
      question: "Current weight in kg?",
    },
    {
      name: "height_cm",
      type: "number",
      required: true,
      question: "Height in cm?",
    },
    {
      name: "primary_goal",
      type: "text",
      required: true,
      allowed_values: ["lose weight", "build muscle", "general fitness"],
      question: "Main goal? (lose weight / build muscle / general fitness)",
    },
    {
      name: "activity_level",
      type: "text",
      required: true,
      allowed_values: ["sedentary", "light", "moderate", "active"],
      question: "Activity level? (sedentary / light / moderate / active)",
    },
    {
      name: "days_per_week",
      type: "integer",
      required: true,
      question: "Training days per week?",
    },
  ],
  output: {
    format: "json",
    template:
      "Create fitness profile for {name}, {age}y, {weight_kg}kg, {height_cm}cm. " +
      "Goal: {primary_goal}. Activity: {activity_level}. Training: {days_per_week} days/week. " +
      "Return JSON with bmi, bmi_category, fitness_profile_summary.",
  },
  chain: {
    on_complete: [
      {
        if: { primary_goal: "lose weight" },
        then: "nutrition_plan",
        carry_fields: [
          "name",
          "age",
          "weight_kg",
          "height_cm",
          "activity_level",
          "days_per_week",
        ],
      },
      {
        if: { primary_goal: "build muscle" },
        then: "strength_plan",
        carry_fields: ["name", "age", "weight_kg", "activity_level"],
      },
      { else: "general_wellness", carry_fields: ["name", "age", "weight_kg"] },
    ],
  },
};

const NUTRITION_GOAL: GoalConfig = {
  id: "nutrition_plan",
  name: "Nutrition Plan",
  persona: { name: "Alex", tone: "helpful and encouraging", language: "en" },
  fields: [

    {
      name: "food_allergies",
      type: "text",
      required: true,
      question:
        "Any food allergies or intolerances? (dairy / gluten / nuts / none)",
    },
    {
      name: "diet_preference",
      type: "text",
      required: true,
      allowed_values: [
        "vegetarian",
        "vegan",
        "non-vegetarian",
        "no preference",
      ],
      question:
        "Diet preference? (vegetarian / vegan / non-veg / no preference)",
    },
    {
      name: "meals_per_day",
      type: "integer",
      required: true,
      question: "Meals per day?",
    },
    {
      name: "cooking_time_mins",
      type: "integer",
      required: true,
      question: "Minutes for meal prep per day?",
    },
    {
      name: "budget_inr",
      type: "integer",
      required: false,
      question: "Daily food budget in ₹? (say flexible if open)",
    },
  ],
  output: {
    format: "json",
    template:
      "Create nutrition plan for {name}, {age}y, {weight_kg}kg, {height_cm}cm. " +
      "Activity: {activity_level}. Diet: {diet_preference}. " +
      "Allergies: {food_allergies}. Meals: {meals_per_day}/day. Prep: {cooking_time_mins} min. " +
      "Return JSON with daily_calories, macros, meal_plan (3 days), sample_indian_meal_plan.",
  },
};

const FITNESS_TURNS = [
  "Priya Sharma",
  "28",
  "65",
  "162",
  "lose weight",
  "moderate",
  "4",
];
const NUTRITION_TURNS = ["lactose intolerant", "vegetarian", "3", "30", "300"];

function detectChain(
  goal: GoalConfig,
  collected: Record<string, unknown>,
): { nextGoalId: string; carry: string[] } | null {
  const chain = goal.chain;
  if (!chain) return null;

  for (const step of chain.on_complete) {
    if (step.if) {
      const match = Object.entries(step.if).every(
        ([k, v]) => String(collected[k]) === v,
      );
      if (match && step.then) {
        return { nextGoalId: step.then, carry: step.carry_fields };
      }
    } else if (step.else) {
      return { nextGoalId: step.else, carry: step.carry_fields };
    }
  }
  return null;
}

function transferFields(
  source: Record<string, unknown>,
  fields: string[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fields) {
    if (f in source) out[f] = source[f];
  }
  return out;
}

async function demo1AutoChain(tn: TrueNorth): Promise<void> {
  div('GOAL 1: Fitness Intake')

  const fitSid = `chain_fitness_${Date.now()}`

  let fitSession: Session
  try {
    fitSession = await tn.sessions.create('fitness_plan', { sessionId: fitSid })
  } catch (err) {
    console.log(col('  Error: ', YLW) + String(err))
    showEstimatedChain()
    return
  }

  console.log(`  Agent: ${fitSession.agentMessage}\n`)

  for (const turn of FITNESS_TURNS) {
    console.log(`  User:  ${turn}`)
    const result = await tn.sessions.message(fitSid, turn)
    console.log(`  Agent: ${result.text}\n`)
    if (result.isComplete) break
  }

  const completedFitness = await tn.sessions.get(fitSid)
  const collectedFields  = completedFitness.collectedFields as Record<string, unknown>

  console.log(col('  ✅ Fitness intake complete!\n', GRN, B))
  await tn.sessions.end(fitSid).catch(() => {})

  div('CHAIN DETECTION')

  const chain = detectChain(FITNESS_GOAL, collectedFields)
  if (!chain) {
    console.log('  No chain defined — session ends here')
    return
  }

  const primaryGoal = String(collectedFields.primary_goal ?? '')
  console.log(`  primary_goal: ${col(JSON.stringify(primaryGoal), YLW)}`)
  console.log(`  → Routes to: ${col(chain.nextGoalId, GRN, B)}`)
  console.log(`  → Carrying:  ${chain.carry.join(', ')}`)

  div('STATE TRANSFER')

  const carried  = transferFields(collectedFields, chain.carry)
  const required = ['name', 'age', 'weight_kg', 'height_cm', 'activity_level',
                    'days_per_week', 'food_allergies', 'diet_preference', 'meals_per_day']
  const missing  = required.filter(r => !(r in carried))
  const coverage = (Object.keys(carried).length / required.length * 100).toFixed(0)

  console.log(`  Carried (${Object.keys(carried).length} fields):`)
  Object.entries(carried).forEach(([k, v]) =>
    console.log(`    ${k.padEnd(22)} → ${k.padEnd(22)} = ${v}`)
  )
  console.log(`\n  Missing (${missing.length}):`)
  missing.forEach(m => console.log(`    ${col('✗', YLW)} ${m}`))
  console.log(`\n  Coverage: ${col(coverage + '%', GRN, B)} `+
    `(${Object.keys(carried).length} of ${required.length} nutrition fields pre-filled)`)

  div('GOAL 2: Nutrition Plan (pre-filled — not re-asked)')

  const nutSid = `chain_nutrition_${Date.now()}`

  console.log('  Pre-filled (not asked again):')
  Object.entries(carried).forEach(([k, v]) =>
    console.log(`    ${k.padEnd(22)} = ${v}`)
  )
  console.log()

  const nutSession = await tn.sessions.create('nutrition_plan', {
    sessionId:  nutSid,
    seedFields: carried,
  })
  console.log(`  Agent: ${nutSession.agentMessage}\n`)

  for (const turn of NUTRITION_TURNS) {
    console.log(`  User:  ${turn}`)
    const result = await tn.sessions.message(nutSid, turn)
    console.log(`  Agent: ${result.text}\n`)

    if (result.isComplete && result.output) {
      console.log(col('  ✅ Nutrition plan complete!\n', GRN, B))
      console.log('  COMBINED OUTPUT:')
      console.log(JSON.stringify(result.output.content, null, 2)
        .split('\n').map(l => '  ' + l).join('\n'))
      break
    }
  }

  await tn.sessions.end(nutSid).catch(() => {})
}

function showEstimatedChain(): void {
  console.log(col("  [Estimated output — API not reachable]\n", DIM));
  const fields = {
    name: "Priya Sharma",
    age: 28,
    weight_kg: 65,
    height_cm: 162,
    primary_goal: "lose weight",
    activity_level: "moderate",
    days_per_week: 4,
  };
  console.log("  FITNESS COLLECTED:");
  Object.entries(fields).forEach(([k, v]) =>
    console.log(`    ${k.padEnd(20)} = ${v}`),
  );
  console.log("\n  CHAIN: lose weight → nutrition_plan");
  console.log(
    "  CARRY: name, age, weight_kg, height_cm, activity_level, days_per_week",
  );
  console.log("\n  NUTRITION (only new fields asked):");
  console.log(
    "    Agent: Any food allergies?  ← first question — skips name/age/weight\n",
  );
}

function demo2ManualTransfer(): void {
  div("MANUAL STATE TRANSFER — Code-level control");
  console.log("  Scenario: Medical intake → Lab test form");
  console.log("  Carry: name, DOB, blood_group, allergies");
  console.log("  Rename: chief_complaint → reason_for_test\n");

  const medicalState: Record<string, unknown> = {
    patient_name: "Rahul Kumar",
    date_of_birth: "10 June 1985",
    chief_complaint: "routine check-up, diabetes monitoring",
    blood_group: "B+",
    known_allergies: "penicillin",
    medications: "metformin 500mg",
  };

  const fieldMap: Array<[string, string]> = [
    ["patient_name", "patient_name"],
    ["date_of_birth", "dob"],
    ["blood_group", "blood_group"],
    ["known_allergies", "allergies"],
    ["chief_complaint", "reason_for_test"],
  ];

  console.log(
    `  ${"Source field".padEnd(26)} ${"Target field".padEnd(26)} Transferred value`,
  );
  console.log("  " + "─".repeat(75));

  const carried: Record<string, unknown> = {};
  fieldMap.forEach(([src, tgt]) => {
    if (src in medicalState) {
      carried[tgt] = medicalState[src];
      const rename = src !== tgt ? col(" (renamed)", YLW) : "";
      console.log(
        `  ${src.padEnd(26)} ${tgt.padEnd(26)} ${medicalState[src]}${rename}`,
      );
    }
  });

  console.log(`\n  Carried ${Object.keys(carried).length} fields`);
  console.log(`  Coverage: 5 of 6 required lab test fields pre-filled\n`);
}

function demo3ChainConfig(): void {
  div("GOAL CHAIN CONFIG — Routing table");

  const testCases = [
    { primary_goal: "lose weight", name: "Priya", age: 28 },
    { primary_goal: "build muscle", name: "Rahul", age: 32 },
    { primary_goal: "run a marathon", name: "Anita", age: 25 },
  ];

  console.log(
    `  ${"primary_goal value".padEnd(30)} ${"Routes to".padEnd(22)} Fields carried`,
  );
  console.log("  " + "─".repeat(72));

  testCases.forEach((tc) => {
    const chain = detectChain(FITNESS_GOAL, tc as any);
    if (chain) {
      console.log(
        `  ${JSON.stringify(tc.primary_goal).padEnd(30)} ${chain.nextGoalId.padEnd(22)} [${chain.carry.join(", ")}]`,
      );
    } else {
      console.log(`  ${JSON.stringify(tc.primary_goal).padEnd(30)} (no match)`);
    }
  });
  console.log();
}

async function main(): Promise<void> {
  console.log();
  console.log(
    col("  TrueNorth Goal Chaining Demo (Node.js / TypeScript)", B, CYN),
  );
  console.log(col(`  API: ${TN_URL} | DEMO: ${DEMO}`, DIM));
  console.log();

  const tn = new TrueNorth({
    apiKey: TN_KEY,
    baseUrl: TN_URL,
    timeout: 90_000,
  });

  if (DEMO === "all" || DEMO === "chain") await demo1AutoChain(tn);
  if (DEMO === "all" || DEMO === "transfer") demo2ManualTransfer();
  if (DEMO === "all" || DEMO === "config") demo3ChainConfig();

  console.log(col("  Goal chaining turns single sessions into journeys.", DIM));
  console.log(
    col("  Each goal carries state — users never repeat themselves.\n", DIM),
  );
}

main().catch(console.error);
