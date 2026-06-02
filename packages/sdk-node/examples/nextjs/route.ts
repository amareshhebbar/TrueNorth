// app/api/chat/route.ts — Next.js App Router example
import { NextRequest, NextResponse } from "next/server";
import { TrueNorthClient } from "@amareshhebbar/truenorth";

const tn = new TrueNorthClient({
  serverUrl: process.env.TRUENORTH_API_URL ?? "http://localhost:8000",
  apiKey: process.env.TRUENORTH_API_KEY,
});

export async function POST(req: NextRequest) {
  const { action, goalId, sessionId, message } = await req.json();

  if (action === "start") {
    const session = await tn.startSession(goalId);
    return NextResponse.json({ sessionId: session.sessionId, message: session.welcomeMessage });
  }

  if (action === "message") {
    const resp = await tn.sendMessage(sessionId, message);
    return NextResponse.json(resp);
  }

  return NextResponse.json({ error: "Unknown action" }, { status: 400 });
}
