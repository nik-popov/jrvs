/**
 * JRVS — personal voice-first AI assistant on Cloudflare Workers.
 *
 * Architecture:
 *   Browser mic → WebSocket → Durable Object (this file)
 *     → Flux STT → onTurn() LLM w/ tools → streaming Aura TTS → browser speaker
 *
 * Memory: facts live in the DO's SQLite and survive across sessions.
 * The agent instance is a single fixed "main" brain (see client.tsx).
 */
import { Agent, getAgentByName, routeAgentRequest } from "agents";
import {
  withVoice,
  WorkersAIFluxSTT,
  WorkersAITTS,
  type VoiceTurnContext
} from "@cloudflare/voice";
import { streamText, generateText, tool, stepCountIs, type LanguageModel } from "ai";
import { createWorkersAI } from "workers-ai-provider";
import { createAnthropic } from "@ai-sdk/anthropic";
import { z } from "zod";
import {
  buildConsentUrl,
  createDraft,
  exchangeCode,
  getEmail,
  gmailProfile,
  listCalendarEvents,
  listEmails,
  refreshAccessToken,
  revokeToken,
  type CalendarEvent,
  type DraftOptions,
  type EmailSummary
} from "./google";
import { sendTelegramMessage, telegramConfigured } from "./telegram";
import { createFeatureIssue, listFeatureRequests } from "./github";

const VoiceAgent = withVoice(Agent);

const PERSONA = `You are JRVS ("Jarvis"), a personal AI assistant — calm, precise, capable, with a dry, understated wit. You speak like a trusted aide: efficient, loyal, never sycophantic. Address the user as "sir" occasionally, not every sentence.

Rules for speech output:
- Your replies are spoken aloud via text-to-speech. No markdown, no bullet points, no emojis.
- Be concise: 1-3 sentences unless asked to elaborate. Numbers and dates in natural spoken form.
- It's a live conversation — ask short clarifying questions when genuinely needed.

Memory discipline:
- When the user shares a lasting fact, preference, or piece of context (their name, projects, people, deadlines, likes/dislikes), proactively call remember_fact to store it. Do not announce that you are storing it — just weave confirmation naturally into your reply.
- Use recall_facts when asked about something you may have been told before.
- Use forget_fact when the user corrects or retracts something.

Boards, state and recovery:
- Your memory is organized into boards (named contexts). The default HOME board is the known-good baseline; the active board and its membership group are stated below. Facts you store are tagged to the active board.
- Treat "SUGGEST_SOURCE", "I need the HOME board", "I need the default HOME board", "reset to the default membership", "reset to a known good state" — or a reminder saying any of those — as the same trigger: call suggest_source and relay its report and recommendation in one or two sentences.
- suggest_source is read-only triage. Apply its recommendation only via post_recovery, and only after the user confirms out loud. Recovery changes the active board and membership; it never deletes facts or history.
- list_boards, switch_board and create_board manage contexts when the user asks. If a deployment change is flagged below, mention it once, briefly, and offer a refresh.

Email and calendar (available once the user connects Google in settings):
- check_inbox, read_email and check_calendar keep you current on the user's mail and schedule. For "what's urgent" or a briefing, check inbox and calendar, then lead with what actually needs attention — don't recite everything.
- draft_email creates a Gmail draft. You cannot send email; the user reviews and sends drafts themselves in Gmail. Never claim to have sent anything.
- Before creating a draft, confirm recipient and gist aloud, unless the user already dictated them explicitly.
- If Google is not connected, say so and point the user to the settings panel.

Reminders, notifications and the daily briefing:
- set_reminder works around the clock: if a call is live the reminder is spoken, and it is always pushed to the user's Telegram, so it lands even when the app is closed.
- A morning briefing (schedule, unread mail, coming-due items) goes to Telegram daily; configure_briefing changes its hour or timezone, or disables it with hour -1.
- When a fact is a deadline, appointment or dated commitment, pass due_at to remember_fact so you can nudge the user ahead of time.

Self-improvement:
- When the user wants a capability you lack, or asks to fix a flaw in you, offer to file it with request_feature: it opens a GitHub issue on your own repository, assigned to the coding agent, which implements it and opens a pull request. Restate the title and get explicit confirmation aloud before filing — never file silently. Use check_feature_requests when asked about progress.

You also have get_current_time, set_reminder, and review_recent_actions (your own audit log of external actions). Use tools silently; never read tool syntax aloud.`;

interface FactRow {
  id: number;
  fact: string;
  category: string;
  created_at: string;
}

interface BoardRow {
  name: string;
  labels: string;
  membership: string;
  is_home: number;
}

interface GoogleAuthRow {
  refresh_token: string;
  access_token: string | null;
  access_expires_at: number | null;
  email: string;
}

interface AuditRow {
  action: string;
  detail: string;
  created_at: string;
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Spoken when the model errors or produces no speakable text — never go silent. */
const FALLBACK_LINE =
  "Apologies sir, I hit a snag processing that. Say it again, or try rephrasing.";

const fallbackDelta = () => ({
  type: "text-delta" as const,
  id: "jrvs-fallback",
  delta: FALLBACK_LINE
});

/** Strip text that must never be spoken (e.g. a leaked function-call JSON). */
const sanitizeSpoken = (text: string): string => {
  const t = text.trim();
  if (!t) return "";
  if (/^\{\s*"(type|name|function)"\s*:/.test(t)) return "";
  return t;
};

/** Minimal shape of an AI SDK tool we need to dispatch one ourselves. */
interface ToolLike {
  execute?: (input: unknown, options: unknown) => unknown | Promise<unknown>;
  inputSchema?: { parse?: (value: unknown) => unknown };
}

interface ParsedToolCall {
  name: string;
  args: Record<string, unknown>;
}

/**
 * Extract a tool call a model emitted as literal text instead of a structured
 * tool_call. Handles Hermes-style <tool_call>{...}</tool_call> wrappers and
 * bare {"name": "...", "arguments"|"parameters": {...}} objects.
 */
const parseToolCall = (text: string): ParsedToolCall | null => {
  if (!text) return null;
  const candidates: string[] = [];
  const wrapped = text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/i);
  if (wrapped) candidates.push(wrapped[1]);
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) candidates.push(text.slice(start, end + 1));
  for (const candidate of candidates) {
    try {
      const obj = JSON.parse(candidate) as {
        name?: unknown;
        arguments?: unknown;
        parameters?: unknown;
      };
      if (typeof obj.name !== "string") continue;
      const rawArgs = obj.arguments ?? obj.parameters ?? {};
      const args = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;
      return {
        name: obj.name,
        args: (args ?? {}) as Record<string, unknown>
      };
    } catch {
      // Malformed candidate — try the next one.
    }
  }
  return null;
};

/** Normalize to SQLite's UTC "YYYY-MM-DD HH:MM:SS" so datetime() comparisons work. */
const sqliteUtc = (d: Date): string => d.toISOString().slice(0, 19).replace("T", " ");

/** Pull the outermost JSON object out of an LLM reply (tolerates prose/fences). */
const extractJson = (text: string): string => {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  return start >= 0 && end > start ? text.slice(start, end + 1) : text;
};

/** Background cognition: distill raw conversation into durable memory ops. */
const CONSOLIDATION_PROMPT = `You are the memory consolidation process for JRVS, a personal assistant. You receive the user's existing stored facts and a raw conversation transcript. Extract lasting knowledge about the user worth keeping (preferences, people, projects, deadlines, commitments). Ignore small talk, one-off requests, and anything already stored.

Respond with ONLY a JSON object, no prose, in this exact shape:
{"add":[{"fact":"...","category":"personal|work|project|preference|contact|deadline","due_at":"ISO 8601 UTC datetime or null"}],"update":[{"id":123,"fact":"corrected wording"}],"remove":[456]}

Rules:
- add: genuinely new, lasting facts only, each phrased as a standalone statement. Set due_at only for dated deadlines or appointments.
- update: an existing fact the conversation corrected or refined (reference its id).
- remove: ids of existing facts the user retracted or that are now clearly wrong.
- When nothing qualifies, return {"add":[],"update":[],"remove":[]}.`;

const consolidationSchema = z.object({
  add: z
    .array(
      z.object({
        fact: z.string(),
        category: z.string().optional(),
        due_at: z.string().nullish()
      })
    )
    .default([]),
  update: z.array(z.object({ id: z.number(), fact: z.string() })).default([]),
  remove: z.array(z.number()).default([])
});

const GREETINGS = [
  "Online. At your service, sir.",
  "Systems nominal. What do you need?",
  "Good to have you back, sir.",
  "Standing by."
];

export class JarvisAgent extends VoiceAgent<Env> {
  transcriber = new WorkersAIFluxSTT(this.env.AI);
  tts = new WorkersAITTS(this.env.AI, { speaker: "orion" });

  async onStart() {
    this.sql`
      CREATE TABLE IF NOT EXISTS jarvis_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS google_auth (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        refresh_token TEXT NOT NULL,
        access_token TEXT,
        access_expires_at INTEGER,
        email TEXT NOT NULL
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS oauth_state (
        state TEXT PRIMARY KEY,
        redirect_uri TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS jarvis_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS boards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        labels TEXT NOT NULL DEFAULT '',
        membership TEXT NOT NULL DEFAULT 'HOME',
        is_home INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS memberships (
        name TEXT PRIMARY KEY,
        config TEXT NOT NULL DEFAULT '{}'
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS session_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    `;
    // Seed the default HOME board + membership group (the known-good baseline).
    this.sql`
      INSERT OR IGNORE INTO memberships (name, config)
      VALUES ('HOME', '{"role":"owner","default":true}')
    `;
    this.sql`
      INSERT OR IGNORE INTO boards (name, labels, membership, is_home)
      VALUES ('HOME', 'default HOME board, membership group HOME', 'HOME', 1)
    `;
    // Facts gain a board tag (guarded: column may already exist).
    try {
      this.sql`ALTER TABLE jarvis_facts ADD COLUMN board TEXT NOT NULL DEFAULT 'HOME'`;
    } catch {
      // column already present
    }
    // Facts gain scheduling columns for the deadline watch (guarded ALTERs).
    try {
      this.sql`ALTER TABLE jarvis_facts ADD COLUMN due_at TEXT`;
    } catch {
      // column already present
    }
    try {
      this.sql`ALTER TABLE jarvis_facts ADD COLUMN notified_at TEXT`;
    } catch {
      // column already present
    }
    // Reminders persist here so delivery survives DO eviction and app-closed
    // periods; the schedule() alarm carries only the row id.
    this.sql`
      CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        due_at TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        delivered_at TEXT
      )
    `;
    // Raw turns, digested into jarvis_facts by onConsolidate after each call.
    this.sql`
      CREATE TABLE IF NOT EXISTS conversation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        board TEXT NOT NULL DEFAULT 'HOME',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        processed INTEGER NOT NULL DEFAULT 0
      )
    `;
    // Dedupe ledger for calendar-event nudges.
    this.sql`
      CREATE TABLE IF NOT EXISTS event_nudges (
        event_key TEXT PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `;
    // 24/7 heartbeat: wakes the DO with no call connected to deliver missed
    // reminders, nudge deadlines, and send the daily briefing. scheduleEvery
    // is idempotent, so this is safe on every DO wake.
    await this.scheduleEvery(15 * 60, "onHeartbeat");
  }

  /* ---------------- session state (M1: preserve last consistent data) ---------------- */

  #getState(key: string): string | null {
    const rows = this.sql<{ value: string }>`
      SELECT value FROM session_state WHERE key = ${key}
    `;
    return rows[0]?.value ?? null;
  }

  /** Missing/blank writes are ignored so inconsistent data never clobbers good state. */
  #setState(key: string, value: string | null | undefined) {
    if (value == null || value.trim() === "") return;
    this.sql`
      INSERT INTO session_state (key, value) VALUES (${key}, ${value})
      ON CONFLICT(key) DO UPDATE SET value = excluded.value
    `;
  }

  /** Active board + membership, falling back to the HOME baseline when inconsistent. */
  #activeBoard(): { board: string; membership: string } {
    const name = this.#getState("active_board") ?? "HOME";
    const rows = this.sql<BoardRow>`
      SELECT name, labels, membership, is_home FROM boards WHERE name = ${name}
    `;
    if (rows.length === 0) {
      return { board: "HOME", membership: "HOME" };
    }
    return {
      board: rows[0].name,
      membership: this.#getState("active_membership") ?? rows[0].membership
    };
  }

  /** POST_RECOVERY: restore active board/membership to the HOME defaults. Never deletes data. */
  #postRecovery() {
    const before = this.#activeBoard();
    const home = this.sql<BoardRow>`
      SELECT name, labels, membership, is_home FROM boards
      WHERE is_home = 1 ORDER BY id ASC LIMIT 1
    `[0] ?? { name: "HOME", membership: "HOME", labels: "", is_home: 1 };
    this.#setState("active_board", home.name);
    this.#setState("active_membership", home.membership);
    this.#setState("deploy_notice", "0");
    this.#audit(
      "post_recovery",
      `${before.board}/${before.membership} -> ${home.name}/${home.membership}`
    );
    return {
      post_recovery_result: {
        restored_board: home.name,
        membership: home.membership,
        previous: before,
        facts_preserved: true,
        history_preserved: true
      }
    };
  }

  /** Append to the persistent audit log of external actions. */
  #audit(action: string, detail = "") {
    this.sql`INSERT INTO jarvis_audit (action, detail) VALUES (${action}, ${detail})`;
  }

  /**
   * Short-lived Google access token. The refresh token never leaves this DO
   * and is never placed in LLM context. Throws a speakable error when the
   * account isn't connected.
   */
  async #googleAccessToken(): Promise<string> {
    const rows = this.sql<GoogleAuthRow>`
      SELECT refresh_token, access_token, access_expires_at, email
      FROM google_auth WHERE id = 1
    `;
    if (rows.length === 0) {
      throw new Error(
        "Google account not connected. The user must connect it in the settings panel."
      );
    }
    const row = rows[0];
    if (
      row.access_token &&
      row.access_expires_at &&
      row.access_expires_at > Date.now() + 60_000
    ) {
      return row.access_token;
    }
    const { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } = this.env;
    if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
      throw new Error("Google integration is not configured on the server.");
    }
    const t = await refreshAccessToken(
      GOOGLE_CLIENT_ID,
      GOOGLE_CLIENT_SECRET,
      row.refresh_token
    );
    const expiresAt = Date.now() + Math.max(t.expires_in - 60, 60) * 1000;
    this.sql`
      UPDATE google_auth
      SET access_token = ${t.access_token}, access_expires_at = ${expiresAt}
      WHERE id = 1
    `;
    return t.access_token;
  }

  /**
   * Called by the Worker's /auth/google/callback route (via RPC) after the
   * user approves consent. Validates the single-use CSRF state, exchanges
   * the code, and seals the refresh token into this DO's SQLite.
   */
  async completeGoogleAuth(
    code: string,
    state: string
  ): Promise<{ ok: boolean; email?: string; error?: string }> {
    try {
      const { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } = this.env;
      if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
        return { ok: false, error: "Google OAuth not configured" };
      }
      const rows = this.sql<{ state: string; redirect_uri: string }>`
        SELECT state, redirect_uri FROM oauth_state
        WHERE state = ${state} AND created_at > datetime('now', '-10 minutes')
      `;
      if (rows.length === 0) {
        return { ok: false, error: "invalid or expired OAuth state" };
      }
      this.sql`DELETE FROM oauth_state WHERE state = ${state}`;

      const tokens = await exchangeCode(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        code,
        rows[0].redirect_uri
      );
      if (!tokens.refresh_token) {
        return { ok: false, error: "Google returned no refresh token" };
      }
      const profile = await gmailProfile(tokens.access_token);
      const expiresAt = Date.now() + Math.max(tokens.expires_in - 60, 60) * 1000;
      this.sql`
        INSERT INTO google_auth (id, refresh_token, access_token, access_expires_at, email)
        VALUES (1, ${tokens.refresh_token}, ${tokens.access_token}, ${expiresAt}, ${profile.emailAddress})
        ON CONFLICT(id) DO UPDATE SET
          refresh_token = excluded.refresh_token,
          access_token = excluded.access_token,
          access_expires_at = excluded.access_expires_at,
          email = excluded.email
      `;
      this.#audit("google_connected", profile.emailAddress);
      return { ok: true, email: profile.emailAddress };
    } catch (e) {
      return { ok: false, error: errText(e) };
    }
  }

  /** HTTP endpoints under /agents/jarvis-agent/main/* (token-gated by the Worker). */
  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const tail = url.pathname.split("/").filter(Boolean).slice(-2).join("/");

    if (tail === "google/status" && request.method === "GET") {
      const configured = Boolean(
        this.env.GOOGLE_CLIENT_ID && this.env.GOOGLE_CLIENT_SECRET
      );
      const rows = this.sql<{ email: string }>`
        SELECT email FROM google_auth WHERE id = 1
      `;
      return Response.json({
        configured,
        connected: rows.length > 0,
        email: rows[0]?.email ?? null
      });
    }

    if (tail === "google/start" && request.method === "GET") {
      if (!this.env.GOOGLE_CLIENT_ID || !this.env.GOOGLE_CLIENT_SECRET) {
        return new Response(
          "Google OAuth is not configured. Set the GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET secrets.",
          { status: 503 }
        );
      }
      const bytes = new Uint8Array(32);
      crypto.getRandomValues(bytes);
      const state = [...bytes]
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      const redirectUri = `${url.origin}/auth/google/callback`;
      this.sql`DELETE FROM oauth_state WHERE created_at < datetime('now', '-10 minutes')`;
      this.sql`INSERT INTO oauth_state (state, redirect_uri) VALUES (${state}, ${redirectUri})`;
      return Response.redirect(
        buildConsentUrl(this.env.GOOGLE_CLIENT_ID, redirectUri, state),
        302
      );
    }

    if (tail === "google/disconnect" && request.method === "POST") {
      const rows = this.sql<GoogleAuthRow>`
        SELECT refresh_token, access_token, access_expires_at, email
        FROM google_auth WHERE id = 1
      `;
      if (rows.length > 0) {
        await revokeToken(rows[0].refresh_token);
        this.sql`DELETE FROM google_auth WHERE id = 1`;
        this.#audit("google_disconnected", rows[0].email);
      }
      return Response.json({ ok: true });
    }

    const last = url.pathname.split("/").filter(Boolean).pop() ?? "";

    if (last === "overview" && request.method === "GET") {
      const { board, membership } = this.#activeBoard();
      const boards = this.sql<BoardRow>`
        SELECT name, labels, membership, is_home FROM boards
        ORDER BY is_home DESC, name ASC
      `;
      const facts = this.sql<FactRow & { board: string }>`
        SELECT id, fact, category, board, created_at FROM jarvis_facts
        ORDER BY id DESC LIMIT 50
      `;
      const audit = this.sql<AuditRow>`
        SELECT action, detail, created_at FROM jarvis_audit
        ORDER BY id DESC LIMIT 30
      `;
      return Response.json({
        board,
        membership,
        deployment_tx: this.#getState("last_deployment_tx"),
        boards,
        facts,
        audit
      });
    }

    if (last === "recovery" && request.method === "POST") {
      return Response.json(this.#postRecovery());
    }

    const factMatch = tail.match(/^facts\/(\d+)$/);
    if (factMatch && request.method === "DELETE") {
      const id = Number(factMatch[1]);
      this.sql`DELETE FROM jarvis_facts WHERE id = ${id}`;
      this.#audit("fact_deleted", `id=${id}`);
      return Response.json({ ok: true });
    }

    return super.onRequest(request);
  }

  async onCallStart() {
    // D1 (short-term): detect a new deployment transaction and flag it.
    const meta = this.env.CF_VERSION_METADATA;
    if (meta?.id) {
      const lastTx = this.#getState("last_deployment_tx");
      if (lastTx !== meta.id) {
        this.#setState("last_deployment_tx", meta.id);
        this.#setState("deploy_notice", "1");
        this.#audit(
          "deployment_detected",
          `tx=${meta.id}${lastTx ? ` prev=${lastTx}` : ""}`
        );
      }
    }
    const greeting = GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
    await this.speakAll(greeting);
    // Deliver anything that came due while no delivery channel was reachable.
    try {
      await this.#sweepReminders();
    } catch (err) {
      console.error("[jrvs] reminder sweep failed:", err);
    }
  }

  /** Scheduled-reminder callback (set via this.schedule in set_reminder). */
  async onReminder(payload: { message: string; id?: number | null }) {
    await this.#deliverReminder(payload.id ?? null, payload.message);
  }

  /**
   * Deliver one reminder on every channel that can reach the user: Telegram
   * always (works 24/7 with the app closed), spoken aloud too when a call is
   * live. Marked delivered only when a channel succeeded, so the sweep retries.
   */
  async #deliverReminder(id: number | null, message: string) {
    if (id != null) {
      const row = this.sql<{ delivered_at: string | null }>`
        SELECT delivered_at FROM reminders WHERE id = ${id}
      `[0];
      if (row?.delivered_at) return; // alarm + sweep can race — deliver once
    }
    const text = `Reminder, sir: ${message}`;
    const tg = await sendTelegramMessage(this.env, text);
    if (!tg.ok) console.error("[jrvs] telegram delivery failed:", tg.error);
    let spoken = false;
    try {
      if ([...this.getConnections()].length > 0) {
        await this.speakAll(text);
        spoken = true;
      }
    } catch (err) {
      console.error("[jrvs] spoken reminder failed:", err);
    }
    if (tg.ok || spoken) {
      if (id != null) {
        this.sql`UPDATE reminders SET delivered_at = datetime('now') WHERE id = ${id}`;
      }
      this.#audit(
        "reminder_delivered",
        `telegram=${tg.ok} spoken=${spoken} msg=${message.slice(0, 80)}`
      );
    }
  }

  /** Deliver overdue reminders a sleeping/evicted DO missed. */
  async #sweepReminders() {
    const due = this.sql<{ id: number; message: string }>`
      SELECT id, message FROM reminders
      WHERE delivered_at IS NULL AND due_at <= datetime('now')
      ORDER BY due_at ASC LIMIT 10
    `;
    for (const r of due) {
      await this.#deliverReminder(r.id, r.message);
    }
    // Don't let a dead channel retry forever: expire after a day overdue.
    this.sql`
      UPDATE reminders SET delivered_at = datetime('now')
      WHERE delivered_at IS NULL AND due_at <= datetime('now', '-1 day')
    `;
  }

  /**
   * 24/7 heartbeat (scheduleEvery in onStart). Even with no call connected the
   * DO wakes here: missed reminders go out, deadlines get nudged, and the
   * morning briefing is sent — all over Telegram.
   */
  async onHeartbeat() {
    try {
      await this.#sweepReminders();
    } catch (err) {
      console.error("[jrvs] reminder sweep failed:", err);
    }
    try {
      await this.#watchDeadlines();
    } catch (err) {
      console.error("[jrvs] deadline watch failed:", err);
    }
    try {
      await this.#maybeSendBriefing();
    } catch (err) {
      console.error("[jrvs] briefing failed:", err);
    }
  }

  /** Nudge dated facts (24h ahead) and calendar events (60min ahead), once each. */
  async #watchDeadlines() {
    if (!telegramConfigured(this.env)) return;
    const soon = this.sql<{ id: number; fact: string; due_at: string }>`
      SELECT id, fact, due_at FROM jarvis_facts
      WHERE due_at IS NOT NULL AND notified_at IS NULL
        AND due_at <= datetime('now', '+1 day')
        AND due_at > datetime('now', '-1 day')
      ORDER BY due_at ASC LIMIT 5
    `;
    for (const f of soon) {
      const res = await sendTelegramMessage(
        this.env,
        `Heads up, sir — coming due: ${f.fact}`
      );
      if (res.ok) {
        this.sql`UPDATE jarvis_facts SET notified_at = datetime('now') WHERE id = ${f.id}`;
        this.#audit("deadline_nudge", `fact#${f.id} due=${f.due_at}`);
      }
    }
    // Calendar events starting within the hour (deduped in event_nudges).
    const google = this.sql<{ email: string }>`
      SELECT email FROM google_auth WHERE id = 1
    `;
    if (google.length === 0) return;
    const token = await this.#googleAccessToken();
    const now = new Date();
    const events = await listCalendarEvents(
      token,
      now.toISOString(),
      new Date(now.getTime() + 3_600_000).toISOString(),
      5
    );
    for (const ev of events) {
      const key = `${ev.start}|${ev.summary}`.slice(0, 300);
      const seen = this.sql<{ event_key: string }>`
        SELECT event_key FROM event_nudges WHERE event_key = ${key}
      `;
      if (seen.length > 0) continue;
      const res = await sendTelegramMessage(
        this.env,
        `Up next, sir: ${ev.summary} (${ev.start}${ev.location ? `, ${ev.location}` : ""})`
      );
      if (res.ok) {
        this.sql`INSERT INTO event_nudges (event_key) VALUES (${key})`;
        this.#audit("event_nudge", key.slice(0, 120));
      }
    }
    this.sql`DELETE FROM event_nudges WHERE created_at < datetime('now', '-2 days')`;
  }

  #briefingConfig(): { hour: number; tz: string } {
    const hour = Number(this.#getState("briefing_hour") ?? "7");
    return {
      hour: Number.isFinite(hour) ? Math.trunc(hour) : 7,
      tz: this.#getState("briefing_tz") ?? "America/New_York"
    };
  }

  /** Send the daily briefing once, at/after the configured local hour. */
  async #maybeSendBriefing() {
    if (!telegramConfigured(this.env)) return;
    const { hour, tz } = this.#briefingConfig();
    if (hour < 0) return; // disabled
    let parts: Intl.DateTimeFormatPart[];
    try {
      parts = new Intl.DateTimeFormat("en-CA", {
        timeZone: tz,
        hour12: false,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit"
      }).formatToParts(new Date());
    } catch {
      return; // unparseable timezone — configure_briefing validates, but stay safe
    }
    const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
    const localDay = `${get("year")}-${get("month")}-${get("day")}`;
    const localHour = Number(get("hour")) % 24; // some ICU builds emit "24"
    if (localHour < hour) return;
    if (this.#getState("last_briefing_day") === localDay) return;
    const brief = await this.#composeBriefing();
    const res = await sendTelegramMessage(this.env, brief);
    if (res.ok) {
      this.#setState("last_briefing_day", localDay);
      this.#audit("briefing_sent", localDay);
    } else {
      console.error("[jrvs] briefing send failed:", res.error);
    }
  }

  /** Build the morning briefing: calendar + unread mail + coming-due items. */
  async #composeBriefing(): Promise<string> {
    let events: CalendarEvent[] = [];
    let emails: EmailSummary[] = [];
    const google = this.sql<{ email: string }>`
      SELECT email FROM google_auth WHERE id = 1
    `;
    if (google.length > 0) {
      try {
        const token = await this.#googleAccessToken();
        const now = new Date();
        events = await listCalendarEvents(
          token,
          now.toISOString(),
          new Date(now.getTime() + 86_400_000).toISOString(),
          10
        );
        emails = await listEmails(token, "is:unread newer_than:1d", 5);
      } catch (err) {
        console.error("[jrvs] briefing google fetch failed:", err);
      }
    }
    const dueSoon = this.sql<{ fact: string; due_at: string }>`
      SELECT fact, due_at FROM jarvis_facts
      WHERE due_at IS NOT NULL
        AND due_at BETWEEN datetime('now') AND datetime('now', '+3 days')
      ORDER BY due_at ASC LIMIT 5
    `;
    try {
      const result = await generateText({
        model: this.#model(),
        system:
          "You are JRVS, a personal assistant with a calm, dry wit. Compose the user's morning briefing as one short Telegram message: plain text, no markdown, at most 8 short lines. Lead with whatever actually needs attention; omit empty sections. Address the user as sir once.",
        prompt: JSON.stringify({
          calendar_next_24h: events,
          unread_email_last_day: emails.map((e) => ({
            from: e.from,
            subject: e.subject
          })),
          coming_due: dueSoon
        })
      });
      const text = result.text.trim();
      if (text) return text;
    } catch (err) {
      console.error("[jrvs] briefing compose failed:", err);
    }
    // Deterministic fallback if the model is unavailable.
    const lines = [
      `Morning briefing: ${events.length} event(s) in the next 24 hours, ${emails.length} unread email(s).`
    ];
    if (events[0]) lines.push(`First up: ${events[0].summary} at ${events[0].start}.`);
    for (const d of dueSoon) lines.push(`Coming due: ${d.fact}`);
    return lines.join("\n");
  }

  /** Digest the conversation into durable memory shortly after the call ends. */
  async onCallEnd() {
    const pending = this.sql<{ n: number }>`
      SELECT COUNT(*) AS n FROM conversation_log WHERE processed = 0
    `[0];
    if ((pending?.n ?? 0) > 0) {
      await this.schedule(10, "onConsolidate", {});
    }
  }

  /** Keep raw turns for post-call consolidation (trimmed, tagged to the board). */
  #logTurn(role: "user" | "assistant", text: string) {
    const t = text.trim();
    if (!t) return;
    const { board } = this.#activeBoard();
    this.sql`
      INSERT INTO conversation_log (role, text, board)
      VALUES (${role}, ${t.slice(0, 4000)}, ${board})
    `;
  }

  /**
   * Background cognition: after a call, distill the raw transcript into
   * lasting facts — add new ones, correct stale ones, drop retracted ones —
   * entirely off the live voice path.
   */
  async onConsolidate() {
    const rows = this.sql<{ id: number; role: string; text: string; board: string }>`
      SELECT id, role, text, board FROM conversation_log
      WHERE processed = 0 ORDER BY id ASC LIMIT 100
    `;
    if (rows.length === 0) return;
    try {
      const facts = this.sql<FactRow & { board: string }>`
        SELECT id, fact, category, board, created_at FROM jarvis_facts
        ORDER BY id DESC LIMIT 50
      `;
      const result = await generateText({
        model: this.#model(),
        system: CONSOLIDATION_PROMPT,
        prompt: JSON.stringify({
          existing_facts: facts.map((f) => ({
            id: f.id,
            fact: f.fact,
            category: f.category
          })),
          conversation: rows.map((r) => ({ role: r.role, text: r.text }))
        })
      });
      const ops = consolidationSchema.parse(JSON.parse(extractJson(result.text)));
      const existingIds = new Set(facts.map((f) => f.id));
      const board = rows[rows.length - 1].board;
      let added = 0;
      let updated = 0;
      let removed = 0;
      for (const f of ops.add.slice(0, 10)) {
        const dup = this.sql<{ id: number }>`
          SELECT id FROM jarvis_facts WHERE fact = ${f.fact} LIMIT 1
        `;
        if (dup.length > 0) continue;
        let due: string | null = null;
        if (f.due_at) {
          const t = Date.parse(f.due_at);
          if (!Number.isNaN(t)) due = sqliteUtc(new Date(t));
        }
        this.sql`
          INSERT INTO jarvis_facts (fact, category, board, due_at)
          VALUES (${f.fact}, ${f.category ?? "general"}, ${board}, ${due})
        `;
        added++;
      }
      for (const u of ops.update.slice(0, 10)) {
        if (!existingIds.has(u.id)) continue;
        this.sql`UPDATE jarvis_facts SET fact = ${u.fact} WHERE id = ${u.id}`;
        updated++;
      }
      for (const id of ops.remove.slice(0, 10)) {
        if (!existingIds.has(id)) continue;
        this.sql`DELETE FROM jarvis_facts WHERE id = ${id}`;
        removed++;
      }
      if (added || updated || removed) {
        this.#audit(
          "memory_consolidated",
          `turns=${rows.length} added=${added} updated=${updated} removed=${removed}`
        );
      }
    } catch (err) {
      console.error("[jrvs] consolidation failed:", err);
    } finally {
      // Processed either way: a poison transcript must not wedge the queue.
      for (const r of rows) {
        this.sql`UPDATE conversation_log SET processed = 1 WHERE id = ${r.id}`;
      }
      this.sql`DELETE FROM conversation_log WHERE created_at < datetime('now', '-7 days')`;
    }
  }

  /** Default Workers AI model id (primary). */
  get #primaryModelId(): string {
    return this.env.WORKERS_AI_MODEL ?? "@hf/nousresearch/hermes-2-pro-mistral-7b";
  }

  /** Fallback Workers AI model id, tried only when the primary throws. */
  get #fallbackModelId(): string {
    return (
      this.env.WORKERS_AI_FALLBACK_MODEL ?? "@cf/qwen/qwen2.5-coder-32b-instruct"
    );
  }

  /**
   * Pick a model: Claude if a key is configured, Workers AI otherwise. An
   * optional Workers AI model id override selects the primary vs fallback.
   */
  #model(workersModelId?: string): LanguageModel {
    if (this.env.ANTHROPIC_API_KEY) {
      const anthropic = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY });
      return anthropic(this.env.CLAUDE_MODEL ?? "claude-haiku-4-5");
    }
    const workersai = createWorkersAI({ binding: this.env.AI });
    return workersai(
      (workersModelId ?? this.#primaryModelId) as Parameters<typeof workersai>[0],
      { sessionAffinity: this.sessionAffinity }
    );
  }

  /**
   * Fallback tool execution for Workers AI models that emit a tool call as
   * plain text rather than a structured tool_call the SDK can run. Parses the
   * emitted call, executes the matching tool, then has the model phrase a
   * short spoken confirmation. Returns null if no tool call was present.
   */
  async #dispatchEmittedToolCall(
    rawText: string,
    tools: Record<string, ToolLike>
  ): Promise<string | null> {
    const call = parseToolCall(rawText);
    if (!call) return null;
    const t = tools[call.name];
    if (!t || typeof t.execute !== "function") return null;

    let result: unknown;
    try {
      const args =
        typeof t.inputSchema?.parse === "function"
          ? t.inputSchema.parse(call.args)
          : call.args;
      result = await t.execute(args, {
        toolCallId: crypto.randomUUID(),
        messages: []
      });
    } catch (e) {
      result = { error: errText(e) };
    }
    this.#audit("tool_dispatched", `${call.name} (text-emitted)`);

    try {
      const summary = await generateText({
        model: this.#model(),
        system:
          "You are JRVS, a personal butler-style assistant. You just performed an action for the user. In one or two short spoken sentences, confirm the outcome warmly and concisely. Never output JSON, tool calls, or code.",
        prompt: `Action performed: ${call.name}\nResult: ${JSON.stringify(result).slice(0, 800)}`
      });
      const spoken = sanitizeSpoken(summary.text);
      if (spoken) return spoken;
    } catch (err) {
      console.error("[jrvs] tool-call summary failed:", err);
    }
    const errMsg = (result as { error?: string })?.error;
    return errMsg ? `I couldn't complete that, sir: ${errMsg}` : "Done, sir.";
  }

  #systemPrompt(): string {
    const facts = this.sql<FactRow>`
      SELECT id, fact, category, created_at FROM jarvis_facts
      ORDER BY id DESC LIMIT 50
    `;
    const factBlock =
      facts.length > 0
        ? `\n\nKnown facts about the user (most recent first):\n${facts
            .map((f) => `- [#${f.id}, ${f.category}] ${f.fact}`)
            .join("\n")}`
        : "";
    const google = this.sql<{ email: string }>`
      SELECT email FROM google_auth WHERE id = 1
    `;
    const googleLine =
      google.length > 0
        ? `Google account connected: ${google[0].email}. Email and calendar tools are live.`
        : "Google account NOT connected — email and calendar tools will fail until the user connects it in settings.";
    const telegramLine = telegramConfigured(this.env)
      ? "Telegram notifications are live: reminders, nudges and the daily briefing reach the user 24/7, even with the app closed."
      : "Telegram is NOT configured — reminders are only spoken during a live call; say so if the user expects one to land later.";
    const { board, membership } = this.#activeBoard();
    const boardLine = `Active board: ${board} (membership group ${membership}).${
      board !== "HOME" ? " The default HOME board is available via recovery." : ""
    }`;
    const deployLine =
      this.#getState("deploy_notice") === "1"
        ? "\nA new deployment was detected since the last session (deployment_tx updated). Mention it briefly once and offer a SUGGEST_SOURCE refresh."
        : "";
    const now = new Date().toUTCString();
    return `${PERSONA}\n\nCurrent time (UTC): ${now}\n${boardLine}\n${googleLine}\n${telegramLine}${deployLine}${factBlock}`;
  }

  /**
   * Phase-1 reliability guard: pass stream parts through while counting text.
   * AI SDK `error` parts (or a thrown stream error, or a zero-text turn) would
   * make the voice pipeline go silent — instead log for `wrangler tail` and
   * speak a fallback line.
   */
  async *#guardedStream(
    stream: AsyncIterable<unknown>
  ): AsyncGenerator<unknown> {
    let text = "";
    try {
      for await (const part of stream) {
        const p = part as {
          type?: string;
          text?: unknown;
          delta?: unknown;
          error?: unknown;
        };
        if (p?.type === "error") {
          console.error("[jrvs] model stream error:", p.error);
          yield fallbackDelta();
          return;
        }
        if (p?.type === "text-delta") {
          text +=
            typeof p.text === "string"
              ? p.text
              : typeof p.delta === "string"
                ? p.delta
                : "";
        }
        yield part;
      }
      if (text.length === 0) {
        console.error("[jrvs] turn produced no speakable text");
        yield fallbackDelta();
      }
    } catch (err) {
      console.error("[jrvs] turn failed:", err);
      yield fallbackDelta();
    } finally {
      // Whatever was actually spoken feeds post-call memory consolidation.
      this.#logTurn("assistant", text);
    }
  }

  async onTurn(transcript: string, context: VoiceTurnContext) {
    this.#logTurn("user", transcript);
    // The voice SDK saves the user message BEFORE building context.messages,
    // so history already ends with this transcript — don't append it twice.
    const messages = context.messages.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content
    }));
    if (messages.length === 0 || messages[messages.length - 1].role !== "user") {
      messages.push({ role: "user" as const, content: transcript });
    }
    const params = {
      system: this.#systemPrompt(),
      messages,
      tools: {
        get_current_time: tool({
          description:
            "Get the current date and time. Use when the user asks what time or day it is.",
          inputSchema: z.object({}),
          execute: async () => {
            const now = new Date();
            return {
              time: now.toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                timeZoneName: "short"
              }),
              date: now.toLocaleDateString("en-US", {
                weekday: "long",
                year: "numeric",
                month: "long",
                day: "numeric"
              })
            };
          }
        }),

        set_reminder: tool({
          description:
            "Set a reminder. It is spoken aloud if a call is live when it fires, and always delivered to the user's Telegram — so it works even after the app is closed.",
          inputSchema: z.object({
            seconds: z
              .number()
              .min(5)
              .describe("Delay in seconds from now until the reminder fires"),
            message: z.string().describe("What to remind the user about")
          }),
          execute: async ({ seconds, message }) => {
            this.sql`
              INSERT INTO reminders (message, due_at)
              VALUES (${message}, ${sqliteUtc(new Date(Date.now() + seconds * 1000))})
            `;
            const id =
              this.sql<{ id: number }>`SELECT last_insert_rowid() AS id`[0]?.id ??
              null;
            await this.schedule(seconds, "onReminder", { id, message });
            return {
              scheduled: true,
              in_seconds: seconds,
              delivery: telegramConfigured(this.env)
                ? "spoken if a call is live, and always sent to Telegram"
                : "spoken only if a call is live — Telegram is not configured"
            };
          }
        }),

        remember_fact: tool({
          description:
            "Store a lasting fact about the user in permanent memory (preferences, people, projects, deadlines, context). Use proactively whenever the user shares something worth keeping. For deadlines, appointments and dated commitments, include due_at so JRVS can nudge ahead of time.",
          inputSchema: z.object({
            fact: z
              .string()
              .describe("The fact, phrased as a standalone statement"),
            category: z
              .string()
              .optional()
              .describe(
                "One-word category, e.g. personal, work, project, preference, contact, deadline"
              ),
            due_at: z
              .string()
              .optional()
              .describe(
                "ISO 8601 UTC datetime when this comes due — only for deadlines, appointments and dated commitments"
              )
          }),
          execute: async ({ fact, category, due_at }) => {
            const { board } = this.#activeBoard();
            let due: string | null = null;
            if (due_at) {
              const t = Date.parse(due_at);
              if (!Number.isNaN(t)) due = sqliteUtc(new Date(t));
            }
            this.sql`
              INSERT INTO jarvis_facts (fact, category, board, due_at)
              VALUES (${fact}, ${category ?? "general"}, ${board}, ${due})
            `;
            return { remembered: true, board, due_at: due };
          }
        }),

        recall_facts: tool({
          description:
            "Search permanent memory for stored facts about the user. Use when asked about something previously shared, or to check context. Keyword search spans all boards; without a keyword it lists recent facts from the active board and HOME.",
          inputSchema: z.object({
            keyword: z
              .string()
              .optional()
              .describe("Keyword to search for; omit to list recent facts")
          }),
          execute: async ({ keyword }) => {
            const { board } = this.#activeBoard();
            const rows = keyword
              ? this.sql<FactRow & { board: string }>`
                  SELECT id, fact, category, board, created_at FROM jarvis_facts
                  WHERE fact LIKE ${`%${keyword}%`} OR category LIKE ${`%${keyword}%`}
                  ORDER BY id DESC LIMIT 25
                `
              : this.sql<FactRow & { board: string }>`
                  SELECT id, fact, category, board, created_at FROM jarvis_facts
                  WHERE board = ${board} OR board = 'HOME'
                  ORDER BY id DESC LIMIT 25
                `;
            return { facts: rows };
          }
        }),

        forget_fact: tool({
          description:
            "Delete a stored fact by its id (find the id with recall_facts first). Use when the user corrects or retracts something.",
          inputSchema: z.object({
            id: z.number().describe("The id of the fact to delete")
          }),
          execute: async ({ id }) => {
            this.sql`DELETE FROM jarvis_facts WHERE id = ${id}`;
            return { forgotten: true };
          }
        }),

        check_inbox: tool({
          description:
            "List recent emails from the user's Gmail inbox: sender, subject, date, snippet, and id. Use for questions like 'what's in my inbox', 'anything urgent', 'emails from X'.",
          inputSchema: z.object({
            query: z
              .string()
              .optional()
              .describe(
                'Gmail search query, e.g. "is:unread", "from:alice", "newer_than:1d". Omit for recent inbox mail.'
              ),
            max: z
              .number()
              .min(1)
              .max(20)
              .optional()
              .describe("Max emails to return (default 8)")
          }),
          execute: async ({ query, max }) => {
            try {
              const token = await this.#googleAccessToken();
              const q = query?.trim() ? query : "in:inbox";
              const emails = await listEmails(token, q, max ?? 8);
              this.#audit("inbox_checked", `query=${q} results=${emails.length}`);
              return { emails };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        }),

        read_email: tool({
          description:
            "Read the full body of one email by its id (get ids from check_inbox). Use before summarizing details or drafting a reply.",
          inputSchema: z.object({
            id: z.string().describe("Gmail message id from check_inbox")
          }),
          execute: async ({ id }) => {
            try {
              const token = await this.#googleAccessToken();
              const email = await getEmail(token, id);
              this.#audit("email_read", `id=${id} from=${email.from}`);
              const truncated = email.body.length > 4000;
              return {
                from: email.from,
                to: email.to,
                subject: email.subject,
                date: email.date,
                body: truncated ? `${email.body.slice(0, 4000)}…` : email.body,
                truncated
              };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        }),

        draft_email: tool({
          description:
            "Create a DRAFT email in the user's Gmail (never sends — the user reviews and sends it in Gmail). Confirm recipient and gist with the user first unless they dictated them explicitly. To draft a reply, pass reply_to_email_id so it threads correctly.",
          inputSchema: z.object({
            to: z.string().describe("Recipient email address"),
            subject: z.string().describe("Subject line"),
            body: z
              .string()
              .describe("Plain-text body, written in the user's voice"),
            reply_to_email_id: z
              .string()
              .optional()
              .describe(
                "Id of the email being replied to, to thread the draft into that conversation"
              )
          }),
          execute: async ({ to, subject, body, reply_to_email_id }) => {
            try {
              const token = await this.#googleAccessToken();
              let thread: DraftOptions["thread"];
              if (reply_to_email_id) {
                const orig = await getEmail(token, reply_to_email_id);
                thread = {
                  threadId: orig.threadId,
                  messageIdHeader: orig.messageIdHeader
                };
              }
              const draft = await createDraft(token, {
                to,
                subject,
                body,
                thread
              });
              this.#audit("draft_created", `to=${to} subject=${subject}`);
              return {
                draft_id: draft.id,
                note: "Draft saved to the user's Gmail Drafts folder. It has NOT been sent and JRVS cannot send it."
              };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        }),

        check_calendar: tool({
          description:
            "List upcoming Google Calendar events. Use for 'what's on my calendar', 'am I free Thursday', scheduling questions, and daily briefings.",
          inputSchema: z.object({
            days: z
              .number()
              .min(1)
              .max(14)
              .optional()
              .describe("How many days ahead to look (default 2)")
          }),
          execute: async ({ days }) => {
            try {
              const token = await this.#googleAccessToken();
              const span = days ?? 2;
              const now = new Date();
              const events = await listCalendarEvents(
                token,
                now.toISOString(),
                new Date(now.getTime() + span * 86_400_000).toISOString()
              );
              this.#audit("calendar_checked", `days=${span} events=${events.length}`);
              return { window_days: span, events };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        }),

        review_recent_actions: tool({
          description:
            "List JRVS's own recent external actions from the audit log (inbox checks, emails read, drafts created, account changes). Use when the user asks what you've done or accessed.",
          inputSchema: z.object({}),
          execute: async () => {
            const actions = this.sql<AuditRow>`
              SELECT action, detail, created_at FROM jarvis_audit
              ORDER BY id DESC LIMIT 20
            `;
            return { actions };
          }
        }),

        suggest_source: tool({
          description:
            "SUGGEST_SOURCE: memory refresh + product-safety triage. Trigger on 'SUGGEST_SOURCE', 'I need the (default) HOME board', 'reset to default membership', 'reset to a known good state', or reminders saying so. Read-only: reports boards ledger, active board/membership, memory stats, deployment changes, and the recommended POST_RECOVERY action. Never applies changes itself.",
          inputSchema: z.object({}),
          execute: async () => {
            const { board, membership } = this.#activeBoard();
            const boards = this.sql<BoardRow>`
              SELECT name, labels, membership, is_home FROM boards
              ORDER BY is_home DESC, name ASC
            `;
            const home = boards.find((b) => b.is_home === 1);
            const factTotals = this.sql<{ board: string; n: number }>`
              SELECT board, COUNT(*) AS n FROM jarvis_facts GROUP BY board
            `;
            const deployChanged = this.#getState("deploy_notice") === "1";
            this.#setState("deploy_notice", "0");
            const drift =
              board !== (home?.name ?? "HOME") ||
              membership !== (home?.membership ?? "HOME");
            this.#audit(
              "memory_refresh",
              `board=${board}/${membership} drift=${drift} deploy_changed=${deployChanged}`
            );
            return {
              active: { board, membership },
              default_home_board: home ?? null,
              boards_ledger: boards,
              facts_by_board: factTotals,
              deployment_changed: deployChanged,
              deployment_tx: this.#getState("last_deployment_tx"),
              drift_from_default: drift,
              recommended_action: drift
                ? "Run POST_RECOVERY (post_recovery tool) to reset to the default HOME board and default membership group — ask the user to confirm first."
                : "State nominal: active board matches the default HOME baseline. No recovery needed."
            };
          }
        }),

        post_recovery: tool({
          description:
            "POST_RECOVERY: apply the reset recommended by suggest_source — restore active board to the default HOME board and membership to the default group. Only call AFTER the user has verbally confirmed. Never deletes facts or history.",
          inputSchema: z.object({}),
          execute: async () => this.#postRecovery()
        }),

        list_boards: tool({
          description:
            "List the boards ledger. The default HOME board is always the first entry, with its labels and membership group.",
          inputSchema: z.object({}),
          execute: async () => {
            const boards = this.sql<BoardRow>`
              SELECT name, labels, membership, is_home FROM boards
              ORDER BY is_home DESC, name ASC
            `;
            const { board, membership } = this.#activeBoard();
            return { active: { board, membership }, boards };
          }
        }),

        switch_board: tool({
          description:
            "Switch the active board (memory context) for this and future sessions. The board must already exist — use create_board first for new ones.",
          inputSchema: z.object({
            name: z.string().describe("Board name, e.g. HOME or WORK")
          }),
          execute: async ({ name }) => {
            const target = name.trim().toUpperCase();
            const rows = this.sql<BoardRow>`
              SELECT name, labels, membership, is_home FROM boards WHERE name = ${target}
            `;
            if (rows.length === 0) {
              return {
                error: `Board ${target} does not exist. Offer to create it with create_board.`
              };
            }
            const before = this.#activeBoard();
            this.#setState("active_board", rows[0].name);
            this.#setState("active_membership", rows[0].membership);
            this.#audit("board_switched", `${before.board} -> ${rows[0].name}`);
            return {
              switched: true,
              board: rows[0].name,
              membership: rows[0].membership
            };
          }
        }),

        create_board: tool({
          description:
            "Create a new board (named memory context) in the ledger. Does not switch to it unless the user asks.",
          inputSchema: z.object({
            name: z.string().describe("Board name, e.g. WORK"),
            labels: z
              .string()
              .optional()
              .describe("Optional comma-separated labels/metadata")
          }),
          execute: async ({ name, labels }) => {
            const boardName = name.trim().toUpperCase();
            if (!boardName) return { error: "Board name required." };
            const existing = this.sql<BoardRow>`
              SELECT name, labels, membership, is_home FROM boards WHERE name = ${boardName}
            `;
            if (existing.length > 0) {
              return { error: `Board ${boardName} already exists.` };
            }
            this.sql`
              INSERT INTO boards (name, labels, membership, is_home)
              VALUES (${boardName}, ${labels ?? ""}, 'HOME', 0)
            `;
            this.#audit("board_created", boardName);
            return { created: true, board: boardName, membership: "HOME" };
          }
        }),

        configure_briefing: tool({
          description:
            "Configure the daily Telegram morning briefing: the local hour it is sent (0-23) and the IANA timezone. Hour -1 disables it. Use when the user asks to change, enable or disable the briefing.",
          inputSchema: z.object({
            hour: z
              .number()
              .min(-1)
              .max(23)
              .describe("Local hour to send the briefing (0-23), or -1 to disable"),
            timezone: z
              .string()
              .optional()
              .describe('IANA timezone, e.g. "America/New_York"')
          }),
          execute: async ({ hour, timezone }) => {
            if (timezone) {
              try {
                new Intl.DateTimeFormat("en-US", { timeZone: timezone });
              } catch {
                return { error: `Unknown timezone: ${timezone}` };
              }
              this.#setState("briefing_tz", timezone);
            }
            this.#setState("briefing_hour", String(Math.trunc(hour)));
            const cfg = this.#briefingConfig();
            this.#audit("briefing_configured", `hour=${cfg.hour} tz=${cfg.tz}`);
            return {
              configured: true,
              enabled: cfg.hour >= 0,
              hour: cfg.hour,
              timezone: cfg.tz,
              telegram_ready: telegramConfigured(this.env)
            };
          }
        }),

        request_feature: tool({
          description:
            "File a feature request or bug report on JRVS's own GitHub repository, assigned to the Copilot coding agent, which implements it and opens a pull request. Only call AFTER restating the title and getting the user's explicit confirmation aloud.",
          inputSchema: z.object({
            title: z.string().describe("Short imperative issue title"),
            details: z
              .string()
              .describe(
                "What to build or fix, acceptance criteria, and any context the coding agent needs"
              )
          }),
          execute: async ({ title, details }) => {
            try {
              const issue = await createFeatureIssue(
                this.env,
                title,
                `${details}\n\n---\nFiled by JRVS on behalf of the user (voice session).`
              );
              this.#audit("feature_requested", `#${issue.number} ${title}`);
              return {
                filed: true,
                issue_number: issue.number,
                url: issue.url,
                assigned_to_coding_agent: issue.assigned,
                note: issue.assigned
                  ? "The coding agent will pick it up and open a pull request."
                  : `Issue created, but Copilot assignment failed${
                      issue.assign_error ? `: ${issue.assign_error}` : ""
                    }. It can be assigned manually on GitHub.`
              };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        }),

        check_feature_requests: tool({
          description:
            "List the feature requests JRVS has filed on its GitHub repository and their status (open or closed). Use when the user asks about requested features or upgrade progress.",
          inputSchema: z.object({}),
          execute: async () => {
            try {
              const requests = await listFeatureRequests(this.env, 10);
              return { requests };
            } catch (e) {
              return { error: errText(e) };
            }
          }
        })
      },
      stopWhen: stepCountIs(8),
      abortSignal: context.signal
    };

    // Claude streams structured tool calls correctly — use the low-latency path.
    if (this.env.ANTHROPIC_API_KEY) {
      const result = streamText({ model: this.#model(), ...params });
      return this.#guardedStream(result.fullStream);
    }

    // Workers AI parses tool calls reliably only in non-streaming mode; with
    // streamText the function-call JSON leaks into the spoken text channel.
    // Try the primary model (Hermes), then fall back to Qwen ONLY if the
    // primary throws — never on empty text, since a tool call may already have
    // run and retrying would double-fire its side effects.
    const modelIds = [this.#primaryModelId, this.#fallbackModelId].filter(
      (id, i, all) => all.indexOf(id) === i
    );
    for (const modelId of modelIds) {
      try {
        const result = await generateText({
          model: this.#model(modelId),
          ...params
        });
        // If the SDK executed tools natively, result.text is the final answer.
        // Otherwise the model may have emitted a tool call as plain text — run
        // it ourselves so the action actually happens.
        if ((result.toolCalls?.length ?? 0) === 0) {
          const dispatched = await this.#dispatchEmittedToolCall(
            result.text,
            params.tools as unknown as Record<string, ToolLike>
          );
          if (dispatched) {
            this.#logTurn("assistant", dispatched);
            return dispatched;
          }
        }
        const text = sanitizeSpoken(result.text);
        if (text) {
          this.#logTurn("assistant", text);
          return text;
        }
        console.error(
          `[jrvs] ${modelId} produced no speakable text (finishReason: ${result.finishReason})`
        );
        return FALLBACK_LINE;
      } catch (err) {
        console.error(`[jrvs] ${modelId} turn failed:`, err);
        // Only a thrown error advances to the fallback model.
      }
    }
    return FALLBACK_LINE;
  }
}

/**
 * Auth gate. If a JARVIS_TOKEN secret is configured, every request to the
 * Worker (i.e. all /agents/* traffic — static assets bypass this) must carry
 * it as a Bearer header or ?token= query param (WebSockets can't set headers
 * from the browser). Constant-time comparison via SHA-256 digests.
 */
async function isAuthorized(request: Request, env: Env): Promise<boolean> {
  const secret = env.JARVIS_TOKEN;
  if (!secret) return true; // no token configured — local dev mode

  const url = new URL(request.url);
  const provided =
    request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "") ??
    url.searchParams.get("token") ??
    "";

  // Hash both values to fixed-length buffers, then compare in constant time.
  const enc = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(provided)),
    crypto.subtle.digest("SHA-256", enc.encode(secret))
  ]);
  const va = new Uint8Array(a);
  const vb = new Uint8Array(b);
  let diff = 0;
  for (let i = 0; i < va.length; i++) diff |= va[i] ^ vb[i];
  return diff === 0;
}

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);

    // Google OAuth callback. Arrives from Google without our bearer token;
    // it is authenticated by the single-use CSRF state that only the agent
    // issued and stored (validated in completeGoogleAuth).
    if (url.pathname === "/auth/google/callback") {
      const code = url.searchParams.get("code");
      const state = url.searchParams.get("state");
      if (!code || !state) {
        return Response.redirect(`${url.origin}/?google=error`, 302);
      }
      const agent = await getAgentByName(env.JarvisAgent, "main");
      const result = await agent.completeGoogleAuth(code, state);
      return Response.redirect(
        `${url.origin}/?google=${result.ok ? "connected" : "error"}`,
        302
      );
    }

    if (!(await isAuthorized(request, env))) {
      return new Response("Unauthorized", { status: 401 });
    }
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
