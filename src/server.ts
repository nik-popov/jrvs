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
  type DraftOptions
} from "./google";

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

const GREETINGS = [
  "Online. At your service, sir.",
  "Systems nominal. What do you need?",
  "Good to have you back, sir.",
  "Standing by."
];

export class JarvisAgent extends VoiceAgent<Env> {
  transcriber = new WorkersAIFluxSTT(this.env.AI);
  tts = new WorkersAITTS(this.env.AI, { speaker: "orion" });

  onStart() {
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
  }

  /** Scheduled-reminder callback (set via this.schedule in set_reminder). */
  async onReminder(payload: { message: string }) {
    await this.speakAll(`Reminder, sir: ${payload.message}`);
  }

  /** Pick the best available model: Claude if a key is configured, Workers AI otherwise. */
  #model(): LanguageModel {
    if (this.env.ANTHROPIC_API_KEY) {
      const anthropic = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY });
      return anthropic(this.env.CLAUDE_MODEL ?? "claude-haiku-4-5");
    }
    const workersai = createWorkersAI({ binding: this.env.AI });
    return workersai(
      (this.env.WORKERS_AI_MODEL ??
        "@cf/zai-org/glm-4.7-flash") as Parameters<typeof workersai>[0],
      { sessionAffinity: this.sessionAffinity }
    );
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
    const { board, membership } = this.#activeBoard();
    const boardLine = `Active board: ${board} (membership group ${membership}).${
      board !== "HOME" ? " The default HOME board is available via recovery." : ""
    }`;
    const deployLine =
      this.#getState("deploy_notice") === "1"
        ? "\nA new deployment was detected since the last session (deployment_tx updated). Mention it briefly once and offer a SUGGEST_SOURCE refresh."
        : "";
    const now = new Date().toUTCString();
    return `${PERSONA}\n\nCurrent time (UTC): ${now}\n${boardLine}\n${googleLine}${deployLine}${factBlock}`;
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
    let textChars = 0;
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
          const t =
            typeof p.text === "string"
              ? p.text
              : typeof p.delta === "string"
                ? p.delta
                : "";
          textChars += t.length;
        }
        yield part;
      }
      if (textChars === 0) {
        console.error("[jrvs] turn produced no speakable text");
        yield fallbackDelta();
      }
    } catch (err) {
      console.error("[jrvs] turn failed:", err);
      yield fallbackDelta();
    }
  }

  async onTurn(transcript: string, context: VoiceTurnContext) {
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
            "Set a reminder that will be spoken aloud after a delay. Use when the user asks to be reminded of something.",
          inputSchema: z.object({
            seconds: z
              .number()
              .min(5)
              .describe("Delay in seconds from now until the reminder fires"),
            message: z.string().describe("What to remind the user about")
          }),
          execute: async ({ seconds, message }) => {
            await this.schedule(seconds, "onReminder", { message });
            return { scheduled: true, in_seconds: seconds };
          }
        }),

        remember_fact: tool({
          description:
            "Store a lasting fact about the user in permanent memory (preferences, people, projects, deadlines, context). Use proactively whenever the user shares something worth keeping.",
          inputSchema: z.object({
            fact: z
              .string()
              .describe("The fact, phrased as a standalone statement"),
            category: z
              .string()
              .optional()
              .describe(
                "One-word category, e.g. personal, work, project, preference, contact"
              )
          }),
          execute: async ({ fact, category }) => {
            const { board } = this.#activeBoard();
            this.sql`
              INSERT INTO jarvis_facts (fact, category, board)
              VALUES (${fact}, ${category ?? "general"}, ${board})
            `;
            return { remembered: true, board };
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
    try {
      const result = await generateText({ model: this.#model(), ...params });
      const text = sanitizeSpoken(result.text);
      if (text) return text;
      console.error(
        "[jrvs] turn produced no speakable text (finishReason:",
        result.finishReason,
        ")"
      );
      return FALLBACK_LINE;
    } catch (err) {
      console.error("[jrvs] turn failed:", err);
      return FALLBACK_LINE;
    }
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
