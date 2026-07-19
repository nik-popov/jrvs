/**
 * Minimal Google OAuth + Gmail/Calendar REST helpers for JRVS.
 *
 * Plain fetch against googleapis.com — no SDK, works anywhere Workers run.
 * Credential discipline: refresh tokens never leave the Durable Object and
 * are never shown to the LLM. Callers pass a short-lived access token in.
 */

const CONSENT_URL = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL = "https://oauth2.googleapis.com/token";
const REVOKE_URL = "https://oauth2.googleapis.com/revoke";
const GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me";
const CALENDAR = "https://www.googleapis.com/calendar/v3";

/**
 * Least-privilege scopes:
 * - gmail.readonly   — list/read mail
 * - gmail.compose    — create drafts (Google has no drafts-only scope;
 *                      our code never calls any send endpoint)
 * - calendar.readonly — read events
 */
export const GOOGLE_SCOPES = [
  "https://www.googleapis.com/auth/gmail.readonly",
  "https://www.googleapis.com/auth/gmail.compose",
  "https://www.googleapis.com/auth/calendar.readonly"
].join(" ");

export interface GoogleTokens {
  access_token: string;
  expires_in: number;
  refresh_token?: string;
}

export function buildConsentUrl(
  clientId: string,
  redirectUri: string,
  state: string
): string {
  const u = new URL(CONSENT_URL);
  u.searchParams.set("client_id", clientId);
  u.searchParams.set("redirect_uri", redirectUri);
  u.searchParams.set("response_type", "code");
  u.searchParams.set("scope", GOOGLE_SCOPES);
  u.searchParams.set("access_type", "offline");
  u.searchParams.set("prompt", "consent"); // guarantees a refresh_token
  u.searchParams.set("state", state);
  return u.toString();
}

async function tokenRequest(
  params: Record<string, string>
): Promise<GoogleTokens> {
  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(params).toString()
  });
  if (!res.ok) {
    const detail = (await res.text()).slice(0, 300);
    throw new Error(`Google token endpoint ${res.status}: ${detail}`);
  }
  return (await res.json()) as GoogleTokens;
}

export function exchangeCode(
  clientId: string,
  clientSecret: string,
  code: string,
  redirectUri: string
): Promise<GoogleTokens> {
  return tokenRequest({
    grant_type: "authorization_code",
    client_id: clientId,
    client_secret: clientSecret,
    code,
    redirect_uri: redirectUri
  });
}

export function refreshAccessToken(
  clientId: string,
  clientSecret: string,
  refreshToken: string
): Promise<GoogleTokens> {
  return tokenRequest({
    grant_type: "refresh_token",
    client_id: clientId,
    client_secret: clientSecret,
    refresh_token: refreshToken
  });
}

/** Best-effort revocation on disconnect. */
export async function revokeToken(token: string): Promise<void> {
  try {
    await fetch(`${REVOKE_URL}?token=${encodeURIComponent(token)}`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" }
    });
  } catch {
    // Revocation is a courtesy; local deletion is what matters.
  }
}

async function gfetch<T>(accessToken: string, url: string): Promise<T> {
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${accessToken}` }
  });
  if (!res.ok) {
    const detail = (await res.text()).slice(0, 300);
    throw new Error(`Google API ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export function gmailProfile(
  accessToken: string
): Promise<{ emailAddress: string }> {
  return gfetch(accessToken, `${GMAIL}/profile`);
}

/* ---------------- Gmail message parsing ---------------- */

interface GmailHeader {
  name: string;
  value: string;
}
interface GmailPart {
  mimeType?: string;
  body?: { data?: string };
  parts?: GmailPart[];
  headers?: GmailHeader[];
}
interface GmailMessage {
  id: string;
  threadId: string;
  snippet?: string;
  payload?: GmailPart;
}

function header(part: GmailPart | undefined, name: string): string {
  const lower = name.toLowerCase();
  return (
    part?.headers?.find((h) => h.name.toLowerCase() === lower)?.value ?? ""
  );
}

function b64urlDecode(data: string): string {
  const bin = atob(data.replace(/-/g, "+").replace(/_/g, "/"));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function b64urlEncode(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function htmlToText(html: string): string {
  return html
    .replace(/<(style|script)[\s\S]*?<\/\1>/gi, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|tr|li|h[1-6])>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Depth-first: prefer a text/plain part, fall back to stripped text/html. */
function extractBody(payload: GmailPart | undefined): string {
  if (!payload) return "";
  let plain = "";
  let html = "";
  const walk = (p: GmailPart) => {
    if (p.body?.data) {
      if (p.mimeType?.startsWith("text/plain") && !plain) {
        plain = b64urlDecode(p.body.data);
      } else if (p.mimeType?.startsWith("text/html") && !html) {
        html = b64urlDecode(p.body.data);
      }
    }
    p.parts?.forEach(walk);
  };
  walk(payload);
  return plain || htmlToText(html);
}

/* ---------------- Gmail operations ---------------- */

export interface EmailSummary {
  id: string;
  threadId: string;
  from: string;
  subject: string;
  date: string;
  snippet: string;
}

export async function listEmails(
  accessToken: string,
  query: string,
  max: number
): Promise<EmailSummary[]> {
  const list = await gfetch<{ messages?: { id: string }[] }>(
    accessToken,
    `${GMAIL}/messages?q=${encodeURIComponent(query)}&maxResults=${max}`
  );
  const ids = (list.messages ?? []).map((m) => m.id);
  return Promise.all(
    ids.map(async (id) => {
      const m = await gfetch<GmailMessage>(
        accessToken,
        `${GMAIL}/messages/${id}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date`
      );
      return {
        id: m.id,
        threadId: m.threadId,
        from: header(m.payload, "From"),
        subject: header(m.payload, "Subject"),
        date: header(m.payload, "Date"),
        snippet: m.snippet ?? ""
      };
    })
  );
}

export interface EmailDetail extends EmailSummary {
  to: string;
  messageIdHeader: string;
  body: string;
}

export async function getEmail(
  accessToken: string,
  id: string
): Promise<EmailDetail> {
  const m = await gfetch<GmailMessage>(
    accessToken,
    `${GMAIL}/messages/${encodeURIComponent(id)}?format=full`
  );
  return {
    id: m.id,
    threadId: m.threadId,
    from: header(m.payload, "From"),
    to: header(m.payload, "To"),
    subject: header(m.payload, "Subject"),
    date: header(m.payload, "Date"),
    messageIdHeader: header(m.payload, "Message-ID"),
    snippet: m.snippet ?? "",
    body: extractBody(m.payload)
  };
}

function encodeSubject(subject: string): string {
  return /^[\x20-\x7E]*$/.test(subject)
    ? subject
    : `=?UTF-8?B?${b64urlEncode(subject).replace(/-/g, "+").replace(/_/g, "/")}?=`;
}

export interface DraftOptions {
  to: string;
  subject: string;
  body: string;
  /** Reply threading: ties the draft to an existing conversation. */
  thread?: { threadId: string; messageIdHeader: string };
}

/** Creates a Gmail draft. Never sends — the user reviews and sends in Gmail. */
export async function createDraft(
  accessToken: string,
  opts: DraftOptions
): Promise<{ id: string }> {
  const headers = [
    `To: ${opts.to}`,
    `Subject: ${encodeSubject(opts.subject)}`,
    'Content-Type: text/plain; charset="UTF-8"',
    "MIME-Version: 1.0"
  ];
  if (opts.thread?.messageIdHeader) {
    headers.push(
      `In-Reply-To: ${opts.thread.messageIdHeader}`,
      `References: ${opts.thread.messageIdHeader}`
    );
  }
  const message: { raw: string; threadId?: string } = {
    raw: b64urlEncode(`${headers.join("\r\n")}\r\n\r\n${opts.body}`)
  };
  if (opts.thread) message.threadId = opts.thread.threadId;

  const res = await fetch(`${GMAIL}/drafts`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });
  if (!res.ok) {
    const detail = (await res.text()).slice(0, 300);
    throw new Error(`Gmail draft creation ${res.status}: ${detail}`);
  }
  const draft = (await res.json()) as { id: string };
  return { id: draft.id };
}

/* ---------------- Calendar ---------------- */

interface GcalTime {
  dateTime?: string;
  date?: string;
}
interface GcalEvent {
  summary?: string;
  location?: string;
  start?: GcalTime;
  end?: GcalTime;
}

export interface CalendarEvent {
  summary: string;
  start: string;
  end: string;
  location?: string;
}

export async function listCalendarEvents(
  accessToken: string,
  timeMin: string,
  timeMax: string,
  max = 20
): Promise<CalendarEvent[]> {
  const u = new URL(`${CALENDAR}/calendars/primary/events`);
  u.searchParams.set("timeMin", timeMin);
  u.searchParams.set("timeMax", timeMax);
  u.searchParams.set("singleEvents", "true");
  u.searchParams.set("orderBy", "startTime");
  u.searchParams.set("maxResults", String(max));
  const data = await gfetch<{ items?: GcalEvent[] }>(accessToken, u.toString());
  return (data.items ?? []).map((e) => ({
    summary: e.summary ?? "(no title)",
    start: e.start?.dateTime ?? e.start?.date ?? "",
    end: e.end?.dateTime ?? e.end?.date ?? "",
    ...(e.location ? { location: e.location } : {})
  }));
}
