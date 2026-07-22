# JRVS

Personal voice-first AI assistant ("Jarvis driver") on Cloudflare Workers.

```
Browser mic → WebSocket → Durable Object (JarvisAgent)
  → Flux STT → LLM + tools → streaming TTS → browser speaker
```

- **Voice pipeline** — `@cloudflare/voice` on Workers AI (Flux STT, Aura TTS). Click the orb or press Space to talk; type when you'd rather not.
- **One persistent brain** — a single Durable Object instance (`main`) holds conversation history, long-term memory (facts), Google credentials, and an audit log in its own SQLite.
- **LLM** — Claude (if `ANTHROPIC_API_KEY` is set) with Workers AI fallback.
- **Email & calendar** — Gmail read/search/draft and Google Calendar read, via least-privilege OAuth. JRVS **cannot send email**: it creates drafts, you review and send them in Gmail (human-in-the-loop by design).
- **Audit log** — every external action (inbox check, email read, draft created, account change) is recorded; ask *"what have you done recently?"*.

## Develop

```sh
npm install
npx wrangler login   # voice models run remotely even in local dev
npm run dev
```

## Deploy

```sh
npm run deploy
```

### Secrets

| Secret | Required | Purpose |
| --- | --- | --- |
| `JARVIS_TOKEN` | recommended | Access token gating all agent traffic. Set it, then enter the same value in the app's settings panel. |
| `ANTHROPIC_API_KEY` | optional | Use Claude for reasoning instead of Workers AI. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | optional | Enables Gmail + Calendar tools. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | optional | 24/7 notifications: reminders, deadline nudges, and the daily briefing land on Telegram even with the app closed. Create a bot with [@BotFather](https://t.me/BotFather); get your chat id from `getUpdates` after messaging the bot. |
| `GITHUB_TOKEN` | optional | Lets JRVS file feature requests on its own repo (`GITHUB_REPO` var), assigned to the Copilot coding agent. Needs issues write access. |

```sh
npx wrangler secret put JARVIS_TOKEN
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler secret put GOOGLE_CLIENT_ID
npx wrangler secret put GOOGLE_CLIENT_SECRET
npx wrangler secret put TELEGRAM_BOT_TOKEN
npx wrangler secret put TELEGRAM_CHAT_ID
npx wrangler secret put GITHUB_TOKEN
```

Optional vars in wrangler.jsonc: `CLAUDE_MODEL`, `WORKERS_AI_MODEL`, `GITHUB_REPO`.

## Google (Gmail + Calendar) setup

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable the **Gmail API** and **Google Calendar API**.
2. Configure the OAuth consent screen (External; adding yourself as a test user is fine for personal use).
3. Create an **OAuth client ID** (type: Web application) with authorized redirect URIs:
   - `https://<your-worker-domain>/auth/google/callback`
   - `http://localhost:5173/auth/google/callback` (for local dev)
4. Store the client id/secret as the secrets above (or in `.dev.vars` locally).
5. Open the app → settings (gear) → **Connect Google** and approve.

Scopes requested: `gmail.readonly`, `gmail.compose` (drafts only — the code never calls a send endpoint), `calendar.readonly`. The refresh token is sealed inside the Durable Object's SQLite and is never placed in LLM context.

## Try it

- "What's urgent in my inbox today?"
- "Read me the email from Alice."
- "Draft a reply saying I'll have the numbers by Friday." *(lands in Gmail Drafts for your review)*
- "What's on my calendar tomorrow?"
- "Remember that the Acme proposal is due next Thursday."
- "Remind me in 20 minutes to stretch."
- "What have you accessed recently?" *(audit log)*