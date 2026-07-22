/**
 * Minimal Telegram Bot API helper for JRVS notifications.
 *
 * Outbound-only: reminders, briefings and nudges are pushed to one chat
 * (TELEGRAM_CHAT_ID) via the bot (TELEGRAM_BOT_TOKEN). Plain text, no
 * parse_mode — nothing to escape, nothing to break. This is what makes
 * JRVS 24/7: Durable Object alarms fire with no call connected, and
 * Telegram is the channel that still reaches the user.
 */

export interface TelegramEnv {
  TELEGRAM_BOT_TOKEN?: string;
  TELEGRAM_CHAT_ID?: string;
}

export function telegramConfigured(env: TelegramEnv): boolean {
  return Boolean(env.TELEGRAM_BOT_TOKEN && env.TELEGRAM_CHAT_ID);
}

/**
 * Send a plain-text message to the configured chat. Never throws —
 * notification failures must not break turns, alarms or consolidation.
 */
export async function sendTelegramMessage(
  env: TelegramEnv,
  text: string
): Promise<{ ok: boolean; error?: string }> {
  if (!telegramConfigured(env)) {
    return {
      ok: false,
      error:
        "Telegram not configured — set the TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets."
    };
  }
  try {
    const res = await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: env.TELEGRAM_CHAT_ID,
          // Telegram hard limit is 4096 chars per message.
          text: text.slice(0, 4096)
        })
      }
    );
    if (!res.ok) {
      const detail = (await res.text()).slice(0, 300);
      return { ok: false, error: `Telegram API ${res.status}: ${detail}` };
    }
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
