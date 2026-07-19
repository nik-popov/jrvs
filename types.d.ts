/**
 * Environment bindings for the jrvs Worker.
 * Keep in sync with wrangler.jsonc. Regenerate with `npm run types`
 * (wrangler types env.d.ts) if you prefer generated types.
 */
declare namespace Cloudflare {
  interface Env {
    AI: Ai;
    JarvisAgent: DurableObjectNamespace<import("./src/server").JarvisAgent>;
    /** Optional auth token (wrangler secret put JARVIS_TOKEN). When unset, auth is disabled (dev). */
    JARVIS_TOKEN?: string;
    /** Optional: use Claude instead of Workers AI (wrangler secret put ANTHROPIC_API_KEY). */
    ANTHROPIC_API_KEY?: string;
    /** Optional Claude model override. Default: claude-haiku-4-5 */
    CLAUDE_MODEL?: string;
    /** Optional Workers AI model override. Default: @cf/zai-org/glm-4.7-flash */
    WORKERS_AI_MODEL?: string;
    /** Optional Google OAuth client id (wrangler secret put GOOGLE_CLIENT_ID). Enables Gmail/Calendar tools. */
    GOOGLE_CLIENT_ID?: string;
    /** Optional Google OAuth client secret (wrangler secret put GOOGLE_CLIENT_SECRET). */
    GOOGLE_CLIENT_SECRET?: string;
    /** Deployment version metadata (wrangler.jsonc version_metadata binding). */
    CF_VERSION_METADATA?: WorkerVersionMetadata;
  }
}
interface Env extends Cloudflare.Env {}
