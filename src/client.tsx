/**
 * JRVS client — voice-first UI.
 *
 * Connects to the JarvisAgent Durable Object over WebSocket via
 * useVoiceAgent: mic PCM streams up, transcripts + TTS audio stream back.
 * Fixed instance name "main" = one persistent Jarvis brain.
 */
import { useVoiceAgent, type VoiceStatus } from "@cloudflare/voice/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const TOKEN_KEY = "jrvs_token";
const AGENT_BASE = "/agents/jarvis-agent/main";

function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

const authHeaders = (token: string): HeadersInit | undefined =>
  token ? { Authorization: `Bearer ${token}` } : undefined;

/**
 * Safari/WebKit's echo cancellation does not reference audio played via Web
 * Audio, so the mic hears JRVS's own TTS. That echo constantly triggered the
 * SDK's barge-in interrupt (audio cutting out mid-sentence) and re-transcribed
 * JRVS's voice as user speech — a feedback loop of echo, stutter, and runaway
 * turns. On WebKit we run half-duplex: the mic is muted while JRVS speaks.
 * (All iOS browsers are WebKit, so they need it too.)
 */
const IS_WEBKIT_AUDIO = (() => {
  const ua = navigator.userAgent;
  const isIOS =
    /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const isSafari =
    /safari/i.test(ua) && !/chrome|chromium|crios|fxios|edg|opr/i.test(ua);
  return isIOS || isSafari;
})();

interface BoardEntry {
  name: string;
  labels: string;
  membership: string;
  is_home: number;
}
interface FactEntry {
  id: number;
  fact: string;
  category: string;
  board: string;
  created_at: string;
}
interface AuditEntry {
  action: string;
  detail: string;
  created_at: string;
}
interface Overview {
  board: string;
  membership: string;
  deployment_tx: string | null;
  boards: BoardEntry[];
  facts: FactEntry[];
  audit: AuditEntry[];
}

const STATUS_LABEL: Record<VoiceStatus, string> = {
  idle: "standby",
  listening: "listening",
  thinking: "thinking",
  speaking: "speaking"
};

function App() {
  const token = useRef(getToken()).current;
  const [textInput, setTextInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  const refreshOverview = useCallback(async () => {
    try {
      const res = await fetch(`${AGENT_BASE}/overview`, {
        headers: authHeaders(token)
      });
      if (res.ok) setOverview((await res.json()) as Overview);
    } catch {
      // overview stays null — chip and panels simply hidden
    }
  }, [token]);

  useEffect(() => {
    void refreshOverview();
  }, [refreshOverview]);

  // Re-sync board/memory/audit state whenever settings open
  useEffect(() => {
    if (showSettings) void refreshOverview();
  }, [showSettings, refreshOverview]);

  // Surface the result of a Google OAuth round-trip (?google=connected|error)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const g = params.get("google");
    if (!g) return;
    setNotice(
      g === "connected"
        ? "Google account connected. Email and calendar are live."
        : "Google connection failed — check the worker logs and try again."
    );
    if (g !== "connected") setShowSettings(true);
    params.delete("google");
    const qs = params.toString();
    window.history.replaceState(
      {},
      "",
      qs ? `?${qs}` : window.location.pathname
    );
  }, []);

  const {
    status,
    transcript,
    interimTranscript,
    metrics,
    audioLevel,
    isMuted,
    connected,
    error,
    startCall,
    endCall,
    toggleMute,
    sendText
  } = useVoiceAgent({
    agent: "jarvis-agent",
    name: "main",
    query: token ? { token } : undefined,
    // WebKit: echo from TTS leaks into the mic (see IS_WEBKIT_AUDIO). Make
    // barge-in require sustained, clearly-above-echo input so residual leak
    // during mute/unmute transitions can't cut playback.
    ...(IS_WEBKIT_AUDIO ? { interruptThreshold: 0.15, interruptChunks: 5 } : {})
  });

  const isInCall = status !== "idle";

  // Half-duplex for WebKit: auto-mute the mic while JRVS is speaking (a muted
  // mic sends nothing and skips level processing, so echo can neither
  // interrupt playback nor start a phantom user turn). The user's own mute
  // intent is tracked separately and always wins.
  const [userMuted, setUserMuted] = useState(false);
  useEffect(() => {
    if (!IS_WEBKIT_AUDIO || !isInCall) return;
    const shouldMute = userMuted || status === "speaking";
    if (isMuted !== shouldMute) toggleMute();
  }, [status, userMuted, isMuted, isInCall, toggleMute]);

  const handleToggleMute = useCallback(() => {
    setUserMuted((m) => !m);
    // Non-WebKit: the SDK mute is driven directly; WebKit reconciles in the
    // effect above.
    if (!IS_WEBKIT_AUDIO) toggleMute();
  }, [toggleMute]);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, interimTranscript]);

  // Spacebar toggles the call (unless typing)
  const toggleCall = useCallback(() => {
    if (isInCall) {
      endCall();
    } else {
      void startCall();
    }
  }, [isInCall, startCall, endCall]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }
      e.preventDefault();
      toggleCall();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleCall]);

  const submitText = (e: React.FormEvent) => {
    e.preventDefault();
    const text = textInput.trim();
    if (!text || !connected) return;
    sendText(text);
    setTextInput("");
  };

  const orbScale = 1 + Math.min(audioLevel * 1.8, 0.35);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className={`conn-dot ${connected ? "on" : "off"}`} />
          JRVS
          {overview && (
            <span
              className="board-chip"
              title={`membership group ${overview.membership}`}
            >
              {overview.board}
              {overview.boards.find((b) => b.name === overview.board)
                ?.is_home === 1
                ? " \u2605"
                : ""}
            </span>
          )}
        </div>
        <div className="topbar-right">
          <span className="status-label" data-status={status}>
            {connected ? STATUS_LABEL[status] : "connecting"}
          </span>
          <button
            type="button"
            className="ghost-btn"
            onClick={() => setShowSettings((s) => !s)}
            aria-label="Settings"
          >
            &#9881;
          </button>
        </div>
      </header>

      {showSettings && (
        <SettingsPanel overview={overview} onRefresh={refreshOverview} />
      )}

      {notice && (
        <div className="banner ok" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}

      {error && (
        <div className="banner error">
          {error}
          {!token && (
            <span>
              {" "}
              — if this deployment requires a token, set it in settings.
            </span>
          )}
        </div>
      )}

      <main className="stage">
        <button
          type="button"
          className="orb"
          data-status={status}
          onClick={toggleCall}
          style={{ ["--level" as string]: orbScale }}
          aria-label={isInCall ? "End call" : "Start call"}
        >
          <span className="orb-ring outer" />
          <span className="orb-ring inner" />
          <span className="orb-core" />
        </button>
        <div className="hint">
          {isInCall
            ? "click or press space to end"
            : "click or press space to talk"}
        </div>
        {isInCall && (
          <button
            type="button"
            className="ghost-btn mute"
            onClick={handleToggleMute}
          >
            {(IS_WEBKIT_AUDIO ? userMuted : isMuted) ? "unmute mic" : "mute mic"}
          </button>
        )}
      </main>

      <section className="transcript">
        {transcript.length === 0 && !interimTranscript ? (
          <div className="transcript-empty">
            {connected
              ? "Say something, or type below."
              : "Connecting to agent..."}
          </div>
        ) : (
          <>
            {transcript.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <span className="msg-role">
                  {m.role === "user" ? "you" : "jrvs"}
                </span>
                <span className="msg-text">{m.text}</span>
              </div>
            ))}
            {interimTranscript && (
              <div className="msg user interim">
                <span className="msg-role">you</span>
                <span className="msg-text">{interimTranscript}</span>
              </div>
            )}
          </>
        )}
        <div ref={transcriptEndRef} />
      </section>

      <form className="composer" onSubmit={submitText}>
        <input
          ref={inputRef}
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Type instead of talking..."
          disabled={!connected}
        />
        <button type="submit" disabled={!connected || !textInput.trim()}>
          Send
        </button>
      </form>

      <footer className="metrics">
        {metrics ? (
          <>
            <span>llm {metrics.llm_ms}ms</span>
            <span>tts {metrics.tts_ms}ms</span>
            <span>first audio {metrics.first_audio_ms}ms</span>
          </>
        ) : (
          <span>voice pipeline on cloudflare workers ai</span>
        )}
      </footer>
    </div>
  );
}

function SettingsPanel({
  overview,
  onRefresh
}: {
  overview: Overview | null;
  onRefresh: () => void | Promise<void>;
}) {
  const [value, setValue] = useState(getToken());

  const save = (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (value.trim()) {
        localStorage.setItem(TOKEN_KEY, value.trim());
      } else {
        localStorage.removeItem(TOKEN_KEY);
      }
    } catch {
      // localStorage unavailable — proceed; token just won't persist
    }
    // Reconnect with the new token
    window.location.reload();
  };

  return (
    <div className="settings-panel">
      <form className="settings" onSubmit={save}>
        <label htmlFor="token">Access token</label>
        <input
          id="token"
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="required if JARVIS_TOKEN is set on the worker"
          autoComplete="off"
        />
        <button type="submit">Save &amp; reconnect</button>
      </form>
      <GoogleSection />
      {overview && <BoardsSection overview={overview} onRefresh={onRefresh} />}
      {overview && <MemorySection facts={overview.facts} onRefresh={onRefresh} />}
      {overview && <AuditSection audit={overview.audit} />}
    </div>
  );
}

function BoardsSection({
  overview,
  onRefresh
}: {
  overview: Overview;
  onRefresh: () => void | Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const token = getToken();

  const reset = async () => {
    if (
      !window.confirm(
        "Reset to the default HOME board and default membership group? Facts and history are preserved."
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      await fetch(`${AGENT_BASE}/recovery`, {
        method: "POST",
        headers: authHeaders(token)
      });
      await onRefresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-section">
      <div className="section-head">
        <span className="section-title">Boards</span>
        <button
          type="button"
          className="ghost-btn"
          onClick={reset}
          disabled={busy}
        >
          Reset to HOME defaults
        </button>
      </div>
      {overview.boards.map((b) => (
        <div key={b.name} className="list-row">
          <span className="row-main">
            {b.name}
            {b.is_home === 1 ? " \u2605" : ""}
            {b.name === overview.board && (
              <span className="row-tag">active</span>
            )}
          </span>
          <span className="row-meta">
            {b.labels || `membership group ${b.membership}`}
          </span>
        </div>
      ))}
      {overview.deployment_tx && (
        <div className="row-meta">deployment_tx {overview.deployment_tx.slice(0, 8)}</div>
      )}
    </div>
  );
}

function MemorySection({
  facts,
  onRefresh
}: {
  facts: FactEntry[];
  onRefresh: () => void | Promise<void>;
}) {
  const token = getToken();

  const remove = async (id: number) => {
    await fetch(`${AGENT_BASE}/facts/${id}`, {
      method: "DELETE",
      headers: authHeaders(token)
    });
    await onRefresh();
  };

  return (
    <div className="settings-section">
      <div className="section-head">
        <span className="section-title">Memory</span>
        <span className="row-meta">
          facts live in the agent's private SQLite, tagged to a board
        </span>
      </div>
      {facts.length === 0 ? (
        <div className="row-meta">nothing stored yet</div>
      ) : (
        facts.map((f) => (
          <div key={f.id} className="list-row">
            <span className="row-main">{f.fact}</span>
            <span className="row-meta">
              {f.board}/{f.category}
            </span>
            <button
              type="button"
              className="row-del"
              onClick={() => remove(f.id)}
              aria-label={`Forget fact ${f.id}`}
            >
              ×
            </button>
          </div>
        ))
      )}
    </div>
  );
}

function AuditSection({ audit }: { audit: AuditEntry[] }) {
  return (
    <div className="settings-section">
      <div className="section-head">
        <span className="section-title">Audit log</span>
      </div>
      {audit.length === 0 ? (
        <div className="row-meta">no actions recorded yet</div>
      ) : (
        audit.map((a, i) => (
          <div key={i} className="list-row">
            <span className="row-main">{a.action}</span>
            <span className="row-meta">
              {a.detail ? `${a.detail} · ` : ""}
              {a.created_at} UTC
            </span>
          </div>
        ))
      )}
    </div>
  );
}

interface GoogleStatus {
  configured: boolean;
  connected: boolean;
  email: string | null;
}

function GoogleSection() {
  const [status, setStatus] = useState<GoogleStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const token = getToken();

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${AGENT_BASE}/google/status`, {
        headers: authHeaders(token)
      });
      if (res.ok) setStatus((await res.json()) as GoogleStatus);
    } catch {
      // status stays null — section shows "unavailable"
    }
  }, [token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const connect = () => {
    window.location.href = `${AGENT_BASE}/google/start${
      token ? `?token=${encodeURIComponent(token)}` : ""
    }`;
  };

  const disconnect = async () => {
    if (!window.confirm("Disconnect Google? JRVS loses email and calendar access.")) {
      return;
    }
    setBusy(true);
    try {
      await fetch(`${AGENT_BASE}/google/disconnect`, {
        method: "POST",
        headers: authHeaders(token)
      });
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  let body: React.ReactNode;
  if (!status) {
    body = <span className="google-state">status unavailable</span>;
  } else if (!status.configured) {
    body = (
      <span className="google-state">
        not configured — set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET on the
        worker
      </span>
    );
  } else if (status.connected) {
    body = (
      <>
        <span className="google-state on">{status.email}</span>
        <button
          type="button"
          className="ghost-btn"
          onClick={disconnect}
          disabled={busy}
        >
          Disconnect
        </button>
      </>
    );
  } else {
    body = (
      <>
        <span className="google-state">not connected</span>
        <button type="button" className="ghost-btn" onClick={connect}>
          Connect Google
        </button>
      </>
    );
  }

  return (
    <div className="settings google-row">
      <label>Gmail &amp; Calendar</label>
      {body}
    </div>
  );
}

const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(<App />);
}
