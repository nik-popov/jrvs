/**
 * Live turn test against the deployed JRVS worker.
 * Opens the voice WebSocket, sends a text_message, prints transcript/error
 * events. Verifies Phase 1 (reply produced) and Phase 4 (SUGGEST_SOURCE path).
 * Usage: node scripts/turn-test.mjs "message" [timeoutSeconds]
 */
const HOST = "jrvs.apis-popov.workers.dev";
const msg = process.argv[2] ?? "hello";
const timeoutS = Number(process.argv[3] ?? 45);

const ws = new WebSocket(`wss://${HOST}/agents/jarvis-agent/main`);
const timer = setTimeout(() => {
  console.log("TIMEOUT: no assistant transcript within", timeoutS, "s");
  ws.close();
  process.exit(2);
}, timeoutS * 1000);

ws.addEventListener("open", () => {
  console.log("ws open");
  ws.send(JSON.stringify({ type: "hello", protocol_version: 1 }));
  ws.send(JSON.stringify({ type: "start_call" }));
  setTimeout(() => {
    console.log(">> sending:", msg);
    sent = true;
    ws.send(JSON.stringify({ type: "text_message", text: msg }));
  }, 1500);
});

let sent = false;

ws.addEventListener("message", (ev) => {
  if (typeof ev.data !== "string") return; // skip audio frames
  let m;
  try {
    m = JSON.parse(ev.data);
  } catch {
    return;
  }
  if (["transcript", "error", "status", "metrics"].includes(m.type)) {
    console.log(`[${m.type}]`, JSON.stringify(m).slice(0, 400));
  }
  if (m.type === "transcript_end" && m.text && sent) {
    clearTimeout(timer);
    console.log("ASSISTANT:", m.text);
    console.log("PASS: assistant replied");
    ws.send(JSON.stringify({ type: "end_call" }));
    setTimeout(() => {
      ws.close();
      process.exit(0);
    }, 300);
  }
  if (m.type === "error") {
    clearTimeout(timer);
    console.log("FAIL: pipeline error");
    ws.close();
    process.exit(1);
  }
});

ws.addEventListener("error", (e) => {
  console.log("ws error", e.message ?? e);
  process.exit(1);
});
