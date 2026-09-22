// Live insights dashboard.
//
// Nudges are added as they arrive and removed when the engine stops listing
// them as active, so expiry and the concurrent cap are visible in the interface
// rather than only in a log. That is deliberate: suppression is the part of
// this system worth watching, and a panel that only ever grows would hide it.

import { VoiceCapture } from "/static/voice.js";

const el = (id) => document.getElementById(id);
const ui = {
  body: document.body,
  grid: el("scenarioGrid"), board: el("board"), stream: el("stream"),
  nudges: el("nudges"), lamp: el("lamp"), clock: el("clock"),
  reset: el("reset"), brand: el("brandContext"),
  cSeen: el("cSeen"), cShown: el("cShown"), cSupp: el("cSupp"),
  cLat: el("cLat"), cExp: el("cExp"), suppBar: el("suppBar"),
  micCard: el("micCard"), micBoard: el("micBoard"), micStream: el("micStream"),
  micNudges: el("micNudges"), micSupp: el("micSupp"),
  liveMic: el("liveMic"), liveRing: el("liveRing"), liveMeter: el("liveMeter"),
  liveMicStatus: el("liveMicStatus"), liveText: el("liveText"), liveSend: el("liveSend"),
  roleAgent: el("roleAgent"), roleCaller: el("roleCaller"),
};

let ws = null;
let shown = 0;
let current = null;

function lamp(text, state) { ui.lamp.textContent = text; ui.lamp.dataset.state = state || "idle"; }

// ── scenario picker ──────────────────────────────────────────────────

function renderScenarios(list) {
  ui.grid.innerHTML = "";
  list.forEach((s) => {
    const silent = s.expect.length === 0;
    const card = document.createElement("button");
    card.type = "button";
    card.className = "scenario-card";
    card.dataset.silent = String(silent);
    card.innerHTML = `
      <span class="scenario-tag"><span class="dot"></span>${silent ? "must stay silent" : s.expect.join(" · ").replace(/_/g, " ")}</span>
      <h2></h2>
      <p class="why"></p>
      <span class="dur"></span>
      <span class="scenario-go">Run call →</span>`;
    card.querySelector("h2").textContent = s.name;
    card.querySelector(".why").textContent = s.why;
    card.querySelector(".dur").textContent = `${s.duration_s}s · real time`;
    card.addEventListener("click", () => run(s));
    ui.grid.appendChild(card);
  });
}

// ── run ──────────────────────────────────────────────────────────────

function run(scenario) {
  ui.micBoard.hidden = true;
  current = scenario;
  shown = 0;
  ui.body.dataset.view = "board";
  ui.board.hidden = false;
  ui.brand.textContent = scenario.name;
  ui.stream.innerHTML = `<p class="stream-empty">Connecting…</p>`;
  ui.nudges.innerHTML = `<p class="empty-note">Listening. Nudges appear here while the call runs.</p>`;
  ui.cSeen.textContent = "0"; ui.cShown.textContent = "0";
  ui.cSupp.textContent = "—"; ui.cLat.textContent = "—";
  ui.cExp.textContent = scenario.expect.length ? scenario.expect.join(", ").replace(/_/g, " ") : "nothing";
  ui.suppBar.style.width = "0%";
  ui.clock.hidden = false;
  ui.clock.textContent = "0.0s";
  ui.reset.hidden = true;
  lamp("connecting", "think");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onopen = () => ws.send(JSON.stringify({ scenario: scenario.id }));

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);

    if (m.type === "stream_start") { ui.stream.innerHTML = ""; lamp("live", "live"); return; }

    if (m.type === "transcript") { addLine(m.at_s, m.speaker, m.text); return; }

    if (m.type === "tick") {
      ui.clock.textContent = `${m.at_s.toFixed(1)}s`;
      syncNudges(m.active);
      if (m.suppression) paintSuppression(m.suppression);
      return;
    }

    if (m.type === "nudge") {
      ui.cLat.textContent = `${m.nudge.latency_ms} ms`;
      return;
    }

    if (m.type === "stream_end") { paintSuppression(m.suppression); return; }

    if (m.type === "done") {
      lamp("complete", "idle");
      ui.reset.hidden = false;
      const pass = m.expected.length === 0
        ? m.fired.length === 0
        : m.expected.every((e) => m.fired.includes(e));
      const v = document.createElement("div");
      v.className = "verdict";
      v.dataset.pass = String(pass);
      v.textContent = pass
        ? (m.expected.length === 0
            ? "Pass — stayed silent, as required on a call with nothing to act on."
            : `Pass — detected ${m.expected.join(", ").replace(/_/g, " ")}.`)
        : `Fail — expected ${m.expected.join(", ") || "nothing"}, fired ${m.fired.join(", ") || "nothing"}.`;
      ui.nudges.parentElement.appendChild(v);
    }
  };

  ws.onclose = () => { if (ui.lamp.dataset.state === "live") lamp("disconnected", "idle"); };
}

function paintSuppression(s) {
  ui.cSeen.textContent = String(s.signals_seen);
  ui.cShown.textContent = String(s.nudges_emitted);
  ui.cSupp.textContent = s.signals_seen
    ? `${Math.round(s.suppression_rate * 100)}%` : "—";
  // The bar shows the share actually SHOWN, so a nearly empty bar reads as
  // heavy suppression at a glance.
  const shownPct = s.signals_seen ? (s.nudges_emitted / s.signals_seen) * 100 : 0;
  ui.suppBar.style.width = `${s.signals_seen ? Math.max(shownPct, 2) : 0}%`;
}

function addLine(at, speaker, text) {
  const empty = ui.stream.querySelector(".stream-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `line ${speaker === "agent" ? "agent" : "caller"}`;
  div.innerHTML = `<span class="t"></span><span class="s"></span><span class="x"></span>`;
  div.querySelector(".t").textContent = `${at.toFixed(1)}s`;
  div.querySelector(".s").textContent = speaker.toUpperCase();
  div.querySelector(".x").textContent = text;
  ui.stream.appendChild(div);
  ui.stream.scrollTop = ui.stream.scrollHeight;
}

function syncNudges(active) {
  const ids = new Set(active.map((n) => n.id));
  [...ui.nudges.querySelectorAll(".nudge")].forEach((node) => {
    if (!ids.has(node.dataset.id)) {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 420);
    }
  });

  active.forEach((n) => {
    let node = ui.nudges.querySelector(`[data-id="${n.id}"]`);
    if (!node) {
      const note = ui.nudges.querySelector(".empty-note");
      if (note) note.remove();
      node = document.createElement("div");
      node.className = "nudge";
      node.dataset.id = n.id;
      node.dataset.type = n.type;
      node.innerHTML = `
        <div class="nudge-top"><span class="nudge-type"></span><span class="nudge-prio"></span></div>
        <div class="nudge-text"></div>
        <div class="nudge-meta">
          <span class="m-conf"></span><span class="m-det"></span>
          <span class="m-ev"></span><span class="m-age"></span>
        </div>`;
      node.querySelector(".nudge-type").textContent = n.type.replace(/_/g, " ");
      node.querySelector(".nudge-prio").textContent = `p${n.priority}`;
      node.querySelector(".nudge-text").textContent = n.text;
      node.querySelector(".m-conf").innerHTML = `conf <em>${n.confidence}</em>`;
      node.querySelector(".m-det").innerHTML = `via <em>${n.detector}</em>`;
      node.querySelector(".m-ev").innerHTML = n.evidence ? `“<em>${n.evidence}</em>”` : "";
      ui.nudges.appendChild(node);
    }
    node.querySelector(".m-age").textContent = `${n.age_s}s old`;
  });
}

// ── live microphone mode ─────────────────────────────────────────────
//
// The speaker is chosen rather than inferred, because the detectors are
// asymmetric: a missing disclosure is only a finding when the agent failed to
// give it, and a spouse mentioned is only an opening when the customer said it.

const RING = 239;
let micWs = null;
let capture = null;
let role = "agent";
const micBars = Array.from(ui.liveMeter.children);

function micStatus(text, tone) {
  ui.liveMicStatus.textContent = text;
  if (tone) ui.liveMicStatus.dataset.tone = tone;
  else ui.liveMicStatus.removeAttribute("data-tone");
}

function paintLevel(level, threshold) {
  const ceiling = Math.max(threshold * 6, 0.09);
  const pct = Math.min(level / ceiling, 1);
  ui.liveRing.style.strokeDashoffset = String(RING * (1 - pct));
  micBars.forEach((bar, i) => {
    const weight = 1 - Math.abs(i - (micBars.length - 1) / 2) / micBars.length;
    bar.style.height = `${Math.min(3 + pct * 19 * (0.45 + weight), 22)}px`;
  });
  ui.liveMeter.classList.toggle("active", pct > 0.06);
}

function setRole(next) {
  role = next;
  ui.roleAgent.setAttribute("aria-pressed", String(next === "agent"));
  ui.roleCaller.setAttribute("aria-pressed", String(next === "caller"));
}

function micLine(speaker, text, at) {
  const empty = ui.micStream.querySelector(".stream-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `line ${speaker === "agent" ? "agent" : "caller"}`;
  div.innerHTML = `<span class="t"></span><span class="s"></span><span class="x"></span>`;
  div.querySelector(".t").textContent = `${at.toFixed(1)}s`;
  div.querySelector(".s").textContent = speaker === "agent" ? "AGENT" : "CUSTOMER";
  div.querySelector(".x").textContent = text;
  ui.micStream.appendChild(div);
  ui.micStream.scrollTop = ui.micStream.scrollHeight;
}

function syncMicNudges(active, suppression) {
  const ids = new Set(active.map((n) => n.id));
  [...ui.micNudges.querySelectorAll(".nudge")].forEach((node) => {
    if (!ids.has(node.dataset.id)) {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 420);
    }
  });
  active.forEach((n) => {
    let node = ui.micNudges.querySelector(`[data-id="${n.id}"]`);
    if (!node) {
      const note = ui.micNudges.querySelector(".empty-note");
      if (note) note.remove();
      node = document.createElement("div");
      node.className = "nudge";
      node.dataset.id = n.id;
      node.dataset.type = n.type;
      node.innerHTML = `
        <div class="nudge-top"><span class="n-type"></span><span class="nudge-prio"></span></div>
        <div class="nudge-text"></div>
        <div class="nudge-meta"><span class="n-conf"></span><span class="n-det"></span><span class="n-ev"></span></div>`;
      node.querySelector(".n-type").textContent = n.type.replace(/_/g, " ");
      node.querySelector(".nudge-prio").textContent = `p${n.priority}`;
      node.querySelector(".nudge-text").textContent = n.text;
      node.querySelector(".n-conf").innerHTML = `conf <em>${n.confidence}</em>`;
      node.querySelector(".n-det").innerHTML = `via <em>${n.detector}</em>`;
      if (n.evidence) node.querySelector(".n-ev").innerHTML = `“<em>${n.evidence}</em>”`;
      ui.micNudges.appendChild(node);
    }
  });
  if (suppression && suppression.signals_seen) {
    ui.micSupp.hidden = false;
    ui.micSupp.textContent = `${suppression.nudges_emitted} shown · ${suppression.signals_seen} seen`;
  }
}

function openMicMode() {
  ui.body.dataset.view = "board";
  ui.micBoard.hidden = false;
  ui.board.hidden = true;
  ui.brand.textContent = "live microphone";
  ui.reset.hidden = false;
  ui.micStream.innerHTML = `<p class="stream-empty">Pick who you are speaking as, then click the mic and talk.</p>`;
  ui.micNudges.innerHTML = `<p class="empty-note">Nudges appear here the moment a detector fires.</p>`;
  ui.micSupp.hidden = true;
  lamp("connecting", "think");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  micWs = new WebSocket(`${proto}://${location.host}/ws/live_mic`);
  micWs.onopen = () => { lamp("listening", "live"); micStatus("Click the mic and talk — it sends when you stop"); };
  micWs.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "transcript") { micLine(m.speaker, m.text, m.at_s); return; }
    if (m.type === "nudge_sync") { syncMicNudges(m.active, m.suppression); lamp("listening", "live"); return; }
    if (m.type === "no_speech") { micStatus(m.message, "warn"); lamp("listening", "live"); return; }
  };
  micWs.onclose = () => lamp("disconnected", "idle");
}

function sendMicText() {
  const text = ui.liveText.value.trim();
  if (!text || !micWs || micWs.readyState !== 1) return;
  micWs.send(JSON.stringify({ type: "text", text, speaker: role }));
  ui.liveText.value = "";
  ui.liveSend.disabled = true;
  lamp("thinking", "think");
}

function startMic() {
  if (!micWs || micWs.readyState !== 1 || (capture && capture.active)) return;
  capture = new VoiceCapture({
    onLevel: paintLevel,
    onState: (state) => {
      if (state === "calibrating") { micStatus("Listening…", "live"); ui.liveMic.dataset.state = "recording"; lamp("recording", "rec"); }
      if (state === "waiting") micStatus("Go ahead", "live");
      if (state === "speaking") micStatus("Hearing you…", "live");
    },
    onResult: ({ blob, spoke }) => {
      ui.liveMic.dataset.state = "";
      paintLevel(0, 1);
      if (!spoke) { micStatus("Didn't catch anything — try again, or type it", "warn"); lamp("listening", "live"); return; }
      micStatus("Sent", "ok");
      ui.liveMic.dataset.state = "sending";
      lamp("transcribing", "think");
      blob.arrayBuffer().then((buf) => {
        let bin = "";
        new Uint8Array(buf).forEach((b) => (bin += String.fromCharCode(b)));
        micWs.send(JSON.stringify({ type: "audio", data: btoa(bin), speaker: role }));
        setTimeout(() => { ui.liveMic.dataset.state = ""; }, 400);
      });
    },
    onError: (err) => {
      ui.liveMic.dataset.state = "";
      micStatus(`Microphone unavailable (${err.name}) — type instead`, "warn");
      lamp("listening", "live");
    },
  });
  capture.start();
}

ui.micCard.addEventListener("click", openMicMode);
ui.roleAgent.addEventListener("click", () => setRole("agent"));
ui.roleCaller.addEventListener("click", () => setRole("caller"));
ui.liveMic.addEventListener("click", () => {
  if (capture && capture.active) capture.stop("manual");
  else startMic();
});
ui.liveText.addEventListener("input", () => { ui.liveSend.disabled = !ui.liveText.value.trim(); });
ui.liveText.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); sendMicText(); } });
ui.liveSend.addEventListener("click", sendMicText);

ui.reset.addEventListener("click", () => {
  ui.body.dataset.view = "picker";
  ui.board.hidden = true;
  ui.micBoard.hidden = true;
  ui.clock.hidden = true;
  ui.reset.hidden = true;
  ui.brand.textContent = "real-time nudges";
  if (micWs) { micWs.close(); micWs = null; }
  if (capture && capture.active) capture.stop("manual");
  const v = document.querySelector(".verdict");
  if (v) v.remove();
  lamp("idle", "idle");
});

fetch("/api/live_scenarios")
  .then((r) => r.json())
  .then((d) => renderScenarios(d.scenarios))
  .catch(() => {
    ui.grid.innerHTML = `<p class="empty-note">Could not load scenarios. Run <code>python -m scripts.make_stereo_calls</code> first.</p>`;
  });
