// Call console.
//
// Two views in one document. The market picker is the landing state because
// three markets on one engine is the most interesting property of the system,
// and it was previously hidden in a corner dropdown.

import { VoiceCapture } from "/static/voice.js";

const el = (id) => document.getElementById(id);
const ui = {
  body: document.body,
  picker: el("picker"), grid: el("marketGrid"), console: el("console"),
  stream: el("stream"), mic: el("mic"), ring: el("ringLevel"), meter: el("meter"),
  micStatus: el("micStatus"), text: el("textInput"),
  lamp: el("lamp"), clock: el("clock"), end: el("endCall"), back: el("backBtn"), brand: el("brandContext"),
  slots: el("slots"), sources: el("sources"),
  nudges: el("nudges"), suppChip: el("suppChip"),
  send: el("send"),
  sheet: el("sheet"), scrim: el("sheetScrim"), statRow: el("statRow"),
  sheetBody: el("sheetBody"), closeSheet: el("closeSheet"),
  player: el("player"),
};

const RING_CIRCUMFERENCE = 239;
// language_name is phrased for the model prompt ("Taglish (Filipino with natural
// English code-switching)"), which is right there and far too long on a card.
const SHORT_LANGUAGE = {
  "English": "English",
  "Taglish (Filipino with natural English code-switching)": "Taglish",
  "Bahasa Indonesia": "Bahasa",
};
const shortLang = (name) => SHORT_LANGUAGE[name] || name.split(/[(,]/)[0].trim();

const SLOT_LABELS = {
  name: "Name", age_band: "Age band", dependents: "Dependents",
  existing_cover: "Existing cover", interest: "Interest", callback: "Callback",
  awareness: "Aware of due date", payment_status: "Payment status",
  difficulty: "Obstacle", payment_plan: "Plan to pay", channel: "Channel",
};

let ws = null;
let capture = null;
let live = false;
let markets = [];
let currentMarket = null;
let slotOrder = [];
let tickTimer = null;
let startedAt = 0;

// ── status ───────────────────────────────────────────────────────────

function lamp(text, state) {
  ui.lamp.textContent = text;
  ui.lamp.dataset.state = state || "idle";
}

function micStatus(text, tone) {
  ui.micStatus.textContent = text;
  if (tone) ui.micStatus.dataset.tone = tone;
  else ui.micStatus.removeAttribute("data-tone");
}

function startClock() {
  startedAt = Date.now();
  ui.clock.hidden = false;
  tickTimer = setInterval(() => {
    const s = Math.floor((Date.now() - startedAt) / 1000);
    ui.clock.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}

function stopClock() { if (tickTimer) clearInterval(tickTimer); tickTimer = null; }

// ── market picker ────────────────────────────────────────────────────

function renderMarkets(list) {
  ui.grid.innerHTML = "";
  list.forEach((m) => {
    const card = document.createElement("button");
    card.className = "market-card";
    card.type = "button";
    const region = m.key.split("_").slice(-1)[0];
    card.innerHTML = `
      <span class="market-flagline"><span class="dot"></span>${region} · ${shortLang(m.language)}</span>
      <h2></h2>
      <p class="sector"></p>
      <dl class="market-meta">
        <div><dt>Voice</dt><dd class="v-voice"></dd></div>
        <div><dt>Flow</dt><dd class="v-flow"></dd></div>
      </dl>
      <span class="market-go">Start call →</span>`;
    card.querySelector("h2").textContent = m.display_name.replace(/\s*\([^)]*\)\s*$/, "");
    card.querySelector(".sector").textContent = m.sector;
    card.querySelector(".v-voice").textContent = m.voice;
    card.querySelector(".v-flow").textContent =
      m.sector.includes("multifinance") ? "instalment reminder" : "lead qualification";
    card.addEventListener("click", () => beginCall(m));
    ui.grid.appendChild(card);
  });
}

function slotsForMarket(market) {
  return market.sector.includes("multifinance")
    ? ["name", "awareness", "payment_status", "difficulty", "payment_plan", "channel"]
    : ["name", "age_band", "dependents", "existing_cover", "interest", "callback"];
}

function renderSlots(filled) {
  filled = filled || {};
  const firstEmpty = slotOrder.find((k) => !filled[k]);
  ui.slots.innerHTML = "";
  slotOrder.forEach((key) => {
    const li = document.createElement("li");
    li.className = "slot";
    li.dataset.filled = String(Boolean(filled[key]));
    li.dataset.pending = String(key === firstEmpty && live);
    li.innerHTML = `<span class="slot-label"></span><span class="slot-value"></span>`;
    li.querySelector(".slot-label").textContent = SLOT_LABELS[key] || key;
    li.querySelector(".slot-value").textContent = filled[key] || "—";
    ui.slots.appendChild(li);
  });
}

// ── live coaching ────────────────────────────────────────────────────
//
// The same nudge engine that powers the Live insights page, running against
// this conversation. Speaker attribution is exact here - the agent's words are
// generated locally and the caller's come back already labelled - so it needs
// none of the channel splitting the recorded pipeline does.

function syncNudges(active, suppression) {
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
        <div class="nudge-top"><span class="n-type"></span><span class="nudge-prio"></span></div>
        <div class="nudge-text"></div>
        <div class="nudge-meta"><span class="n-conf"></span><span class="n-det"></span><span class="n-ev"></span></div>`;
      node.querySelector(".n-type").textContent = n.type.replace(/_/g, " ");
      node.querySelector(".nudge-prio").textContent = `p${n.priority}`;
      node.querySelector(".nudge-text").textContent = n.text;
      node.querySelector(".n-conf").innerHTML = `conf <em>${n.confidence}</em>`;
      node.querySelector(".n-det").innerHTML = `via <em>${n.detector}</em>`;
      if (n.evidence) node.querySelector(".n-ev").innerHTML = `“<em>${n.evidence}</em>”`;
      ui.nudges.appendChild(node);
    }
  });

  if (suppression && suppression.signals_seen) {
    ui.suppChip.hidden = false;
    ui.suppChip.textContent =
      `${suppression.nudges_emitted} shown · ${suppression.signals_seen} seen`;
  }
}

// ── transcript ───────────────────────────────────────────────────────

function addTurn(who, text, badges) {
  const empty = ui.stream.querySelector(".stream-empty");
  if (empty) empty.remove();
  const wrap = document.createElement("div");
  wrap.className = `turn ${who}`;
  wrap.innerHTML = `<div class="turn-who"></div><div class="bubble"></div>`;
  wrap.querySelector(".turn-who").textContent = who === "agent" ? "Agent" : "You";
  wrap.querySelector(".bubble").textContent = text;
  if (badges && badges.length) {
    const row = document.createElement("div");
    row.className = "badges";
    badges.forEach((b) => {
      const s = document.createElement("span");
      s.className = `badge ${b.kind}`;
      s.textContent = b.label;
      row.appendChild(s);
    });
    wrap.appendChild(row);
  }
  ui.stream.appendChild(wrap);
  ui.stream.scrollTop = ui.stream.scrollHeight;
}

function addSources(ids) {
  if (!ids || !ids.length) return;
  const note = ui.sources.querySelector(".empty-note");
  if (note) note.remove();
  ids.forEach((id) => {
    if (ui.sources.querySelector(`[data-id="${id}"]`)) return;
    const chip = document.createElement("div");
    chip.className = "source-chip";
    chip.dataset.id = id;
    chip.textContent = id;
    ui.sources.appendChild(chip);
  });
}

// ── call lifecycle ───────────────────────────────────────────────────

function beginCall(market) {
  currentMarket = market;
  slotOrder = slotsForMarket(market);
  ui.brand.textContent = market.display_name;
  ui.body.dataset.view = "call";
  ui.console.hidden = false;
  ui.back.hidden = false;
  ui.stream.innerHTML = `<p class="stream-empty">Connecting…</p>`;
  ui.sources.innerHTML = `<p class="empty-note">Every factual answer is retrieved and cited. Sources appear here as the agent uses them.</p>`;
  ui.nudges.innerHTML = `<p class="empty-note">Signals from this call appear here as it runs — the same engine as Live insights, watching in real time.</p>`;
  ui.suppChip.hidden = true;
  renderSlots({});
  connect(market.key);
}

function connect(marketKey) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", market: marketKey }));

  ws.onmessage = (event) => {
    const m = JSON.parse(event.data);

    if (m.type === "call_started") {
      live = true;
      ui.stream.innerHTML = "";
      ui.mic.disabled = false;
      ui.text.disabled = false;
      ui.send.disabled = true;
      ui.end.hidden = false;
      lamp("live", "live");
      micStatus("Click the mic to speak — it sends when you stop");
      startClock();
      return;
    }

    if (m.type === "caller_turn") { addTurn("caller", m.text); lamp("thinking", "think"); return; }

    if (m.type === "agent_turn") {
      const badges = [];
      if (m.citations && m.citations.length) {
        badges.push({ kind: "verified", label: `grounded · ${m.citations.join(", ")}` });
      }
      if (m.refused) badges.push({ kind: "refused", label: `refused at ${m.refusal_gate} gate` });
      if (m.state === "escalated") badges.push({ kind: "escalate", label: "escalated to a human" });
      addTurn("agent", m.text, badges);
      addSources(m.citations);
      renderSlots(m.slots);
      play(m.audio);
      if (live) lamp("live", "live");
      return;
    }

    if (m.type === "nudge_sync") { syncNudges(m.active, m.suppression); return; }
    if (m.type === "nudge") { return; }   // the sync that follows renders it

    if (m.type === "turn_error") {
      addTurn("agent", m.message, [{ kind: "refused", label: "temporary error" }]);
      micStatus("Something glitched — try that again", "warn");
      lamp("live", "live");
      return;
    }

    if (m.type === "no_speech") { micStatus(m.message, "warn"); lamp("live", "live"); return; }

    if (m.type === "call_ended") {
      live = false;
      stopClock();
      ui.mic.disabled = true;
      ui.text.disabled = true;
      ui.send.disabled = true;
      ui.end.hidden = true;
      lamp("ended", "idle");
      showSheet(m);
      return;
    }

    if (m.type === "error") addTurn("agent", m.message);
  };

  ws.onclose = () => {
    live = false;
    stopClock();
    ui.mic.disabled = true;
    ui.text.disabled = true;
    ui.send.disabled = true;
    ui.end.hidden = true;
    if (ui.lamp.dataset.state !== "idle") lamp("disconnected", "idle");
  };
}

function play(b64) {
  if (!b64) return;
  ui.player.src = "data:audio/mpeg;base64," + b64;
  ui.player.play().catch(() => { /* autoplay blocked; transcript still shows */ });
}

// ── summary sheet ────────────────────────────────────────────────────

function stat(k, v, tone, i) {
  const d = document.createElement("div");
  d.className = "stat";
  if (tone) d.dataset.tone = tone;
  d.style.animationDelay = `${0.05 + i * 0.05}s`;
  d.innerHTML = `<div class="stat-k"></div><div class="stat-v"></div>`;
  d.querySelector(".stat-k").textContent = k;
  d.querySelector(".stat-v").textContent = v;
  return d;
}

function showSheet(msg) {
  const s = msg.summary || {};
  const a = msg.actions || {};
  ui.statRow.innerHTML = "";
  const stats = [
    ["Outcome", s.final_state, s.escalated ? "signal" : null],
    ["Slots", `${s.slots_filled}/${s.slots_total}`, null],
    ["Grounded", s.grounded_answers, s.grounded_answers ? "verified" : null],
    ["Refusals", s.refusals, s.refusals ? "caution" : null],
  ];
  if (a.lead) stats.push(["Lead", `${a.lead.band} · ${a.lead.score}`, "verified"]);
  stats.forEach(([k, v, tone], i) => ui.statRow.appendChild(stat(k, v, tone, i)));

  let html = "";
  if (s.latency) {
    html += "<h3>Latency, this call</h3>";
    Object.entries(s.latency).forEach(([k, v]) => {
      html += `<div class="kv"><span>${k}</span><strong>p50 ${v.p50_ms} ms · p95 ${v.p95_ms} ms · n=${v.count}</strong></div>`;
    });
  }
  if (a.escalation) {
    html += `<h3>Handover</h3><div class="kv"><span>Escalation record written</span><strong>${a.escalation.delivered ? "webhook delivered" : "stored locally"}</strong></div>`;
  }
  html += "<h3>Transcript</h3><div class='transcript-lines' id='tlines'></div>";
  ui.sheetBody.innerHTML = html;

  const lines = (msg.transcript || "").split("\n").filter(Boolean);
  const host = document.getElementById("tlines");
  lines.forEach((line, i) => {
    const isAgent = line.startsWith("AGENT");
    const div = document.createElement("div");
    div.className = `tline ${isAgent ? "agent" : "caller"}`;
    div.style.animationDelay = `${0.25 + i * 0.04}s`;
    div.innerHTML = `<span class="who"></span><span class="what"></span>`;
    div.querySelector(".who").textContent = isAgent ? "AGENT" : "CALLER";
    div.querySelector(".what").textContent = line.replace(/^(AGENT|CALLER)\s*:\s*/, "");
    host.appendChild(div);
  });

  ui.scrim.hidden = false;
  ui.sheet.hidden = false;
}

function hideSheet() {
  ui.scrim.hidden = true;
  ui.sheet.hidden = true;
  ui.body.dataset.view = "picker";
  ui.console.hidden = true;
  ui.back.hidden = true;
  ui.brand.textContent = "lead qualification";
  ui.clock.hidden = true;
  lamp("idle", "idle");
}

function goBack() {
  // Closing the socket, not sending "end", is what the server actually keys
  // off: its WebSocketDisconnect handler persists the transcript and creates
  // the lead if any turns happened, exactly as a dropped tab would. "Back" is
  // therefore a lighter exit than "End call" - no summary sheet - not a path
  // that skips saving what the call already produced.
  if (ws) { try { ws.close(); } catch (e) { /* already closing */ } ws = null; }
  if (capture && capture.active) capture.stop("manual");
  live = false;
  stopClock();
  ui.scrim.hidden = true;
  ui.sheet.hidden = true;
  ui.body.dataset.view = "picker";
  ui.console.hidden = true;
  ui.back.hidden = true;
  ui.brand.textContent = "lead qualification";
  ui.clock.hidden = true;
  lamp("idle", "idle");
}

// ── microphone ───────────────────────────────────────────────────────

const bars = Array.from(ui.meter.children);

function paintLevel(level, threshold) {
  // Ring fills against a ceiling a little above the speech threshold so normal
  // speech uses most of the arc rather than pinning it.
  const ceiling = Math.max(threshold * 6, 0.09);
  const pct = Math.min(level / ceiling, 1);
  ui.ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - pct));

  bars.forEach((bar, i) => {
    // Centre bars react most, giving the meter a shape rather than a flat block.
    const weight = 1 - Math.abs(i - (bars.length - 1) / 2) / bars.length;
    const h = 3 + pct * 19 * (0.45 + weight);
    bar.style.height = `${Math.min(h, 22)}px`;
  });
  ui.meter.classList.toggle("active", pct > 0.06);
}

function startMic() {
  if (!live || (capture && capture.active)) return;
  capture = new VoiceCapture({
    onLevel: paintLevel,
    onState: (state) => {
      if (state === "calibrating") { micStatus("Listening…", "live"); ui.mic.dataset.state = "recording"; lamp("recording", "rec"); }
      if (state === "waiting") micStatus("Go ahead — I'm listening", "live");
      if (state === "speaking") micStatus("Hearing you…", "live");
    },
    onResult: ({ blob, spoke, reason }) => {
      ui.mic.dataset.state = "";
      paintLevel(0, 1);
      if (!spoke) {
        micStatus("Didn't catch anything — try again, or type instead", "warn");
        lamp("live", "live");
        return;
      }
      micStatus(reason === "silence" ? "Sent" : "Sent", "ok");
      ui.mic.dataset.state = "sending";
      lamp("transcribing", "think");
      blob.arrayBuffer().then((buf) => {
        let bin = "";
        new Uint8Array(buf).forEach((b) => (bin += String.fromCharCode(b)));
        ws.send(JSON.stringify({ type: "audio", data: btoa(bin) }));
        setTimeout(() => { ui.mic.dataset.state = ""; }, 400);
      });
    },
    onError: (err) => {
      ui.mic.dataset.state = "";
      micStatus(`Microphone unavailable (${err.name}) — type your reply instead`, "warn");
      lamp("live", "live");
    },
  });
  capture.start();
}

function stopMic() { if (capture && capture.active) capture.stop("manual"); }

ui.mic.addEventListener("click", () => {
  if (capture && capture.active) stopMic();
  else startMic();
});

// Spacebar stays available as hold-to-talk for anyone who prefers an explicit
// boundary, and as a fallback if the detector misjudges a noisy room.
let spaceHeld = false;
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== ui.text && live) {
    e.preventDefault(); spaceHeld = true; startMic();
  }
  if (e.key === "Escape" && !ui.sheet.hidden) hideSheet();
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && spaceHeld && document.activeElement !== ui.text) {
    e.preventDefault(); spaceHeld = false; stopMic();
  }
});

// ── wiring ───────────────────────────────────────────────────────────

function sendText() {
  const text = ui.text.value.trim();
  if (!text || !live) return;
  ws.send(JSON.stringify({ type: "text", text }));
  ui.text.value = "";
  ui.send.disabled = true;
  lamp("thinking", "think");
}

ui.text.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); sendText(); }
});
ui.text.addEventListener("input", () => {
  ui.send.disabled = !ui.text.value.trim() || !live;
});
ui.send.addEventListener("click", sendText);

ui.end.addEventListener("click", () => { if (ws && live) ws.send(JSON.stringify({ type: "end" })); });
ui.back.addEventListener("click", goBack);
ui.closeSheet.addEventListener("click", hideSheet);
ui.scrim.addEventListener("click", hideSheet);

fetch("/api/markets")
  .then((r) => r.json())
  .then((d) => { markets = d.markets; renderMarkets(markets); })
  .catch(() => {
    ui.grid.innerHTML = `<p class="empty-note">Could not load markets. Is the server running?</p>`;
  });
