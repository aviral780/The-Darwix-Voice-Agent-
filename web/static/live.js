// Live insights dashboard.
//
// Nudges are added as they arrive and removed when the engine stops listing
// them as active, so expiry and the concurrent cap are visible in the interface
// rather than only in a log. That is deliberate: suppression is the part of
// this system worth watching, and a panel that only ever grows would hide it.

const el = (id) => document.getElementById(id);
const ui = {
  body: document.body,
  grid: el("scenarioGrid"), board: el("board"), stream: el("stream"),
  nudges: el("nudges"), lamp: el("lamp"), clock: el("clock"),
  reset: el("reset"), brand: el("brandContext"),
  cSeen: el("cSeen"), cShown: el("cShown"), cSupp: el("cSupp"),
  cLat: el("cLat"), cExp: el("cExp"), suppBar: el("suppBar"),
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

ui.reset.addEventListener("click", () => {
  ui.body.dataset.view = "picker";
  ui.board.hidden = true;
  ui.clock.hidden = true;
  ui.reset.hidden = true;
  ui.brand.textContent = "real-time nudges";
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
