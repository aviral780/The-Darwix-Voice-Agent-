// Knowledge base explorer.
//
// The point of this page is that the refusal design is inspectable. Anyone can
// type a question the corpus cannot answer and watch which gate stops it, with
// the component scores and the threshold visible rather than described.

const el = (id) => document.getElementById(id);
const ui = {
  switch: el("corpusSwitch"), q: el("q"), form: el("askForm"), btn: el("askBtn"),
  examples: el("examples"),
  gates: el("gates"), g1: el("gate1"), g1v: el("gate1Verdict"), g1n: el("gate1Note"),
  g2: el("gate2"), g2v: el("gate2Verdict"), g2n: el("gate2Note"),
  answer: el("answerBox"),
  resultsWrap: el("resultsWrap"), results: el("results"), table: el("resultTable"),
  wVec: el("wVec"), wBm: el("wBm"),
  funnel: el("funnel"), notes: el("pipelineNotes"), comp: el("composition"),
};

// Each set pairs questions the corpus answers with ones it must refuse. The
// refusals are the interesting half and are marked so they invite a click.
const EXAMPLES = {
  prulife_ph: [
    { q: "What happens if I miss a premium payment?" },
    { q: "How long does it take to process a claim?" },
    { q: "Can I pay my premium through GCash?" },
    { q: "What is my current policy account balance?", kind: "refuse" },
    { q: "What's the weather in Cebu tomorrow?", kind: "refuse" },
  ],
  fifgroup_id: [
    { q: "Apa itu AMITRA?" },
    { q: "Mbak, apa itu AMITRA ya?" },
    { q: "Berapa plafond pinjaman DANASTRA?" },
    { q: "Siapa presiden Indonesia sekarang?", kind: "refuse" },
  ],
};

let corpora = [];
let active = "prulife_ph";

// ── corpus switch ────────────────────────────────────────────────────

function renderSwitch() {
  ui.switch.innerHTML = "";
  corpora.forEach((c) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${c.name} · ${c.records ?? "?"} records`;
    b.setAttribute("aria-pressed", String(c.key === active));
    b.addEventListener("click", () => {
      active = c.key;
      renderSwitch();
      renderExamples();
      loadOverview();
      ui.gates.hidden = true;
      ui.resultsWrap.hidden = true;
      ui.q.value = "";
    });
    ui.switch.appendChild(b);
  });
}

function renderExamples() {
  ui.examples.innerHTML = "";
  (EXAMPLES[active] || []).forEach((ex) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "example";
    if (ex.kind) b.dataset.kind = ex.kind;
    b.textContent = ex.q;
    b.addEventListener("click", () => { ui.q.value = ex.q; search(); });
    ui.examples.appendChild(b);
  });
}

// ── search ───────────────────────────────────────────────────────────

async function search() {
  const query = ui.q.value.trim();
  if (!query) return;
  ui.btn.disabled = true;
  ui.btn.textContent = "Searching…";

  let data;
  try {
    const res = await fetch("/api/kb/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, source: active }),
    });
    data = await res.json();
  } catch (err) {
    ui.btn.disabled = false; ui.btn.textContent = "Search";
    ui.answer.textContent = "Search failed. Is the server running?";
    ui.gates.hidden = false;
    return;
  }

  ui.btn.disabled = false;
  ui.btn.textContent = "Search";
  if (data.error) { ui.answer.textContent = data.error; ui.gates.hidden = false; return; }

  paintGates(data);
  paintResults(data);
}

function paintGates(d) {
  const passed1 = d.passed_retrieval_gate;
  ui.g1.dataset.pass = String(passed1);
  ui.g1v.textContent = passed1 ? "Passed" : "Stopped here";
  const top = d.results[0];
  ui.g1n.textContent = top
    ? `Best passage scored ${top.score.toFixed(3)} against a gate of ${d.threshold}.`
    : "Nothing retrieved.";

  const reachedGate2 = d.gate !== "retrieval";
  const passed2 = !d.refused;
  ui.g2.dataset.pass = reachedGate2 ? String(passed2) : "";
  if (!reachedGate2) {
    ui.g2v.textContent = "Not reached";
    ui.g2n.textContent = "The score gate already stopped this query.";
  } else {
    ui.g2v.textContent = passed2 ? "Answered" : "Declined";
    ui.g2n.textContent = passed2
      ? "The model found the answer in the retrieved passages and cited it."
      : "The model read the passages and judged that they do not answer the question.";
  }

  ui.answer.dataset.refused = String(d.refused);
  ui.answer.innerHTML = "";
  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = d.refused ? "What the caller would hear" : "Answer";
  const body = document.createElement("div");
  if (d.refused) {
    body.textContent =
      "It declines and offers to have a person confirm, rather than answering from a passage that does not contain the answer.";
  } else {
    renderCited(body, d.answer);
  }
  ui.answer.append(lbl, body);
  ui.gates.hidden = false;
}

function paintResults(d) {
  ui.wVec.textContent = d.weights.vector;
  ui.wBm.textContent = d.weights.bm25;
  ui.results.innerHTML = "";

  // A shared scale across rows, so bar lengths are comparable to each other and
  // to the threshold marker. Scaling each row to its own max would make every
  // query look identical.
  const scaleMax = Math.max(d.threshold * 1.6, ...d.results.map((r) => r.score), 0.2);

  d.results.forEach((r, i) => {
    const pass = r.score >= d.threshold;
    const row = document.createElement("div");
    row.className = "result";
    row.dataset.cited = String(r.cited);
    row.dataset.pass = String(pass);
    row.innerHTML = `
      <div class="result-top">
        <span class="result-rank"></span>
        <span class="result-title"></span>
        <span class="result-head-meta"></span>
      </div>
      <div class="result-heading"></div>
      <div class="bar-row">
        <div class="bar-track">
          <div class="bar-fill"><span class="seg vec"></span><span class="seg bm"></span></div>
          <div class="threshold"></div>
        </div>
        <span class="bar-value"></span>
      </div>
      <details class="result-detail">
        <summary></summary>
        <div class="result-body"></div>
        <a class="src" target="_blank" rel="noopener"></a>
      </details>`;

    row.querySelector(".result-rank").textContent = `${r.rank}.`;
    row.querySelector(".result-title").textContent = r.title;
    row.querySelector(".result-heading").textContent = r.heading;

    const meta = row.querySelector(".result-head-meta");
    if (r.cited) meta.appendChild(pill("cited by the answer", "cited"));
    if (r.pii) meta.appendChild(pill("PII redacted", "pii"));
    meta.appendChild(pill(r.category));
    meta.appendChild(pill(`${r.words}w`));

    const vecPct = (r.vector_part / scaleMax) * 100;
    const bmPct = (r.bm25_part / scaleMax) * 100;
    // Animate from zero on the next frame so the transition actually runs.
    const vecSeg = row.querySelector(".seg.vec");
    const bmSeg = row.querySelector(".seg.bm");
    vecSeg.style.width = "0%"; bmSeg.style.width = "0%";
    requestAnimationFrame(() => {
      vecSeg.style.width = `${vecPct}%`;
      bmSeg.style.width = `${bmPct}%`;
    });
    vecSeg.title = `semantic ${r.vector.toFixed(3)} × ${d.weights.vector} = ${r.vector_part.toFixed(3)}`;
    bmSeg.title = `keyword ${r.bm25.toFixed(3)} × ${d.weights.bm25} = ${r.bm25_part.toFixed(3)}`;

    row.querySelector(".threshold").style.left = `${(d.threshold / scaleMax) * 100}%`;
    row.querySelector(".bar-value").textContent = r.score.toFixed(3);
    row.querySelector(".result-detail summary").textContent =
      `${r.explain} — read the passage`;
    row.querySelector(".result-body").textContent = r.content;
    const src = row.querySelector(".src");
    src.href = r.url; src.textContent = r.url;

    row.style.animation = `riseIn .4s var(--ease-out) ${0.03 * i}s both`;
    ui.results.appendChild(row);
  });

  // Table view, so identity and value never depend on the bars alone.
  ui.table.innerHTML =
    "<thead><tr><th>#</th><th>Passage</th><th>Semantic</th><th>Keyword</th><th>Fused</th><th>Gate</th></tr></thead><tbody>" +
    d.results.map((r) => `<tr>
      <td class="num">${r.rank}</td>
      <td>${escapeHtml(r.chunk_id)}</td>
      <td class="num">${r.vector.toFixed(3)}</td>
      <td class="num">${r.bm25.toFixed(3)}</td>
      <td class="num">${r.score.toFixed(3)}</td>
      <td>${r.score >= d.threshold ? "pass" : "below"}</td>
    </tr>`).join("") + "</tbody>";

  ui.resultsWrap.hidden = false;
}

// Citation markers are rendered as chips rather than left as raw brackets. On
// this page they are the evidence, not noise, so they should read as a source
// reference and not as punctuation the model forgot to strip.
function renderCited(host, text) {
  const re = /\[([a-z0-9_]+_c\d+)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) host.appendChild(document.createTextNode(text.slice(last, m.index)));
    const chip = document.createElement("span");
    chip.className = "cite-chip";
    chip.textContent = m[1];
    chip.title = "Passage this claim came from";
    host.appendChild(chip);
    last = m.index + m[0].length;
  }
  if (last < text.length) host.appendChild(document.createTextNode(text.slice(last)));
}

function pill(text, kind) {
  const s = document.createElement("span");
  s.className = "pill" + (kind ? ` ${kind}` : "");
  s.textContent = text;
  return s;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── pipeline + composition ───────────────────────────────────────────

async function loadOverview() {
  ui.funnel.innerHTML = `<p class="loading">Loading…</p>`;
  ui.comp.innerHTML = `<p class="loading">Loading…</p>`;
  const d = await fetch(`/api/kb/overview?source=${active}`).then((r) => r.json());
  if (d.error) { ui.funnel.innerHTML = `<p class="loading">${d.error}</p>`; ui.comp.innerHTML = ""; return; }

  const top = d.steps[0].value || 1;
  ui.funnel.innerHTML = "";
  d.steps.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "stage";
    div.innerHTML = `
      <div class="stage-top"><span class="stage-label"></span><span class="stage-count"></span></div>
      <div class="stage-bar"><span></span></div>`;
    div.querySelector(".stage-label").textContent = s.label;
    div.querySelector(".stage-count").textContent = s.value;
    const bar = div.querySelector(".stage-bar span");
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = `${(s.value / top) * 100}%`; }, 40 + i * 70);
    if (s.drop) {
      const drop = document.createElement("div");
      drop.className = "stage-drop";
      drop.textContent = `−${s.drop} · ${s.reason}`;
      div.appendChild(drop);
    }
    ui.funnel.appendChild(div);
  });

  ui.notes.innerHTML = [
    ["Records kept", d.final_records],
    ["Chunks indexed", d.chunks],
    ["Chunk size (words)", `${d.chunk_words.min}–${d.chunk_words.max}, median ${d.chunk_words.median}`],
    ["Boilerplate lines removed", d.boilerplate_removed],
    ["Dates normalised to ISO", d.dates_normalised],
    ["Records with PII redacted", `${d.pii_records}${d.pii_types.length ? " · " + d.pii_types.join(", ") : ""}`],
  ].map(([k, v]) => `<div class="note-row"><span>${k}</span><strong>${escapeHtml(String(v))}</strong></div>`).join("");

  const maxChunks = Math.max(...d.composition.map((c) => c.chunks), 1);
  ui.comp.innerHTML = "";
  d.composition.forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "comp-row";
    row.innerHTML = `<span class="comp-label"></span><span class="comp-count"></span>
                     <div class="comp-bar"><span></span></div>`;
    row.querySelector(".comp-label").textContent = c.category.replace(/_/g, " ");
    row.querySelector(".comp-count").textContent = c.chunks;
    const bar = row.querySelector(".comp-bar span");
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = `${(c.chunks / maxChunks) * 100}%`; }, 40 + i * 40);
    ui.comp.appendChild(row);
  });
}

// ── init ─────────────────────────────────────────────────────────────

ui.form.addEventListener("submit", (e) => { e.preventDefault(); search(); });

fetch("/api/kb/corpora")
  .then((r) => r.json())
  .then((d) => {
    corpora = d.corpora.filter((c) => c.built);
    if (!corpora.length) {
      ui.switch.innerHTML = `<span class="loading">No index built yet.</span>`;
      return;
    }
    active = corpora[0].key;
    renderSwitch();
    renderExamples();
    loadOverview();
  });
