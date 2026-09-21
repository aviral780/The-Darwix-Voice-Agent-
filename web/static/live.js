// Live nudge dashboard.
//
// Nudges are rendered as they arrive and removed when the engine stops listing
// them as active, so expiry and the concurrent cap are visible in the UI rather
// than only in the logs. That is deliberate: suppression is the part of this
// system worth watching, and a panel that only ever adds rows would hide it.

const els = {
  scenario: document.getElementById('scenario'),
  run: document.getElementById('run'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  nudges: document.getElementById('nudges'),
  clock: document.getElementById('clock'),
  cSeen: document.getElementById('cSeen'),
  cShown: document.getElementById('cShown'),
  cSupp: document.getElementById('cSupp'),
  cLat: document.getElementById('cLat'),
};

let ws = null, shown = 0;

function setStatus(t, c) { els.status.textContent = t; els.status.className = 'badge ' + (c || 'idle'); }

function addLine(at, speaker, text) {
  const empty = els.transcript.querySelector('.empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'line ' + (speaker === 'agent' ? 'agent' : 'caller');
  div.innerHTML = `<span class="t">${at.toFixed(1)}s</span>` +
                  `<span class="s">${speaker.toUpperCase()}</span>` +
                  `<span class="x"></span>`;
  div.querySelector('.x').textContent = text;
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function renderActive(active) {
  const ids = new Set(active.map(n => n.id));
  [...els.nudges.querySelectorAll('.nudge')].forEach(el => {
    if (!ids.has(el.dataset.id)) { el.classList.add('fading'); setTimeout(() => el.remove(), 400); }
  });
  active.forEach(n => {
    let el = els.nudges.querySelector(`[data-id="${n.id}"]`);
    if (!el) {
      const muted = els.nudges.querySelector('.muted');
      if (muted) muted.remove();
      el = document.createElement('div');
      el.className = 'nudge ' + n.type;
      el.dataset.id = n.id;
      el.innerHTML = `<div class="top"><span>${n.type.replace(/_/g, ' ')}</span>` +
                     `<span>p${n.priority}</span></div>` +
                     `<div class="txt"></div><div class="meta"></div>`;
      el.querySelector('.txt').textContent = n.text;
      els.nudges.appendChild(el);
    }
    el.querySelector('.meta').textContent =
      `conf ${n.confidence} · ${n.detector} · "${n.evidence}" · ${n.age_s}s old`;
  });
}

els.run.onclick = () => {
  els.transcript.innerHTML = '';
  els.nudges.innerHTML = '<div class="muted">Listening…</div>';
  shown = 0; els.cSeen.textContent = '0'; els.cShown.textContent = '0';
  els.cSupp.textContent = '0'; els.cLat.textContent = '—';
  els.run.disabled = true;
  setStatus('connecting', 'think');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onopen = () => ws.send(JSON.stringify({ scenario: els.scenario.value }));

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'stream_start') { setStatus('live', 'live'); return; }
    if (m.type === 'transcript') { addLine(m.at_s, m.speaker, m.text); return; }
    if (m.type === 'tick') { els.clock.textContent = m.at_s.toFixed(1) + 's'; renderActive(m.active); return; }
    if (m.type === 'nudge') {
      shown += 1;
      els.cShown.textContent = shown;
      els.cLat.textContent = m.nudge.latency_ms + ' ms';
      return;
    }
    if (m.type === 'stream_end') {
      els.cSeen.textContent = m.suppression.signals_seen;
      els.cSupp.textContent = Math.round(m.suppression.suppression_rate * 100) + '%';
      return;
    }
    if (m.type === 'done') {
      setStatus('complete', 'idle');
      els.run.disabled = false;
      const ok = m.expected.length === 0 ? m.fired.length === 0
                                         : m.expected.every(e => m.fired.includes(e));
      addLine(0, 'agent', `— ${ok ? 'PASS' : 'FAIL'}: expected [${m.expected.join(', ') || 'nothing'}], fired [${m.fired.join(', ') || 'nothing'}]`);
    }
  };
  ws.onclose = () => { els.run.disabled = false; };
};

fetch('/api/live_scenarios').then(r => r.json()).then(d => {
  d.scenarios.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = `${s.name} (${s.duration_s}s)`;
    els.scenario.appendChild(o);
  });
});
