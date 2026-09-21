// Browser call client.
//
// Push-to-talk rather than voice activity detection. VAD on an open mic clips
// the first syllable of a reply and fires on background noise, and neither is
// something to be debugging during a recorded demo. Holding a button makes the
// turn boundary explicit and reproducible.
//
// The text path sends to the same state machine over the same socket, so it is a
// genuine fallback rather than a separate mock: if a microphone fails mid-demo,
// the call continues without restarting.

const els = {
  transcript: document.getElementById('transcript'),
  talk: document.getElementById('talk'),
  text: document.getElementById('textInput'),
  start: document.getElementById('start'),
  end: document.getElementById('end'),
  status: document.getElementById('status'),
  state: document.getElementById('state'),
  callId: document.getElementById('callId'),
  voice: document.getElementById('voice'),
  slots: document.getElementById('slots'),
  grounding: document.getElementById('grounding'),
  market: document.getElementById('market'),
  player: document.getElementById('player'),
  summary: document.getElementById('summary'),
  summaryBody: document.getElementById('summaryBody'),
  closeSummary: document.getElementById('closeSummary'),
};

let ws = null, recorder = null, chunks = [], recording = false, active = false;
const SLOT_LABELS = {
  name: 'Name', age_band: 'Age band', dependents: 'Dependents',
  existing_cover: 'Existing cover', interest: 'Interest', callback: 'Callback',
};

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = 'badge ' + (cls || 'idle');
}

function addMessage(who, text, tags) {
  const empty = els.transcript.querySelector('.empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.innerHTML = `<div class="who">${who === 'agent' ? 'AGENT' : 'YOU'}</div>`;
  div.appendChild(document.createTextNode(text));
  if (tags && tags.length) {
    const row = document.createElement('div');
    row.className = 'tags';
    tags.forEach(t => {
      const span = document.createElement('span');
      span.className = 'tag ' + (t.kind || '');
      span.textContent = t.label;
      row.appendChild(span);
    });
    div.appendChild(row);
  }
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function renderSlots(slots, total) {
  els.slots.innerHTML = '';
  Object.entries(SLOT_LABELS).forEach(([key, label]) => {
    const value = slots[key];
    const li = document.createElement('li');
    li.innerHTML = `<span class="label">${label}</span>` +
      `<span class="val ${value ? '' : 'pending'}">${value || '—'}</span>`;
    els.slots.appendChild(li);
  });
}

function addSources(citations) {
  if (!citations || !citations.length) return;
  const muted = els.grounding.querySelector('.muted');
  if (muted) muted.remove();
  citations.forEach(id => {
    if (els.grounding.querySelector(`[data-id="${id}"]`)) return;
    const div = document.createElement('div');
    div.dataset.id = id;
    div.innerHTML = `<span class="tag cite">${id}</span>`;
    els.grounding.appendChild(div);
  });
}

function playAudio(b64) {
  if (!b64) return;
  els.player.src = 'data:audio/mpeg;base64,' + b64;
  els.player.play().catch(() => {/* autoplay blocked; transcript still shows */});
}

// --- socket ---------------------------------------------------------------

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => ws.send(JSON.stringify({ type: 'start', market: els.market.value }));

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'call_started') {
      active = true;
      els.callId.textContent = msg.call_id;
      els.voice.textContent = msg.voice;
      els.talk.disabled = false;
      els.text.disabled = false;
      els.end.disabled = false;
      els.start.disabled = true;
      setStatus('live', 'live');
      return;
    }

    if (msg.type === 'caller_turn') { addMessage('caller', msg.text); setStatus('thinking', 'think'); return; }

    if (msg.type === 'agent_turn') {
      const tags = [];
      if (msg.citations && msg.citations.length) {
        tags.push({ label: 'grounded: ' + msg.citations.join(', '), kind: 'cite' });
      }
      if (msg.refused) tags.push({ label: 'refused at ' + msg.refusal_gate + ' gate', kind: 'refuse' });
      addMessage('agent', msg.text, tags);
      addSources(msg.citations);
      els.state.textContent = msg.state;
      renderSlots(msg.slots || {}, msg.slots_total);
      playAudio(msg.audio);
      if (active) setStatus('live', 'live');
      return;
    }

    if (msg.type === 'no_speech') { addMessage('agent', msg.message); setStatus('live', 'live'); return; }

    if (msg.type === 'call_ended') {
      active = false;
      setStatus('ended', 'idle');
      els.talk.disabled = true; els.text.disabled = true;
      els.end.disabled = true; els.start.disabled = false;
      showSummary(msg);
      return;
    }

    if (msg.type === 'error') addMessage('agent', msg.message);
  };

  ws.onclose = () => {
    active = false;
    setStatus('disconnected', 'idle');
    els.talk.disabled = true; els.text.disabled = true;
    els.end.disabled = true; els.start.disabled = false;
  };
}

function showSummary(msg) {
  const s = msg.summary || {};
  const a = msg.actions || {};
  let html = `<div class="state-row"><span>Final state</span><strong>${s.final_state}</strong></div>`;
  html += `<div class="state-row"><span>Slots filled</span><strong>${s.slots_filled}/${s.slots_total}</strong></div>`;
  html += `<div class="state-row"><span>Grounded answers</span><strong>${s.grounded_answers}</strong></div>`;
  html += `<div class="state-row"><span>Refusals</span><strong>${s.refusals}</strong></div>`;
  if (a.lead) html += `<div class="state-row"><span>Lead created</span><strong>${a.lead.band} (score ${a.lead.score})</strong></div>`;
  if (a.escalation) html += `<div class="state-row"><span>Escalation</span><strong>written</strong></div>`;
  if (s.latency) {
    html += '<h2 style="font-size:12px;color:#8b93a3;margin:16px 0 8px">LATENCY (ms)</h2>';
    html += '<pre>' + Object.entries(s.latency).map(([k, v]) =>
      `${k.padEnd(12)} p50 ${String(v.p50_ms).padStart(7)}   p95 ${String(v.p95_ms).padStart(7)}   n=${v.count}`
    ).join('\n') + '</pre>';
  }
  if (msg.transcript) {
    html += '<h2 style="font-size:12px;color:#8b93a3;margin:16px 0 8px">TRANSCRIPT</h2>';
    html += '<pre>' + msg.transcript.replace(/</g, '&lt;') + '</pre>';
  }
  els.summaryBody.innerHTML = html;
  els.summary.classList.remove('hidden');
}

// --- microphone -----------------------------------------------------------

async function startRecording() {
  if (recording || !active) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    chunks = [];
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      // A blob this small is a mis-click or a bumped key, not speech.
      if (blob.size < 1200) { setStatus('live', 'live'); return; }
      const buffer = await blob.arrayBuffer();
      let binary = '';
      new Uint8Array(buffer).forEach(b => binary += String.fromCharCode(b));
      ws.send(JSON.stringify({ type: 'audio', data: btoa(binary) }));
      setStatus('transcribing', 'think');
    };
    recorder.start();
    recording = true;
    els.talk.classList.add('recording');
    els.talk.textContent = 'Release to send';
    setStatus('recording', 'rec');
  } catch (err) {
    addMessage('agent', 'Microphone unavailable: ' + err.message + ' — you can type instead.');
  }
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  els.talk.classList.remove('recording');
  els.talk.textContent = 'Hold to talk';
  if (recorder && recorder.state !== 'inactive') recorder.stop();
}

// --- wiring ---------------------------------------------------------------

els.start.onclick = () => {
  els.transcript.innerHTML = '';
  els.grounding.innerHTML = '<div class="muted">Sources cited by the agent appear here.</div>';
  connect();
};
els.end.onclick = () => { if (ws && active) ws.send(JSON.stringify({ type: 'end' })); };
els.closeSummary.onclick = () => els.summary.classList.add('hidden');

els.talk.addEventListener('mousedown', startRecording);
els.talk.addEventListener('mouseup', stopRecording);
els.talk.addEventListener('mouseleave', stopRecording);
els.talk.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
els.talk.addEventListener('touchend', e => { e.preventDefault(); stopRecording(); });

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.repeat && document.activeElement !== els.text) {
    e.preventDefault(); startRecording();
  }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space' && document.activeElement !== els.text) { e.preventDefault(); stopRecording(); }
});

els.text.addEventListener('keydown', e => {
  if (e.key === 'Enter' && els.text.value.trim() && active) {
    ws.send(JSON.stringify({ type: 'text', text: els.text.value.trim() }));
    els.text.value = '';
    setStatus('thinking', 'think');
  }
});

fetch('/api/markets').then(r => r.json()).then(data => {
  data.markets.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = `${m.display_name}`;
    els.market.appendChild(opt);
  });
});
renderSlots({}, 6);
