# Darwix AI — AI Engineer Assessment

Four working prototypes: a knowledge-grounded voice agent, a production knowledge
base, localised bots for the Philippines and Indonesia, and live nudges generated
while a call is still running.

Everything runs on free tiers. No paid API, no credit card.

---

## If you only have five minutes

Every recording is in this repo. You don't have to run anything.

| What | Where |
|---|---|
| Q1 — five recorded calls, with transcripts | [`docs/q1_call_results_en_PH.md`](docs/q1_call_results_en_PH.md) |
| Q2 — retrieval results, 14 queries with verdicts | [`docs/retrieval_results.md`](docs/retrieval_results.md) |
| Q3 — localisation, with the Taglish and Bahasa calls | [`docs/q3_localization.md`](docs/q3_localization.md) |
| Q3 — ASR testing, 85 trials including regional accent | [`docs/q3_asr_report.md`](docs/q3_asr_report.md) |
| Q4 — latency, suppression and false-positive analysis | [`docs/latency_report.md`](docs/latency_report.md) |
| Why this model, measured not assumed | [`docs/model_selection.md`](docs/model_selection.md) |
| Known gaps, stated plainly | [`docs/limitations.md`](docs/limitations.md) |

Audio files:

- `evidence/calls/` — nine call recordings (five English, four localised), MP3
- `evidence/live_calls/` — four stereo calls used for the live nudge tests, WAV
- `evidence/transcripts/` — a transcript for every call
- `data/leads/` — the lead records the agent actually created
- `evidence/escalations/` — the handover records written when a caller asked for a person

If you want the single most interesting result, it is in
[`docs/retrieval_results.md`](docs/retrieval_results.md): the scores for questions
the system *can* answer and questions it *cannot* overlap, so no confidence
threshold separates them. That is why there are two gates instead of one.

---

## Running it yourself

A step-by-step PDF is in [`Setup-Guide.pdf`](Setup-Guide.pdf).

You need Python 3.11 or newer, `ffmpeg`, and a free Groq API key from
[console.groq.com](https://console.groq.com) (no card required).

```bash
git clone <this repo>
cd darwix-voice-agent

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env          # then paste your Groq key into it

.venv/bin/python -m uvicorn agent.server:app --port 8000
```

Open **http://127.0.0.1:8000**.

The knowledge base is already built and committed, so there is nothing to scrape
and nothing to wait for. First start takes about ten seconds while the embedding
model loads.

### Three pages

| Page | What it is |
|---|---|
| `/` | **Call console.** Pick a market, talk to the agent. Live coaching runs alongside the call. |
| `/kb` | **Knowledge base explorer.** Type a question, watch both gates decide. |
| `/live` | **Live insights.** Play a recorded call, or speak into it yourself. |

On the call console, click the mic and talk. It sends when you stop speaking.
If your microphone is awkward, type instead — same engine either way.

The nudge engine runs on the live call too, in the panel on the right. Say
*"my wife and two kids depend on me"* and a cross-sell nudge appears while you
are still talking.

On `/live` you can also skip the recordings and use your own voice. Pick whether
you are speaking as the agent or the customer, because the detectors are
asymmetric: a missing disclosure only counts against the agent, and a mention of
a spouse is only an opening when the customer says it.

Two things worth trying on the call console:

- Ask something it knows: *"What happens if I miss a premium payment?"*
- Ask something it can't know: *"What is my current policy account balance?"*

The second one refuses and offers a human. That behaviour is the point.

---

## What each question does

**Q1 — Voice agent.** Life insurance lead qualification for the Philippines. A
state machine, not a prompt: greeting, six qualification slots, objection
handling, escalation. Every factual answer is retrieved from the Q2 knowledge
base and cited. When it doesn't know, it says so.

**Q2 — Knowledge base.** 139 pages from Pru Life UK Philippines, cleaned down to
116 records and 814 chunks. Boilerplate removed, near-duplicates dropped, dates
normalised, PII redacted. Hybrid retrieval: embeddings for paraphrase, BM25 for
product names people say verbatim.

**Q3 — Localised bots.** Philippines in Taglish, Indonesia in Bahasa. Same engine,
different config file. The Indonesian bot reads from its own corpus because it is
a different sector — multifinance, not insurance.

**Q4 — Live nudges.** Two ways in. Recorded calls play at real speed with the two
speakers on separate stereo channels; each channel is cut into utterances where
the speaker actually pauses, and nothing is processed before it has been heard.
A live call feeds the same detectors turn by turn. Rules catch what is
deterministic (a disclosure the agent has now skipped), the model judges what is
not (whether someone is getting frustrated), and each nudge marks the line that
caused it.

---

## Results

| | Result |
|---|---|
| Knowledge base answers correctly or refuses correctly | 14 / 14 |
| Live nudge scenarios pass | 4 / 4 |
| Live nudge latency (from speaker stopping) | p50 1106 ms, p95 1380 ms |
| ASR trials, scored against exact references | 85 |
| Recorded calls | 9 |

Every number here is regenerated by a script in `scripts/`. None of them are
typed by hand.

---

## How it fits together

See [`docs/architecture.md`](docs/architecture.md). The short version: one voice
engine, configured three ways, reading from two knowledge bases, with the live
nudge pipeline tapping the same audio.

---

## Stack

| | |
|---|---|
| Speech to text | Groq Whisper large v3 turbo |
| Language model | Qwen3.8-27B on Groq, chosen by benchmark |
| Embeddings | multilingual MiniLM via ONNX, runs locally, no PyTorch |
| Keyword search | BM25 |
| Speech | edge-tts, native Filipino and Indonesian voices |
| Server | FastAPI and WebSockets |
| Frontend | Plain HTML, CSS and JavaScript. No build step. |

The embedding model runs locally on CPU in about 130 MB. Everything else is a
free-tier API call.

---

## A note on what isn't here

There is no real telephony. The brief allowed "a callable number or a web calling
interface" and every free telephony trial needs a card, so this is browser-based.
That means no PSTN leg, and no packet loss or codec degradation in the tests.

The honest list of gaps is in [`docs/limitations.md`](docs/limitations.md). The
biggest one: I don't speak Tagalog or Bahasa Indonesia, so the localisation is
researched, not validated by a native speaker.
