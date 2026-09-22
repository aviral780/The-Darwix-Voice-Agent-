"""Web call interface for the voice agent.

The brief allows "a callable number or a web calling interface". A browser
interface was chosen over telephony for a reason worth stating: every free
telephony trial requires a card, and a demo that cannot be reproduced by whoever
reviews it is worth less than one they can open and use. The trade-off is real
and it is recorded in docs/limitations.md - there is no PSTN leg here, so real
network jitter, packet loss and codec degradation are untested.

Each browser tab is one CallSession. Audio arrives as a WebM/Opus blob from
MediaRecorder, goes to Whisper, through the state machine, and comes back as MP3
from edge-tts with the transcript.

The state machine in agent/flow.py is deliberately synchronous: it is far easier
to test and reason about that way, and it can be driven from a script with no
event loop at all. It is network-bound rather than CPU-bound, so the server runs
it with asyncio.to_thread. Calling it directly would block the event loop for the
whole model round trip and stall every other call on the server.

There is also a text path that bypasses microphone capture entirely. It runs the
identical state machine, and it exists because a live microphone is the least
reliable part of any recorded demo.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import asr, config, tts
from core.timing import LatencyRecorder
from agent.actions import finalise_call
from agent.flow import CallSession, State

app = FastAPI(title="Darwix Voice Agent")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Serve static assets uncached.

    This is a demo server that is edited while it runs, and a cached stylesheet
    silently shows the reviewer an old interface. Correct behaviour that looks
    broken because of a stale asset is worse than slow.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

WEB_DIR = config.ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.on_event("startup")
async def warm_up() -> None:
    """Load the index and the ONNX embedding model before the first caller.

    Measured cold, the first retrieval took 2.1 seconds against roughly 90ms
    warm, because loading the embedding model happens lazily on first use. On a
    phone call that lands entirely on the first person to ask a question, so it
    is paid here at boot instead.
    """
    def _warm() -> None:
        from kb.retrieve import get_retriever
        get_retriever().search("premium payment grace period")

    await asyncio.to_thread(_warm)
    print("index and embedding model warm; ready on http://127.0.0.1:8000")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/live")
def live_dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "live.html")


@app.get("/api/live_scenarios")
def live_scenarios() -> dict:
    path = config.EVIDENCE_DIR / "live_calls" / "scenarios.json"
    if not path.exists():
        return {"scenarios": []}
    import json as _json
    scenarios = _json.loads(path.read_text())
    return {"scenarios": [
        {"id": s["id"], "name": s["name"], "why": s["why"],
         "expect": s["expect"], "duration_s": s["duration_s"]}
        for s in scenarios
    ]}


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket) -> None:
    """Stream a call and push nudges to the dashboard as they are generated.

    The stream runs in a worker thread because live/stream.py is synchronous and
    paces itself against the audio clock. Events are handed back through a queue
    so the socket keeps sending while the thread blocks on its next chunk.
    """
    await websocket.accept()
    try:
        message = json.loads(await websocket.receive_text())
        scenario_id = message.get("scenario", "")

        import json as _json
        scenarios = _json.loads(
            (config.EVIDENCE_DIR / "live_calls" / "scenarios.json").read_text())
        match = next((s for s in scenarios if s["id"] == scenario_id), None)
        if match is None:
            await websocket.send_json({"type": "error", "message": "Unknown scenario"})
            return

        from live.stream import process

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        task = asyncio.create_task(asyncio.to_thread(
            process, config.ROOT / match["path"], match["id"], True, on_event))

        while True:
            if task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(event)

        result = await task
        await websocket.send_json({
            "type": "done",
            "expected": match["expect"],
            "fired": [n["type"] for n in result.nudges],
            "suppression": result.suppression,
            "latency": result.latency,
        })
    except WebSocketDisconnect:
        pass


@app.get("/kb")
def kb_explorer() -> FileResponse:
    return FileResponse(WEB_DIR / "kb.html")


@app.get("/api/kb/corpora")
def kb_corpora() -> dict:
    """The corpora available to query, with the state of each."""
    from kb.sources import SOURCES

    out = []
    for key, source in SOURCES.items():
        meta_path = source.index_dir / "meta.json"
        report_path = source.clean_dir / "cleaning_report.json"
        entry = {
            "key": key,
            "name": source.name,
            "base": source.base,
            "language": source.language,
            "sector": source.sector,
            "threshold": source.min_retrieval_score,
            "built": meta_path.exists(),
        }
        if meta_path.exists():
            entry.update(json.loads(meta_path.read_text()))
        if report_path.exists():
            report = json.loads(report_path.read_text())
            entry["records"] = report["stages"].get("final_records", 0)
        out.append(entry)
    return {"corpora": out}


@app.get("/api/kb/overview")
def kb_overview(source: str = "prulife_ph") -> dict:
    """Pipeline attrition and corpus composition for one corpus."""
    from kb.sources import get as get_source

    src = get_source(source)
    report_path = src.clean_dir / "cleaning_report.json"
    if not report_path.exists():
        return {"error": f"No cleaning report for {source}. Run the pipeline first."}
    report = json.loads(report_path.read_text())
    stages = report["stages"]
    dropped = report["dropped"]

    # Attrition, in the order the pipeline applies it. Each step carries the
    # reason, because a count with no reason is not auditable.
    steps = [
        {"label": "Pages fetched", "value": stages.get("fetched", 0), "drop": 0, "reason": ""},
        {"label": "Content extracted", "value": stages.get("extracted", 0),
         "drop": len(dropped.get("extraction_failed", [])), "reason": "no main content found"},
    ]
    running = stages.get("extracted", 0)
    for key, label in [("irrelevant_category", "off-topic section"),
                       ("thin_content", "too little content to answer from"),
                       ("failed_extraction", "extraction produced a navigation widget"),
                       ("duplicates", "exact or near-duplicate of a kept page")]:
        n = len(dropped.get(key, []))
        if n:
            running -= n
            steps.append({"label": f"After removing {label}", "value": running,
                          "drop": n, "reason": label})

    chunks_path = src.index_dir / "chunks.json"
    chunks = json.loads(chunks_path.read_text()) if chunks_path.exists() else []
    from collections import Counter
    composition = Counter(c["category"] for c in chunks)

    return {
        "source": source,
        "steps": steps,
        "final_records": stages.get("final_records", 0),
        "chunks": len(chunks),
        "boilerplate_lines": stages.get("boilerplate_lines_identified", 0),
        "boilerplate_removed": stages.get("boilerplate_lines_removed", 0),
        "dates_normalised": stages.get("dates_normalised", 0),
        "pii_records": stages.get("pii_records_flagged", 0),
        "pii_types": stages.get("pii_types_seen", []),
        "composition": [{"category": c, "chunks": n} for c, n in composition.most_common()],
        "chunk_words": {
            "min": min((c["word_count"] for c in chunks), default=0),
            "median": sorted(c["word_count"] for c in chunks)[len(chunks) // 2] if chunks else 0,
            "max": max((c["word_count"] for c in chunks), default=0),
        },
    }


@app.post("/api/kb/search")
async def kb_search(payload: dict) -> dict:
    """Run a query through the full two-gate path and report both gates.

    This is the endpoint that makes the refusal design inspectable: it returns
    the component scores, where the threshold sits, which gate fired, and what
    the agent would actually have said.
    """
    query = (payload.get("query") or "").strip()
    source = payload.get("source", "prulife_ph")
    language = payload.get("language", "")
    if not query:
        return {"error": "Enter a question."}

    from core.config import MarketConfig, available_markets
    from kb.answer import answer_question
    from kb.retrieve import get_retriever

    # Filler stripping is market-specific, so use the market that reads this
    # corpus. Searching without it would not be what the agent does.
    fillers: list[str] = []
    for key in available_markets():
        market = MarketConfig.load(key)
        if market.kb_source == source:
            fillers = market.query_fillers
            if not language:
                language = market.language_name
            break

    retriever = get_retriever(source)
    recorder = LatencyRecorder("kb-search")

    results = await asyncio.to_thread(
        retriever.search, query, config.TOP_K, None, recorder, fillers)

    answer = await asyncio.to_thread(
        answer_question, query, config.TOP_K, 3, "", recorder, source, language, fillers)

    return {
        "query": query,
        "source": source,
        "threshold": retriever.min_score,
        "weights": {"vector": config.VECTOR_WEIGHT, "bm25": config.BM25_WEIGHT},
        "passed_retrieval_gate": retriever.is_answerable(results),
        "answer": answer.text,
        "refused": answer.refused,
        "gate": answer.gate,
        "citations": answer.citations,
        "results": [
            {
                "rank": r.rank,
                "chunk_id": r.chunk["chunk_id"],
                "title": r.chunk["title"],
                "heading": " > ".join(r.chunk["heading_path"]) or r.chunk["title"],
                "category": r.chunk["category"],
                "url": r.source_url,
                "words": r.chunk["word_count"],
                "pii": r.chunk["pii"],
                "content": r.chunk["content"],
                "score": round(r.score, 4),
                "vector": round(r.vector_score, 4),
                "bm25": round(r.bm25_score, 4),
                # The weighted parts that actually sum to the fused score.
                "vector_part": round(config.VECTOR_WEIGHT * r.vector_score, 4),
                "bm25_part": round(config.BM25_WEIGHT * r.bm25_score, 4),
                "cited": r.chunk["chunk_id"] in answer.citations,
                "explain": r.explain(),
            }
            for r in results
        ],
        "latency": recorder.summary(),
    }


@app.get("/api/markets")
def markets() -> dict:
    from core.config import MarketConfig, available_markets
    out = []
    for key in available_markets():
        market = MarketConfig.load(key)
        out.append({
            "key": key,
            "display_name": market.display_name,
            "language": market.language_name,
            "sector": market.sector,
            "voice": market.tts_voice,
        })
    return {"markets": out}


async def speak(session: CallSession, text: str) -> str:
    """Synthesize agent speech and return it base64-encoded for the browser."""
    if not text.strip():
        return ""
    audio = await tts.synthesize_timed(
        text,
        voice=session.market.tts_voice,
        recorder=session.recorder,
        stage="tts",
    )
    return base64.b64encode(audio).decode()


def turn_payload(session: CallSession, turn, audio_b64: str) -> dict:
    return {
        "type": "agent_turn",
        "text": turn.text,
        "audio": audio_b64,
        "state": session.state.value,
        "citations": turn.citations,
        "refused": turn.refused,
        "refusal_gate": turn.refusal_gate,
        "slots": session.slots,
        "slots_total": len(session.market.qualification),
        "ended": session.state in (State.ENDED, State.ESCALATED),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session: CallSession | None = None

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            kind = message.get("type")

            if kind == "start":
                session = CallSession(market_key=message.get("market", "en_PH"))
                greeting = session.open()
                await websocket.send_json({
                    "type": "call_started",
                    "call_id": session.call_id,
                    "market": session.market.display_name,
                    "voice": session.market.tts_voice,
                })
                await websocket.send_json(
                    turn_payload(session, session.turns[-1], await speak(session, greeting))
                )
                continue

            if session is None:
                await websocket.send_json({"type": "error", "message": "No active call. Press Start."})
                continue

            # --- caller speech ------------------------------------------------
            if kind == "audio":
                audio = base64.b64decode(message["data"])
                # Whisper is seeded with market vocabulary so terms like
                # "bancassurance" and "rider" are recognised rather than guessed.
                prompt = ", ".join(session.market.terminology.keys())
                text = await asyncio.to_thread(
                    asr.transcribe_bytes,
                    audio,
                    filename="turn.webm",
                    language=session.market.asr_language,
                    prompt=prompt,
                    recorder=session.recorder,
                )
                if not text.strip():
                    await websocket.send_json({
                        "type": "no_speech",
                        "message": "I didn't catch that - could you say it again?",
                    })
                    continue
                await websocket.send_json({"type": "caller_turn", "text": text})

            elif kind == "text":
                text = message.get("text", "").strip()
                if not text:
                    continue
                await websocket.send_json({"type": "caller_turn", "text": text})

            elif kind == "end":
                summary = session.summary()
                path = session.save()
                actions = finalise_call(summary)
                await websocket.send_json({
                    "type": "call_ended",
                    "summary": {k: v for k, v in summary.items() if k != "turns"},
                    "transcript_path": str(path),
                    "actions": actions,
                    "transcript": session.transcript_text(),
                })
                session = None
                continue

            else:
                continue

            # --- agent turn ---------------------------------------------------
            turn = await asyncio.to_thread(session.handle, text)
            await websocket.send_json(turn_payload(session, turn, await speak(session, turn.text)))

            # A call that reached a terminal state runs its business actions
            # immediately, so an escalation is never lost to a closed tab.
            if session.state in (State.ENDED, State.ESCALATED):
                summary = session.summary()
                session.save()
                actions = finalise_call(summary)
                await websocket.send_json({
                    "type": "call_ended",
                    "summary": {k: v for k, v in summary.items() if k != "turns"},
                    "actions": actions,
                    "transcript": session.transcript_text(),
                })
                session = None

    except WebSocketDisconnect:
        # A caller closing the tab mid-call is a real outcome, not an error.
        # Persist whatever was collected rather than discarding the call.
        if session is not None and session.turns:
            summary = session.summary()
            session.save()
            finalise_call(summary)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
