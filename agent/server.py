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
from agent.actions import finalise_call
from agent.flow import CallSession, State

app = FastAPI(title="Darwix Voice Agent")

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
