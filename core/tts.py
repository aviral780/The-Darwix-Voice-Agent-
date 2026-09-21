"""Text to speech via edge-tts.

Chosen because it needs no API key and no credit card, and because it has
genuine native neural voices for both Question 3 markets. That last point is
what makes the localisation credible: a Filipino script read by an American
voice would undercut the whole exercise.

Output is MP3. The browser plays it directly, and ffmpeg converts it when a
WAV is needed for the evidence pack.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

from core.timing import LatencyRecorder

# Verified against `edge-tts --list-voices` on the machine this was built on.
VOICES = {
    "en_PH": "en-PH-JamesNeural",
    "en_PH_female": "en-PH-RosaNeural",
    "fil_PH": "fil-PH-BlessicaNeural",
    "fil_PH_male": "fil-PH-AngeloNeural",
    "id_ID": "id-ID-GadisNeural",
    "id_ID_male": "id-ID-ArdiNeural",
}

DEFAULT_VOICE = "en-US-AriaNeural"


async def synthesize_async(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def synthesize(
    text: str,
    voice: str | None = None,
    market_key: str = "",
    rate: str = "+0%",
    recorder: LatencyRecorder | None = None,
    stage: str = "tts",
) -> bytes:
    """Synthesize speech and return MP3 bytes."""
    voice = voice or VOICES.get(market_key, DEFAULT_VOICE)

    def _run() -> bytes:
        return asyncio.run(synthesize_async(text, voice, rate=rate))

    if recorder is not None:
        with recorder.span(stage, voice=voice, chars=len(text)):
            return _run()
    return _run()


def synthesize_to_file(
    text: str,
    path: str | Path,
    voice: str | None = None,
    market_key: str = "",
    rate: str = "+0%",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(synthesize(text, voice=voice, market_key=market_key, rate=rate))
    return path


async def list_voices(locale_prefix: str = "") -> list[dict]:
    """List available voices, optionally filtered by locale prefix such as 'fil'."""
    voices = await edge_tts.list_voices()
    if locale_prefix:
        voices = [v for v in voices if v["ShortName"].lower().startswith(locale_prefix.lower())]
    return voices
