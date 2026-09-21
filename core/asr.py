"""Speech recognition via Groq Whisper.

Two things here matter more than the API call itself.

Language hinting: Whisper auto-detects language, and on short Taglish or
code-switched utterances it frequently guesses wrong and then transcribes
phonetically into the wrong script. Passing an explicit language code per
market fixes most of that. The Philippines market is the awkward case, since
Whisper has no Taglish code, so it is hinted as Tagalog and the observed
behaviour is documented in the Q3 write-up rather than hidden.

Chunk sizing: Question 4 feeds audio in short segments. Whisper degrades badly
on segments under roughly one second because it has too little context, so the
streamer keeps a rolling overlap instead of sending bare chunks.
"""

from __future__ import annotations

import io
from pathlib import Path

from groq import Groq

from core import config
from core.timing import LatencyRecorder

_client: Groq | None = None

# Whisper has no code for mixed speech. Taglish is hinted as Tagalog, which in
# testing preserved the English fragments intact rather than transliterating
# them; that observation is reported in docs/q3-localization.md.
LANGUAGE_HINTS = {
    "en_PH": "en",
    "fil_PH": "tl",
    "id_ID": "id",
}

MIN_USEFUL_AUDIO_S = 0.8


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_groq_key())
    return _client


def transcribe_file(
    path: str | Path,
    language: str | None = None,
    prompt: str = "",
    recorder: LatencyRecorder | None = None,
    stage: str = "asr",
) -> str:
    """Transcribe an audio file from disk."""
    path = Path(path)
    with path.open("rb") as handle:
        return _transcribe(handle.read(), path.name, language, prompt, recorder, stage)


def transcribe_bytes(
    audio: bytes,
    filename: str = "chunk.wav",
    language: str | None = None,
    prompt: str = "",
    recorder: LatencyRecorder | None = None,
    stage: str = "asr",
) -> str:
    """Transcribe raw audio bytes. Used by the live streaming path."""
    return _transcribe(audio, filename, language, prompt, recorder, stage)


def _transcribe(
    audio: bytes,
    filename: str,
    language: str | None,
    prompt: str,
    recorder: LatencyRecorder | None,
    stage: str,
) -> str:
    buffer = io.BytesIO(audio)
    buffer.name = filename

    kwargs = {
        "file": (filename, buffer),
        "model": config.ASR_MODEL,
        "response_format": "text",
    }
    if language:
        kwargs["language"] = language
    if prompt:
        # Whisper biases toward vocabulary in the prompt. Seeding it with market
        # terminology measurably improves recognition of words like "angsuran"
        # and "bancassurance" that generic training data under-represents.
        kwargs["prompt"] = prompt

    def _run() -> str:
        result = client().audio.transcriptions.create(**kwargs)
        return (result if isinstance(result, str) else getattr(result, "text", "")).strip()

    if recorder is not None:
        with recorder.span(stage, bytes=len(audio)):
            return _run()
    return _run()


def language_for_market(market_key: str) -> str:
    return LANGUAGE_HINTS.get(market_key, "en")
