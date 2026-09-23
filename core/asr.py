"""Speech recognition via Groq Whisper.

Two things here matter more than the API call itself.

Language hinting: an explicit language code is passed per market. It is worth
being precise about what that buys, because the measurement in
scripts/asr_report.py did not support the original reason for adding it.

The assumption was that auto-detect misfires on short code-switched utterances
and that hinting fixes it. Across 85 trials on clean synthetic speech the hint
made almost no difference: mean WER 0.177 with `tl` against 0.185 auto-detect
for Filipino, and identical at 0.119 for Indonesian. Whisper's auto-detect is
simply good on clean audio.

The hint is kept for two narrower reasons. It removes a failure mode rather
than improving the average - auto-detect has to commit to one language for the
segment, and a wrong commitment on a short utterance mistranscribes the whole
of it - and it makes behaviour deterministic per market, which matters when
comparing runs. It is not the accuracy win it was introduced as, and the report
says so.

Segmenting: the live pipeline sends one whole utterance per request, cut where
the speaker actually paused, rather than fixed-length chunks. Whisper is far
more accurate with a complete sentence than with a fragment cut mid-word, and
there are fewer requests to make.

Rate limits: that is one request per utterance, which on a talkative call runs
past the free tier's twenty a minute. A rate-limited request moves straight to
the second Whisper model, which has its own quota, and only waits if both are
limited - a dropped request here is a line missing from the transcript and a
detector that never saw it, and a request that waits out the limit is a line
that appears fifteen seconds after it was said. Measured: four calls back to
back on turbo alone pushed individual transcripts to sixteen seconds late.
"""

from __future__ import annotations

import io
import time
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


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
# A per-minute request limit clears within seconds; a daily one does not, and is
# not worth waiting for mid-call.
MAX_RETRY_WAIT_S = 8.0


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
        from core.llm import retry_after_seconds

        models = list(dict.fromkeys([config.ASR_MODEL, config.ASR_FALLBACK_MODEL]))
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            waits: list[float] = []
            for model in models:
                buffer.seek(0)            # a failed attempt consumed the upload
                kwargs["model"] = model
                try:
                    result = client().audio.transcriptions.create(**kwargs)
                    return (result if isinstance(result, str) else getattr(result, "text", "")).strip()
                except Exception as exc:
                    if getattr(exc, "status_code", None) not in RETRYABLE_STATUS:
                        raise
                    last_error = exc
                    print(f"asr: {model} rate limited or unavailable, trying the next option ({str(exc)[:120]})")
                    hint = retry_after_seconds(str(exc))
                    waits.append(hint if hint is not None else 0.6 * (2 ** attempt))
            # Every model is limited right now; wait for the soonest to clear.
            wait = min(waits)
            if attempt == MAX_ATTEMPTS - 1 or wait > MAX_RETRY_WAIT_S:
                break
            time.sleep(wait + 0.05)
        raise last_error if last_error else RuntimeError("transcription failed")

    if recorder is not None:
        with recorder.span(stage, bytes=len(audio)):
            return _run()
    return _run()


def language_for_market(market_key: str) -> str:
    return LANGUAGE_HINTS.get(market_key, "en")
