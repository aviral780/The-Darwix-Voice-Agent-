"""Real-time call processing: audio in, nudges out while the call is running.

The brief is explicit that analysing a finished recording does not qualify. This
consumes audio in chunks at the pace the audio actually plays, and emits nudges
as the call proceeds. Replaying a recording at real-time speed is the permitted
form and is what the test suite uses, because it makes runs comparable; the same
loop accepts live microphone chunks.

Speaker separation is done with stereo channels rather than diarisation. Contact
centre recorders put the agent on one channel and the customer on the other, and
Whisper does not diarise at all. Splitting channels and transcribing each gives
true attribution instead of a guess; mono input still works but is flagged as
unattributed, because pretending to know who spoke would corrupt every
agent-side compliance rule.

Two timing decisions shape the pipeline:

Chunks are 4 seconds. Shorter and Whisper loses the context it needs and its
accuracy drops sharply; longer and the nudge arrives after the moment has passed.

Rules run on every chunk because they cost microseconds. The model judge runs
only when enough new speech has accumulated to be worth a round trip, which is
what keeps the end-to-end number low without giving up the judgement calls.
"""

from __future__ import annotations

import array
import json
import math
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from core import asr, config
from core.timing import LatencyRecorder, format_markdown_table
from live.nudges import Nudge, NudgeEngine
from live.signals import Signal, model_signals, rule_signals

CHUNK_SECONDS = 4.0
# Below this RMS a chunk is treated as silence and never sent to the ASR.
# Whisper hallucinates on silence - it returns "you", "Thank you." or
# "Thanks for watching!" for an empty segment, which lands in the transcript as
# if someone said it and then feeds the signal detectors. In a stereo call each
# speaker is silent roughly half the time, so gating on energy also removes
# about half the ASR calls and their latency.
SILENCE_RMS = 180.0

# Phrases Whisper emits for near-silence, kept as a second line of defence for
# chunks that carry line noise above the energy gate.
SILENCE_ARTIFACTS = {
    "you", "thank you.", "thank you", "thanks for watching!", "thanks for watching",
    "bye.", "bye", ".", "...", "um", "uh", "so", "and the road.",
}
SAMPLE_RATE = 16000
# Judge with the model once this much unjudged speech has built up.
JUDGE_EVERY_CHARS = 180
JUDGE_WINDOW_TURNS = 6


@dataclass
class Chunk:
    index: int
    start_s: float
    agent_audio: bytes
    caller_audio: bytes
    mono_audio: bytes = b""
    stereo: bool = True
    agent_rms: float = 0.0
    caller_rms: float = 0.0
    mono_rms: float = 0.0


@dataclass
class StreamResult:
    call_id: str
    source: str
    stereo: bool
    duration_s: float
    chunks: int
    transcript: list[dict] = field(default_factory=list)
    nudges: list[dict] = field(default_factory=list)
    suppression: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    end_to_end_ms: list[float] = field(default_factory=list)


def probe_channels(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip() or 1)
    except ValueError:
        return 1


def to_wav(path: Path, destination: Path, channel: int | None = None) -> Path:
    """Decode to 16kHz mono WAV, optionally extracting one stereo channel."""
    filters = ["-ac", "1"]
    if channel is not None:
        filters = ["-af", f"pan=mono|c0=c{channel}"]
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), *filters, "-ar", str(SAMPLE_RATE),
         "-acodec", "pcm_s16le", str(destination)],
        capture_output=True, check=True,
    )
    return destination


def read_frames(path: Path) -> tuple[bytes, int, float]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
        duration = handle.getnframes() / float(rate)
    return frames, rate, duration


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of 16-bit PCM."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def is_artifact(text: str) -> bool:
    """Whether a transcription is a known silence hallucination."""
    return text.strip().lower() in SILENCE_ARTIFACTS


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM in a WAV container so it can be posted to the ASR API."""
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def chunk_source(path: Path, chunk_seconds: float = CHUNK_SECONDS) -> Iterator[Chunk]:
    """Yield fixed-length chunks, split by speaker when the source is stereo."""
    channels = probe_channels(path)
    stereo = channels >= 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if stereo:
            agent_wav = to_wav(path, tmp_path / "agent.wav", channel=0)
            caller_wav = to_wav(path, tmp_path / "caller.wav", channel=1)
            agent_pcm, rate, duration = read_frames(agent_wav)
            caller_pcm, _, _ = read_frames(caller_wav)
            mono_pcm = b""
        else:
            mono_wav = to_wav(path, tmp_path / "mono.wav")
            mono_pcm, rate, duration = read_frames(mono_wav)
            agent_pcm = caller_pcm = b""

        step = int(rate * chunk_seconds) * 2      # 16-bit samples
        total = len(agent_pcm if stereo else mono_pcm)
        index = 0
        for offset in range(0, total, step):
            agent_pcm_chunk = agent_pcm[offset:offset + step] if stereo else b""
            caller_pcm_chunk = caller_pcm[offset:offset + step] if stereo else b""
            mono_pcm_chunk = mono_pcm[offset:offset + step] if not stereo else b""
            yield Chunk(
                index=index,
                start_s=offset / 2 / rate,
                agent_audio=wav_bytes(agent_pcm_chunk, rate) if stereo else b"",
                caller_audio=wav_bytes(caller_pcm_chunk, rate) if stereo else b"",
                mono_audio=wav_bytes(mono_pcm_chunk, rate) if not stereo else b"",
                stereo=stereo,
                agent_rms=rms(agent_pcm_chunk),
                caller_rms=rms(caller_pcm_chunk),
                mono_rms=rms(mono_pcm_chunk),
            )
            index += 1


def process(
    source: Path,
    call_id: str = "",
    realtime: bool = True,
    on_event: Callable[[dict], None] | None = None,
    terminology: str = "premium, policy, beneficiary, rider, lapse, coverage, bancassurance",
) -> StreamResult:
    """Stream a call and emit nudges as it plays.

    realtime=True paces the loop to the audio's own duration, which is what makes
    this a live-processing test rather than batch analysis. Setting it False runs
    as fast as the API allows and is only for checking correctness.
    """
    call_id = call_id or source.stem
    recorder = LatencyRecorder(f"live-{call_id}")
    engine = NudgeEngine()

    transcript: list[dict] = []
    emitted: list[dict] = []
    end_to_end: list[float] = []
    unjudged_chars = 0
    asr_calls = 0
    asr_possible = 0
    stereo = probe_channels(source) >= 2
    duration = 0.0

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    emit({"type": "stream_start", "call_id": call_id, "source": str(source), "stereo": stereo})

    wall_start = time.perf_counter()

    for chunk in chunk_source(source):
        duration = chunk.start_s + CHUNK_SECONDS

        # Pace to the audio. Without this the loop runs as fast as the network
        # allows and the latency numbers would describe batch throughput rather
        # than what an agent on a live call experiences.
        if realtime:
            target = chunk.start_s
            drift = target - (time.perf_counter() - wall_start)
            if drift > 0:
                time.sleep(drift)

        chunk_received = time.perf_counter()

        # --- transcription ---------------------------------------------------
        agent_text = caller_text = ""
        with recorder.span("asr", chunk=chunk.index):
            if chunk.stereo:
                if chunk.agent_rms >= SILENCE_RMS:
                    agent_text = asr.transcribe_bytes(
                        chunk.agent_audio, f"a{chunk.index}.wav", language="en", prompt=terminology)
                    asr_calls += 1
                if chunk.caller_rms >= SILENCE_RMS:
                    caller_text = asr.transcribe_bytes(
                        chunk.caller_audio, f"c{chunk.index}.wav", language="en", prompt=terminology)
                    asr_calls += 1
            elif chunk.mono_rms >= SILENCE_RMS:
                caller_text = asr.transcribe_bytes(
                    chunk.mono_audio, f"m{chunk.index}.wav", language="en", prompt=terminology)
                asr_calls += 1
            asr_possible += 2 if chunk.stereo else 1

        for speaker, text in (("agent", agent_text), ("caller", caller_text)):
            if text.strip() and not is_artifact(text):
                entry = {"at_s": round(chunk.start_s, 1),
                         "speaker": speaker if chunk.stereo else "unattributed",
                         "text": text.strip()}
                transcript.append(entry)
                emit({"type": "transcript", **entry})
                unjudged_chars += len(text)

        # --- signal extraction ------------------------------------------------
        signals: list[Signal] = []
        full_agent = " ".join(t["text"] for t in transcript if t["speaker"] == "agent")
        recent_caller = " ".join(t["text"] for t in transcript[-4:] if t["speaker"] != "agent")

        with recorder.span("signals_rule", chunk=chunk.index):
            signals.extend(rule_signals(full_agent, recent_caller, at_seconds=chunk.start_s))

        if unjudged_chars >= JUDGE_EVERY_CHARS:
            window = "\n".join(
                f"{t['speaker'].upper()}: {t['text']}" for t in transcript[-JUDGE_WINDOW_TURNS:])
            signals.extend(model_signals(window, chunk.start_s, recorder=recorder))
            unjudged_chars = 0

        # --- nudge control ----------------------------------------------------
        with recorder.span("nudge_control", chunk=chunk.index):
            new_nudges: list[Nudge] = engine.offer(signals)

        for nudge in new_nudges:
            # End to end: the chunk arriving through to a nudge ready to display.
            latency_ms = (time.perf_counter() - chunk_received) * 1000
            end_to_end.append(latency_ms)
            recorder.record("end_to_end", latency_ms, nudge=nudge.nudge_id)
            payload = {**nudge.to_dict(), "latency_ms": round(latency_ms, 1)}
            emitted.append(payload)
            # Nested, not spread. The nudge dict has its own "type" field holding
            # the signal type, and spreading it overwrote the event envelope's
            # type - so every listener saw "compliance_gap" where it expected
            # "nudge" and silently ignored the event.
            emit({"type": "nudge", "nudge": payload})

        # Suppression counts go out on every tick, not only at the end. The
        # interesting number on this dashboard is how much was found and not
        # shown, and a panel reading "3 shown, 0 seen" mid-call reads as broken.
        emit({"type": "tick", "at_s": round(chunk.start_s, 1),
              "active": [n.to_dict() for n in engine.active()],
              "suppression": engine.stats.as_dict()})

    result = StreamResult(
        call_id=call_id,
        source=str(source),
        stereo=stereo,
        duration_s=round(duration, 1),
        chunks=len([t for t in transcript]),
        transcript=transcript,
        nudges=emitted,
        suppression={
            **engine.stats.as_dict(),
            "asr_calls_made": asr_calls,
            "asr_calls_skipped_silent": asr_possible - asr_calls,
        },
        latency=recorder.summary(),
        end_to_end_ms=[round(x, 1) for x in end_to_end],
    )
    emit({"type": "stream_end", "suppression": result.suppression, "latency": result.latency})
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m live.stream <audio-file> [--fast]")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    fast = "--fast" in sys.argv

    def printer(event: dict) -> None:
        if event["type"] == "transcript":
            print(f"  [{event['at_s']:6.1f}s] {event['speaker'][:6].upper():8s} {event['text'][:78]}")
        elif event["type"] == "nudge":
            n = event["nudge"]
            print(f"  >>> NUDGE [{n['type']}] {n['text']}"
                  f"  ({n['latency_ms']}ms, conf {n['confidence']})")

    result = process(path, realtime=not fast, on_event=printer)
    print("\n" + format_markdown_table(result.latency, "Latency"))
    print("\nsuppression:", json.dumps(result.suppression, indent=2))
