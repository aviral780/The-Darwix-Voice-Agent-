"""Real-time call processing: audio in, nudges out while the call is running.

The brief is explicit that analysing a finished recording does not qualify. This
consumes a call at the pace it plays and emits transcript and nudges as it goes.
Replaying a recording at real-time speed is the permitted form and is what the
test suite uses; the loop only ever looks at audio up to the current playback
position, so it behaves exactly as it would on a live line.

That last property is the one the first version got wrong. It cut audio into
fixed 4-second windows and started work on each window at the window's START,
which meant transcribing up to four seconds of speech that had not been played
yet. Text appeared before the words were heard and a compliance nudge fired
while the agent was still in the sentence that caused it. It also flattered the
latency numbers, because the clock started before the speech existed. A live
system cannot read ahead, so this one no longer does.

Segmentation is by speech, not by clock. Each channel runs its own voice
activity detector over 100 ms frames. An utterance opens when the voice starts
and closes after HANGOVER_S of silence, and only then is it transcribed - whole.
Fixed windows cut sentences in half wherever the boundary fell, and the halves
of one speaker's sentence ended up either side of the other speaker's line.

Speaker separation uses stereo channels rather than diarisation: agent left,
customer right, the way contact centre recorders lay out a call. Mono input
still works but is marked unattributed rather than guessed, because every
agent-side compliance rule depends on knowing who spoke.

Transcription and the model judge run in worker threads. The frame loop never
blocks on a network call, so the other speaker starting to talk while one
utterance is being transcribed is still noticed on time.
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from core import asr
from core.timing import LatencyRecorder, format_markdown_table
from live.nudges import Nudge, NudgeEngine
from live.signals import Signal, model_signals, rule_signals

SAMPLE_RATE = 16000
FRAME_S = 0.1

# Voice activity. Frames above this RMS count as speech. Tuned against clean
# recordings; a real phone line would need it recalibrated against line noise.
SPEECH_RMS = 180.0
# Two consecutive voiced frames open an utterance, so a click does not.
ONSET_FRAMES = 2
# Silence needed to close an utterance. Long enough to ride over the pause
# between two sentences in one turn, short enough that text lands soon after
# the speaker stops. This is the floor on transcript latency: nothing can be
# transcribed until the detector is sure the speaker has finished. At 0.45 s a
# third of turns were split at a sentence pause, each split costing a request
# against the free tier's twenty a minute; 0.6 s keeps most turns whole.
HANGOVER_S = 0.6
# A speaker who never pauses is still transcribed in pieces of at most this long.
MAX_UTTERANCE_S = 12.0
# Audio kept either side of the detected speech so the first and last
# syllables are not clipped.
PRE_ROLL_S = 0.15
POST_ROLL_S = 0.15

TICK_EVERY_S = 0.25
# How long to wait for the listener to confirm playback before starting anyway.
START_TIMEOUT_S = 10.0

# The model judge reads intent - frustration, buying interest - so it runs when
# the customer finishes speaking and enough has been said to judge.
JUDGE_MIN_CHARS = 100
JUDGE_WINDOW_TURNS = 6

ASR_WORKERS = 2

# Phrases Whisper emits for near-silence. The detector means segments are
# voiced audio now, so these should not occur; kept as a second line of defence.
SILENCE_ARTIFACTS = {
    "you", "thank you.", "thank you", "thanks for watching!", "thanks for watching",
    "bye.", "bye", ".", "...", "um", "uh", "so",
}


@dataclass
class Utterance:
    utt_id: str
    speaker: str
    start_s: float
    end_s: float = 0.0
    text: str = ""


@dataclass
class StreamResult:
    call_id: str
    source: str
    stereo: bool
    duration_s: float
    utterances: int
    transcript: list[dict] = field(default_factory=list)
    nudges: list[dict] = field(default_factory=list)
    suppression: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    end_to_end_ms: list[float] = field(default_factory=list)
    transcript_lag_ms: list[float] = field(default_factory=list)
    # Raw durations per stage, so a suite can compute real percentiles across
    # runs instead of percentiles of per-run averages.
    raw: dict[str, list[float]] = field(default_factory=dict)


# ── audio ────────────────────────────────────────────────────────────────


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


def _decode(path: Path, channel: int | None) -> np.ndarray:
    """Decode to 16 kHz mono int16, optionally one channel of a stereo file."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        filters = ["-af", f"pan=mono|c0=c{channel}"] if channel is not None else ["-ac", "1"]
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), *filters, "-ar", str(SAMPLE_RATE),
             "-acodec", "pcm_s16le", str(out)],
            capture_output=True, check=True,
        )
        with wave.open(str(out), "rb") as handle:
            return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def load_channels(path: Path) -> tuple[dict[str, np.ndarray], bool, float]:
    """Return {speaker: samples}, whether the source was stereo, and duration."""
    stereo = probe_channels(path) >= 2
    if stereo:
        channels = {"agent": _decode(path, 0), "caller": _decode(path, 1)}
    else:
        channels = {"unattributed": _decode(path, None)}
    duration = max(len(s) for s in channels.values()) / SAMPLE_RATE
    return channels, stereo, duration


def wav_bytes(samples: np.ndarray) -> bytes:
    """Wrap int16 samples in a WAV container for the ASR upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def is_artifact(text: str) -> bool:
    return text.strip().lower() in SILENCE_ARTIFACTS


# ── voice activity ───────────────────────────────────────────────────────


class ChannelVAD:
    """Opens and closes utterances on one channel, one frame at a time."""

    def __init__(self, speaker: str, samples: np.ndarray) -> None:
        self.speaker = speaker
        self.samples = samples
        self.frame_len = int(SAMPLE_RATE * FRAME_S)
        self.run = 0
        self.current: Utterance | None = None
        self.last_voiced_end = 0.0
        self.count = 0

    def _voiced(self, index: int) -> bool:
        frame = self.samples[index * self.frame_len:(index + 1) * self.frame_len]
        if frame.size == 0:
            return False
        level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return level >= SPEECH_RMS

    def step(self, index: int) -> list[tuple[str, Utterance]]:
        events: list[tuple[str, Utterance]] = []
        frame_end = (index + 1) * FRAME_S
        voiced = self._voiced(index)

        if self.current is None:
            self.run = self.run + 1 if voiced else 0
            if self.run >= ONSET_FRAMES:
                self.count += 1
                start = (index - ONSET_FRAMES + 1) * FRAME_S
                self.current = Utterance(
                    utt_id=f"{self.speaker[0]}{self.count}", speaker=self.speaker, start_s=round(start, 2))
                self.last_voiced_end = frame_end
                events.append(("start", self.current))
            return events

        if voiced:
            self.last_voiced_end = frame_end
        silent_for = frame_end - self.last_voiced_end
        if silent_for >= HANGOVER_S or frame_end - self.current.start_s >= MAX_UTTERANCE_S:
            events.append(("end", self._close()))
        return events

    def flush(self) -> list[tuple[str, Utterance]]:
        return [("end", self._close())] if self.current is not None else []

    def _close(self) -> Utterance:
        utt = self.current
        utt.end_s = round(self.last_voiced_end, 2)
        self.current = None
        self.run = 0
        return utt

    def segment(self, utt: Utterance) -> np.ndarray:
        start = max(0, int((utt.start_s - PRE_ROLL_S) * SAMPLE_RATE))
        end = min(len(self.samples), int((utt.end_s + POST_ROLL_S) * SAMPLE_RATE))
        return self.samples[start:end]


# ── pipeline ─────────────────────────────────────────────────────────────


def process(
    source: Path,
    call_id: str = "",
    realtime: bool = True,
    on_event: Callable[[dict], None] | None = None,
    start_gate: threading.Event | None = None,
    stop_flag: threading.Event | None = None,
    terminology: str = "premium, policy, beneficiary, rider, lapse, coverage, bancassurance",
) -> StreamResult:
    """Stream a call and emit transcript and nudges as it plays.

    realtime=True paces the loop to the audio's own clock, which is what makes
    this live processing rather than batch analysis. start_gate, when given, is
    waited on before the clock starts, so the pipeline and the listener's audio
    begin together. stop_flag ends the run early - a listener who leaves should
    not keep spending transcription calls on a call nobody is watching.
    """
    call_id = call_id or source.stem
    recorder = LatencyRecorder(f"live-{call_id}")
    engine = NudgeEngine()
    executor = ThreadPoolExecutor(max_workers=ASR_WORKERS)

    channels, stereo, duration = load_channels(source)
    vads = [ChannelVAD(speaker, samples) for speaker, samples in channels.items()]
    total_frames = int(np.ceil(duration / FRAME_S))

    transcript: list[dict] = []
    turns: list[tuple[str, str]] = []
    emitted: list[dict] = []
    end_to_end: list[float] = []
    transcript_lag: list[float] = []
    pending: list[tuple[Utterance, ChannelVAD, Future, float]] = []
    judging: list[tuple[Future, Utterance, float]] = []
    unjudged = 0
    asr_calls = 0

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    def stopped() -> bool:
        return stop_flag is not None and stop_flag.is_set()

    emit({"type": "ready", "call_id": call_id, "stereo": stereo, "duration_s": round(duration, 1)})
    if start_gate is not None:
        start_gate.wait(timeout=START_TIMEOUT_S)
    if stopped():
        executor.shutdown(wait=False, cancel_futures=True)
        return _result(call_id, source, stereo, duration, transcript, emitted, engine,
                       asr_calls, recorder, end_to_end, transcript_lag)

    emit({"type": "stream_start", "call_id": call_id, "source": str(source), "stereo": stereo})
    wall_start = time.perf_counter()

    def speech_end_wall(utt: Utterance, closed_wall: float) -> float:
        # Latency is measured from the moment the speaker stopped talking. In
        # real time that is a point on the audio clock; in fast mode there is no
        # audio clock, so the moment the utterance was closed stands in for it.
        return wall_start + utt.end_s if realtime else closed_wall

    def transcribe(samples: np.ndarray, name: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        text = asr.transcribe_bytes(wav_bytes(samples), name, language="en", prompt=terminology)
        return text, (time.perf_counter() - t0) * 1000

    def judge(window: str, at_s: float) -> tuple[list[Signal], float]:
        t0 = time.perf_counter()
        return model_signals(window, at_s), (time.perf_counter() - t0) * 1000

    def publish(new: list[Nudge], utt: Utterance, reference: float) -> None:
        for nudge in new:
            latency_ms = (time.perf_counter() - reference) * 1000
            end_to_end.append(latency_ms)
            recorder.record("end_to_end", latency_ms, nudge=nudge.nudge_id)
            payload = {**nudge.to_dict(), "latency_ms": round(latency_ms, 1)}
            emitted.append(payload)
            # Nested, not spread: the nudge has its own "type" field.
            emit({"type": "nudge", "nudge": payload, "at_s": utt.end_s})
        if new:
            tick(utt.end_s)

    def tick(at_s: float) -> None:
        emit({"type": "tick", "at_s": round(at_s, 2),
              "active": [n.to_dict() for n in engine.active()],
              "suppression": engine.stats.as_dict()})

    def handle_transcribed(utt: Utterance, future: Future, closed_wall: float) -> None:
        nonlocal unjudged
        try:
            text, asr_ms = future.result()
            recorder.record("asr", asr_ms, utt=utt.utt_id)
        except Exception as exc:
            print(f"asr failed for {utt.utt_id}: {type(exc).__name__}: {exc}")
            text = ""

        text = (text or "").strip()
        if not text or is_artifact(text):
            emit({"type": "utterance_empty", "utt_id": utt.utt_id})
            return

        utt.text = text
        reference = speech_end_wall(utt, closed_wall)
        lag_ms = (time.perf_counter() - reference) * 1000
        transcript_lag.append(lag_ms)
        recorder.record("transcript_lag", lag_ms, utt=utt.utt_id)

        entry = {"utt_id": utt.utt_id, "speaker": utt.speaker, "text": text,
                 "start_s": utt.start_s, "end_s": utt.end_s, "at_s": utt.end_s}
        transcript.append(entry)
        emit({"type": "transcript", **entry})

        turns.append((utt.speaker, text))
        unjudged += len(text)
        is_agent = utt.speaker == "agent"

        # Rules read the utterance that just finished, with the conversation so
        # far for disclosure timing. Each nudge carries the id of the line that
        # caused it, so the dashboard can mark that line.
        with recorder.span("signals_rule", utt=utt.utt_id):
            signals = rule_signals(text if is_agent else "", "" if is_agent else text,
                                   at_seconds=utt.end_s, turns=turns)
        for signal in signals:
            signal.meta["utt"] = utt.utt_id
        with recorder.span("nudge_control", utt=utt.utt_id):
            new = engine.offer(signals)
        publish(new, utt, reference)

        if not is_agent and unjudged >= JUDGE_MIN_CHARS:
            unjudged = 0
            window = "\n".join(
                f"{'AGENT' if s == 'agent' else 'CALLER'}: {t}" for s, t in turns[-JUDGE_WINDOW_TURNS:])
            judging.append((executor.submit(judge, window, utt.end_s), utt, reference))

    def handle_judged(future: Future, utt: Utterance, reference: float) -> None:
        try:
            signals, judge_ms = future.result()
            recorder.record("signal_llm", judge_ms, utt=utt.utt_id)
        except Exception as exc:
            print(f"judge failed after {utt.utt_id}: {type(exc).__name__}: {exc}")
            return
        for signal in signals:
            signal.meta["utt"] = utt.utt_id
        with recorder.span("nudge_control", utt=utt.utt_id):
            new = engine.offer(signals)
        publish(new, utt, reference)

    def drain(block: bool = False) -> None:
        # Transcripts are handled in the order utterances closed, so rules see
        # the conversation in the order it was spoken even when two transcription
        # calls finish out of order.
        while pending and (block or pending[0][2].done()):
            utt, vad, future, closed_wall = pending.pop(0)
            handle_transcribed(utt, future, closed_wall)
        for item in [j for j in judging if block or j[0].done()]:
            judging.remove(item)
            handle_judged(*item)

    def close(utt: Utterance, vad: ChannelVAD) -> None:
        nonlocal asr_calls
        asr_calls += 1
        future = executor.submit(transcribe, vad.segment(utt), f"{utt.utt_id}.wav")
        pending.append((utt, vad, future, time.perf_counter()))

    next_tick = 0.0
    for index in range(total_frames):
        if stopped():
            break
        frame_end = (index + 1) * FRAME_S

        # Never look at audio before it has played.
        if realtime:
            wait = wall_start + frame_end - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

        for vad in vads:
            for kind, utt in vad.step(index):
                if kind == "start":
                    emit({"type": "speech_start", "utt_id": utt.utt_id, "speaker": utt.speaker,
                          "start_s": utt.start_s, "at_s": utt.start_s})
                else:
                    close(utt, vad)

        drain()
        if frame_end >= next_tick:
            tick(frame_end)
            next_tick = frame_end + TICK_EVERY_S

    if not stopped():
        for vad in vads:
            for _, utt in vad.flush():
                close(utt, vad)
        # Finish whatever is still being transcribed or judged, in order.
        while pending or judging:
            drain(block=True)
        tick(duration)
    executor.shutdown(wait=not stopped(), cancel_futures=stopped())

    result = _result(call_id, source, stereo, duration, transcript, emitted, engine,
                     asr_calls, recorder, end_to_end, transcript_lag)
    emit({"type": "stream_end", "suppression": result.suppression, "latency": result.latency})
    return result


def _result(call_id, source, stereo, duration, transcript, emitted, engine,
            asr_calls, recorder, end_to_end, transcript_lag) -> StreamResult:
    return StreamResult(
        call_id=call_id,
        source=str(source),
        stereo=stereo,
        duration_s=round(duration, 1),
        utterances=len(transcript),
        transcript=transcript,
        nudges=emitted,
        suppression={**engine.stats.as_dict(), "utterances": len(transcript), "asr_calls": asr_calls},
        latency=recorder.summary(),
        end_to_end_ms=[round(x, 1) for x in end_to_end],
        transcript_lag_ms=[round(x, 1) for x in transcript_lag],
        raw={stage: recorder.durations(stage) for stage in recorder.stages()},
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m live.stream <audio-file> [--fast]")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    fast = "--fast" in sys.argv

    def printer(event: dict) -> None:
        if event["type"] == "speech_start":
            print(f"  [{event['start_s']:5.1f}s] {event['speaker'][:6].upper():7s} ...speaking")
        elif event["type"] == "transcript":
            print(f"  [{event['end_s']:5.1f}s] {event['speaker'][:6].upper():7s} {event['text'][:74]}")
        elif event["type"] == "nudge":
            n = event["nudge"]
            print(f"      >>> NUDGE [{n['type']}] {n['text'][:60]}  ({n['latency_ms']}ms)")

    result = process(path, realtime=not fast, on_event=printer)
    print("\n" + format_markdown_table(result.latency, "Latency"))
    print("\nsuppression:", json.dumps(result.suppression, indent=2))
