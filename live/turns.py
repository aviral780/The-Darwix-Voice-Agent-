"""Nudge generation for a turn-based conversation.

live/stream.py consumes fixed audio chunks and has to work out who spoke from
stereo channels. A live call does not need any of that: the agent's words are
generated here and the caller's come back from the transcriber already
attributed, so speaker labelling is exact rather than inferred.

That makes this the better path for the two live surfaces. Same detectors, same
suppression, same safety check as the recorded pipeline - only the segmentation
differs, turns instead of chunks.

Rules run on every turn because they cost microseconds. The model judge runs only
once enough new speech has accumulated, and always off the critical path: the
caller has already heard their answer by the time it runs, so its latency costs
the conversation nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.timing import LatencyRecorder
from live.nudges import Nudge, NudgeEngine
from live.signals import Signal, model_signals, rule_signals

JUDGE_EVERY_CHARS = 160
WINDOW_TURNS = 6


@dataclass
class Turn:
    speaker: str      # agent | caller
    text: str
    at_s: float = 0.0


@dataclass
class TurnNudger:
    """Accumulates a conversation and emits nudges as it develops."""

    label: str = "live"
    turns: list[Turn] = field(default_factory=list)
    _unjudged: int = 0

    def __post_init__(self) -> None:
        self.engine = NudgeEngine()
        self.recorder = LatencyRecorder(f"nudge-{self.label}")

    # --- transcript ---------------------------------------------------------

    def add(self, speaker: str, text: str, at_s: float = 0.0) -> None:
        if text.strip():
            self.turns.append(Turn(speaker=speaker, text=text.strip(), at_s=at_s))
            self._unjudged += len(text)

    @property
    def agent_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.speaker == "agent")

    @property
    def recent_caller_text(self) -> str:
        return " ".join(t.text for t in self.turns[-4:] if t.speaker == "caller")

    # --- detection ----------------------------------------------------------

    def run_rules(self, at_s: float = 0.0) -> list[Nudge]:
        """Deterministic detectors. Microseconds, so safe on every turn."""
        with self.recorder.span("signals_rule"):
            signals: list[Signal] = rule_signals(
                self.agent_text, self.recent_caller_text, at_seconds=at_s)
        with self.recorder.span("nudge_control"):
            return self.engine.offer(signals)

    def should_judge(self) -> bool:
        return self._unjudged >= JUDGE_EVERY_CHARS

    def run_judge(self, at_s: float = 0.0) -> list[Nudge]:
        """Model judgement over the recent window. Call this off the hot path."""
        if not self.turns:
            return []
        self._unjudged = 0
        window = "\n".join(
            f"{t.speaker.upper()}: {t.text}" for t in self.turns[-WINDOW_TURNS:])
        signals = model_signals(window, at_s, recorder=self.recorder)
        with self.recorder.span("nudge_control"):
            return self.engine.offer(signals)

    # --- output -------------------------------------------------------------

    def active(self) -> list[dict]:
        return [n.to_dict() for n in self.engine.active()]

    def stats(self) -> dict:
        return self.engine.stats.as_dict()

    def latency(self) -> dict:
        return self.recorder.summary()
