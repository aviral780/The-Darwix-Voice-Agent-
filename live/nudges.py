"""Nudge control: deciding which signals an agent actually sees.

Detection is the easy half. A system that surfaces every signal it finds is
useless on a real call, because an agent mid-conversation can act on roughly one
prompt at a time and will start ignoring the panel entirely if it chatters. The
brief names this directly: excessive low-value alerts are a rejection condition.

Five controls, each for a failure observed while testing against real call
transcripts:

  confidence floor   a weak guess is worse than silence
  duplicate suppress the same signal re-fires on every window while the words
                     stay inside it, so one cross-sell cue became eleven nudges
  per-type cooldown  even genuinely new instances of the same type need spacing
  expiry             a nudge about something said ninety seconds ago is no
                     longer actionable and should disappear on its own
  concurrent cap     only the highest-priority few are shown at once

Compliance signals are deliberately exempt from the concurrent cap and given a
long cooldown rather than a short one: a missed disclosure that is suppressed to
reduce clutter is the one failure this system must not have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from live.signals import Signal, SignalType

# Below this a signal is discarded outright.
MIN_CONFIDENCE = 0.55
# Compliance is held to a lower bar: a missed disclosure costs more than a
# false alarm, which is the opposite of the trade-off for a cross-sell hint.
MIN_CONFIDENCE_COMPLIANCE = 0.40

# Seconds before the same signal type may fire again.
COOLDOWN_S = {
    SignalType.COMPLIANCE_GAP: 45.0,
    SignalType.RISKY_STATEMENT: 20.0,
    SignalType.RISING_FRUSTRATION: 40.0,
    SignalType.PAYMENT_DIFFICULTY: 60.0,
    SignalType.BUYING_SIGNAL: 30.0,
    SignalType.MISSED_CROSS_SELL: 60.0,
    SignalType.CALLBACK_NEEDED: 60.0,
    SignalType.TOPIC_SHIFT: 90.0,
}
DEFAULT_COOLDOWN_S = 45.0

EXPIRY_S = 90.0
MAX_CONCURRENT = 3


@dataclass
class Nudge:
    nudge_id: str
    type: SignalType
    text: str
    confidence: float
    evidence: str
    detector: str
    priority: int
    created_at: float
    at_seconds: float
    meta: dict = field(default_factory=dict)

    def age_s(self, now: float | None = None) -> float:
        return (now or time.time()) - self.created_at

    def expired(self, now: float | None = None) -> bool:
        return self.age_s(now) > EXPIRY_S

    def to_dict(self, now: float | None = None) -> dict:
        return {
            "id": self.nudge_id,
            "type": self.type.value,
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "detector": self.detector,
            "priority": self.priority,
            "age_s": round(self.age_s(now), 1),
            "at_seconds": round(self.at_seconds, 1),
            **({"disclosure": self.meta["disclosure"]} if "disclosure" in self.meta else {}),
        }


@dataclass
class SuppressionStats:
    """Why signals were dropped. Reported with the latency numbers, because
    'we generated 9 nudges' means nothing without 'from 47 signals'."""
    seen: int = 0
    low_confidence: int = 0
    duplicate: int = 0
    cooldown: int = 0
    emitted: int = 0

    def as_dict(self) -> dict:
        return {
            "signals_seen": self.seen,
            "dropped_low_confidence": self.low_confidence,
            "dropped_duplicate": self.duplicate,
            "dropped_cooldown": self.cooldown,
            "nudges_emitted": self.emitted,
            "suppression_rate": (
                round(1 - self.emitted / self.seen, 3) if self.seen else 0.0
            ),
        }


class NudgeEngine:
    def __init__(self, clock=time.time) -> None:
        # Injectable clock so replay tests can run faster than real time while
        # still exercising the real cooldown and expiry arithmetic.
        self._clock = clock
        self._last_fired: dict[SignalType, float] = {}
        self._seen_keys: set[str] = set()
        self._active: list[Nudge] = []
        self._history: list[Nudge] = []
        self.stats = SuppressionStats()
        self._counter = 0

    def _dedupe_key(self, signal: Signal) -> str:
        """What counts as 'the same nudge'.

        Keyed on type plus the triggering evidence rather than the nudge text,
        because the model rewords the same observation differently each window.
        Text-keyed deduplication let the same frustration cue through four times
        with four phrasings.
        """
        if "disclosure" in signal.meta:
            return f"{signal.type.value}:{signal.meta['disclosure']}"
        return f"{signal.type.value}:{signal.evidence.strip().lower()[:60]}"

    def offer(self, signals: list[Signal]) -> list[Nudge]:
        """Filter signals and return only the nudges that should be shown now."""
        now = self._clock()
        emitted: list[Nudge] = []

        # Highest priority first, so if a cooldown is about to be consumed it is
        # spent on the more important signal.
        for signal in sorted(signals, key=lambda s: -s.priority):
            self.stats.seen += 1

            floor = (MIN_CONFIDENCE_COMPLIANCE
                     if signal.type in (SignalType.COMPLIANCE_GAP, SignalType.RISKY_STATEMENT)
                     else MIN_CONFIDENCE)
            if signal.confidence < floor:
                self.stats.low_confidence += 1
                continue

            key = self._dedupe_key(signal)
            if key in self._seen_keys:
                self.stats.duplicate += 1
                continue

            cooldown = COOLDOWN_S.get(signal.type, DEFAULT_COOLDOWN_S)
            last = self._last_fired.get(signal.type)
            if last is not None and (now - last) < cooldown:
                self.stats.cooldown += 1
                continue

            self._counter += 1
            nudge = Nudge(
                nudge_id=f"n{self._counter:03d}",
                type=signal.type,
                text=signal.nudge,
                confidence=signal.confidence,
                evidence=signal.evidence,
                detector=signal.detector,
                priority=signal.priority,
                created_at=now,
                at_seconds=signal.at_seconds,
                meta=signal.meta,
            )
            self._seen_keys.add(key)
            self._last_fired[signal.type] = now
            self._active.append(nudge)
            self._history.append(nudge)
            emitted.append(nudge)
            self.stats.emitted += 1

        return emitted

    def active(self) -> list[Nudge]:
        """Currently displayable nudges: unexpired, ranked, capped.

        Compliance is exempt from the cap. Hiding a disclosure reminder to keep
        the panel tidy is the one trade-off this system should never make.
        """
        now = self._clock()
        self._active = [n for n in self._active if not n.expired(now)]
        ranked = sorted(self._active, key=lambda n: (-n.priority, n.created_at))

        critical_types = (SignalType.COMPLIANCE_GAP, SignalType.RISKY_STATEMENT)
        critical = [n for n in ranked if n.type in critical_types]
        others = [n for n in ranked if n.type not in critical_types]
        return critical + others[:MAX_CONCURRENT]

    def dismiss(self, nudge_id: str) -> bool:
        before = len(self._active)
        self._active = [n for n in self._active if n.nudge_id != nudge_id]
        return len(self._active) < before

    def history(self) -> list[Nudge]:
        return list(self._history)
