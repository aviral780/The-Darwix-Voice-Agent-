"""Latency measurement.

Question 4 is graded on measured latency, not on claims about it, so timing
is built in rather than bolted on. Every pipeline stage opens a span; the
recorder aggregates spans into per-stage and end-to-end percentiles that get
written straight into the latency report.

Percentiles are computed with linear interpolation (the same method numpy
uses) so that small sample counts don't quantise to a single observation.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Span:
    """One timed stage. Duration is filled in when the context manager exits."""

    stage: str
    started_at: float
    duration_ms: float = 0.0
    meta: dict = field(default_factory=dict)


class LatencyRecorder:
    """Collects spans and reports percentiles per stage.

    Not thread-safe by design: each call/session gets its own recorder, which
    keeps concurrent calls from polluting each other's numbers.
    """

    def __init__(self, label: str = "default") -> None:
        self.label = label
        self._spans: list[Span] = []

    @contextmanager
    def span(self, stage: str, **meta) -> Iterator[Span]:
        span = Span(stage=stage, started_at=time.perf_counter(), meta=meta)
        try:
            yield span
        finally:
            span.duration_ms = (time.perf_counter() - span.started_at) * 1000.0
            self._spans.append(span)

    def record(self, stage: str, duration_ms: float, **meta) -> None:
        """Record a stage measured elsewhere (e.g. a client-reported display time)."""
        self._spans.append(
            Span(stage=stage, started_at=time.perf_counter(), duration_ms=duration_ms, meta=meta)
        )

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def durations(self, stage: str) -> list[float]:
        return [s.duration_ms for s in self._spans if s.stage == stage]

    def stages(self) -> list[str]:
        # dict preserves insertion order, so stages come back in the order first seen,
        # which is the pipeline order and reads correctly in the report.
        return list(dict.fromkeys(s.stage for s in self._spans))

    def summary(self) -> dict:
        out = {}
        for stage in self.stages():
            values = self.durations(stage)
            out[stage] = {
                "count": len(values),
                "p50_ms": round(percentile(values, 50), 1),
                "p95_ms": round(percentile(values, 95), 1),
                "min_ms": round(min(values), 1),
                "max_ms": round(max(values), 1),
                "mean_ms": round(sum(values) / len(values), 1),
            }
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "summary": self.summary(),
            "raw_spans": [
                {"stage": s.stage, "duration_ms": round(s.duration_ms, 2), **s.meta}
                for s in self._spans
            ],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def merge(self, other: "LatencyRecorder") -> None:
        """Fold another recorder's spans in. Used to aggregate per-call recorders
        into one report across a whole test run."""
        self._spans.extend(other._spans)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile. Returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def format_markdown_table(summary: dict, title: str = "") -> str:
    """Render a summary as a markdown table for the latency report."""
    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| Stage | Count | P50 (ms) | P95 (ms) | Mean (ms) | Max (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for stage, stats in summary.items():
        lines.append(
            f"| {stage} | {stats['count']} | {stats['p50_ms']} | "
            f"{stats['p95_ms']} | {stats['mean_ms']} | {stats['max_ms']} |"
        )
    return "\n".join(lines)
