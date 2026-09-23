"""Run the live-processing test suite and produce the Q4 evidence.

Four scenarios, each checking something different:

  missed cross-sell   signals the agent talked past
  compliance gap      a disclosure skipped and a guarantee promised
  rising frustration  a judgement call rules cannot make
  noisy and ambiguous must produce nothing

The fourth matters most. Generating a useful nudge is not hard; staying quiet
when there is nothing to say is, and a system that chatters through a rambling
call is one an agent stops reading. It is scored as a pass only when it emits
nothing.

Every run is real time: audio is fed at the pace it plays, so the latency
numbers describe what an agent on a live call would experience rather than batch
throughput.

Run:  python -m scripts.run_live_tests
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from core import config
from core.timing import LatencyRecorder, format_markdown_table, percentile
from live.stream import process


COOLDOWN_S = 60


def run() -> dict:
    scenarios_path = config.EVIDENCE_DIR / "live_calls" / "scenarios.json"
    if not scenarios_path.exists():
        raise FileNotFoundError("No scenarios. Run `python -m scripts.make_stereo_calls` first.")
    scenarios = json.loads(scenarios_path.read_text())

    results = []
    all_latency = LatencyRecorder("live-suite")
    all_end_to_end: list[float] = []
    all_lag: list[float] = []

    for index, scenario in enumerate(scenarios):
        if index:
            # Each call is measured as a single live call runs. Back to back, four
            # calls exhaust the free tier's per-minute transcription quota and
            # later lines wait for it to clear - measured at up to sixteen seconds
            # late - which says something about the free tier, not about the
            # pipeline, and is reported as such rather than hidden in a P95.
            print(f"\n  (cooling down {COOLDOWN_S}s so the per-minute quota clears)")
            time.sleep(COOLDOWN_S)
        print(f"\n{'=' * 74}\n{scenario['name']}  ({scenario['duration_s']}s, real time)")
        print(f"  expect: {scenario['expect'] or 'NO NUDGES'}")

        def show(event: dict) -> None:
            if event["type"] == "nudge":
                n = event["nudge"]
                print(f"    >>> [{n['type']}] {n['text'][:62]}")
                print(f"        {n['latency_ms']}ms  conf={n['confidence']}  via={n['detector']}")

        result = process(config.ROOT / scenario["path"], call_id=scenario["id"],
                         realtime=True, on_event=show)

        fired = [n["type"] for n in result.nudges]
        expected = set(scenario["expect"])

        if not expected:
            passed = len(fired) == 0
            reason = ("stayed silent as required" if passed
                      else f"emitted {len(fired)} nudge(s) on a call with nothing to act on: {fired}")
        else:
            missing = expected - set(fired)
            passed = not missing
            reason = ("all expected signals detected" if passed
                      else f"missed {sorted(missing)}")

        print(f"  -> {'PASS' if passed else 'FAIL'}: {reason}")
        print(f"     nudges={len(fired)} from {result.suppression['signals_seen']} signals "
              f"({result.suppression['suppression_rate']:.0%} suppressed)")

        # Raw measurements, not per-run means repeated: percentiles of repeated
        # averages flatten the tail and understate P95.
        for stage, values in result.raw.items():
            for value in values:
                all_latency.record(stage, value)
        all_end_to_end.extend(result.end_to_end_ms)
        all_lag.extend(result.transcript_lag_ms)

        results.append({
            "id": scenario["id"],
            "name": scenario["name"],
            "why": scenario["why"],
            "audio": scenario["path"],
            "duration_s": scenario["duration_s"],
            "expected": sorted(expected),
            "fired": fired,
            "passed": passed,
            "reason": reason,
            "nudges": result.nudges,
            "transcript": result.transcript,
            "suppression": result.suppression,
            "latency": result.latency,
            "end_to_end_ms": result.end_to_end_ms,
            "transcript_lag_ms": result.transcript_lag_ms,
        })

    report = {
        "generated_at": date.today().isoformat(),
        "scenarios": results,
        "passed": sum(1 for r in results if r["passed"]),
        "total": len(results),
        "aggregate_latency": all_latency.summary(),
        "end_to_end": {
            "count": len(all_end_to_end),
            "p50_ms": round(percentile(all_end_to_end, 50), 1),
            "p95_ms": round(percentile(all_end_to_end, 95), 1),
            "max_ms": round(max(all_end_to_end), 1) if all_end_to_end else 0.0,
        },
        "transcript_lag": {
            "count": len(all_lag),
            "p50_ms": round(percentile(all_lag, 50), 1),
            "p95_ms": round(percentile(all_lag, 95), 1),
        },
    }

    out = config.EVIDENCE_DIR / "live_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (config.DOCS_DIR / "latency_report.md").write_text(to_markdown(report))

    print(f"\n{'=' * 74}")
    print(f"{report['passed']}/{report['total']} scenarios passed")
    print(f"end-to-end p50 {report['end_to_end']['p50_ms']}ms  "
          f"p95 {report['end_to_end']['p95_ms']}ms")
    print(f"transcript lag p50 {report['transcript_lag']['p50_ms']}ms  "
          f"p95 {report['transcript_lag']['p95_ms']}ms")
    print(f"-> {out}\n-> docs/latency_report.md")
    return report


def to_markdown(report: dict) -> str:
    e2e = report["end_to_end"]
    lines = [
        "# Question 4 — live processing, latency and suppression",
        "",
        f"Generated by `python -m scripts.run_live_tests` on {report['generated_at']}.",
        "",
        "Every run feeds audio at the pace it actually plays, and the pipeline never looks",
        "at audio that has not played yet, so these numbers are what an agent on a live",
        "call would experience rather than batch throughput.",
        "",
        "## How a call is processed",
        "",
        "Each channel runs its own voice activity detector over 100 ms frames. An utterance",
        "opens when the voice starts and closes after 600 ms of silence, and is then",
        "transcribed whole. Rules run on each finished utterance, with the conversation so",
        "far for context; the model judge runs when the customer finishes speaking and",
        "enough has been said to judge intent.",
        "",
        "An earlier version cut audio into fixed 4-second windows and started work on each",
        "window at the moment it *began*, which meant transcribing up to four seconds of",
        "speech that had not been heard yet. Text appeared before its words, a compliance",
        "nudge fired mid-sentence, and latency was measured from a moment that could",
        "precede the speech itself. The numbers below are measured from when the speaker",
        "actually stopped talking, so they are larger than the old ones and correct.",
        "",
        "## End-to-end latency",
        "",
        "From the speaker finishing the utterance that caused it to the nudge being ready",
        "to display: the 600 ms needed to be sure they have stopped, transcription, rules,",
        "nudge control, and for model-judged nudges the judge call.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Nudges measured | {e2e['count']} |",
        f"| **P50** | **{e2e['p50_ms']} ms** |",
        f"| **P95** | **{e2e['p95_ms']} ms** |",
        f"| Max | {e2e['max_ms']} ms |",
        "",
        "## Transcript lag",
        "",
        "From the speaker finishing to their words appearing. The dashboard opens the",
        "speaker's line the moment they start talking, so what trails is the text, not",
        "the line.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Utterances | {report['transcript_lag']['count']} |",
        f"| **P50** | **{report['transcript_lag']['p50_ms']} ms** |",
        f"| **P95** | **{report['transcript_lag']['p95_ms']} ms** |",
        "",
        "## Per-component latency",
        "",
        format_markdown_table(report["aggregate_latency"]),
        "",
        "Percentiles are over every individual measurement across all four calls.",
        "",
        "`asr` is a network round trip to a hosted Whisper endpoint. `signals_rule` and",
        "`nudge_control` are local and cost microseconds, which is why the rules run on",
        "every finished utterance while the model judge runs only when the customer has",
        "said enough to judge.",
        "",
        "## Scenario results",
        "",
        "| Scenario | Expected | Fired | Result |",
        "|---|---|---|---|",
    ]
    for r in report["scenarios"]:
        expected = ", ".join(r["expected"]) or "*nothing*"
        fired = ", ".join(r["fired"]) or "*nothing*"
        lines.append(f"| {r['name']} | {expected} | {fired} | {'PASS' if r['passed'] else 'FAIL'} |")

    lines += ["", f"**{report['passed']}/{report['total']} passed.**", ""]

    lines += ["## Suppression", "",
              "Detection is the easy half. These are the counts of what was found against what",
              "was actually shown:", "",
              "| Scenario | Signals seen | Nudges shown | Suppressed | Utterances transcribed |",
              "|---|---|---|---|---|"]
    for r in report["scenarios"]:
        s = r["suppression"]
        lines.append(
            f"| {r['name']} | {s['signals_seen']} | {s['nudges_emitted']} | "
            f"{s['suppression_rate']:.0%} | {s.get('utterances', 0)} |"
        )

    rewritten = sum(r["suppression"].get("unsafe_text_rewritten", 0) for r in report["scenarios"])
    dropped = sum(r["suppression"].get("unsafe_text_dropped", 0) for r in report["scenarios"])
    lines += [
        "",
        "## Nudge safety",
        "",
        "A nudge is advice given to a human mid-call, and on a regulated sales call some",
        "advice is itself the violation. This was not hypothetical. On the compliance",
        "scenario the model judge produced:",
        "",
        "> \"Provide a specific projected return or example to address the caller's interest.\"",
        "",
        "four seconds after a compliance rule had fired telling the agent that returns are",
        "never guaranteed. The model was being helpful about sales and had no way to know",
        "it was recommending the exact conduct the rule above exists to prevent.",
        "",
        "Model-written nudge text is now checked against prohibited-advice patterns before",
        "it can reach a human. A flagged nudge keeps its signal - the detection was correct,",
        "the caller really was interested - but its wording is replaced with vetted text.",
        "The model classifies; what a human is told comes from a reviewed source. Where no",
        "vetted wording exists for that signal type, the nudge is dropped, because advice",
        "that cannot be made safe is worse than silence.",
        "",
        f"In this run: **{rewritten} rewritten, {dropped} dropped.**",
        "",
        "The guard applies to model-generated text only, and scoping it that way was itself",
        "a fix. Checking every nudge flagged the rule-authored compliance text - *\"Guarantee",
        "language used. Correct it now: returns and approval are never guaranteed\"* - because",
        "it quotes the very word it exists to police. With no replacement defined for that",
        "type it would have been dropped silently, disabling the two most important nudges",
        "in the system in the name of safety.",
        "",
        "## False-positive analysis",
        "",
        "The noisy-and-ambiguous scenario is the false-positive test. It contains filler,",
        "hesitation, abandoned sentences and small talk about traffic - the shape of a real",
        "call where nothing actionable happens. A detector keyed on words alone fires",
        "repeatedly here.",
        "",
        "Observed sources of false positives while building this, and what each cost:",
        "",
        "- **Repeated firing on a cue still inside the window.** One mention of a spouse",
        "  produced a nudge on every subsequent window while those words remained in it.",
        "  Fixed by keying deduplication on the triggering evidence rather than on the",
        "  nudge text, since the model rewords the same observation each pass.",
        "- **Whisper hallucinating on silence.** An empty channel returns \"you\" or",
        "  \"Thank you.\", which entered the transcript as speech and then fed the detectors.",
        "  Segmenting by voice activity means only voiced audio is ever sent now.",
        "- **A disclosure flagged before it could have been given.** \"State that returns",
        "  are not guaranteed\" fired while the agent was still in the sentence that mentioned",
        "  investment. A disclosure now becomes a gap only when the agent speaks again",
        "  without it - the point at which a supervisor would actually step in.",
        "- **Ambiguous agreement read as frustration.** \"Fine, whatever\" and \"fine, that",
        "  works\" differ only in context. This is why frustration is judged by the model on",
        "  a window rather than matched as a keyword.",
        "",
        "## Behaviour at 10x scale",
        "",
        "One call issues one ASR request per utterance - roughly fifteen to twenty a",
        "minute on these recordings - so ten concurrent calls is about three requests a",
        "second against the free tier's twenty a minute per model. **The free tier is the",
        "binding constraint well before the architecture is**, and it is what would be",
        "replaced first.",
        "",
        "This is measured, not estimated. Run back to back with no gap, four calls",
        "exhausted the per-minute quota and individual transcripts arrived up to sixteen",
        "seconds after their words. Failing over to a second Whisper model with its own",
        "quota cut that to seven. With a one-minute gap between calls - which is how this",
        "suite runs, so each call is measured the way a single live call would be - no",
        "request was rate limited at all.",
        "",
        "What changes, in the order it would bite:",
        "",
        "1. **ASR throughput.** The fix is a streaming ASR connection per call rather than",
        "   one upload per utterance. That also closes the transcript lag: a streaming",
        "   recogniser returns words while they are spoken instead of after the pause.",
        "2. **Voice activity matters more, not less.** Only voiced audio is sent; on real",
        "   recordings with hold music and dead air the saving is larger.",
        "3. **Rules and nudge control do not move.** Both are local and measured in",
        "   microseconds, so they stay negligible at any realistic concurrency.",
        "4. **The model judge is the first thing to batch.** It already runs only when a",
        "   customer finishes speaking; across many calls it would batch across calls too.",
        "",
        "## Behaviour on degraded audio",
        "",
        "Untested and stated as such. These recordings are clean synthetic speech on",
        "separate channels. Real calls bring packet loss, codec compression, crosstalk and",
        "background noise, and all of them hit the transcription step, which everything",
        "downstream depends on.",
        "",
        "The specific risk is not that transcription gets worse in general. It is that a",
        "*partially* wrong transcript still reads as fluent, so a compliance rule can match",
        "on a phrase nobody said. The voice activity threshold would also need",
        "recalibrating: SPEECH_RMS is tuned against clean audio and would treat line noise",
        "on a real line as speech, which keeps utterances open and delays every transcript.",
        "",
        "Mono input is handled but attributed as `unattributed` rather than guessed, because",
        "every agent-side compliance rule depends on knowing who spoke. Claiming attribution",
        "the system does not have would make those rules quietly wrong instead of honestly",
        "unavailable.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
