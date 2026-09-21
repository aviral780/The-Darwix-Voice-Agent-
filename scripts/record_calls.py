"""Record the required test calls as audio, transcripts and results.

The brief asks for at least three recorded calls covering a cooperative
customer, an objection, incomplete or conflicting details, an out-of-scope
question and a human-assistance request. Five are recorded, one per scenario.

Both sides are synthesised. The agent speaks in its market voice; the caller
speaks in a different voice so the recording is followable, and the two are
concatenated into one MP3 per call. Scripting the caller is what makes these
reproducible: the same five calls can be re-run after any change and compared,
which a live human reading lines into a microphone cannot give.

What is NOT simulated is the part being tested. Every agent turn runs the real
state machine, the real retrieval and the real model. Only the caller is
scripted.

Run:  python -m scripts.record_calls [market_key]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from core import config, tts
from core.timing import LatencyRecorder, format_markdown_table
from agent.actions import finalise_call
from agent.flow import CallSession

# A distinctly different voice for the caller so turns are tellable apart.
CALLER_VOICE = "en-US-GuyNeural"

SCENARIOS = [
    {
        "id": "01_cooperative",
        "name": "Cooperative customer",
        "why": "the baseline path: caller engages, every slot fills, lead is created",
        "lines": [
            "Yes, I have a few minutes.",
            "My name is Marco Reyes.",
            "I'm in my thirties.",
            "Yes, my wife and our two kids depend on me.",
            "No, I don't have any insurance at the moment.",
            "I think protection is more important for me right now.",
            "Afternoon would be better for me.",
        ],
    },
    {
        "id": "02_objection",
        "name": "Price objection",
        "why": "caller pushes back on cost; agent must acknowledge and answer from the knowledge base, not improvise a price",
        "lines": [
            "Okay, go ahead.",
            "I'm Elena Cruz.",
            "Forties.",
            "Honestly, insurance is too expensive for me right now.",
            "Yes, one son in college.",
            "Only what my employer provides.",
            "Something that also builds savings, maybe.",
            "Morning is fine.",
        ],
    },
    {
        "id": "03_incomplete_conflicting",
        "name": "Incomplete and conflicting details",
        "why": "caller is vague, contradicts themselves, then stonewalls; the agent must reprompt, take the corrected value, and eventually skip a slot rather than loop",
        "lines": [
            "I guess so.",
            "Hmm.",
            "Does it really matter?",
            "Fine, it's Dan.",
            "I'm in my twenties. Actually no, I turned thirty-one last month.",
            "I don't know, maybe.",
            "Not sure.",
            "Look, I really can't say.",
            "Yes I have insurance. Well, I had it, but it lapsed last year.",
            "Whenever, I don't mind.",
        ],
    },
    {
        "id": "04_out_of_scope",
        "name": "Out-of-scope question",
        "why": "the hallucination test: the agent must refuse rather than answer from outside the knowledge base",
        "lines": [
            "Sure, okay.",
            "Paolo Santos.",
            "Late twenties.",
            "What's the weather going to be like in Cebu tomorrow?",
            "Okay. What's the interest rate on a personal loan at BPI?",
            "Fine. What is my current policy account balance?",
            "Alright, no dependents yet.",
            "Morning works.",
        ],
    },
    {
        "id": "05_human_request",
        "name": "Human assistance request",
        "why": "caller asks for a person; escalation must fire immediately and write a handover record",
        "lines": [
            "Yes, briefly.",
            "It's Grace Lim.",
            "Fifties.",
            "Actually, can I just speak to a real person please?",
        ],
    },
]


def concat_mp3(parts: list[bytes], destination: Path) -> bool:
    """Join MP3 segments into one file using ffmpeg's concat demuxer."""
    if not parts:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = []
        for i, blob in enumerate(parts):
            piece = tmp_path / f"{i:03d}.mp3"
            piece.write_bytes(blob)
            files.append(piece)
        listing = tmp_path / "list.txt"
        listing.write_text("\n".join(f"file '{f}'" for f in files))
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(destination)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"    ffmpeg failed: {result.stderr.decode()[:200]}")
            return False
    return True


def record(scenario: dict, market_key: str) -> dict:
    session = CallSession(market_key=market_key)
    audio_parts: list[bytes] = []

    print(f"\n  {scenario['name']}")
    greeting = session.open()
    print(f"    AGENT : {greeting[:88]}")
    audio_parts.append(tts.synthesize(greeting, voice=session.market.tts_voice,
                                      recorder=session.recorder, stage="tts"))

    for line in scenario["lines"]:
        print(f"    CALLER: {line[:88]}")
        audio_parts.append(tts.synthesize(line, voice=CALLER_VOICE))

        turn = session.handle(line)
        if not turn.text:
            continue

        marker = ""
        if turn.citations:
            marker = f"   [cited {', '.join(turn.citations)}]"
        elif turn.grounded:
            marker = "   [grounded]"
        elif turn.refused:
            marker = f"   [refused at {turn.refusal_gate} gate]"
        print(f"    AGENT : {turn.text[:88]}{marker}")

        audio_parts.append(tts.synthesize(turn.text, voice=session.market.tts_voice,
                                          recorder=session.recorder, stage="tts"))
        if session.state.value in ("ended", "escalated"):
            break

    summary = session.summary()
    actions = finalise_call(summary)

    calls_dir = config.EVIDENCE_DIR / "calls"
    audio_path = calls_dir / f"{scenario['id']}_{market_key}.mp3"
    has_audio = concat_mp3(audio_parts, audio_path)

    transcripts = config.EVIDENCE_DIR / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    (transcripts / f"{scenario['id']}_{market_key}.txt").write_text(
        f"{scenario['name']}\n{scenario['why']}\n"
        f"market: {market_key}   call_id: {session.call_id}\n"
        f"{'=' * 78}\n{session.transcript_text()}\n"
    )
    session.save(transcripts)

    print(f"    -> {summary['final_state']}, "
          f"{len(summary['slots'])}/{summary['slots_total']} slots, "
          f"{summary['grounded_answers']} grounded, {summary['refusals']} refusals")

    return {
        "scenario": scenario["id"],
        "name": scenario["name"],
        "why": scenario["why"],
        "call_id": session.call_id,
        "market": market_key,
        "audio": str(audio_path.relative_to(config.ROOT)) if has_audio else "",
        "transcript": f"evidence/transcripts/{scenario['id']}_{market_key}.txt",
        "final_state": summary["final_state"],
        "slots": summary["slots"],
        "slots_filled": f"{len(summary['slots'])}/{summary['slots_total']}",
        "objections_raised": summary["objections_raised"],
        "grounded_answers": summary["grounded_answers"],
        "refusals": summary["refusals"],
        "escalated": summary["escalated"],
        "actions": actions,
        "latency": summary["latency"],
        "turns": summary["turns"],
    }


def run(market_key: str = "en_PH") -> list[dict]:
    print(f"recording {len(SCENARIOS)} calls for market {market_key}")
    results = [record(s, market_key) for s in SCENARIOS]

    combined = LatencyRecorder(f"calls-{market_key}")
    for result in results:
        for stage, stats in result["latency"].items():
            # Reconstruct spans from the per-call summaries so the aggregate
            # percentiles are computed across every turn of every call.
            for _ in range(stats["count"]):
                combined.record(stage, stats["mean_ms"])

    out = config.EVIDENCE_DIR / f"call_results_{market_key}.json"
    out.write_text(json.dumps({"market": market_key, "calls": results}, indent=2, ensure_ascii=False))
    (config.DOCS_DIR / f"q1_call_results_{market_key}.md").write_text(to_markdown(results, market_key))

    print(f"\n{'=' * 70}")
    print(f"{len(results)} calls recorded -> evidence/calls/, evidence/transcripts/")
    print(f"results -> {out}")
    return results


def to_markdown(results: list[dict], market_key: str) -> str:
    lines = [
        f"# Question 1 — recorded test calls ({market_key})",
        "",
        "Generated by `python -m scripts.record_calls`.",
        "",
        "Five calls covering the coverage the brief asks for: a cooperative customer, an",
        "objection, incomplete and conflicting details, an out-of-scope question and a",
        "human-assistance request.",
        "",
        "The caller side is scripted and synthesised so the suite is reproducible and can",
        "be re-run after any change. Every agent turn is the real state machine, the real",
        "retrieval and the real model.",
        "",
        "| # | Scenario | Final state | Slots | Grounded | Refusals | Escalated | Audio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        audio = f"[mp3]({r['audio']})" if r["audio"] else "—"
        lines.append(
            f"| {i} | {r['name']} | {r['final_state']} | {r['slots_filled']} | "
            f"{r['grounded_answers']} | {r['refusals']} | {'yes' if r['escalated'] else 'no'} | {audio} |"
        )

    for r in results:
        lines += [
            "", f"## {r['name']}", "",
            f"*{r['why']}*", "",
            f"- Call ID: `{r['call_id']}`",
            f"- Transcript: [`{r['transcript']}`]({r['transcript']})",
            f"- Slots collected: `{r['slots']}`",
        ]
        if r["objections_raised"]:
            lines.append(f"- Objections raised: {', '.join(r['objections_raised'])}")
        if r["actions"].get("lead"):
            lead = r["actions"]["lead"]
            lines.append(f"- Lead created: **{lead['band']}** (score {lead['score']})")
        if r["actions"].get("escalation"):
            lines.append("- Escalation record written for human handover")
        lines += ["", "```"]
        for turn in r["turns"]:
            who = "CALLER" if turn["speaker"] == "caller" else "AGENT "
            mark = ""
            if turn.get("citations"):
                mark = f"   [cited: {', '.join(turn['citations'])}]"
            elif turn.get("refused"):
                mark = f"   [refused at {turn['refusal_gate']} gate]"
            elif turn.get("grounded"):
                mark = "   [grounded in knowledge base]"
            lines.append(f"{who}: {turn['text']}{mark}")
        lines += ["```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "en_PH")
