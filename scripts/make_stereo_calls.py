"""Generate stereo call recordings for the live-processing tests.

Agent on the left channel, customer on the right, which is how contact centre
recorders lay out a call and what gives live/stream.py true speaker attribution
instead of a diarisation guess.

Each turn is rendered to its own channel with matched silence on the other, so
the two tracks stay aligned and a chunk taken at any offset contains the right
speaker's audio on the right channel.

The four scenarios exist to exercise the four cases the brief names, including
the one that must produce nothing: a noisy, rambling call where a system that
fires on keywords alone would bury the agent in useless prompts.

Run:  python -m scripts.make_stereo_calls
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from core import config, tts

AGENT_VOICE = "en-PH-RosaNeural"
CALLER_VOICE = "en-US-GuyNeural"

SCENARIOS = [
    {
        "id": "live_01_missed_cross_sell",
        "name": "Missed cross-sell opportunity",
        "expect": ["missed_cross_sell"],
        "why": "the customer names a family and a health scare; both are openings the agent talks past",
        "turns": [
            ("agent", "Good morning, this is Maya from Pru Life UK. How can I help you today?"),
            ("caller", "Hi, I just want to check the status of my policy payment."),
            ("agent", "Of course, I can help with that. Let me pull up your record."),
            ("caller", "Thanks. My wife has been asking about it, and we have two kids in school now."),
            ("agent", "Right, I can see the payment went through last week."),
            ("caller", "Good. My father was in hospital last month, it made me think about all this."),
            ("agent", "I understand. Is there anything else you need today?"),
            ("caller", "No, that's all. Thank you."),
        ],
    },
    {
        "id": "live_02_compliance_gap",
        "name": "Skipped disclosure and risky statement",
        "expect": ["compliance_gap", "risky_statement"],
        "why": "the agent pitches an investment product without the required disclosure, then promises guaranteed returns",
        "turns": [
            ("agent", "Good afternoon! I wanted to tell you about our PRULink investment funds."),
            ("caller", "Okay, I'm listening."),
            ("agent", "These funds have shown very strong growth and I would recommend the equity fund for you."),
            ("caller", "How much could I make?"),
            ("agent", "Honestly, it is guaranteed returns of around ten percent. There is no risk at all."),
            ("caller", "That sounds too good to be true."),
            ("agent", "Trust me, you will definitely be approved and it will definitely pay out."),
            ("caller", "Let me think about it."),
        ],
    },
    {
        "id": "live_03_rising_frustration",
        "name": "Rising frustration",
        "expect": ["rising_frustration"],
        "why": "the customer repeats themselves and escalates in tone; the agent keeps talking past them",
        "turns": [
            ("agent", "Thanks for calling, how can I help?"),
            ("caller", "I have called three times already about this claim."),
            ("agent", "Let me check the reference number for you."),
            ("caller", "I gave the reference number to the last two people I spoke to."),
            ("agent", "I understand. Could you confirm it once more for me?"),
            ("caller", "This is honestly ridiculous. Nobody ever calls me back and I keep repeating myself."),
            ("agent", "I do apologise for the inconvenience."),
            ("caller", "Fine, whatever. Just tell me when it will be sorted."),
        ],
    },
    {
        "id": "live_04_noisy_ambiguous",
        "name": "Noisy and ambiguous call",
        "expect": [],
        "why": "small talk, hesitation and half-finished sentences; a keyword-triggered system would fire repeatedly here and it must not",
        "turns": [
            ("agent", "Good morning, Maya speaking."),
            ("caller", "Hi, sorry, one second, the signal here is not great."),
            ("agent", "No problem, take your time."),
            ("caller", "Yeah so, um, I was just, well, I wanted to ask about, hold on."),
            ("agent", "Take your time, I'm here."),
            ("caller", "Sorry about that. The traffic today is unbelievable. Anyway, where was I."),
            ("agent", "You were about to ask me something."),
            ("caller", "Right, yes, it was, hmm, actually I think I already sorted it out. Never mind."),
            ("agent", "Not a problem at all. Anything else?"),
            ("caller", "No, that's it. Thanks, bye."),
        ],
    },
]


def synth(text: str, voice: str, path: Path) -> float:
    """Synthesize one turn and return its duration in seconds."""
    path.write_bytes(tts.synthesize(text, voice=voice))
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip() or 0.0)


def silence(seconds: float, path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", f"{seconds:.3f}", "-acodec", "libmp3lame", str(path)],
        capture_output=True, check=True,
    )
    return path


def concat(files: list[Path], destination: Path, tmp: Path) -> Path:
    listing = tmp / f"{destination.stem}_list.txt"
    listing.write_text("\n".join(f"file '{f}'" for f in files))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-acodec", "libmp3lame", str(destination)],
        capture_output=True, check=True,
    )
    return destination


def build(scenario: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        agent_track: list[Path] = []
        caller_track: list[Path] = []

        for i, (speaker, text) in enumerate(scenario["turns"]):
            voice = AGENT_VOICE if speaker == "agent" else CALLER_VOICE
            turn_file = tmp / f"{i:02d}_{speaker}.mp3"
            duration = synth(text, voice, turn_file)
            gap = silence(duration, tmp / f"{i:02d}_gap.mp3")
            if speaker == "agent":
                agent_track.append(turn_file)
                caller_track.append(gap)
            else:
                agent_track.append(gap)
                caller_track.append(turn_file)

        left = concat(agent_track, tmp / "left.mp3", tmp)
        right = concat(caller_track, tmp / "right.mp3", tmp)

        destination = out_dir / f"{scenario['id']}.wav"
        # amerge puts the two mono inputs on separate channels of one stereo file.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(left), "-i", str(right),
             "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]", "-map", "[a]",
             "-ac", "2", "-ar", "16000", "-acodec", "pcm_s16le", str(destination)],
            capture_output=True, check=True,
        )

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=channels",
         "-of", "json", str(destination)],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "why": scenario["why"],
        "expect": scenario["expect"],
        "path": str(destination.relative_to(config.ROOT)),
        "duration_s": round(float(info["format"]["duration"]), 1),
        "channels": info["streams"][0]["channels"],
        "turns": scenario["turns"],
    }


def run() -> list[dict]:
    out_dir = config.EVIDENCE_DIR / "live_calls"
    built = []
    for scenario in SCENARIOS:
        print(f"building {scenario['id']} ...", end=" ", flush=True)
        info = build(scenario, out_dir)
        print(f"{info['duration_s']}s, {info['channels']}ch")
        built.append(info)
    (out_dir / "scenarios.json").write_text(json.dumps(built, indent=2))
    return built


if __name__ == "__main__":
    run()
