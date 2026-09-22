"""Check that a localised call never switches to English unexpectedly.

The brief names unexpected English switching as a failure for Question 3, and
the first recording of these markets did exactly that: escalation and refusal
were hardcoded English in the flow, so a call conducted entirely in Bahasa
answered in English at the handover.

Fixing the four strings is not enough on its own, because the next hardcoded
line would reintroduce it silently. This scans every line the agent can speak -
the market config and the literals in flow.py - for English markers, and drives
a scripted call in each market to check the turns actually produced.

Run:  python -m scripts.check_language_leak
"""

from __future__ import annotations

import re
import sys

from core.config import MarketConfig, available_markets

# Words that are English rather than borrowed finance vocabulary. Terms like
# "premium", "policy" and "tenor" are genuinely used in both markets and must
# not be flagged; these are function words that only appear in English prose.
ENGLISH_MARKERS = re.compile(
    r"\b(of course|i'?ll|you'?ll|won'?t|thank you for your time|no problem at all|"
    r"that'?s a fair point|we can skip|i appreciate|have a good day|"
    r"they'?ll have your details|repeat yourself|let me|i'?m sorry|"
    r"could you please|i don'?t have)\b",
    re.I,
)

NON_ENGLISH_MARKETS = ["fil_PH", "id_ID"]

PROBE_LINES = {
    "fil_PH": ["Opo, sige po.", "Si Ana po.", "Ayaw ko na po, pwede po bang makausap ng tao?"],
    "id_ID": ["Iya Mbak.", "Saya Budi.", "Saya mau bicara dengan petugas saja."],
}


def scan_config(market_key: str) -> list[tuple[str, str]]:
    cfg = MarketConfig.load(market_key)
    findings = []
    fields = {
        "greeting": cfg.greeting, "refusal_line": cfg.refusal_line, "closing": cfg.closing,
        "escalation_line": cfg.escalation_line, "objection_ack": cfg.objection_ack,
        "skip_line": cfg.skip_line, "declined_line": cfg.declined_line,
    }
    for name, value in fields.items():
        if not value:
            findings.append((name, "EMPTY - flow would speak an empty turn"))
        elif ENGLISH_MARKERS.search(value):
            findings.append((name, f"English marker: {ENGLISH_MARKERS.search(value).group(0)!r}"))
    for slot in cfg.qualification:
        if ENGLISH_MARKERS.search(slot["question"]):
            findings.append((f"slot:{slot['slot']}", "English marker in question"))
    return findings


def scan_flow_literals() -> list[str]:
    """Any user-facing string literal left in flow.py is a latent leak."""
    from pathlib import Path
    source = Path("agent/flow.py").read_text()
    # Strings passed as the spoken text of a turn.
    suspects = re.findall(r'self\.add\(\s*"agent",\s*(?:f?")([^"]{12,})"', source)
    return [s for s in suspects if ENGLISH_MARKERS.search(s)]


def probe_call(market_key: str) -> list[tuple[str, str]]:
    from agent.flow import CallSession
    session = CallSession(market_key=market_key)
    findings = []
    turns = [session.open()]
    for line in PROBE_LINES[market_key]:
        turn = session.handle(line)
        if turn.text:
            turns.append(turn.text)
    for text in turns:
        match = ENGLISH_MARKERS.search(text)
        if match:
            findings.append((text[:70], match.group(0)))
    return findings


def run() -> int:
    failures = 0

    literals = scan_flow_literals()
    print("=== flow.py hardcoded spoken strings ===")
    if literals:
        failures += len(literals)
        for lit in literals:
            print(f"  LEAK: {lit[:78]}")
    else:
        print("  none - every spoken line comes from market config")

    for market_key in NON_ENGLISH_MARKETS:
        print(f"\n=== {market_key} config ===")
        findings = scan_config(market_key)
        if findings:
            failures += len(findings)
            for field, why in findings:
                print(f"  LEAK: {field}: {why}")
        else:
            print("  clean")

        print(f"--- {market_key} live call probe ---")
        probe = probe_call(market_key)
        if probe:
            failures += len(probe)
            for text, marker in probe:
                print(f"  LEAK ({marker!r}): {text}")
        else:
            print("  clean - no English markers in any spoken turn")

    print(f"\n{'PASS: no language leaks' if failures == 0 else f'FAIL: {failures} leak(s)'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
