"""Signal extraction from a live call.

Signals fall into two kinds and they are detected differently, because treating
them the same makes both worse.

Deterministic signals are rules. A required disclosure is either spoken or it is
not, and that is a string match over the agent's turns, not a judgement call.
Sending it to a model would add 300ms and a chance of being wrong about
something that is not in doubt.

Judgement signals are a model call. Whether a caller is getting frustrated, or
has let slip that they are about to buy, is not something a keyword list decides
well. "Fine, whatever" is frustration; "fine, that works" is agreement.

The split matters for latency too. Rules run in microseconds on every chunk. The
model runs on a window, and only when the window has enough new speech to be
worth judging, because a nudge that arrives after the moment has passed is not a
nudge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class SignalType(str, Enum):
    COMPLIANCE_GAP = "compliance_gap"
    MISSED_CROSS_SELL = "missed_cross_sell"
    RISING_FRUSTRATION = "rising_frustration"
    BUYING_SIGNAL = "buying_signal"
    PAYMENT_DIFFICULTY = "payment_difficulty"
    CALLBACK_NEEDED = "callback_needed"
    RISKY_STATEMENT = "risky_statement"
    TOPIC_SHIFT = "topic_shift"


# Priority decides what survives when several nudges compete for one agent's
# attention. Compliance outranks everything: a missed cross-sell costs a sale,
# a missed disclosure costs a regulator's fine.
PRIORITY = {
    SignalType.COMPLIANCE_GAP: 100,
    SignalType.RISKY_STATEMENT: 95,
    SignalType.RISING_FRUSTRATION: 80,
    SignalType.PAYMENT_DIFFICULTY: 70,
    SignalType.BUYING_SIGNAL: 60,
    SignalType.MISSED_CROSS_SELL: 50,
    SignalType.CALLBACK_NEEDED: 40,
    SignalType.TOPIC_SHIFT: 20,
}


@dataclass
class Signal:
    type: SignalType
    confidence: float
    evidence: str                 # the words that triggered it
    nudge: str                    # what the agent should be told
    detector: str = "rule"        # rule | model
    speaker: str = ""
    at_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.type, 0)


# --- deterministic rules ----------------------------------------------------

# Required disclosures for an insurance sales call. Each must be spoken by the
# agent before the call ends; the trigger is what makes it due.
DISCLOSURES = [
    {
        "id": "not_a_deposit",
        # Stems, not whole words: \binvest\b does not match "investment" and \bfund\b
        # does not match "funds", which silently missed real disclosure gaps.
        "trigger": re.compile(r"\b(invest\w*|funds?|returns?|savings|growth|VUL|unit[- ]linked)\b", re.I),
        "satisfied": re.compile(r"\b(not a deposit|not guaranteed|subject to market|investment risk|value may (go down|fall))\b", re.I),
        "nudge": "Investment is being pitched without the disclosure. State that returns are not guaranteed and this is not a deposit.",
    },
    {
        "id": "licensed_advisor",
        "trigger": re.compile(r"\b(recommend\w*|should (get|buy|take)|best (plan|option) for you|advis\w+)\b", re.I),
        "satisfied": re.compile(r"\b(licensed (advisor|agent)|financial adviser|subject to (assessment|underwriting))\b", re.I),
        "nudge": "Advice given without the suitability caveat. Note that a licensed advisor must confirm it.",
    },
    {
        "id": "recording_notice",
        "trigger": re.compile(r"\b(personal|details|date of birth|address|ID number)\b", re.I),
        "satisfied": re.compile(r"\b(record(ed|ing)|data privacy|consent|Data Privacy Act)\b", re.I),
        "nudge": "Personal details discussed without the privacy notice. Give the recording and data-privacy notice now.",
    },
]

# Phrases that must never be said on a regulated sales call.
RISKY_PHRASES = [
    (re.compile(r"\bguarantee(d)?\s+(returns?|profit|payout|approval)\b", re.I),
     "Guarantee language used. Correct it now: returns and approval are never guaranteed."),
    (re.compile(r"\b(no risk|risk[- ]free|can't lose|cannot lose)\b", re.I),
     "Risk-free language used. Retract and state the actual risk."),
    (re.compile(r"\b(definitely|certainly)\s+(approved|covered|pay out)\b", re.I),
     "Certainty of approval implied. Restate as subject to underwriting."),
]

# Openings the agent left on the table.
CROSS_SELL_CUES = [
    (re.compile(r"\b(my |our )?(wife|husband|spouse|partner|kids?|children|son|daughter|family)\b", re.I),
     "Family mentioned. Family protection or an education plan may fit.",
     SignalType.MISSED_CROSS_SELL),
    (re.compile(r"\b(second|another|other)\s+(car|vehicle|house|property|policy)\b", re.I),
     "Second asset mentioned. A multi-policy discount may apply.",
     SignalType.MISSED_CROSS_SELL),
    (re.compile(r"\b(hospital|surgery|illness|diagnos(is|ed)|medical bills?|confine(d|ment))\b", re.I),
     "Health concern raised. A health or critical-illness rider is relevant.",
     SignalType.MISSED_CROSS_SELL),
    (re.compile(r"\b(retire|retirement|pension|old age)\b", re.I),
     "Retirement mentioned. A savings or retirement plan is relevant.",
     SignalType.MISSED_CROSS_SELL),
]

PAYMENT_DIFFICULTY_CUES = re.compile(
    r"\b(can'?t afford|too expensive|no money|tight budget|lost my job|unemployed|"
    r"salary (cut|reduced)|struggling|behind on|late payment|miss(ed|ing) a payment)\b", re.I)

CALLBACK_CUES = re.compile(
    r"\b(call me (back|later)|not a good time|busy right now|driving|in a meeting|"
    r"can you call|ring me)\b", re.I)

BUYING_CUES = re.compile(
    r"\b(how do I (sign|apply|start)|what'?s the next step|send me the (form|details|quote)|"
    r"I'?m interested|sounds good|let'?s do it|how much would it be for me)\b", re.I)


def rule_signals(
    agent_text: str,
    caller_text: str,
    at_seconds: float = 0.0,
    turns: list[tuple[str, str]] | None = None,
) -> list[Signal]:
    """Run every deterministic detector.

    agent_text and caller_text are what is new in this pass - normally the one
    utterance that has just finished. turns, when given, is the conversation so
    far as (speaker, text) and is what disclosure timing is judged against.

    A disclosure is not a gap the moment its trigger is said. The first version
    fired "state that returns are not guaranteed" while the agent was still in
    the sentence that mentioned investment - before they had any chance to give
    it. A supervisor would not interrupt there. The gap is real once the agent
    has spoken again and still not given it, so that is when this fires: the
    trigger makes the disclosure due, and the agent's next turn without it makes
    it overdue.
    """
    found: list[Signal] = []

    if turns is not None:
        agent_turns = [text for speaker, text in turns if speaker == "agent"]
        for rule in DISCLOSURES:
            trigger_at = next(
                (i for i, (_, text) in enumerate(turns) if rule["trigger"].search(text)), None)
            if trigger_at is None:
                continue
            if any(rule["satisfied"].search(text) for text in agent_turns):
                continue
            later_agent = [text for speaker, text in turns[trigger_at + 1:] if speaker == "agent"]
            if not later_agent:
                continue            # still within the agent's chance to give it
            found.append(Signal(
                type=SignalType.COMPLIANCE_GAP,
                confidence=0.95,
                evidence=_first_match(rule["trigger"], turns[trigger_at][1]),
                nudge=rule["nudge"],
                detector="rule",
                speaker="agent",
                at_seconds=at_seconds,
                meta={"disclosure": rule["id"]},
            ))
    else:
        combined = f"{agent_text}\n{caller_text}"
        for rule in DISCLOSURES:
            if rule["trigger"].search(combined) and not rule["satisfied"].search(agent_text):
                found.append(Signal(
                    type=SignalType.COMPLIANCE_GAP,
                    confidence=0.95,
                    evidence=_first_match(rule["trigger"], combined),
                    nudge=rule["nudge"],
                    detector="rule",
                    speaker="agent",
                    at_seconds=at_seconds,
                    meta={"disclosure": rule["id"]},
                ))

    for pattern, nudge in RISKY_PHRASES:
        match = pattern.search(agent_text)
        if match:
            found.append(Signal(
                type=SignalType.RISKY_STATEMENT, confidence=0.97, evidence=match.group(0),
                nudge=nudge, detector="rule", speaker="agent", at_seconds=at_seconds))

    for pattern, nudge, signal_type in CROSS_SELL_CUES:
        match = pattern.search(caller_text)
        if match:
            found.append(Signal(
                type=signal_type, confidence=0.75, evidence=match.group(0),
                nudge=nudge, detector="rule", speaker="caller", at_seconds=at_seconds))

    for pattern, signal_type, nudge, confidence in [
        (PAYMENT_DIFFICULTY_CUES, SignalType.PAYMENT_DIFFICULTY,
         "Payment difficulty signalled. Offer an approved payment-support option, not a discount.", 0.85),
        (CALLBACK_CUES, SignalType.CALLBACK_NEEDED,
         "Caller wants to be reached later. Agree a specific callback time before closing.", 0.85),
        (BUYING_CUES, SignalType.BUYING_SIGNAL,
         "Buying signal. Move to the next concrete step rather than adding information.", 0.8),
    ]:
        match = pattern.search(caller_text)
        if match:
            found.append(Signal(type=signal_type, confidence=confidence, evidence=match.group(0),
                                nudge=nudge, detector="rule", speaker="caller", at_seconds=at_seconds))
    return found


def _first_match(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


# --- model-judged signals ---------------------------------------------------

JUDGE_PROMPT = """You monitor a live insurance sales call and flag only what the agent
should act on RIGHT NOW. Most windows contain nothing worth flagging. Saying nothing
is the correct and common answer.

RECENT CONVERSATION:
{window}

Return JSON: {{"signals": [...]}}. Each entry has:
  type        one of: rising_frustration, buying_signal, payment_difficulty,
              callback_needed, topic_shift
  confidence  0.0 to 1.0
  evidence    the caller's exact words that show it, quoted from above
  nudge       one short imperative sentence telling the agent what to do

Rules:
- Flag only what is clearly present in the words above. Do not infer mood from
  punctuation or brevity.
- "Fine, whatever" is frustration. "Fine, that works" is agreement. Read carefully.
- Return {{"signals": []}} when nothing is clearly present. This is expected.
- Never flag something already obvious from a single keyword; that is handled elsewhere.
- Maximum two signals.
"""

JUDGED_TYPES = {
    SignalType.RISING_FRUSTRATION, SignalType.BUYING_SIGNAL,
    SignalType.PAYMENT_DIFFICULTY, SignalType.CALLBACK_NEEDED, SignalType.TOPIC_SHIFT,
}


def model_signals(window: str, at_seconds: float, recorder=None) -> list[Signal]:
    """Judge the recent window with the model. Returns [] on any failure."""
    from core.llm import chat_json

    result = chat_json(
        [{"role": "user", "content": JUDGE_PROMPT.format(window=window)}],
        temperature=0.0,
        max_tokens=280,
        recorder=recorder,
        stage="signal_llm",
    )

    signals: list[Signal] = []
    for entry in (result.get("signals") or [])[:2]:
        try:
            signal_type = SignalType(entry["type"])
        except (KeyError, ValueError):
            continue
        if signal_type not in JUDGED_TYPES:
            continue
        signals.append(Signal(
            type=signal_type,
            confidence=float(entry.get("confidence", 0.5)),
            evidence=str(entry.get("evidence", ""))[:160],
            nudge=str(entry.get("nudge", ""))[:200],
            detector="model",
            speaker="caller",
            at_seconds=at_seconds,
        ))
    return signals
