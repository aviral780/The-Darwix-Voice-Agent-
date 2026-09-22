"""Conversation state machine for the voice agent.

The agent is a state machine that calls a model, not a model asked to behave
like an agent. The distinction matters in practice: a prompt holding the whole
script will, under pressure from a talkative caller, skip qualification slots,
answer from memory rather than the knowledge base, and forget to escalate. An
explicit state machine cannot, because the transitions are code.

Each turn runs the same four steps:

  1. classify  what the caller just did (one fast model call, JSON)
  2. transition the state machine decides where that takes the call
  3. respond    the state decides how the reply is produced
  4. record     transcript, timings and slots are appended for the evidence pack

Only one of those steps lets the model choose words freely, and even then it is
answering from retrieved context under the two-gate rule in kb/answer.py. Every
factual claim a caller hears is either read from the market config or retrieved
and cited. The model is never the source of a fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from core import config
from core.config import MarketConfig
from core.llm import chat, chat_json
from core.timing import LatencyRecorder
from kb.answer import REFUSAL_TOKEN, answer_question, strip_citations


class State(str, Enum):
    GREETING = "greeting"        # opened, waiting for consent to continue
    QUALIFYING = "qualifying"    # working through the slots
    ANSWERING = "answering"      # caller asked something; answer then resume
    OBJECTION = "objection"      # caller pushed back; address then resume
    ESCALATED = "escalated"      # handing to a human, terminal
    CLOSING = "closing"          # wrapping up
    ENDED = "ended"              # terminal


class Intent(str, Enum):
    AGREE = "agree"
    DECLINE = "decline"
    ANSWER = "answer"            # answering the question the agent asked
    QUESTION = "question"        # asking the agent something
    OBJECTION = "objection"
    ESCALATION = "escalation"
    UNCLEAR = "unclear"


@dataclass
class Turn:
    index: int
    speaker: str                 # caller | agent
    text: str
    state: str = ""
    intent: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    refusal_gate: str = ""
    slot_filled: str = ""
    latency_ms: float = 0.0
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


CLASSIFIER_PROMPT = """Classify the caller's message in an insurance qualification call.

The call is conducted in {language}.
The agent just asked: "{pending}"
That question is asking for: {expects}
The caller replied: "{utterance}"

Return JSON with these keys:
  intent      one of: agree, decline, answer, question, objection, escalation, unclear
  objection   if intent is objection, one of: {objection_keys}. Otherwise "".
  slot_value  the caller's answer as a short plain string, ONLY if it is genuinely
              {expects}. Otherwise "".
  value_fits  true only if slot_value is genuinely {expects}. false otherwise.
  escalate    true if the caller wants a human, asks about their own specific policy
              or account, complains, or is repeatedly frustrated. Otherwise false.

Rules:
- "answer" means they responded to what was asked, even partially.
- A reply that both answers and asks something back is "question".
- Reluctance about cost, timing, trust or need is "objection", not "decline".
- "decline" means they do not want to continue the call at all.
- A non-answer such as "I don't know", "not sure", "maybe" or "does it matter" is
  NOT an answer. Set intent to unclear, slot_value to "" and value_fits to false.
- If the caller corrects themselves, take the corrected value. "I'm in my twenties,
  actually no, I turned thirty-one" means thirty-one.
- If the caller answers a DIFFERENT question than the one asked, set value_fits to
  false. Telling the agent about their insurance when asked their age does not
  answer the age question.
"""


@dataclass
class CallSession:
    """One call. Holds state, slots and the full transcript."""

    market_key: str = "en_PH"
    call_id: str = field(default_factory=lambda: uuid4().hex[:12])
    state: State = State.GREETING
    slot_index: int = 0
    slots: dict[str, str] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    objections_raised: list[str] = field(default_factory=list)
    slot_attempts: dict[str, int] = field(default_factory=dict)
    skipped_slots: list[str] = field(default_factory=list)
    escalation_reason: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.market = MarketConfig.load(self.market_key)
        self.recorder = LatencyRecorder(f"call-{self.call_id}")

    # --- slots ---------------------------------------------------------------

    @property
    def pending_slot(self) -> dict | None:
        if self.slot_index < len(self.market.qualification):
            return self.market.qualification[self.slot_index]
        return None

    @property
    def pending_question(self) -> str:
        """The next qualification question to ask."""
        slot = self.pending_slot
        return slot["question"] if slot else self.market.closing

    @property
    def last_agent_utterance(self) -> str:
        """What the agent actually last said.

        This is what the classifier must be given, and getting it wrong was a
        real defect. During the greeting the agent asks for consent ("do you have
        two or three minutes?") while pending_question already points at the first
        slot ("may I know who I'm speaking with?"). Handing the classifier the
        slot question meant "yes, I have a few minutes" was measured against a
        request for a name, judged not to be one, and marked unclear - and since
        the greeting state only advances on agreement, the call then reprompted
        forever. Three of the five recorded calls died this way.
        """
        for turn in reversed(self.turns):
            if turn.speaker == "agent" and turn.text.strip():
                return turn.text
        return self.market.greeting.strip()

    # A caller who cannot or will not answer must not be asked the same thing
    # forever. After this many failed attempts the slot is skipped and the call
    # moves on with a partial lead, which is worth more than a call that loops
    # until the caller hangs up.
    MAX_SLOT_ATTEMPTS = 3

    def note_failed_attempt(self) -> bool:
        """Record a failed attempt at the pending slot. True if it was skipped."""
        slot = self.pending_slot
        if not slot:
            return False
        key = slot["slot"]
        self.slot_attempts[key] = self.slot_attempts.get(key, 0) + 1
        if self.slot_attempts[key] >= self.MAX_SLOT_ATTEMPTS:
            self.skipped_slots.append(key)
            self.slot_index += 1
            return True
        return False

    def fill_slot(self, value: str) -> str:
        slot = self.pending_slot
        if not slot or not value.strip():
            return ""
        self.slots[slot["slot"]] = value.strip()
        self.slot_index += 1
        return slot["slot"]

    # --- transcript ----------------------------------------------------------

    def add(self, speaker: str, text: str, **kwargs) -> Turn:
        turn = Turn(index=len(self.turns), speaker=speaker, text=text,
                    state=self.state.value, **kwargs)
        self.turns.append(turn)
        return turn

    # --- the call ------------------------------------------------------------

    def open(self) -> str:
        """First thing the caller hears."""
        greeting = self.market.greeting.strip()
        self.add("agent", greeting)
        return greeting

    def classify(self, utterance: str) -> dict:
        """One fast model call to decide what the caller did."""
        slot = self.pending_slot
        prompt = CLASSIFIER_PROMPT.format(
            language=self.market.language_name,
            pending=self.last_agent_utterance,
            expects=(slot or {}).get("expects", "a reply to the question"),
            utterance=utterance,
            objection_keys=", ".join(self.market.objections) or "none",
        )
        result = chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=160,
            recorder=self.recorder,
            stage="classify",
        )
        # A failed parse must not drop the call, so default to unclear and let
        # the reprompt path handle it.
        intent = result.get("intent", "")
        if intent not in {i.value for i in Intent}:
            intent = Intent.UNCLEAR.value
        return {
            "intent": intent,
            "objection": result.get("objection", "") or "",
            "slot_value": str(result.get("slot_value", "") or ""),
            "value_fits": bool(result.get("value_fits", False)),
            "escalate": bool(result.get("escalate", False)),
        }

    def handle(self, utterance: str) -> Turn:
        """Process one caller utterance and produce the agent's reply."""
        with self.recorder.span("turn") as turn_span:
            self.add("caller", utterance)

            if self.state in (State.ESCALATED, State.ENDED):
                return self.add("agent", "", intent="terminal")

            signal = self.classify(utterance)

            # Escalation overrides every other path, in any state.
            if signal["escalate"] or signal["intent"] == Intent.ESCALATION.value:
                return self._escalate("caller requested a human or raised an account-specific matter")

            if signal["intent"] == Intent.DECLINE.value:
                return self._close(declined=True)

            if signal["intent"] == Intent.QUESTION.value:
                return self._answer_question(utterance)

            if signal["intent"] == Intent.OBJECTION.value:
                return self._handle_objection(utterance, signal["objection"])

            if self.state == State.GREETING:
                if signal["intent"] == Intent.AGREE.value:
                    self.state = State.QUALIFYING
                    return self.add("agent", self.pending_question, intent=signal["intent"])

                # People often skip the yes and simply start answering. Treating
                # that as consent is both truer to how the calls actually go and
                # the thing that stops a missed "yes" from trapping the call in
                # this state with no way forward.
                if signal["intent"] == Intent.ANSWER.value and signal["value_fits"]:
                    self.state = State.QUALIFYING
                    filled = self.fill_slot(signal["slot_value"])
                    if self.pending_slot is None:
                        return self._close()
                    return self.add("agent", self.pending_question,
                                    intent=signal["intent"], slot_filled=filled)

                return self._reprompt(signal["intent"])

            if self.state in (State.QUALIFYING, State.ANSWERING, State.OBJECTION):
                self.state = State.QUALIFYING
                if signal["intent"] == Intent.ANSWER.value:
                    # A value that does not fit the slot is worse than no value.
                    # Before this check the conflicting-details call recorded a
                    # name of "31" and an age band of "had insurance but it
                    # lapsed last year", which is a lead the sales team cannot use.
                    if not signal["value_fits"] or not signal["slot_value"]:
                        return self._reprompt(Intent.UNCLEAR.value)
                    filled = self.fill_slot(signal["slot_value"])
                    if self.pending_slot is None:
                        return self._close()
                    return self.add("agent", self.pending_question,
                                    intent=signal["intent"], slot_filled=filled)
                return self._reprompt(signal["intent"])

            return self._reprompt(signal["intent"])

    # --- response paths ------------------------------------------------------

    def _answer_question(self, question: str) -> Turn:
        """Grounded answer from the knowledge base, then steer back to the flow."""
        self.state = State.ANSWERING
        answer = answer_question(
            question,
            max_sentences=3,
            extra_instructions=(
                self._answer_only_instructions() if self.pending_slot
                else self._voice_instructions()
            ),
            recorder=self.recorder,
            source_key=self.market.kb_source,
            answer_language=self.market.language_name,
        )

        if answer.refused:
            # The refusal line is fixed text from the market config, so a
            # caller can never be told something invented in place of an answer.
            text = self.market.refusal_line.strip()
            if self.pending_slot:
                text = f"{text} {self.pending_question}"
            return self.add("agent", text, intent=Intent.QUESTION.value,
                            refused=True, refusal_gate=answer.gate)

        spoken = strip_citations(answer.text)
        if self.pending_slot:
            spoken = f"{spoken} {self.pending_question}"
        return self.add("agent", spoken, intent=Intent.QUESTION.value,
                        citations=answer.citations,
                        retrieved_ids=answer.retrieved_ids,
                        grounded=answer.grounded)

    def _handle_objection(self, utterance: str, objection_key: str) -> Turn:
        """Address pushback using the config's stance plus retrieved facts."""
        self.state = State.OBJECTION
        if objection_key:
            self.objections_raised.append(objection_key)
        stance = self.market.objections.get(objection_key, "")

        answer = answer_question(
            utterance,
            max_sentences=3,
            extra_instructions=(
                f"{self._answer_only_instructions() if self.pending_slot else self._voice_instructions()}\n"
                f"The caller raised an objection. Stance to take: {stance}\n"
                "Acknowledge the concern in your first sentence before anything else. "
                "If the CONTEXT has no supporting fact, acknowledge the concern and say "
                f"you can have an advisor follow up, rather than replying {REFUSAL_TOKEN}."
            ),
            recorder=self.recorder,
        )

        if answer.refused:
            # A configured stance is vetted business guidance, so it is a better
            # answer than a generic refusal when the knowledge base cannot help.
            # The Indonesian collections corpus has no hardship content, and
            # falling through to the refusal line answered a customer who had
            # just lost their job with "I don't have that detail".
            if stance:
                text = self._speak_stance(utterance, stance)
            else:
                text = f"{self.market.objection_ack.strip()} {self.market.refusal_line.strip()}".strip()
        else:
            text = strip_citations(answer.text)

        if self.pending_slot:
            text = f"{text} {self.pending_question}"
        return self.add("agent", text, intent=Intent.OBJECTION.value,
                        citations=answer.citations,
                        retrieved_ids=answer.retrieved_ids,
                        grounded=answer.grounded,
                        refused=answer.refused, refusal_gate=answer.gate)

    def _speak_stance(self, utterance: str, stance: str) -> str:
        """Answer an objection from the market's configured stance.

        Used when the knowledge base has nothing to support the objection. The
        stance is authored guidance in the market file, not model invention, so
        the model is only putting vetted content into the market's own language
        and register rather than deciding what to say.
        """
        return chat(
            [
                {"role": "system", "content":
                 f"{self._voice_instructions()}\n"
                 f"Respond to the caller's objection following this guidance exactly: {stance}\n"
                 "Acknowledge the concern in your first sentence. State no figure, rate or "
                 "product detail that is not in the guidance. Do not end with a question. "
                 "Reply in the language and register described above."},
                {"role": "user", "content": utterance},
            ],
            temperature=0.3,
            max_tokens=160,
            recorder=self.recorder,
            stage="stance",
        ).strip()

    def _reprompt(self, intent: str) -> Turn:
        """The caller said something that did not advance the call."""
        if self.state == State.QUALIFYING and self.note_failed_attempt():
            # Slot abandoned. Move to the next question rather than insisting.
            if self.pending_slot is None:
                return self._close()
            return self.add("agent", f"{self.market.skip_line.strip()} {self.pending_question}".strip(),
                            intent="slot_skipped")
        text = chat(
            [
                {"role": "system", "content":
                 f"{self.market.persona}\n{self._voice_instructions()}\n"
                 "The caller's reply was unclear. In one short sentence, acknowledge "
                 "that and ask the pending question again in different words. "
                 "Do not invent any fact."},
                {"role": "user", "content":
                 f"The question to ask again: "
                 f"{self.market.greeting if self.state == State.GREETING else self.pending_question}"},
            ],
            temperature=0.4,
            max_tokens=90,
            recorder=self.recorder,
            stage="reprompt",
        )
        return self.add("agent", text.strip(), intent=intent)

    def _escalate(self, reason: str) -> Turn:
        self.state = State.ESCALATED
        self.escalation_reason = reason
        return self.add("agent", self.market.escalation_line.strip(),
                        intent=Intent.ESCALATION.value)

    def _close(self, declined: bool = False) -> Turn:
        self.state = State.CLOSING
        text = (self.market.declined_line.strip() if declined
                else self.market.closing.strip())
        turn = self.add("agent", text, intent="close")
        self.state = State.ENDED
        return turn

    def _voice_instructions(self) -> str:
        return (
            f"{self.market.persona}\n"
            + "\n".join(f"- {rule}" for rule in self.market.style_rules)
        )

    def _answer_only_instructions(self) -> str:
        """Instructions for turns where the flow appends the next question itself.

        Without this the model ends its answer with an offer or a question of its
        own, the pending slot question is appended after it, and the caller hears
        two questions in one breath - which breaks the market's own style rule and
        reliably confuses people on a phone line.
        """
        return (
            f"{self._voice_instructions()}\n"
            "- Do not end with a question, an offer, or 'would you like'. Stop after "
            "the information. The call flow asks the next question itself."
        )

    # --- output --------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "call_id": self.call_id,
            "market": self.market_key,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "final_state": self.state.value,
            "slots": self.slots,
            "slots_filled": len(self.slots),
            "slots_total": len(self.market.qualification),
            "objections_raised": self.objections_raised,
            "skipped_slots": self.skipped_slots,
            "slot_attempts": self.slot_attempts,
            "escalated": self.state == State.ESCALATED,
            "escalation_reason": self.escalation_reason,
            "grounded_answers": sum(1 for t in self.turns if t.grounded),
            "answers_with_stated_citation": sum(1 for t in self.turns if t.citations),
            "refusals": sum(1 for t in self.turns if t.refused),
            "turns": [asdict(t) for t in self.turns],
            "latency": self.recorder.summary(),
        }

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or (config.EVIDENCE_DIR / "transcripts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"call_{self.market_key}_{self.call_id}.json"
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False))
        return path

    def transcript_text(self) -> str:
        lines = []
        for turn in self.turns:
            who = "CALLER" if turn.speaker == "caller" else "AGENT "
            marker = ""
            if turn.citations:
                marker = f"   [cited: {', '.join(turn.citations)}]"
            elif turn.grounded:
                marker = f"   [grounded in: {', '.join(turn.retrieved_ids[:3])}]"
            elif turn.refused:
                marker = f"   [refused at {turn.refusal_gate} gate]"
            lines.append(f"{who}: {turn.text}{marker}")
        return "\n".join(lines)
