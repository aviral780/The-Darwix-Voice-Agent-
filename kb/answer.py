"""Grounded answer generation: the second gate.

Retrieval decides which passages the model may look at. This decides whether
those passages actually answer the question, and it is the gate that matters,
because the first one cannot read.

The evaluation in kb/eval_queries.py shows why both are needed. Asking for
tomorrow's weather in Cebu retrieves a typhoon customer-support advisory with a
higher similarity score than several legitimate questions about premiums. The
vocabulary genuinely overlaps. Only something that reads the passage can tell
that it does not answer the question.

The prompt therefore forbids outside knowledge, requires a citation for every
claim, and requires the exact token NO_ANSWER_IN_CONTEXT when the passages fall
short. Qwen follows this reliably; it was one of the four axes the model
benchmark tested for precisely this reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import config
from core.llm import chat
from core.timing import LatencyRecorder
from kb.retrieve import Result, get_retriever

REFUSAL_TOKEN = "NO_ANSWER_IN_CONTEXT"

# What language each corpus is written in, used to decide whether a question
# needs translating before retrieval.
CORPUS_LANGUAGE = {
    "prulife_ph": "English",
    "fifgroup_id": "Bahasa Indonesia",
}

SYSTEM_PROMPT = """You answer questions using ONLY the numbered CONTEXT passages provided.

How to decide whether you can answer:
- If any passage contains the answer, or part of it, ANSWER. Give the part that is
  supported, cite it, and say plainly which part you do not have.
- Only if no passage contains any part of the answer, reply with exactly {token}
  and nothing else.
- A passage that shares a topic with the question but does not address what was
  asked is not an answer. Weather, other companies' products, other countries'
  rules, and anything specific to one customer's account are not in scope here.

How to write the answer:
- Never use knowledge from outside the CONTEXT. Never guess, estimate or infer a
  figure, date, percentage or condition that is not written in the passages.
- Cite the passage id in square brackets after each claim, like [kb_payments_050_c03].
- At most {max_sentences} sentences, in plain spoken language suitable for reading
  aloud on a phone call. No markdown, no bullet points, no headings.

{extra}"""


@dataclass
class Answer:
    text: str
    refused: bool
    citations: list[str] = field(default_factory=list)      # ids the model stated
    retrieved_ids: list[str] = field(default_factory=list)  # ids that were in context
    invalid_citations: list[str] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    gate: str = ""            # which gate refused: retrieval | generation | none

    @property
    def grounded(self) -> bool:
        """Whether this answer was produced from retrieved context.

        Distinct from having citation markers. The model emits markers most of
        the time but not every time, and an answer is grounded by construction
        because the context is all it was given. Treating a missing marker as
        "not grounded" would under-report grounding; treating it as a citation
        would invent provenance. Both are recorded separately instead.
        """
        return not self.refused and bool(self.retrieved_ids)

    @property
    def sources(self) -> list[dict]:
        """Unique source pages behind the cited chunks, for display or logging."""
        wanted = set(self.citations)
        seen: dict[str, dict] = {}
        for result in self.results:
            if result.chunk["chunk_id"] in wanted or not wanted:
                seen.setdefault(result.source_url, {
                    "title": result.chunk["title"],
                    "url": result.source_url,
                    "citation": result.citation,
                })
        return list(seen.values())


# Chunk ids end in _c<n>; authored FAQ ids do not. The first version matched
# only the chunk shape, so a citation of an authored answer survived into the
# spoken text and the caller heard "faq who you are" read aloud.
CITATION_RE = re.compile(r"\[((?:faq_[a-z0-9_]+)|(?:[a-z0-9_]+_c\d+))\]")


def extract_citations(text: str) -> list[str]:
    return sorted(set(CITATION_RE.findall(text)))


def strip_citations(text: str) -> str:
    """Remove citation markers for the spoken version.

    A caller should not hear "kb payments zero five zero c zero three" read out,
    but the markers have to survive long enough to be verified and logged, so
    they are stripped at the point of speech rather than at generation.
    """
    return re.sub(r"\s*" + CITATION_RE.pattern, "", text).strip()


def answer_question(
    question: str,
    top_k: int | None = None,
    max_sentences: int = 3,
    extra_instructions: str = "",
    recorder: LatencyRecorder | None = None,
    source_key: str = "prulife_ph",
    answer_language: str = "",
    query_fillers: list[str] | None = None,
    authored: dict[str, str] | None = None,
) -> Answer:
    retriever = get_retriever(source_key)
    corpus_language = CORPUS_LANGUAGE.get(source_key, "English")

    if answer_language and not answer_language.lower().startswith(corpus_language.lower()[:4]):
        results = retriever.search_cross_lingual(
            question, corpus_language, top_k=top_k, recorder=recorder,
            fillers=query_fillers)
    else:
        results = retriever.search(question, top_k=top_k, recorder=recorder,
                                   fillers=query_fillers)

    # Authored passages from the market file sit alongside the retrieved ones.
    # They are vetted content, so the "answer only from context" rule still holds;
    # the corpus simply is not the only legitimate source. Without them the agent
    # refuses its own name.
    authored_block = ""
    if authored:
        authored_block = "\n\n".join(
            f"[faq_{key}] {text.strip()}" for key, text in authored.items())

    # Gate 1: retrieval confidence. Skipped when authored context is available,
    # since a question the market file answers does not depend on the corpus.
    if not retriever.is_answerable(results) and not authored_block:
        return Answer(text="", refused=True, results=results, gate="retrieval")

    language_rule = ""
    if answer_language:
        # The Indonesian and Filipino markets read from corpora whose language
        # may differ from the one the caller is speaking, so the output language
        # is stated explicitly rather than left to match the context.
        language_rule = (
            f"\n- Reply in {answer_language}. The CONTEXT may be in another language; "
            "translate what you need from it, but never add anything it does not say."
        )

    system = SYSTEM_PROMPT.format(
        token=REFUSAL_TOKEN,
        max_sentences=max_sentences,
        extra=extra_instructions + language_rule,
    )
    context = retriever.context_block(results) if retriever.is_answerable(results) else ""
    if authored_block:
        context = f"{authored_block}\n\n{context}".strip()
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    raw = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=320,
        recorder=recorder,
        stage="generation",
    )

    # Gate 2: the model read the passages and judged them insufficient.
    if REFUSAL_TOKEN in raw:
        return Answer(text="", refused=True, results=results, gate="generation")

    retrieved_ids = [r.chunk["chunk_id"] for r in results]
    retrieved_ids += [f"faq_{k}" for k in (authored or {})]
    stated = extract_citations(raw)

    # A citation naming a chunk that was never in context is a fabricated
    # reference. It has not been seen in testing, but an unverified citation is
    # worse than none: it looks like provenance while pointing nowhere.
    valid = [c for c in stated if c in retrieved_ids]
    invalid = [c for c in stated if c not in retrieved_ids]

    return Answer(
        text=raw.strip(),
        refused=False,
        citations=valid,
        retrieved_ids=retrieved_ids,
        invalid_citations=invalid,
        results=results,
        gate="none",
    )


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What happens if I miss a premium payment?"
    a = answer_question(q)
    print(f"Q: {q}")
    if a.refused:
        print(f"REFUSED at the {a.gate} gate")
    else:
        print(f"A: {a.text}")
        print(f"cited: {a.citations}")
        for s in a.sources:
            print(f"  - {s['citation']}  {s['url']}")
