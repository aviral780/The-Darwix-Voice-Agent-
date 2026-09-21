"""Stage 3: split clean records into retrievable chunks.

Chunking is heading-aware rather than fixed-width. The extractor emits markdown
headings, so a section boundary is known rather than guessed, and a chunk can be
cut where the document itself changes subject.

This matters for a voice agent specifically. A fixed 500-character window will
happily split "What happens if I miss a payment?" from its answer, and the
retriever then returns a chunk containing a question and no answer. The agent
either refuses when it should not have, or stitches an answer together from
context it does not have. Cutting on headings keeps a question and its answer in
the same chunk.

Every chunk keeps its parent record's provenance so a citation can name the page
it came from, which is what makes the "grounded and traceable" requirement
verifiable rather than asserted.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core import config

TARGET_WORDS = 180          # comfortable for a spoken answer plus context
MAX_WORDS = 320             # hard ceiling before a section is split further
MIN_WORDS = 25              # below this a chunk carries no answerable content
OVERLAP_WORDS = 30          # carried between splits of one oversized section

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    record_id: str
    title: str                     # parent page title
    heading: str                   # section heading this chunk sits under
    content: str
    category: str
    source: str
    source_url: str
    version: str
    pii: bool
    chunk_index: int
    word_count: int
    heading_path: list[str] = field(default_factory=list)

    def citation(self) -> str:
        """Human-readable source reference, spoken or displayed with an answer."""
        where = " > ".join(self.heading_path) if self.heading_path else self.title
        return f"{self.title} — {where} ({self.source_url})"

    def embedding_text(self) -> str:
        """What actually gets embedded.

        The heading path is prepended because a chunk's body often uses pronouns
        that only resolve against its heading ("it covers ages 18-65"). Without
        the heading the embedding loses the subject entirely.
        """
        prefix = " > ".join(self.heading_path)
        return f"{self.title}. {prefix}. {self.content}" if prefix else f"{self.title}. {self.content}"


def split_sections(body: str) -> list[tuple[list[str], str]]:
    """Split a document into (heading_path, text) sections on markdown headings.

    The stack holds (level, heading) pairs rather than bare strings. Pages here
    routinely skip levels - an h2 with no h1 above it, or an h4 directly under an
    h2 - so trimming the stack by list position attributes a section to whatever
    heading happens to sit at that index. Trimming by the heading's own level is
    what keeps a top-level section from inheriting its sibling as a parent.
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append(([h for _, h in stack], text))
        buffer.clear()

    for line in body.split("\n"):
        match = HEADING_RE.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            # Drop every heading at this level or deeper, then push this one.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buffer.append(line)
    flush()
    return sections


def split_long_text(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split an oversized section on sentence boundaries with a small overlap."""
    # Bullet lists and tables often contain no terminal punctuation at all, so a
    # naive sentence split returns the whole section as one "sentence" and the
    # size ceiling is never enforced. Split on line breaks too, and hard-split
    # any single unit that still exceeds the target on its own.
    units = [u for u in re.split(r"(?<=[.!?])\s+|\n", text) if u.strip()]
    sentences: list[str] = []
    for unit in units:
        words = unit.split()
        if len(words) <= target:
            sentences.append(unit)
        else:
            for i in range(0, len(words), target):
                sentences.append(" ".join(words[i:i + target]))
    pieces: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if count + words > target and current:
            pieces.append(" ".join(current))
            # Carry the tail forward so a fact spanning the cut is not orphaned.
            tail = " ".join(current).split()[-overlap:]
            current = [" ".join(tail)] if tail else []
            count = len(tail)
        current.append(sentence)
        count += words

    if current:
        piece = " ".join(current)
        # Fold a too-small remainder back into the previous piece.
        if pieces and len(piece.split()) < MIN_WORDS:
            pieces[-1] = pieces[-1] + " " + piece
        else:
            pieces.append(piece)
    return pieces


def chunk_record(record: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = split_sections(record["content"])

    # A page with no headings is still one section, handled by the same path.
    if not sections:
        sections = [([], record["content"])]

    index = 0
    for heading_path, text in sections:
        words = len(text.split())
        pieces = [text] if words <= MAX_WORDS else split_long_text(text)

        for piece in pieces:
            if len(piece.split()) < MIN_WORDS:
                continue
            chunks.append(Chunk(
                chunk_id=f"{record['record_id']}_c{index:02d}",
                record_id=record["record_id"],
                title=record["title"],
                heading=heading_path[-1] if heading_path else record["title"],
                content=piece.strip(),
                category=record["category"],
                source=record["source"],
                source_url=record["source_url"],
                version=record["version"],
                pii=record["pii"],
                chunk_index=index,
                word_count=len(piece.split()),
                heading_path=heading_path,
            ))
            index += 1
    return chunks


def run() -> list[Chunk]:
    path = config.CLEAN_DIR / "records.json"
    if not path.exists():
        raise FileNotFoundError("No cleaned records. Run `python -m kb.clean` first.")
    records = json.loads(path.read_text())

    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(chunk_record(record))

    out = config.CLEAN_DIR / "chunks.json"
    out.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False))

    sizes = [c.word_count for c in chunks]
    print(f"records            {len(records)}")
    print(f"chunks             {len(chunks)}")
    if sizes:
        print(f"words per chunk    min {min(sizes)}  median {sorted(sizes)[len(sizes)//2]}  max {max(sizes)}")
    print(f"with heading path  {sum(1 for c in chunks if c.heading_path)}/{len(chunks)}")
    print(f"-> {out}")
    return chunks


if __name__ == "__main__":
    run()
