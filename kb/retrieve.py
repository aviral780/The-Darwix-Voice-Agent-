"""Stage 5: hybrid retrieval with citations.

Scores from the two indexes are not comparable as they stand. Cosine similarity
over normalised vectors is bounded in [0, 1]; BM25 is unbounded and its scale
shifts with query length and corpus statistics.

Per-query min-max normalisation is the usual fix and is wrong here. It forces
the best result of every query to 1.0, including queries the corpus cannot
answer at all, which destroys exactly the signal the refusal threshold depends
on. A question about car insurance would score as confidently as one about
premium payments.

Instead BM25 is squashed with x / (x + k), which is monotonic, bounded in [0, 1]
and absolute: a weak match stays weak no matter what else the query returned.
The fused score is therefore comparable across queries, which is what makes
MIN_RETRIEVAL_SCORE meaningful and lets the agent refuse rather than improvise.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core import config
from kb.sources import get as get_source
from core.timing import LatencyRecorder

# Saturation constant for BM25. Set so that a typical strong lexical match on
# this corpus lands near 0.6-0.8 rather than saturating immediately.
BM25_SATURATION = 6.0


def normalise_query(query: str, fillers: list[str]) -> str:
    """Strip vocatives and discourse particles before retrieval.

    These carry real meaning in conversation and none in a search. Removing them
    is safe because they are matched as whole tokens against a curated per-market
    list, and the original text is what the model still answers; only the string
    used to find passages is trimmed. If stripping would empty the query, the
    original is used.
    """
    if not fillers:
        return query
    pattern = r"\b(?:" + "|".join(re.escape(f) for f in fillers) + r")\b"
    trimmed = re.sub(pattern, " ", query, flags=re.I)
    trimmed = re.sub(r"\s+", " ", trimmed).strip(" ,.?!")
    return trimmed if len(trimmed.split()) >= 2 else query


@dataclass
class Result:
    chunk: dict
    score: float
    vector_score: float
    bm25_score: float
    rank: int

    @property
    def citation(self) -> str:
        path = " > ".join(self.chunk["heading_path"]) or self.chunk["title"]
        return f"{self.chunk['title']} — {path}"

    @property
    def source_url(self) -> str:
        return self.chunk["source_url"]

    def explain(self) -> str:
        """Why this chunk matched. Used in the retrieval evaluation report."""
        parts = []
        if self.vector_score >= 0.5:
            parts.append(f"strong semantic match ({self.vector_score:.2f})")
        elif self.vector_score >= 0.3:
            parts.append(f"moderate semantic match ({self.vector_score:.2f})")
        else:
            parts.append(f"weak semantic match ({self.vector_score:.2f})")
        if self.bm25_score >= 0.5:
            parts.append(f"strong keyword overlap ({self.bm25_score:.2f})")
        elif self.bm25_score >= 0.2:
            parts.append(f"some keyword overlap ({self.bm25_score:.2f})")
        else:
            parts.append(f"little keyword overlap ({self.bm25_score:.2f})")
        return "; ".join(parts)


class Retriever:
    def __init__(self, source_key: str = "prulife_ph") -> None:
        source = get_source(source_key)
        if not (source.index_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"No index for {source_key}. Run `python -m kb.build_index {source_key}` first.")
        self.source_key = source_key
        self.min_score = source.min_retrieval_score
        self.chunks = json.loads((source.index_dir / "chunks.json").read_text())
        self.vectors = np.load(source.index_dir / "vectors.npz")["vectors"]
        self.bm25 = pickle.loads((source.index_dir / "bm25.pkl").read_bytes())
        self.meta = json.loads((source.index_dir / "meta.json").read_text())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        recorder: LatencyRecorder | None = None,
        fillers: list[str] | None = None,
    ) -> list[Result]:
        top_k = top_k or config.TOP_K
        query = normalise_query(query, fillers or [])

        def _run() -> list[Result]:
            from kb.build_index import embed_query, tokenize

            query_vector = embed_query(query)
            vector_scores = self.vectors @ query_vector          # cosine, vectors are normalised
            raw_bm25 = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
            bm25_scores = raw_bm25 / (raw_bm25 + BM25_SATURATION)

            fused = config.VECTOR_WEIGHT * vector_scores + config.BM25_WEIGHT * bm25_scores

            order = np.argsort(-fused)
            results: list[Result] = []
            for position in order:
                chunk = self.chunks[position]
                if category and chunk["category"] != category:
                    continue
                results.append(Result(
                    chunk=chunk,
                    score=float(fused[position]),
                    vector_score=float(vector_scores[position]),
                    bm25_score=float(bm25_scores[position]),
                    rank=len(results) + 1,
                ))
                if len(results) >= top_k:
                    break
            return results

        if recorder is not None:
            with recorder.span("retrieval", query_chars=len(query)):
                return _run()
        return _run()

    def search_cross_lingual(
        self,
        query: str,
        corpus_language: str,
        top_k: int | None = None,
        recorder: LatencyRecorder | None = None,
        fillers: list[str] | None = None,
    ) -> list[Result]:
        """Search a corpus written in a different language from the question.

        The Taglish and Bahasa markets both read from corpora that are not in
        the caller's language, and searching directly degrades in two ways at
        once. BM25 can only match the English loanwords that survive in the
        question - "premium" in a Filipino sentence - so its signal comes from
        one token. The multilingual embedding does carry across, but less
        sharply than a same-language pair.

        Measured: "What happens if I miss a premium payment?" retrieved the right
        chunk at 0.591, while the same question in Taglish retrieved a different,
        worse chunk at 0.510 - still above the gate, so the agent did not refuse
        for lack of confidence. It answered from the wrong passage, and the
        generation gate then refused a question the corpus could answer.

        So the question is also asked in the corpus's language and the two result
        sets are merged on the better score per chunk. One extra model call and
        one extra embedding, for a retrieval that finds the passage that exists.
        """
        from core.llm import chat

        translated = ""
        try:
            translated = chat(
                [{"role": "system", "content":
                  f"Translate the user's question into {corpus_language}. Keep financial "
                  "and product terms exactly as written. Reply with the translation only."},
                 {"role": "user", "content": query}],
                temperature=0.0,
                max_tokens=90,
                recorder=recorder,
                stage="query_translate",
            ).strip()
        except Exception:
            # Translation is an enhancement, not a dependency. If it fails the
            # original query still runs and retrieval degrades rather than dies.
            translated = ""

        merged: dict[str, Result] = {}
        for variant in filter(None, [query, translated]):
            for result in self.search(variant, top_k=top_k, recorder=recorder, fillers=fillers):
                existing = merged.get(result.chunk["chunk_id"])
                if existing is None or result.score > existing.score:
                    merged[result.chunk["chunk_id"]] = result

        ranked = sorted(merged.values(), key=lambda r: -r.score)[: (top_k or config.TOP_K)]
        for rank, result in enumerate(ranked, 1):
            result.rank = rank
        return ranked

    def is_answerable(self, results: list[Result]) -> bool:
        """Whether the corpus actually supports an answer.

        The agent calls this before generating. When it returns False the agent
        says it does not have the information and offers escalation, which is
        what keeps an ungrounded answer from being invented.
        """
        return bool(results) and results[0].score >= self.min_score

    def context_block(self, results: list[Result]) -> str:
        """Format results for the model, tagged so answers can cite a source."""
        blocks = []
        for r in results:
            path = " > ".join(r.chunk["heading_path"]) or r.chunk["title"]
            blocks.append(f"[{r.chunk['chunk_id']}] {r.chunk['title']} — {path}\n{r.chunk['content']}")
        return "\n\n".join(blocks)


@lru_cache(maxsize=4)
def get_retriever(source_key: str = "prulife_ph") -> Retriever:
    """One cached retriever per corpus. Loading an index and the ONNX model
    takes seconds, which would otherwise be paid on every turn of a live call."""
    return Retriever(source_key)


if __name__ == "__main__":
    import sys
    r = get_retriever()
    query = " ".join(sys.argv[1:]) or "What happens if I miss a premium payment?"
    results = r.search(query)
    print(f"query: {query}")
    print(f"answerable: {r.is_answerable(results)}  (threshold {config.MIN_RETRIEVAL_SCORE})\n")
    for res in results:
        print(f"{res.rank}. [{res.score:.3f}] {res.citation}")
        print(f"   {res.explain()}")
        print(f"   {res.source_url}")
        print(f"   {res.chunk['content'][:180]}...\n")
