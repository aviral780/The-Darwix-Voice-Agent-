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
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core import config
from core.timing import LatencyRecorder

# Saturation constant for BM25. Set so that a typical strong lexical match on
# this corpus lands near 0.6-0.8 rather than saturating immediately.
BM25_SATURATION = 6.0


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
    def __init__(self) -> None:
        if not (config.INDEX_DIR / "meta.json").exists():
            raise FileNotFoundError("No index. Run `python -m kb.build_index` first.")
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())
        self.vectors = np.load(config.INDEX_DIR / "vectors.npz")["vectors"]
        self.bm25 = pickle.loads((config.INDEX_DIR / "bm25.pkl").read_bytes())
        self.meta = json.loads((config.INDEX_DIR / "meta.json").read_text())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
        recorder: LatencyRecorder | None = None,
    ) -> list[Result]:
        top_k = top_k or config.TOP_K

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

    def is_answerable(self, results: list[Result]) -> bool:
        """Whether the corpus actually supports an answer.

        The agent calls this before generating. When it returns False the agent
        says it does not have the information and offers escalation, which is
        what keeps an ungrounded answer from being invented.
        """
        return bool(results) and results[0].score >= config.MIN_RETRIEVAL_SCORE

    def context_block(self, results: list[Result]) -> str:
        """Format results for the model, tagged so answers can cite a source."""
        blocks = []
        for r in results:
            path = " > ".join(r.chunk["heading_path"]) or r.chunk["title"]
            blocks.append(f"[{r.chunk['chunk_id']}] {r.chunk['title']} — {path}\n{r.chunk['content']}")
        return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide singleton. Loading the index and ONNX model takes seconds,
    which would otherwise be paid on every turn of a live call."""
    return Retriever()


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
