"""Stage 4: build the hybrid retrieval index.

Two indexes are built over the same chunks.

Dense: multilingual-e5-small through fastembed, which runs the model as ONNX.
This avoids a PyTorch dependency entirely, which matters on an 8GB machine, and
the multilingual model means one index serves English, Tagalog and Bahasa
queries without a second embedding pass per market.

Sparse: BM25 over tokenised chunk text.

Both are needed. Callers say product names verbatim - "PRULink", "PRUHealth",
"bancassurance" - and dense retrieval is weakest exactly there, because a rare
proper noun barely moves a sentence embedding. BM25 matches those tokens
directly. Conversely BM25 fails on paraphrase, and callers rarely use the
site's wording. Fusing the two covers both failure modes; the weights live in
core/config.py and were set against the evaluation set in kb/eval_queries.py.

Prefixes are model-specific. The e5 family is trained with "query: " and
"passage: " and loses accuracy without them; paraphrase-multilingual-MiniLM is
not, and adding them would inject a constant token into every embedding. The
prefix is therefore derived from the model name rather than hardcoded.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from core import config

# Only the e5 family uses these.
_IS_E5 = "e5" in config.EMBED_MODEL.lower()
PASSAGE_PREFIX = "passage: " if _IS_E5 else ""
QUERY_PREFIX = "query: " if _IS_E5 else ""

_model: TextEmbedding | None = None


def model() -> TextEmbedding:
    global _model
    if _model is None:
        # Downloads the ONNX weights to .fastembed_cache on first run.
        _model = TextEmbedding(model_name=config.EMBED_MODEL)
    return _model


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25. Keeps digits, which carry meaning here
    (ages, day counts, percentages)."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_passages(texts: list[str]) -> np.ndarray:
    vectors = list(model().embed([PASSAGE_PREFIX + t for t in texts]))
    matrix = np.vstack(vectors).astype("float32")
    # Normalise so cosine similarity is a plain dot product at query time.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def embed_query(text: str) -> np.ndarray:
    vector = next(iter(model().embed([QUERY_PREFIX + text]))).astype("float32")
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def run() -> dict:
    path = config.CLEAN_DIR / "chunks.json"
    if not path.exists():
        raise FileNotFoundError("No chunks. Run `python -m kb.chunk` first.")
    chunks = json.loads(path.read_text())

    print(f"embedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    texts = []
    for c in chunks:
        prefix = " > ".join(c["heading_path"])
        texts.append(f"{c['title']}. {prefix}. {c['content']}" if prefix else f"{c['title']}. {c['content']}")

    vectors = embed_passages(texts)
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.INDEX_DIR / "vectors.npz", vectors=vectors)
    (config.INDEX_DIR / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    meta = {
        "embed_model": config.EMBED_MODEL,
        "dims": int(vectors.shape[1]),
        "chunks": len(chunks),
        "vector_weight": config.VECTOR_WEIGHT,
        "bm25_weight": config.BM25_WEIGHT,
        "min_retrieval_score": config.MIN_RETRIEVAL_SCORE,
    }
    (config.INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"vectors {vectors.shape}  ->  {config.INDEX_DIR}")
    return meta


if __name__ == "__main__":
    run()
