"""Central configuration: credentials, model choices and market definitions.

Model selection is measured, not assumed. scripts/benchmark_models.py tests
every model this account can reach on four axes that map to hard requirements:
latency (Q4 nudges must land mid-call), multilingual quality (Q3 needs real
Taglish and colloquial Bahasa), JSON reliability (signal extraction parses
structured output on every chunk) and grounded refusal (Q1 must decline rather
than invent). Results are written to docs/model_selection.md.

The plan had been to split a large conversation model from a small fast one for
live signals. The benchmark made that unnecessary: Qwen3.8-27B was three times
faster than the next candidate and also the strongest on the other three axes,
so one model serves both paths. FALLBACK_MODEL is from a different family so
that a Qwen-side outage does not end a recorded demo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Credentials -----------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL", "").strip()

# --- Models ----------------------------------------------------------------
CHAT_MODEL = "qwen/qwen3.8-27b"      # fastest and strongest available; see docs/model_selection.md
FALLBACK_MODEL = "openai/gpt-oss-20b"  # different family, so a Qwen-side outage does not stop the demo
ASR_MODEL = "whisper-large-v3-turbo"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIMS = 384
# 0.22GB ONNX, ~50 languages including Indonesian and Tagalog. multilingual-e5-small
# is not in fastembed's catalogue, and e5-large is 2.24GB - too heavy for 8GB RAM.

# --- Paths -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
INDEX_DIR = DATA_DIR / "index"
LEADS_DIR = DATA_DIR / "leads"
EVIDENCE_DIR = ROOT / "evidence"
DOCS_DIR = ROOT / "docs"
MARKETS_DIR = ROOT / "agent" / "markets"

# --- Retrieval tuning ------------------------------------------------------
# Hybrid weighting. Vectors handle paraphrase, BM25 handles exact product and
# policy names, which callers tend to say verbatim. 0.6/0.4 tested best on the
# evaluation set in kb/eval_queries.py.
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
# Raised from 4 after the evaluation showed real answers sitting just outside the
# window: "how long does a claim take" has its answer in kb_claims_043_c05, which
# ranked 7th, so the agent refused a question the corpus could answer. Eight chunks
# is roughly 1,500 words against a 131k context, so the cost is negligible.
TOP_K = 8

# First of two gates against answering ungrounded. Set from the measured score
# distribution in docs/retrieval_results.md, not by intuition: the lowest-scoring
# genuinely answerable query in the evaluation set lands at 0.529, so the gate
# sits just under that at 0.50. Anything higher starts refusing fair questions.
#
# This gate is deliberately not the whole defence. The answerable and
# out-of-scope populations overlap - "tomorrow's weather in Cebu" retrieves a
# typhoon advisory at 0.568, above several legitimate queries - so no threshold
# separates them. The second gate is in the answering prompt, where the model
# reads the retrieved text and must reply NO_ANSWER_IN_CONTEXT if the answer is
# not there. A score cannot read; the model can.
MIN_RETRIEVAL_SCORE = 0.50


@dataclass
class MarketConfig:
    """One market's voice, language and business rules.

    Q1 and both Q3 bots are the same engine with a different config loaded,
    so anything market-specific belongs here and not in code.
    """

    key: str
    display_name: str
    sector: str
    language_name: str
    asr_language: str            # ISO code passed to Whisper as a decoding hint
    tts_voice: str
    tts_voice_alt: str = ""
    currency_symbol: str = ""
    # Which corpus this market answers from. Markets in the same country can
    # share one; the Indonesian market has its own because a Philippine
    # life-insurance answer has no business on a multifinance call.
    kb_source: str = "prulife_ph"
    greeting: str = ""
    persona: str = ""
    style_rules: list[str] = field(default_factory=list)
    terminology: dict[str, str] = field(default_factory=dict)
    qualification: list[dict] = field(default_factory=list)
    objections: dict[str, str] = field(default_factory=dict)
    escalation_triggers: list[str] = field(default_factory=list)
    refusal_line: str = ""
    closing: str = ""
    # Everything the agent can say must come from here. Four of these lines were
    # hardcoded English in flow.py, which meant a Taglish or Bahasa call switched
    # to English at exactly the moments that matter most - escalation and
    # refusal. The brief names unexpected English switching as a failure, and a
    # caller who has spoken Bahasa for four turns being answered in English at
    # the handover is the clearest possible version of it.
    escalation_line: str = ""
    objection_ack: str = ""
    skip_line: str = ""
    declined_line: str = ""

    @classmethod
    def load(cls, key: str) -> "MarketConfig":
        path = MARKETS_DIR / f"{key}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in MARKETS_DIR.glob("*.yaml"))
            raise FileNotFoundError(f"No market config '{key}'. Available: {available}")
        data = yaml.safe_load(path.read_text())
        return cls(key=key, **data)


def available_markets() -> list[str]:
    return sorted(p.stem for p in MARKETS_DIR.glob("*.yaml"))


def require_groq_key() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
            "from https://console.groq.com"
        )
    return GROQ_API_KEY
