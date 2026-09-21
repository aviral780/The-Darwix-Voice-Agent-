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
EMBED_MODEL = "intfloat/multilingual-e5-small"  # ONNX via fastembed, 384 dims
EMBED_DIMS = 384

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
TOP_K = 4

# Below this fused score the agent refuses to answer and offers escalation,
# rather than letting the model improvise. Tuned so the deliberately
# out-of-scope evaluation queries fall under it.
MIN_RETRIEVAL_SCORE = 0.30


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
    greeting: str = ""
    persona: str = ""
    style_rules: list[str] = field(default_factory=list)
    terminology: dict[str, str] = field(default_factory=dict)
    qualification: list[dict] = field(default_factory=list)
    objections: dict[str, str] = field(default_factory=dict)
    escalation_triggers: list[str] = field(default_factory=list)
    refusal_line: str = ""
    closing: str = ""

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
