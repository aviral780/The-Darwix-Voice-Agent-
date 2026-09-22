"""Knowledge base source definitions.

One entry per corpus. Adding a market means adding a source here and a market
YAML in agent/markets/; nothing in the pipeline changes, which is the property
that makes the Q3 markets cheap to add rather than a second implementation.

Each source keeps its own raw, cleaned and index directories so corpora never
mix. A Philippine life-insurance answer must not surface on an Indonesian
multifinance call, and separating at the index is a stronger guarantee than
filtering at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config


@dataclass
class Source:
    key: str
    name: str
    base: str
    sitemap: str
    language: str
    sector: str
    # (url path prefix, category) - first match wins.
    sections: list[tuple[str, str]] = field(default_factory=list)
    noise_prefix: str = ""
    noise_sample: int = 0
    request_delay_s: float = 1.5

    # Retrieval gate, calibrated per corpus. A single global threshold is wrong:
    # scores depend on the corpus's own vocabulary, chunk length and language, so
    # a number tuned on English insurance prose systematically over-refuses on
    # Indonesian multifinance content. Each is set from its own evaluation set.
    min_retrieval_score: float = 0.50

    @property
    def raw_dir(self):
        return config.RAW_DIR / self.key

    @property
    def clean_dir(self):
        return config.CLEAN_DIR / self.key

    @property
    def index_dir(self):
        return config.INDEX_DIR / self.key


SOURCES: dict[str, Source] = {
    "prulife_ph": Source(
        key="prulife_ph",
        name="Pru Life UK Philippines",
        base="https://www.prulifeuk.com.ph",
        sitemap="https://www.prulifeuk.com.ph/en/sitemap.xml",
        language="en",
        sector="life and health insurance",
        sections=[
            ("en/products/investment", "product_investment"),
            ("en/products/life-protection", "product_life"),
            ("en/products/health", "product_health"),
            ("en/products/comparison", "product_comparison"),
            ("en/products/product-enquiry", "product_enquiry"),
            ("en/services-and-claims/claims", "claims"),
            ("en/services-and-claims/payments", "payments"),
            ("en/services-and-claims/pruservices", "services"),
            ("en/customer-support", "customer_support"),
            ("en/knowledge-corner/understanding-insurance", "education_insurance"),
            ("en/knowledge-corner/planning-your-finances", "education_finance"),
            ("en/takaful", "product_takaful"),
            ("en/about-us/know-more-about-pru", "company"),
        ],
        noise_prefix="en/about-us/newsroom",
        noise_sample=12,
        # Derived by scripts/calibrate_thresholds.py: weakest in-scope query 0.529.
        min_retrieval_score=0.499,
    ),
    "fifgroup_id": Source(
        key="fifgroup_id",
        name="FIF Group Indonesia",
        base="https://fifgroup.co.id",
        sitemap="https://fifgroup.co.id/sitemap.xml",
        language="id",
        sector="multifinance and consumer finance",
        sections=[
            ("faq", "faq"),
            ("fifastra", "product_motor"),
            ("danastra", "product_loan"),
            ("amitra", "product_sharia"),
            ("spektra", "product_goods"),
            ("fgc", "product_services"),
            ("cabang-fifgroup", "branches"),
            ("anti-fraud", "fraud_safety"),
            ("corporate", "company"),
            ("hubungi-kami", "customer_support"),
            ("karir", "careers"),
        ],
        # Derived the same way: weakest in-scope query 0.413. The gap from the
        # Philippine corpus's 0.499 is the point - one global threshold would have
        # refused fair Indonesian questions outright.
        min_retrieval_score=0.383,
    ),
}


def get(key: str) -> Source:
    if key not in SOURCES:
        raise KeyError(f"Unknown source '{key}'. Available: {sorted(SOURCES)}")
    return SOURCES[key]
