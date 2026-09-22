"""Stage 2 of the knowledge base: turn raw HTML into clean, traceable records.

The brief lists six requirements for this stage. Each maps to a function here,
and each reports what it removed so the result is auditable rather than a black
box:

  website extraction / document parsing  -> extract_main_content
  remove nav, headers, footers, repeats  -> find_boilerplate + strip_boilerplate
  handle failures, flag source errors    -> validate_record
  remove duplicate / near-duplicate      -> simhash + dedupe
  standardise headings, dates, terms     -> normalise_text, normalise_heading
  identify and protect PII               -> detect_pii, redact_pii

Two design notes.

Boilerplate is detected statistically rather than by CSS selector. Any line
appearing on more than BOILERPLATE_PAGE_RATIO of pages is site furniture by
definition, which keeps working when the site's markup changes and catches
repeated blocks that are not in a nav element at all.

Near-duplicate detection uses SimHash over token shingles rather than exact
hashing, because the real duplication on this source is a product description
reused across pages with one sentence changed. Exact hashing misses all of it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from html import unescape
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from core import config
from kb.sources import get as get_source

# A line must appear on at least this share of pages to count as site furniture.
BOILERPLATE_PAGE_RATIO = 0.25
BOILERPLATE_MAX_WORDS = 25          # long paragraphs repeating are content, not chrome

SIMHASH_BITS = 64
SIMHASH_DISTANCE = 4                # <= this Hamming distance counts as near-duplicate
SHINGLE_SIZE = 4

MIN_CONTENT_WORDS = 40              # below this a page carries no answerable information

# Categories that are legitimate pages but do not belong in a qualification
# knowledge base. Kept through extraction so the report can show them being
# dropped for the right reason.
IRRELEVANT_CATEGORIES = {"newsroom"}

# Terminology is inconsistent across the source: the same concept appears with
# different spelling, spacing and hyphenation. Retrieval and the voice agent both
# suffer when these are treated as different terms.
#
# Only genuine variant-to-canonical mappings belong here. An earlier version also
# listed terms that were already canonical ("rider" -> "rider"), which did nothing
# except lowercase them wherever they began a sentence or heading. Replacement now
# preserves the original leading case, so normalising a term mid-sentence cannot
# corrupt one that starts a heading.
TERMINOLOGY = {
    r"\bvariable unit linked\b": "variable unit-linked",
    r"\bunit linked\b": "unit-linked",
    r"\bpre need\b": "pre-need",
    r"\bbanc assurance\b": "bancassurance",
    r"\bpolicy holder\b": "policyholder",
    r"\bpolicy holders\b": "policyholders",
    # "face amount" and "sum assured" are the same thing; the source uses both,
    # and a caller asking about one must retrieve chunks phrased with the other.
    r"\bface amount\b": "sum assured",
}


def _match_case(original: str, replacement: str) -> str:
    """Apply the original's leading case to the replacement."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# Philippine formats specifically. A generic phone regex both misses local
# numbers and fires on policy numbers and peso amounts.
PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "ph_mobile": re.compile(r"(?:\+63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}\b"),
    "ph_landline": re.compile(r"\(0?2\)\s?\d{4}[\s-]?\d{4}\b|\b\(0?\d{2,3}\)\s?\d{3}[\s-]?\d{4}\b"),
    "tin": re.compile(r"\bTIN[:\s#]*\d{3}-?\d{3}-?\d{3}(?:-?\d{3,5})?\b", re.I),
    "policy_number": re.compile(r"\b(?:policy|contract)\s*(?:no\.?|number|#)[:\s]*([A-Z0-9-]{6,})\b", re.I),
    "sss_gsis": re.compile(r"\b(?:SSS|GSIS)[:\s#]*\d{2}-?\d{7}-?\d\b", re.I),
}

DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.I), "dmy"),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "slash"),
]
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}


@dataclass
class CleanRecord:
    record_id: str
    title: str
    content: str
    category: str
    source: str
    source_url: str
    version: str
    pii: bool
    pii_types: list[str] = field(default_factory=list)
    word_count: int = 0
    retrieved_at: str = ""
    warnings: list[str] = field(default_factory=list)


# --- extraction -------------------------------------------------------------

# Wrapper elements whose class or id marks them as page chrome rather than
# content. Matched as a subtree so the whole banner goes, not just its text.
CHROME_MARKERS = re.compile(
    r"cookie|privacy-statement|disclaimer-modal|modal|nav|footer|header|"
    r"breadcrumb|skip-link|social|share-",
    re.I,
)

CONTENT_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "dt", "dd"]


def extract_structured(html: str) -> str:
    """Walk the main content container, keeping heading structure.

    The source runs on Adobe Experience Manager, which nests content in deep
    generic grid divs. trafilatura's precision mode discards nearly all of it:
    on the premium payment page it returned 393 words against 4,519 here, and
    the part it dropped was the list of payment channels, which is exactly what
    a caller phones in to ask about.

    Headings are emitted as markdown so the chunker downstream can split on
    section boundaries instead of on arbitrary character counts.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "button"]):
        tag.decompose()

    root = soup.find("main") or soup.find(id="main-content") or soup.body
    if root is None:
        return ""

    for attr in ("class", "id"):
        for element in root.find_all(attrs={attr: CHROME_MARKERS}):
            element.decompose()

    parts: list[str] = []
    seen: set[str] = set()
    for element in root.find_all(CONTENT_TAGS):
        text = element.get_text(" ", strip=True)
        if not text or len(text) < 3:
            continue
        # AEM nests the same string inside several wrappers; keep the first.
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        if element.name.startswith("h"):
            parts.append(f"\n{'#' * int(element.name[1])} {text}")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    return "\n".join(parts).strip()


# Keys that hold content in a client-rendered site's inline state payload.
PAYLOAD_CONTENT_KEYS = ("faqs", "articles", "items", "posts", "contents")
# Fields within such an entry, in the order they should be read.
PAYLOAD_TITLE_KEYS = ("title", "question", "name", "heading")
PAYLOAD_BODY_KEYS = ("description", "answer", "content", "body", "text")


def _strip_html(fragment: str) -> str:
    """Flatten an HTML fragment stored inside a JSON field."""
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", fragment, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _collect_payload_entries(node, found: list) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PAYLOAD_CONTENT_KEYS and isinstance(value, list):
                found.extend(v for v in value if isinstance(v, dict))
            else:
                _collect_payload_entries(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_payload_entries(value, found)


def extract_from_payload(html: str) -> str:
    """Recover content from a client-rendered page's inline JSON state.

    The Indonesian source renders entirely in the browser: 534KB of HTML yields
    116 words of body text, and every FAQ answer arrives by JavaScript. Walking
    the DOM cannot see any of it.

    It does not have to. The page ships its state as inline JSON, so the answers
    are already in the HTML that was fetched - just not where an HTML parser
    looks. Reading them needs no extra request and no browser, which keeps the
    crawl to exactly the pages robots.txt permits.
    """
    entries: list[dict] = []
    for match in re.finditer(r"<script[^>]*>\s*(\{.{200,}?\})\s*</script>", html, re.S):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        _collect_payload_entries(payload, entries)

    parts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        title = next((str(entry[k]) for k in PAYLOAD_TITLE_KEYS if entry.get(k)), "")
        body = next((str(entry[k]) for k in PAYLOAD_BODY_KEYS if entry.get(k)), "")
        if not body:
            continue
        body = _strip_html(body)
        key = (title + body)[:120].lower()
        if not body or key in seen:
            continue
        seen.add(key)
        if title:
            parts.append(f"## {_strip_html(title)}")
        parts.append(body)
    return "\n".join(parts).strip()


def extract_main_content(html: str) -> tuple[str, str]:
    """Return (title, body), taking whichever extractor recovered more content.

    Three strategies run and the one recovering the most text wins. Structured
    DOM extraction wins on most server-rendered pages; trafilatura wins where the
    main container is missing or unusual; the JSON payload reader wins on
    client-rendered sites where the DOM holds almost nothing. Each costs
    milliseconds, and running all three removes a whole class of silent
    extraction failure rather than tuning for one site's markup.
    """
    candidates = [
        extract_structured(html),
        trafilatura.extract(html, include_comments=False, include_tables=True,
                            favor_recall=True, no_fallback=False) or "",
        extract_from_payload(html),
    ]
    body = max((c.strip() for c in candidates), key=lambda c: len(c.split()))

    meta = trafilatura.extract_metadata(html)
    title = (getattr(meta, "title", "") or "").strip()
    return title, body


# --- normalisation ----------------------------------------------------------

def normalise_text(text: str) -> str:
    """Unicode, whitespace, currency and terminology normalisation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Peso amounts appear as PHP, Php, P, ₱ and "pesos". A voice agent reading
    # these aloud needs one consistent form.
    text = re.sub(r"[Pp]hp\s?([\d,]+)", r"PHP \1", text)
    text = re.sub(r"₱\s?([\d,]+)", r"PHP \1", text)
    text = re.sub(r"\bP\s?([\d,]{4,})", r"PHP \1", text)

    for pattern, replacement in TERMINOLOGY.items():
        text = re.sub(
            pattern,
            lambda m, r=replacement: _match_case(m.group(0), r),
            text,
            flags=re.I,
        )

    return text.strip()


def normalise_dates(text: str) -> tuple[str, int]:
    """Rewrite dates to ISO (YYYY-MM-DD). Returns (text, count changed)."""
    count = 0

    def dmy(m):
        nonlocal count
        count += 1
        return f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"

    def mdy(m):
        nonlocal count
        count += 1
        return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"

    def slash(m):
        nonlocal count
        count += 1
        # Ambiguous by nature. The source is a PH site, which uses M/D/YYYY.
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    for pattern, kind in DATE_PATTERNS:
        text = pattern.sub({"dmy": dmy, "mdy": mdy, "slash": slash}[kind], text)
    return text, count


def normalise_heading(title: str) -> str:
    """Strip the site suffix and collapse casing noise."""
    title = re.sub(r"\s*[|\-–]\s*Pru ?Life ?UK.*$", "", title, flags=re.I).strip()
    title = re.sub(r"\s+", " ", title)
    if title.isupper() and len(title) > 8:
        title = title.title()
    return title


# --- boilerplate ------------------------------------------------------------

def find_boilerplate(bodies: list[str]) -> set[str]:
    """Lines repeating across a large share of pages are site furniture."""
    if not bodies:
        return set()
    counts: Counter[str] = Counter()
    for body in bodies:
        seen = {ln.strip() for ln in body.split("\n") if ln.strip()}
        counts.update(seen)
    threshold = max(2, int(len(bodies) * BOILERPLATE_PAGE_RATIO))
    return {
        line for line, n in counts.items()
        if n >= threshold and len(line.split()) <= BOILERPLATE_MAX_WORDS
    }


def strip_boilerplate(body: str, boilerplate: set[str]) -> tuple[str, int]:
    kept, removed = [], 0
    for line in body.split("\n"):
        if line.strip() in boilerplate:
            removed += 1
        else:
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), removed


# --- near-duplicate detection -----------------------------------------------

def simhash(text: str, bits: int = SIMHASH_BITS) -> int:
    """SimHash over word shingles. Catches reuse with small edits."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < SHINGLE_SIZE:
        shingles = tokens or ["_empty_"]
    else:
        shingles = [" ".join(tokens[i:i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1)]

    vector = [0] * bits
    for shingle in shingles:
        h = int.from_bytes(__import__("hashlib").blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(bits):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    result = 0
    for bit in range(bits):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe(records: list[CleanRecord]) -> tuple[list[CleanRecord], list[dict]]:
    """Drop exact and near-duplicate records, keeping the longest of each group."""
    # Longest first so the survivor is the most complete version.
    ordered = sorted(records, key=lambda r: r.word_count, reverse=True)
    kept: list[tuple[int, CleanRecord]] = []
    dropped: list[dict] = []

    for record in ordered:
        fingerprint = simhash(record.content)
        match = None
        for existing_hash, existing in kept:
            distance = hamming(fingerprint, existing_hash)
            if distance <= SIMHASH_DISTANCE:
                match = (existing, distance)
                break
        if match:
            dropped.append({
                "record_id": record.record_id,
                "source_url": record.source_url,
                "duplicate_of": match[0].record_id,
                "hamming_distance": match[1],
                "reason": "exact duplicate" if match[1] == 0 else "near-duplicate",
            })
        else:
            kept.append((fingerprint, record))

    return [r for _, r in kept], dropped


# --- PII --------------------------------------------------------------------

def detect_pii(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for label, pattern in PII_PATTERNS.items():
        matches = [m.group(0) for m in pattern.finditer(text)]
        if matches:
            found[label] = sorted(set(matches))
    return found


def redact_pii(text: str, found: dict[str, list[str]]) -> str:
    """Replace detected PII with a typed placeholder.

    Redaction rather than deletion: the surrounding sentence often carries the
    answer ("call us on <number>"), so removing the line would lose information
    the agent needs while keeping the identifier out of the index.
    """
    for label, values in found.items():
        for value in values:
            text = text.replace(value, f"[{label.upper()}_REDACTED]")
    return text


# --- validation -------------------------------------------------------------

def validate_record(record: CleanRecord) -> list[str]:
    """Flag obvious source errors rather than indexing them silently."""
    warnings = []
    if record.word_count < MIN_CONTENT_WORDS:
        warnings.append(f"thin content ({record.word_count} words)")
    if not record.title:
        warnings.append("missing title")
    if re.search(r"lorem ipsum|test page|coming soon", record.content, re.I):
        warnings.append("placeholder text detected")
    if record.content.count("�") > 0:
        warnings.append("encoding errors present")
    # A page that is mostly a single repeated line usually failed extraction.
    lines = [ln for ln in record.content.split("\n") if ln.strip()]
    if lines and len(set(lines)) / len(lines) < 0.4:
        warnings.append("highly repetitive content, extraction may have failed")
    return warnings


# --- pipeline ---------------------------------------------------------------

def run(source_key: str = "prulife_ph") -> dict:
    source = get_source(source_key)
    manifest_path = source.raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No raw manifest for {source_key}. Run `python -m kb.scrape {source_key}` first.")
    manifest = json.loads(manifest_path.read_text())

    fetched = [r for r in manifest["records"] if r["status"] in ("ok", "cached")]
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": manifest["source"],
        "source_key": source.key,
        "stages": {},
        "dropped": {},
        "warnings": [],
    }
    report["stages"]["fetched"] = len(fetched)
    report["stages"]["fetch_failures"] = len(manifest["records"]) - len(fetched)

    # 1. extract
    extracted: list[dict] = []
    extraction_failures: list[dict] = []
    for entry in fetched:
        path = source.raw_dir / entry["cache_file"]
        if not path.exists():
            extraction_failures.append({"source_url": entry["url"], "reason": "cache file missing"})
            continue
        title, body = extract_main_content(path.read_text(encoding="utf-8"))
        if not body:
            extraction_failures.append({"source_url": entry["url"], "reason": "no main content extracted"})
            continue
        extracted.append({"url": entry["url"], "category": entry["category"], "title": title, "body": body})

    report["stages"]["extracted"] = len(extracted)
    report["dropped"]["extraction_failed"] = extraction_failures

    # 2. boilerplate, computed across the corpus then removed per page
    boilerplate = find_boilerplate([e["body"] for e in extracted])
    report["stages"]["boilerplate_lines_identified"] = len(boilerplate)
    total_boilerplate_removed = 0

    # 3. normalise, redact, build records
    records: list[CleanRecord] = []
    dates_normalised = 0
    for i, entry in enumerate(extracted, 1):
        body, removed = strip_boilerplate(entry["body"], boilerplate)
        total_boilerplate_removed += removed

        body = normalise_text(body)
        body, n_dates = normalise_dates(body)
        dates_normalised += n_dates

        found = detect_pii(body)
        if found:
            body = redact_pii(body, found)

        title = normalise_heading(entry["title"])
        record = CleanRecord(
            record_id=f"kb_{entry['category']}_{i:03d}",
            title=title,
            content=body,
            category=entry["category"],
            source=f"{entry['category'].replace('_', ' ')} / website section",
            source_url=entry["url"],
            version="1.0",
            pii=bool(found),
            pii_types=sorted(found),
            word_count=len(body.split()),
            retrieved_at=manifest["fetched_at"],
        )
        record.warnings = validate_record(record)
        records.append(record)

    report["stages"]["boilerplate_lines_removed"] = total_boilerplate_removed
    report["stages"]["dates_normalised"] = dates_normalised
    report["stages"]["pii_records_flagged"] = sum(1 for r in records if r.pii)
    report["stages"]["pii_types_seen"] = sorted({t for r in records for t in r.pii_types})

    # 4. drop irrelevant sections
    irrelevant = [r for r in records if r.category in IRRELEVANT_CATEGORIES]
    records = [r for r in records if r.category not in IRRELEVANT_CATEGORIES]
    report["dropped"]["irrelevant_category"] = [
        {"record_id": r.record_id, "source_url": r.source_url, "category": r.category} for r in irrelevant
    ]

    # 5. drop thin pages
    thin = [r for r in records if r.word_count < MIN_CONTENT_WORDS]
    records = [r for r in records if r.word_count >= MIN_CONTENT_WORDS]
    report["dropped"]["thin_content"] = [
        {"record_id": r.record_id, "source_url": r.source_url, "word_count": r.word_count} for r in thin
    ]

    # 5b. drop pages the validator judged to be failed extractions. These are
    # navigation widgets rather than documents - a fund-report page whose whole
    # body is a list of month names, for example. Indexing one means a retrieval
    # hit that returns "December November October" as an answer, so detecting it
    # and keeping it anyway would be worse than not detecting it at all.
    broken = [r for r in records if any("extraction may have failed" in w for w in r.warnings)]
    records = [r for r in records if r not in broken]
    report["dropped"]["failed_extraction"] = [
        {"record_id": r.record_id, "source_url": r.source_url, "warnings": r.warnings} for r in broken
    ]

    # 6. deduplicate
    records, duplicates = dedupe(records)
    report["dropped"]["duplicates"] = duplicates

    report["stages"]["final_records"] = len(records)
    report["warnings"] = [
        {"record_id": r.record_id, "warnings": r.warnings} for r in records if r.warnings
    ]
    report["by_category"] = dict(Counter(r.category for r in records))

    source.clean_dir.mkdir(parents=True, exist_ok=True)
    (source.clean_dir / "records.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False)
    )
    (source.clean_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    print(f"fetched              {report['stages']['fetched']}")
    print(f"extracted            {report['stages']['extracted']}")
    print(f"boilerplate lines    {len(boilerplate)} identified, {total_boilerplate_removed} removed")
    print(f"dates normalised     {dates_normalised}")
    print(f"PII flagged          {report['stages']['pii_records_flagged']} records {report['stages']['pii_types_seen']}")
    print(f"dropped irrelevant   {len(irrelevant)}")
    print(f"dropped thin         {len(thin)}")
    print(f"dropped broken       {len(broken)}")
    print(f"dropped duplicates   {len(duplicates)}")
    print(f"final records        {len(records)}")
    print(f"records with warning {len(report['warnings'])}")
    return report


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "prulife_ph")
