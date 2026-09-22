"""Stage 1 of the knowledge base: fetch source pages.

Source is Pru Life UK Philippines, a real insurer whose public site matches the
market the Question 1 and Question 3 agents serve. Real content is used
deliberately: it carries genuine duplication, boilerplate and inconsistent
terminology, which is what the cleaning stage has to prove it can handle.
Sun Life PH was the first choice but sits behind DataDome bot protection, so
it was dropped rather than worked around.

Three rules govern this module.

Politeness: robots.txt is fetched and enforced, requests are serialised with a
delay, and the crawl is limited to the sitemap's public pages.

Idempotence: every response is cached to data/raw/ on first fetch and read from
there afterwards. Re-running the pipeline costs no requests to the source, which
matters when iterating on the cleaning logic.

Honest failure: pages that fail are recorded in the manifest with the reason
rather than silently dropped, because the brief asks for extraction failures to
be handled and obvious source errors flagged.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from core import config
from kb.sources import Source, get as get_source

USER_AGENT = "DarwixAssessmentBot/1.0 (knowledge base build; contact via repository)"

REQUEST_DELAY_S = 1.5   # no Crawl-delay in robots.txt, so a conservative default
TIMEOUT_S = 25
MAX_RETRIES = 2

@dataclass
class FetchRecord:
    url: str
    category: str
    status: str            # ok | http_error | network_error | too_small | cached
    http_status: int | None
    bytes: int
    cache_file: str
    fetched_at: str
    error: str = ""


def categorize(url: str, source: Source) -> str | None:
    path = urlparse(url).path.lstrip("/")
    for prefix, category in source.sections:
        if path.startswith(prefix):
            return category
    if source.noise_prefix and path.startswith(source.noise_prefix):
        return "newsroom"
    return None


def load_robots(source: Source) -> urllib.robotparser.RobotFileParser:
    """Fetch and parse robots.txt using this crawler's declared identity.

    RobotFileParser.read() issues its own request as "Python-urllib/3.x", and
    sites behind a CDN commonly 403 that agent. The parser treats a 403 as
    disallow-all, so a site that plainly permits crawling reads as forbidding it
    - which is how the Indonesian source first appeared to block all 24 of its
    pages while serving the same robots.txt with Allow: / to this bot.

    The fetch is done here with the same User-Agent used for every other request
    so the site sees one consistent, declared identity and its rules are applied
    to the agent that will actually be crawling. A failure to read still falls
    back to disallow-all rather than assuming permission.
    """
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(f"{source.base}/robots.txt")
    try:
        response = httpx.get(
            f"{source.base}/robots.txt",
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_S,
            follow_redirects=True,
        )
    except Exception as exc:
        print(f"  robots.txt unreachable ({type(exc).__name__}); treating as disallow-all")
        parser.disallow_all = True
        return parser

    if response.status_code == 200:
        parser.parse(response.text.splitlines())
    elif response.status_code in (401, 403):
        parser.disallow_all = True
    elif response.status_code >= 400:
        # No robots.txt published. The standard reads this as unrestricted.
        parser.allow_all = True
    return parser


def fetch_sitemap(client: httpx.Client, source: Source) -> list[str]:
    response = client.get(source.sitemap)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]


def select_urls(urls: list[str], source: Source) -> list[tuple[str, str]]:
    """Pick the pages worth indexing, plus a bounded sample of newsroom noise."""
    chosen: list[tuple[str, str]] = []
    noise: list[tuple[str, str]] = []
    for url in urls:
        category = categorize(url, source)
        if category is None:
            continue
        if category == "newsroom":
            noise.append((url, category))
        else:
            chosen.append((url, category))
    return chosen + noise[: source.noise_sample]


def cache_path(url: str, source: Source) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    slug = urlparse(url).path.strip("/").replace("/", "_")[:70] or "index"
    return source.raw_dir / f"{slug}__{digest}.html"


def fetch_one(client: httpx.Client, url: str, category: str, source: Source) -> FetchRecord:
    path = cache_path(url, source)
    now = datetime.now(timezone.utc).isoformat()

    if path.exists() and path.stat().st_size > 0:
        return FetchRecord(url, category, "cached", 200, path.stat().st_size, path.name, now)

    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.get(url)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                if response.status_code in (404, 410):
                    break  # permanent, no point retrying
                time.sleep(1.5 * (attempt + 1))
                continue

            body = response.text
            # A page far smaller than a real article usually means a JS shell or
            # an interstitial rather than content. Flag it instead of indexing it.
            if len(body) < 2000:
                return FetchRecord(
                    url, category, "too_small", response.status_code, len(body), "", now,
                    "response under 2KB, likely a JS shell or redirect page",
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            return FetchRecord(url, category, "ok", response.status_code, len(body), path.name, now)

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))

    status = "http_error" if last_error.startswith("HTTP") else "network_error"
    return FetchRecord(url, category, status, None, 0, "", now, last_error)


def run(source_key: str = "prulife_ph", limit: int | None = None) -> dict:
    source = get_source(source_key)
    source.raw_dir.mkdir(parents=True, exist_ok=True)

    robots = load_robots(source)
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-PH,en;q=0.9"}

    with httpx.Client(headers=headers, timeout=TIMEOUT_S, follow_redirects=True) as client:
        urls = fetch_sitemap(client, source)
        selected = select_urls(urls, source)

        blocked = [(u, c) for u, c in selected if not robots.can_fetch(USER_AGENT, u)]
        allowed = [(u, c) for u, c in selected if robots.can_fetch(USER_AGENT, u)]
        if limit:
            allowed = allowed[:limit]

        print(f"sitemap: {len(urls)} urls, {len(selected)} in scope, "
              f"{len(blocked)} blocked by robots.txt, fetching {len(allowed)}")

        records: list[FetchRecord] = []
        for i, (url, category) in enumerate(allowed, 1):
            record = fetch_one(client, url, category, source)
            records.append(record)
            if record.status not in ("ok", "cached"):
                print(f"  [{i}/{len(allowed)}] {record.status}: {url} ({record.error})")
            elif i % 20 == 0 or i == len(allowed):
                print(f"  [{i}/{len(allowed)}] ...")
            if record.status == "ok":
                time.sleep(source.request_delay_s)  # only sleep on a real network call

    manifest = {
        "source_key": source.key,
        "source_name": source.name,
        "source": source.base,
        "language": source.language,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sitemap_urls": len(urls),
        "in_scope": len(selected),
        "blocked_by_robots": [u for u, _ in blocked],
        "counts": {
            status: sum(1 for r in records if r.status == status)
            for status in sorted({r.status for r in records})
        },
        "by_category": {
            category: sum(1 for r in records if r.category == category)
            for category in sorted({r.category for r in records})
        },
        "records": [asdict(r) for r in records],
    }
    out = source.raw_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\n{manifest['counts']}")
    print(f"manifest -> {out}")
    return manifest


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "prulife_ph"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(key, limit)
