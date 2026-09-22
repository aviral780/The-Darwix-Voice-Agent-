"""Calibrate the retrieval gate for each corpus from its own evaluation set.

A single global threshold is wrong. Fused scores depend on the corpus's
vocabulary, chunk length and language, so a number tuned on English insurance
prose does not transfer to Indonesian multifinance content - it systematically
over-refuses, because the same semantic match simply scores lower there.

Each corpus gets in-scope queries it should answer and out-of-scope queries it
must refuse, and the gate is placed just under the weakest in-scope score. It is
deliberately not placed midway between the populations: the generation gate in
kb/answer.py is the one that reads the text, so the retrieval gate's job is only
to drop the clearly unsupported without silencing fair questions.

Run:  python -m scripts.calibrate_thresholds
"""

from __future__ import annotations

import json
from datetime import date

from core import config
from core.timing import percentile
from kb.retrieve import get_retriever

EVAL_SETS: dict[str, dict] = {
    "prulife_ph": {
        "in_scope": [
            "What happens if I miss a premium payment?",
            "How long does it take to process a claim?",
            "Can I pay my premium through GCash?",
            "What investment funds can I choose from for a PRULink policy?",
            "Does the health plan cover critical illness diagnosis?",
            "What documents do I need to file a death claim?",
        ],
        "out_of_scope": [
            "How do I file a car insurance claim for a windscreen crack in Germany?",
            "What is the interest rate on a personal loan at a bank in Jakarta?",
            "Can you tell me tomorrow's weather in Cebu?",
        ],
    },
    "fifgroup_id": {
        "in_scope": [
            "Bagaimana cara mengajukan pembiayaan motor?",
            "Berapa plafond pinjaman DANASTRA?",
            "Apa itu AMITRA?",
            "Apa saja syarat pengajuan pembiayaan?",
            "Berapa lama tenor pembiayaan yang tersedia?",
            "Dokumen apa yang dibutuhkan untuk pengajuan?",
        ],
        "out_of_scope": [
            "Bagaimana cara klaim asuransi jiwa di Filipina?",
            "Berapa harga tiket pesawat ke Bali besok?",
            "Siapa presiden Indonesia saat ini?",
        ],
    },
}

# Headroom below the weakest in-scope query, so a slightly harder phrasing of a
# fair question is not refused outright.
MARGIN = 0.03


def calibrate(source_key: str, queries: dict) -> dict:
    retriever = get_retriever(source_key)

    def top_scores(items: list[str]) -> list[tuple[str, float]]:
        out = []
        for q in items:
            results = retriever.search(q, top_k=4)
            out.append((q, results[0].score if results else 0.0))
        return out

    in_scope = top_scores(queries["in_scope"])
    out_scope = top_scores(queries["out_of_scope"])

    in_min = min(s for _, s in in_scope)
    out_max = max(s for _, s in out_scope)
    suggested = round(max(0.05, in_min - MARGIN), 3)

    return {
        "source": source_key,
        "in_scope": [{"query": q, "score": round(s, 3)} for q, s in in_scope],
        "out_of_scope": [{"query": q, "score": round(s, 3)} for q, s in out_scope],
        "in_scope_min": round(in_min, 3),
        "in_scope_median": round(percentile([s for _, s in in_scope], 50), 3),
        "out_of_scope_max": round(out_max, 3),
        "separable": in_min > out_max,
        "current": retriever.min_score,
        "suggested": suggested,
        "would_refuse_in_scope": [q for q, s in in_scope if s < suggested],
        "would_answer_out_of_scope": [q for q, s in out_scope if s >= suggested],
    }


def run() -> dict:
    report = {"generated_at": date.today().isoformat(), "margin": MARGIN, "sources": {}}
    for key, queries in EVAL_SETS.items():
        result = calibrate(key, queries)
        report["sources"][key] = result
        print(f"\n=== {key}")
        print(f"  in-scope  min {result['in_scope_min']}  median {result['in_scope_median']}")
        print(f"  out-scope max {result['out_of_scope_max']}   separable={result['separable']}")
        print(f"  current {result['current']}  ->  suggested {result['suggested']}")
        if result["would_refuse_in_scope"]:
            print(f"  at suggested, still refuses: {result['would_refuse_in_scope']}")
        if result["would_answer_out_of_scope"]:
            print(f"  at suggested, passes to generation gate: "
                  f"{len(result['would_answer_out_of_scope'])} out-of-scope query(ies)")

    out = config.DOCS_DIR / "threshold_calibration.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    run()
