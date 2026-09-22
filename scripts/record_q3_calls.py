"""Record the Question 3 localised calls.

Two calls per market, covering what the brief asks for between them:
cooperative customer, sector-specific objection, mixed English and finance
terms, colloquial speech, human escalation, and an Indonesian regional accent.

The caller in the Indonesian accent call is voiced with jv-ID-SitiNeural rather
than the Jakarta-standard id-ID voice, so the call exercises the weakness the
ASR report measured (Javanese-accented speech at roughly fifteen times the word
error rate) instead of only describing it.

Run:  python -m scripts.record_q3_calls
"""

from __future__ import annotations

import json
import sys

from core import config, tts
from agent.actions import finalise_call
from agent.flow import CallSession
from scripts.record_calls import concat_mp3

SCENARIOS = {
    "fil_PH": [
        {
            "id": "q3_ph_01_cooperative_taglish",
            "name": "Cooperative customer, natural Taglish",
            "caller_voice": "fil-PH-AngeloNeural",
            "why": "the baseline localised path, with the caller code-switching the way a real caller does",
            "lines": [
                "Opo, meron po akong konting oras.",
                "Si Marco Reyes po.",
                "Nasa thirties po ako.",
                "Opo, yung asawa ko po at dalawang anak.",
                "Wala pa po, yung sa company lang po namin.",
                "Yung protection po siguro, yung sigurado.",
                "Hapon po mas okay sa akin.",
            ],
        },
        {
            "id": "q3_ph_02_objection_escalation",
            "name": "Price objection, finance terms, then asks for a person",
            "caller_voice": "fil-PH-AngeloNeural",
            "why": "sector objection and mixed English finance vocabulary, ending in escalation that must stay in Taglish",
            "lines": [
                "Sige po, pakinggan ko muna.",
                "Si Elena po.",
                "Forties na po ako.",
                "Ano po ba yung sum assured tapos may rider pa po ba dun?",
                "Ay naku po, medyo mahal po yata yan para sa amin ngayon.",
                "Pwede po ba akong makausap na lang ng totoong tao?",
            ],
        },
    ],
    "id_ID": [
        {
            "id": "q3_id_01_instalment_reminder",
            "name": "Instalment reminder, colloquial Bahasa",
            "caller_voice": "id-ID-ArdiNeural",
            "why": "the everyday multifinance call, with colloquial register and finance terminology",
            "lines": [
                "Iya Mbak, ada sebentar.",
                "Saya Budi Santoso.",
                "Sudah, saya sudah terima notifikasinya.",
                "Belum saya bayar, Mbak.",
                "Tenor saya berapa lama ya, dan kena denda berapa kalau telat?",
                "Minggu depan saya bayar.",
                "Biasanya lewat transfer bank.",
            ],
        },
        {
            "id": "q3_id_02_hardship_javanese_accent",
            "name": "Payment difficulty, Javanese-accented caller",
            "caller_voice": "jv-ID-SitiNeural",
            "why": "regional accent plus financial hardship, the combination the ASR report shows is handled worst",
            "lines": [
                "Iya, sebentar saja ya Mbak.",
                "Nama saya Siti Rahayu.",
                "Sudah tahu, tapi ada masalah.",
                "Belum bisa bayar, Mbak.",
                "Maaf, saya baru kena PHK jadi lagi kesulitan sekali.",
                "Saya mau bicara dengan petugas yang bisa bantu keringanan.",
            ],
        },
    ],
}


def record(scenario: dict, market_key: str) -> dict:
    session = CallSession(market_key=market_key)
    parts: list[bytes] = []

    print(f"\n  {scenario['name']}  [{market_key}]")
    greeting = session.open()
    print(f"    AGENT : {greeting[:90]}")
    parts.append(tts.synthesize(greeting, voice=session.market.tts_voice,
                                recorder=session.recorder, stage="tts"))

    for line in scenario["lines"]:
        print(f"    CALLER: {line[:90]}")
        parts.append(tts.synthesize(line, voice=scenario["caller_voice"]))

        turn = session.handle(line)
        if not turn.text:
            continue
        marker = ""
        if turn.citations:
            marker = f"   [cited {', '.join(turn.citations)}]"
        elif turn.grounded:
            marker = "   [grounded]"
        elif turn.refused:
            marker = f"   [refused at {turn.refusal_gate} gate]"
        print(f"    AGENT : {turn.text[:90]}{marker}")
        parts.append(tts.synthesize(turn.text, voice=session.market.tts_voice,
                                    recorder=session.recorder, stage="tts"))
        if session.state.value in ("ended", "escalated"):
            break

    summary = session.summary()
    actions = finalise_call(summary)

    audio_path = config.EVIDENCE_DIR / "calls" / f"{scenario['id']}.mp3"
    has_audio = concat_mp3(parts, audio_path)

    transcripts = config.EVIDENCE_DIR / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    (transcripts / f"{scenario['id']}.txt").write_text(
        f"{scenario['name']}\n{scenario['why']}\n"
        f"market: {market_key}   caller voice: {scenario['caller_voice']}   "
        f"call_id: {session.call_id}\n{'=' * 78}\n{session.transcript_text()}\n"
    )
    session.save(transcripts)

    print(f"    -> {summary['final_state']}, {len(summary['slots'])}/{summary['slots_total']} slots, "
          f"{summary['grounded_answers']} grounded, {summary['refusals']} refusals")

    return {
        "id": scenario["id"], "name": scenario["name"], "why": scenario["why"],
        "market": market_key, "caller_voice": scenario["caller_voice"],
        "call_id": session.call_id,
        "audio": str(audio_path.relative_to(config.ROOT)) if has_audio else "",
        "transcript": f"evidence/transcripts/{scenario['id']}.txt",
        "final_state": summary["final_state"], "slots": summary["slots"],
        "slots_filled": f"{len(summary['slots'])}/{summary['slots_total']}",
        "grounded_answers": summary["grounded_answers"], "refusals": summary["refusals"],
        "escalated": summary["escalated"], "objections_raised": summary["objections_raised"],
        "actions": actions, "latency": summary["latency"], "turns": summary["turns"],
    }


def run(markets: list[str] | None = None) -> list[dict]:
    markets = markets or ["fil_PH", "id_ID"]
    results = []
    for market_key in markets:
        print(f"\n=== {market_key} ===")
        for scenario in SCENARIOS[market_key]:
            results.append(record(scenario, market_key))

    out = config.EVIDENCE_DIR / "q3_call_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{len(results)} calls -> {out}")
    return results


if __name__ == "__main__":
    run(sys.argv[1:] or None)
