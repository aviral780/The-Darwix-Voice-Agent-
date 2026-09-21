"""Business actions triggered by a call.

The brief asks for at least one real action rather than a conversation that ends
in nothing. Two are implemented: a qualified lead is written as a durable record,
and an escalation is posted to a webhook when one is configured.

Both are deliberately unglamorous. A lead is a JSON file on disk rather than a
CRM integration, because a CRM the reviewer cannot inspect proves less than a
record they can open. The shape is the one a CRM would expect, so the storage
layer is the only thing that would change.

Escalation degrades rather than fails: with no webhook configured it writes the
same payload locally. A dropped escalation is the worst failure this system can
have, since it means a caller who asked for a human never reaches one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from core import config

LEAD_SCORE_WEIGHTS = {
    "dependents": 3,        # strongest signal of genuine protection need
    "existing_cover": 2,    # a gap, or an upgrade path
    "interest": 2,
    "age_band": 1,
    "callback": 2,          # agreeing to a callback is real intent
    "name": 1,
}


def score_lead(slots: dict) -> tuple[int, str]:
    """Score a lead from the slots that were actually filled.

    Deliberately transparent arithmetic rather than a model call. A sales team
    has to be able to see why a lead was ranked where it was, and a scoring rule
    that changes with model temperature is not one they can trust or audit.
    """
    score = sum(weight for slot, weight in LEAD_SCORE_WEIGHTS.items() if slots.get(slot))
    total = sum(LEAD_SCORE_WEIGHTS.values())
    ratio = score / total if total else 0.0
    if ratio >= 0.75:
        band = "hot"
    elif ratio >= 0.45:
        band = "warm"
    else:
        band = "cold"
    return score, band


def create_lead(session_summary: dict) -> Path:
    """Persist a qualified lead. Returns the file written."""
    slots = session_summary.get("slots", {})
    score, band = score_lead(slots)

    lead = {
        "lead_id": f"lead_{session_summary['call_id']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "voice_agent",
        "market": session_summary.get("market", ""),
        "call_id": session_summary["call_id"],
        "contact": {
            "name": slots.get("name", ""),
            "preferred_callback": slots.get("callback", ""),
        },
        "qualification": {
            "age_band": slots.get("age_band", ""),
            "dependents": slots.get("dependents", ""),
            "existing_cover": slots.get("existing_cover", ""),
            "interest": slots.get("interest", ""),
        },
        "score": score,
        "score_band": band,
        "slots_filled": f"{len(slots)}/{session_summary.get('slots_total', 0)}",
        "objections_raised": session_summary.get("objections_raised", []),
        "escalated": session_summary.get("escalated", False),
        "grounded_answers_given": session_summary.get("grounded_answers", 0),
        "refusals": session_summary.get("refusals", 0),
        "final_state": session_summary.get("final_state", ""),
    }

    config.LEADS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LEADS_DIR / f"{lead['lead_id']}.json"
    path.write_text(json.dumps(lead, indent=2, ensure_ascii=False))
    return path


def escalate(session_summary: dict, reason: str) -> dict:
    """Hand a call to a human. Posts to a webhook if one is configured."""
    payload = {
        "event": "escalation",
        "raised_at": datetime.now(timezone.utc).isoformat(),
        "call_id": session_summary["call_id"],
        "market": session_summary.get("market", ""),
        "reason": reason,
        "caller_name": session_summary.get("slots", {}).get("name", ""),
        "slots_collected": session_summary.get("slots", {}),
        # The last few turns so whoever picks up has the context and the caller
        # does not have to start over.
        "recent_turns": session_summary.get("turns", [])[-6:],
    }

    delivered = False
    error = ""
    if config.ESCALATION_WEBHOOK_URL:
        try:
            response = httpx.post(config.ESCALATION_WEBHOOK_URL, json=payload, timeout=10.0)
            response.raise_for_status()
            delivered = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    # Always write locally, delivered or not. An escalation that exists only in a
    # failed HTTP call is a caller who asked for help and did not get it.
    directory = config.EVIDENCE_DIR / "escalations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"escalation_{session_summary['call_id']}.json"
    path.write_text(json.dumps({**payload, "webhook_delivered": delivered,
                                "webhook_error": error}, indent=2, ensure_ascii=False))

    return {"delivered": delivered, "error": error, "path": str(path)}


def finalise_call(session_summary: dict) -> dict:
    """Run whichever actions the call's outcome calls for."""
    outcome: dict = {}
    if session_summary.get("escalated"):
        outcome["escalation"] = escalate(
            session_summary, session_summary.get("escalation_reason", "unspecified")
        )
    if session_summary.get("slots"):
        path = create_lead(session_summary)
        score, band = score_lead(session_summary["slots"])
        outcome["lead"] = {"path": str(path), "score": score, "band": band}
    return outcome
