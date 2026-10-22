from __future__ import annotations
import json
from .scene_evidence import evidence_catalog

RULES = 'Check the candidate scene outline against existing state memory. Report only confirmed, hard contradictions between them.\nM IDs denote retrieved states established before this episode; S IDs denote evidence in the candidate scenes. An episode outline is a plan, not past history.\n\nReport a conflict only when all three conditions hold:\n1. At least one cited M state clearly applies to the action;\n2. At least one S span from the same scene clearly contradicts that state;\n3. The scene does not show a valid transition explaining the change.\n\nExamples: someone confirmed dead acts in the present; a destroyed sole original reappears without explanation; a person explicitly unaware of a secret uses it without learning it; someone established elsewhere acts on-site without a move.\nContent that does not need to be reported includes, but is not limited to, the following categories: compatible details, ordinary new information, writing advice, insufficient evidence, missing setup alone, uncertain object identity, legitimate movement/transactions/discoveries/state changes shown in the scene, or flashbacks, hallucinations, voice-over, and keepsakes.\n\nOutput a JSON object:\n{"conflicts":[{"scene_id":"2-1","state_ids":["M0001"],"scene_span_ids":["S0001"],"reason":"A person confirmed dead acts in the present without an established explanation."}]}\nIf there are no confirmed hard conflicts, output {"conflicts":[]}.'


def _json(raw: str):
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def validate_conflicts(raw: str, scenes: list[dict], states: dict) -> dict:
    value = _json(raw)
    if isinstance(value, list):
        conflicts = value
    elif isinstance(value, dict) and isinstance(value.get("conflicts"), list):
        conflicts = value["conflicts"]
    else:
        raise ValueError('Only {"conflicts": [...]} is required')
    spans = evidence_catalog(scenes)
    scene_ids = {scene["scene_id"] for scene in scenes}
    normalized = []
    seen = set()
    for index, item in enumerate(conflicts):
        if not isinstance(item, dict):
            raise ValueError(f"conflicts[{index}] must be an object")
        scene_id = item.get("scene_id")
        state_ids = item.get("state_ids")
        span_ids = item.get("scene_span_ids")
        reason = item.get("reason")
        if not isinstance(scene_id, str) or scene_id not in scene_ids:
            raise ValueError(f"conflicts[{index}] has an unknown scene_id")
        if (
            not isinstance(state_ids, list)
            or not state_ids
            or any((not isinstance(ref, str) or ref not in states for ref in state_ids))
        ):
            raise ValueError(f"conflicts[{index}].state_ids must cite existing M IDs")
        if (
            not isinstance(span_ids, list)
            or not span_ids
            or any((not isinstance(ref, str) or ref not in spans for ref in span_ids))
        ):
            raise ValueError(
                f"conflicts[{index}].scene_span_ids must cite existing S IDs"
            )
        if any((spans[ref]["scene_id"] != scene_id for ref in span_ids)):
            raise ValueError(f"conflicts[{index}] cites S evidence from another scene")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise ValueError(
                f"conflicts[{index}].reason must contain 1 to 300 characters"
            )
        state_ids = list(dict.fromkeys(state_ids))
        span_ids = list(dict.fromkeys(span_ids))
        key = (scene_id, tuple(state_ids), tuple(span_ids), reason.strip())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "scene_id": scene_id,
                "state_ids": state_ids,
                "scene_span_ids": span_ids,
                "reason": reason.strip(),
                "classification": "hard_conflict",
            }
        )
    return {
        "status": "fail" if normalized else "pass",
        "action": "repair_candidate" if normalized else "keep",
        "conflicts": normalized,
        "checks": normalized,
        "auto_repair_executed": False,
    }
