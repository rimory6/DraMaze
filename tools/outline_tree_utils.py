from __future__ import annotations
from typing import Any

CANDIDATE_SCORE_DEFAULTS = {"logic": 7, "drive": 7, "novelty": 6, "appeal": 6}


def event_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("event", "")).strip()
    return ""


def extract_stage_payload(outline: dict[str, Any], stage_name: str) -> dict[str, Any]:
    framework = outline.get("framework", [])
    if not isinstance(framework, list):
        raise ValueError("outline.framework must be an array")
    for stage in framework:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("stage", "")).strip() == stage_name:
            events = [
                text
                for item in stage.get("event_list", [])
                if (text := event_text(item))
            ]
            if not events:
                raise ValueError(f"stage `{stage_name}` has no usable events")
            return {"stage": stage_name, "seed_events": events}
    raise ValueError(f"Stage not found: {stage_name}")


def collect_previous_stage_events(
    outline: dict[str, Any], target_stage_name: str
) -> list[str]:
    framework = outline.get("framework", [])
    if not isinstance(framework, list):
        raise ValueError("outline.framework must be an array")
    previous_events: list[str] = []
    for stage in framework:
        if not isinstance(stage, dict):
            continue
        current_name = str(stage.get("stage", "")).strip()
        if current_name == target_stage_name:
            return previous_events
        previous_events.extend(
            (text for item in stage.get("event_list", []) if (text := event_text(item)))
        )
    raise ValueError(f"Stage not found: {target_stage_name}")


class NodeIdFactory:
    def __init__(self) -> None:
        self._next = 1

    def new_id(self) -> str:
        value = self._next
        self._next += 1
        return f"n{value:04d}"


def sanitize_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def sanitize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def parse_candidates(payload: Any, branch_factor: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("Candidates must be a JSON object")
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates must be an array")
    normalized: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event", "")).strip()
        if not event or event in seen_events:
            continue
        seen_events.add(event)
        normalized.append(
            {
                "event": event,
                "progress_to_anchor": sanitize_int(
                    item.get("progress_to_anchor"), default=50, minimum=0, maximum=100
                ),
                "anchor_reached": sanitize_bool(
                    item.get("anchor_reached"), default=False
                ),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        if len(normalized) >= branch_factor:
            break
    if not normalized:
        raise ValueError("Model API returned no usable candidates")
    return normalized


def parse_candidate_step_scores(
    payload: Any, expected_count: int
) -> list[dict[str, Any] | None]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("Candidate scores must be a JSON object")
    raw_scores = payload.get("scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("scores must be an array")
    score_map: dict[int, dict[str, Any]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        candidate_index = sanitize_int(
            item.get("candidate_index"), default=-1, minimum=-1, maximum=10000
        )
        if candidate_index < 1 or candidate_index > expected_count:
            continue
        score = {}
        for field in CANDIDATE_SCORE_DEFAULTS:
            try:
                value = int(item[field])
            except (KeyError, TypeError, ValueError, OverflowError):
                score[field] = None
            else:
                score[field] = max(1, min(10, value))
        score["reason"] = str(item.get("reason", "")).strip()
        score_map[candidate_index - 1] = score
    return [score_map.get(index) for index in range(expected_count)]


def choose_top_candidates_for_expansion(
    scored_candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        scored_candidates,
        key=lambda item: (
            item["expansion_score"],
            item["logic"],
            item["drive"],
            item["novelty"],
            item["appeal"],
            item["progress_to_anchor"],
            1 if item["anchor_reached"] else 0,
        ),
        reverse=True,
    )
    return ranked[: max(1, top_k)]


def create_node_payload(
    *,
    node_id: str,
    event: str,
    depth: int,
    progress_to_anchor: int,
    anchor_reached: bool,
    reason: str,
    selected_for_expansion: bool,
    logic: int | None = None,
    drive: int | None = None,
    novelty: int | None = None,
    appeal: int | None = None,
    evaluation_reason: str = "",
    expansion_score: float | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "event": event,
        "depth": depth,
        "progress_to_anchor": progress_to_anchor,
        "anchor_reached": anchor_reached,
        "generation_reason": reason,
        "selected_for_expansion": selected_for_expansion,
        "logic": logic,
        "drive": drive,
        "novelty": novelty,
        "appeal": appeal,
        "evaluation_reason": evaluation_reason,
        "expansion_score": expansion_score,
        "children": [],
    }
