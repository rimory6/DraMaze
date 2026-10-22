import json
import re
import json_repair
from typing import Any

_JSON_FENCE = re.compile("```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```", re.S)


def _extract_json_candidate(text: str) -> str:
    raw = text.strip()
    if not raw:
        raise ValueError("Model API returned empty content")
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        return fenced.group(1)
    start_object = raw.find("{")
    end_object = raw.rfind("}")
    start_array = raw.find("[")
    end_array = raw.rfind("]")
    candidates = []
    if start_object != -1 and end_object != -1 and (end_object > start_object):
        candidates.append(raw[start_object : end_object + 1])
    if start_array != -1 and end_array != -1 and (end_array > start_array):
        candidates.append(raw[start_array : end_array + 1])
    if not candidates:
        return raw
    return max(candidates, key=len)


def parse_json_response(text: str) -> Any:
    candidate = _extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is not None:
            return json_repair.loads(candidate)
        raise
