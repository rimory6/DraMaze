from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

STATE_UPDATE_PROMPT = '\nYou maintain the current state of a serialized drama. Update characters, relationships, and assets or environments using only the final screenplay of this episode.\n\nState is a snapshot that remains true at the episode\'s end. Give each entity a few short, stable attributes covering distinct aspects, with a one-sentence value for each.\n\n## Current episode\nEpisode {{EpisodeNumber}}\n## Episode outline\n{{EpisodeOutline}}\n## Complete state snapshot before this episode\n{{PreviousState}}\n## Final episode screenplay\n{{EpisodeScript}}\n\nUpdate rules:\n1. Output only entities and attributes actually affected by this episode. The program preserves omitted entities and attributes verbatim; never remove them merely to compress the state.\n2. Use `set` to add or replace an attribute. Each value must be one brief sentence describing a fact true at this episode\'s end.\n3. Use `delete` only when this episode clearly invalidates an old attribute without replacing it. If there is a new value, put it in `set`. Include `delete` for every reported entity; use [] when there is nothing to delete.\n4. Use `delete_characters`, `delete_relationships`, or `delete_assets` only when an entire entity is no longer worth retaining in the story\'s current state.\n5. Character attributes may cover identity and allegiance, physical condition, location and circumstances, emotional state and goals, capabilities and limits, and knowledge. Maximum eight attributes per character.\n6. Relationship attributes may cover nature of the bond, trust, conflict and obligations, promises and boundaries. Maximum six attributes per relationship. Form a relationship key by joining all participants\' names with `|`. On first appearance, you may supply `participants`.\n7. Record character-related state under `characters`, not `assets`. Assets cover concrete objects, props, evidence, places, environments, organizations, and world rules; attributes may cover possession and control, location, condition, function, and access. Maximum six attributes per asset.\n8. Do not give a character knowledge possessed only by the audience; do not present plans as completed or infer changes from the outline that do not occur in the screenplay.\n9. In `summary`, recount the episode\'s important actions, outcomes, changes to characters, relationships, and assets, and ending circumstances in causal order, in approximately 120–220 English words for use by later episodes.\n\nOutput a valid JSON object only. Use English for every dynamic key and value:\n{\n  "patch": {\n    "characters": {"Character name": {"set": {"state attribute": "One-sentence state at episode end."}, "delete": []}},\n    "relationships": {"Character A|Character B": {"participants": ["Character A", "Character B"], "set": {"relationship attribute": "One-sentence current state."}, "delete": []}},\n    "assets": {"Object or place name": {"kind": "object|environment|place|organization|rule|other", "set": {"state attribute": "One-sentence current state."}, "delete": []}},\n    "delete_characters": [], "delete_relationships": [], "delete_assets": []\n  },\n  "summary": "Episode synopsis"\n}\n{% if ValidationFeedback %}The previous output failed validation: {{ValidationFeedback}}{% endif %}\n'
MAX_ATTRS = {"characters": 8, "relationships": 6, "assets": 6}


def _safe_name(value: str) -> str:
    return (
        re.sub("[^0-9A-Za-z_.\\-\\u4e00-\\u9fff]+", "_", value or "story")[:120]
        or "story"
    )


def _atomic_dump(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".state_", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def empty_state() -> Dict[str, Dict[str, Any]]:
    return {"characters": {}, "relationships": {}, "assets": {}}


def state_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sentence(value: Any) -> str:
    text = re.sub("\\s+", " ", str(value or "")).strip()
    if not text:
        raise ValueError("A state value must not be empty")
    return text


def _string_list(value: Any, field: str) -> List[str]:
    if not isinstance(value, list) or any(
        (not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a list of strings")
    return [item.strip() for item in value]


def _relationship_key(value: str) -> str:
    names = _relationship_participants(value)
    return "|".join(sorted(dict.fromkeys(names))) if len(names) >= 2 else value.strip()


def _relationship_participants(value: str) -> List[str]:
    names = [name.strip() for name in re.split("[|｜/、]", value) if name.strip()]
    return list(dict.fromkeys(names))


def validate_extraction(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("patch"), dict):
        raise ValueError("Missing patch object")
    patch = value["patch"]
    for category in ("characters", "relationships", "assets"):
        if not isinstance(patch.get(category), dict):
            raise ValueError(f"patch.{category} must be an object")
        for entity, operation in patch[category].items():
            if (
                not isinstance(entity, str)
                or not entity.strip()
                or (not isinstance(operation, dict))
            ):
                raise ValueError(f"Invalid entity or operation in patch.{category}")
            if not isinstance(operation.get("set"), dict):
                raise ValueError(f"{entity}.set must be an object")
            if "delete" not in operation:
                operation["delete"] = []
            _string_list(operation.get("delete"), f"{entity}.delete")
            if len(operation["set"]) > MAX_ATTRS[category]:
                raise ValueError(f"Too many attributes in one update for {entity}")
            for attribute, state_value in operation["set"].items():
                if not isinstance(attribute, str) or not attribute.strip():
                    raise ValueError(f"{entity} has an empty attribute name")
                _sentence(state_value)
            if category == "relationships":
                key_participants = _relationship_participants(entity)
                if len(key_participants) < 2:
                    raise ValueError(
                        f"Relationship key {entity} must join at least two names with |"
                    )
                if "participants" in operation:
                    participants = _string_list(
                        operation["participants"], f"{entity}.participants"
                    )
                    if len(participants) < 2:
                        raise ValueError(
                            f"{entity}.participants requires at least two names"
                        )
    for field in ("delete_characters", "delete_relationships", "delete_assets"):
        _string_list(patch.get(field), f"patch.{field}")
    summary = value.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be nonempty text")
    return value


def apply_patch(
    previous: Dict[str, Any], extraction: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    validate_extraction(extraction)
    state = copy.deepcopy(previous or empty_state())
    for category in empty_state():
        state.setdefault(category, {})
    patch = extraction["patch"]
    changed: List[Dict[str, str]] = []
    deleted: List[Dict[str, str]] = []
    delete_fields = {
        "characters": "delete_characters",
        "relationships": "delete_relationships",
        "assets": "delete_assets",
    }
    for category, delete_field in delete_fields.items():
        for entity in patch[delete_field]:
            if category == "relationships":
                entity = _relationship_key(entity)
            if state[category].pop(entity, None) is not None:
                deleted.append(
                    {"category": category, "entity": entity, "attribute": "*"}
                )
        for entity, operation in patch[category].items():
            if category == "relationships":
                entity = _relationship_key(entity)
            current = state[category].setdefault(entity, {})
            if category == "relationships":
                if "participants" not in current:
                    supplied = operation.get("participants")
                    current["participants"] = (
                        list(supplied)
                        if supplied
                        else _relationship_participants(entity)
                    )
                attrs = current.setdefault("states", {})
            elif category == "assets":
                current["kind"] = operation.get("kind", current.get("kind", "other"))
                attrs = current.setdefault("states", {})
            else:
                attrs = current
            for attribute in operation["delete"]:
                if attrs.pop(attribute, None) is not None:
                    deleted.append(
                        {"category": category, "entity": entity, "attribute": attribute}
                    )
            for attribute, state_value in operation["set"].items():
                attrs[attribute.strip()] = _sentence(state_value)
                changed.append(
                    {
                        "category": category,
                        "entity": entity,
                        "attribute": attribute.strip(),
                    }
                )
            if len(attrs) > MAX_ATTRS[category]:
                protected = {attribute.strip() for attribute in operation["set"]}
                removable = [
                    attribute for attribute in attrs if attribute not in protected
                ]
                overflow = len(attrs) - MAX_ATTRS[category]
                if len(removable) < overflow:
                    raise ValueError(
                        f"{entity} exceeds {MAX_ATTRS[category]} state attributes after update"
                    )
                for attribute in removable[:overflow]:
                    attrs.pop(attribute)
                    deleted.append(
                        {"category": category, "entity": entity, "attribute": attribute}
                    )
    return (state, {"changed": changed, "deleted": deleted})


def _tokens(text: str) -> List[str]:
    lowered = str(text or "").lower()
    latin = re.findall("[a-z0-9_]+", lowered)
    chinese_blocks = re.findall("[\\u4e00-\\u9fff]+", lowered)
    result = list(latin)
    for block in chinese_blocks:
        result.extend(
            block
            if len(block) == 1
            else [block[i : i + 2] for i in range(len(block) - 1)]
        )
    return result


def flatten_state(
    state: Dict[str, Any], metadata: Dict[str, int] | None = None
) -> List[Dict[str, Any]]:
    metadata = metadata or {}
    records: List[Dict[str, Any]] = []
    for category in ("characters", "relationships", "assets"):
        for entity, raw in state.get(category, {}).items():
            if category == "characters":
                attrs, participants, kind = (raw, [entity], "character")
            else:
                attrs = raw.get("states", {})
                participants = (
                    raw.get("participants") or _relationship_participants(entity)
                    if category == "relationships"
                    else [entity]
                )
                kind = raw.get("kind", "relationship")
            for attribute, value in attrs.items():
                key = f"{category}\x1f{entity}\x1f{attribute}"
                records.append(
                    {
                        "category": category,
                        "entity": entity,
                        "attribute": attribute,
                        "value": value,
                        "participants": participants,
                        "kind": kind,
                        "last_changed_episode": int(metadata.get(key, -1)),
                    }
                )
    return records


def retrieve_state(
    state: Dict[str, Any],
    query: str,
    episode_id: int,
    metadata: Dict[str, int] | None = None,
    max_records: int = 20,
) -> Tuple[str, Dict[str, Any]]:
    records = flatten_state(state, metadata)
    query_tokens = _tokens(query)
    query_counts = Counter(query_tokens)
    document_tokens = [
        _tokens(f"{r['entity']} {r['attribute']} {r['value']}") for r in records
    ]
    document_frequency = Counter(
        (token for tokens in document_tokens for token in set(tokens))
    )
    average_length = sum(map(len, document_tokens)) / max(len(document_tokens), 1)
    scored = []
    for record, tokens in zip(records, document_tokens):
        counts = Counter(tokens)
        bm25 = 0.0
        for token, qfreq in query_counts.items():
            if not counts[token]:
                continue
            idf = math.log(
                1
                + (len(records) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            tf = (
                counts[token]
                * 2.2
                / (
                    counts[token]
                    + 1.2 * (0.25 + 0.75 * len(tokens) / max(average_length, 1))
                )
            )
            bm25 += idf * tf * min(qfreq, 2)
        mentioned = record["entity"] in query
        participants = [name for name in record["participants"] if name]
        participant_hits = sum((name in query for name in participants))
        entity_boost = (
            2.4
            if mentioned
            else (
                2.0
                if len(participants) >= 2 and participant_hits == len(participants)
                else 0.75 * participant_hits
            )
        )
        age = max(0, episode_id - record["last_changed_episode"])
        recency = (
            0.45 * math.exp(-age / 5) if record["last_changed_episode"] >= 0 else 0.0
        )
        record = dict(record)
        record["score"] = round(bm25 + entity_boost + recency, 5)
        record["matched"] = bool(bm25 > 0 or entity_boost > 0)
        record["reasons"] = {
            "bm25": round(bm25, 5),
            "entity_boost": entity_boost,
            "recency": round(recency, 5),
        }
        scored.append(record)
    scored.sort(
        key=lambda item: (item["score"], item["last_changed_episode"]), reverse=True
    )
    selected, used = ([], 0)
    for record in scored:
        if len(selected) >= max_records:
            break
        if not record["matched"]:
            continue
        line_size = len(record["entity"] + record["attribute"] + record["value"]) + 20
        selected.append(record)
        used += line_size
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in selected:
        grouped[record["category"]][record["entity"]].append(
            {record["attribute"]: record["value"]}
        )
    labels = {
        "characters": "Character states",
        "relationships": "Relationship states",
        "assets": "Assets, objects, and environmental states",
    }
    lines = [
        "Current-state facts retrieved for this episode; all remain true before it begins:"
    ]
    for category in ("characters", "relationships", "assets"):
        lines.append(f"### {labels[category]}")
        if not grouped[category]:
            lines.append("- No strongly relevant state")
        for entity, attrs in grouped[category].items():
            lines.append(f"- {entity}")
            for pair in attrs:
                attribute, value = next(iter(pair.items()))
                lines.append(f"  - {attribute}: {value}")
    return (
        "\n".join(lines),
        {
            "query": query,
            "max_records": max_records,
            "selected_chars": used,
            "selected": selected,
            "total_records": len(records),
            "selected_records": len(selected),
        },
    )


class StateMemory:
    def __init__(self, story_id: str):
        output_root = os.path.abspath(
            os.environ.get("DRAMA_OUTPUT_DIR") or os.path.join(os.getcwd(), "output")
        )
        root = os.path.abspath(
            os.environ.get("DRAMA_STATE_MEMORY_DIR")
            or os.path.join(output_root, "worldstate")
        )
        self.story_id = story_id
        self.story_dir = os.path.join(root, _safe_name(story_id))
        self.path = os.path.join(self.story_dir, "state_memory.json")
        self.data = self._load()
        self.last_retrieval = None

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if loaded.get("story_id") == self.story_id:
                return loaded
        return {
            "schema_version": 1,
            "story_id": self.story_id,
            "state": empty_state(),
            "metadata": {},
            "history": [],
            "episode_summaries": [],
            "last_updated_episode": -1,
        }

    def retrieve(self, query: str, episode_id: int) -> Tuple[str, Dict[str, Any]]:
        max_records = int(os.environ.get("DRAMA_STATE_MEMORY_LIMIT", "20"))
        context, audit = retrieve_state(
            self.data["state"],
            query,
            episode_id,
            self.data.get("metadata"),
            max_records,
        )
        audit.update(
            {
                "schema_version": 1,
                "story_id": self.story_id,
                "episode_id": episode_id,
                "last_updated_episode": self.data.get("last_updated_episode", -1),
                "state_sha256": state_digest(self.data["state"]),
            }
        )
        self.last_retrieval = audit
        return (context, audit)

    def previous_state_json(self) -> str:
        return json.dumps(self.data["state"], ensure_ascii=False, indent=2)

    def recent_summaries(self, episode_id: int, limit: int = 5) -> str:
        rows = [
            row
            for row in self.data.get("episode_summaries", [])
            if row["episode_id"] < episode_id
        ][-limit:]
        return (
            "\n".join(
                (f"- Episode {row['episode_id'] + 1}: {row['summary']}" for row in rows)
            )
            or "No previous episode synopsis"
        )

    def commit(self, episode_id: int, extraction: Dict[str, Any]) -> Dict[str, Any]:
        if episode_id > self.data.get("last_updated_episode", -1) + 1:
            raise ValueError("State memory must be updated in episode order")
        if episode_id <= self.data.get("last_updated_episode", -1):
            history = [
                row for row in self.data["history"] if row["episode_id"] < episode_id
            ]
            self.data["state"] = copy.deepcopy(
                history[-1]["state"] if history else empty_state()
            )
            self.data["history"] = history
            self.data["episode_summaries"] = [
                row
                for row in self.data["episode_summaries"]
                if row["episode_id"] < episode_id
            ]
            self.data["metadata"] = {
                k: v for k, v in self.data.get("metadata", {}).items() if v < episode_id
            }
        next_state, changes = apply_patch(self.data["state"], extraction)
        for item in changes["changed"]:
            key = f"{item['category']}\x1f{item['entity']}\x1f{item['attribute']}"
            self.data["metadata"][key] = episode_id
        for item in changes["deleted"]:
            prefix = f"{item['category']}\x1f{item['entity']}\x1f"
            if item["attribute"] == "*":
                self.data["metadata"] = {
                    k: v
                    for k, v in self.data["metadata"].items()
                    if not k.startswith(prefix)
                }
            else:
                self.data["metadata"].pop(prefix + item["attribute"], None)
        self.data["state"] = next_state
        self.data["history"].append(
            {
                "episode_id": episode_id,
                "state": copy.deepcopy(next_state),
                "changes": changes,
            }
        )
        summary = re.sub("\\s+", " ", extraction["summary"]).strip()[:500]
        self.data["episode_summaries"].append(
            {"episode_id": episode_id, "summary": summary}
        )
        self.data["last_updated_episode"] = episode_id
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_dump(self.path, self.data)
        return {"state": next_state, "summary": summary, **changes}
