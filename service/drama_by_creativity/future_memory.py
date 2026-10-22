from __future__ import annotations
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
import jinja2
from .api_client import LLM
from .llm_inference import llm_inference_json
from .utils import get_model_config

EXTRACT_MEMORY_PROMPT = '\nYou maintain future-driving narrative memory for a serialized short-form drama. From this episode\'s final screenplay, extract events, details, clues, and foreshadowing that may matter later. Link them only to real nodes in the read-only Future Map.\n\nFD types (distinguish carefully):\n1. `event`: A consequential action, change, or outcome that has happened in this episode and directly supports or realizes an FM goal—not a static fact.\n2. `detail`: A clearly presented, significant detail of a character action, object, or event. Ignoring it would break continuity of knowledge, ownership, or causality, or it may be referred to in future plot developments.\n3. `clue`: A concrete lead, piece of evidence, anomaly, or traceable indication that may have long-term impact or recur throughout the plot, pointing toward a later discovery, truth, action, or conflict but not yet fully revealed or paid off.\n4. `foreshadowing`: Setup specifically directed toward a future FM goal. It establishes anticipation, a condition, or a portent without yet fulfilling that goal.\n\nExtraction and linking:\n1. Include only what the final screenplay clearly shows, states, or establishes. Do not infer or invent events.\n2. Keep information with substantial main-story value. Include only material that significantly serves a future goal or can plausibly drive later events or plant an important clue. Omit random lines, incidental atmosphere, bystander reactions, bit-part backstory, repetition, material that serves only momentary emotion, ordinary actions, thematic similarity alone, and tenuous associations.\n3. Assign each FD item its single best kind; do not duplicate content across types.\n4. `fm_links.target` must name an existing FM node; `contribution` must concretely explain the connection.\n5. One FD item may link to several nodes if its longer-term effects support each, but each link needs a clear basis.\n6. `contribution_type` is `setup`, `advance`, `turn`, `complete`, or `payoff`: prerequisite, progress, pivotal turn, goal fulfillment, or realization of earlier setup.\n7. `long_term_importance` (0–1) measures lasting influence on future episodes and possibly the ending; `importance` measures weight within this episode. Do not conflate them.\n\n## Current episode\nEpisode {{EpisodeNumber}}\n## Episode outline\n{{EpisodeOutline}}\n## Read-only Future Map\n{{FutureMap}}\n## Final episode screenplay\n{{EpisodeScript}}\n\nOutput one JSON object only, with all descriptive text in English:\n{\n  "future_driving_nodes": [{\n    "temp_id": "D1",\n    "kind": "event|detail|clue|foreshadowing",\n    "title": "Short title, no more than 10 English words",\n    "summary": "One or two sentences fully describing what the episode establishes.",\n    "participants": ["Character name"],\n    "entities": ["Clue, object, organization, or place"],\n    "evidence": "Short verbatim evidence from the screenplay",\n    "importance": 0.0,\n    "long_term_importance": 0.0,\n    "fm_links": [{"target": "FM001", "contribution_type": "setup|advance|turn|complete|payoff",\n      "contribution": "How this content serves that future goal", "confidence": 0.0}]\n  }]\n}\n'
_ALLOWED_FUTURE_NODE_KINDS = {"event", "detail", "clue", "foreshadowing"}
_ALLOWED_FUTURE_CONTRIBUTION_TYPES = {"setup", "advance", "turn", "complete", "payoff"}


def _safe_name(value: str) -> str:
    value = re.sub("[^0-9A-Za-z_.\\-\\u4e00-\\u9fff]+", "_", value or "story")
    return value[:120] or "story"


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _atomic_json_dump(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".memory_", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class FutureNarrativeMemory:
    def __init__(self, story_id: str):
        self.story_id = story_id
        output_root = os.environ.get("DRAMA_OUTPUT_DIR") or os.path.join(
            os.getcwd(), "output"
        )
        self.output_root = os.path.abspath(output_root)
        configured_root = os.environ.get("DRAMA_MEMORY_DIR")
        self.root = os.path.abspath(
            configured_root or os.path.join(self.output_root, "future_narramem")
        )
        self.story_dir = os.path.join(self.root, _safe_name(story_id))
        self.state_path = os.path.join(self.story_dir, "memory_state.json")
        self.future_map_path = os.path.join(
            self.output_root, "future_map", "future_map.json"
        )
        self.episode_contributions_path = os.path.join(
            self.output_root, "future_map", "episode_contributions.json"
        )
        self.future_driving_path = os.path.join(
            self.output_root, "future_map", "future_driving_graph.json"
        )
        self.data = self._load()

    def _load_future_map(self) -> Tuple[str, Set[str], Dict[str, Any]]:
        try:
            with open(self.future_map_path, "r", encoding="utf-8") as file:
                future_map = json.load(file)
            raw_nodes = (
                _as_list(future_map.get("nodes"))
                if isinstance(future_map, dict)
                else []
            )
            nodes = []
            valid_ids: Set[str] = set()
            for raw in raw_nodes:
                if not isinstance(raw, dict):
                    continue
                node_id = str(raw.get("node_id", "")).strip().upper()
                if not re.fullmatch("FM\\d+", node_id) or node_id in valid_ids:
                    continue
                valid_ids.add(node_id)
                nodes.append(
                    {
                        "node_id": node_id,
                        "parent_id": raw.get("parent_id"),
                        "depth": raw.get("depth"),
                        "goal": _clip(raw.get("goal"), 500),
                        "requirement": _clip(raw.get("requirement"), 500),
                    }
                )
            if not nodes:
                raise ValueError("nodes must not be empty")
            compact = {
                "map_type": future_map.get("map_type", "backward_requirement_tree"),
                "root_id": future_map.get("root_id", "FM001"),
                "nodes": nodes,
            }
            return (
                json.dumps(compact, ensure_ascii=False, indent=2),
                valid_ids,
                compact,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return (
                "No Future Map is available; output an empty future_driving_nodes array.",
                set(),
                {},
            )

    def _load_future_driving_graph(self) -> Dict[str, Any]:
        empty = {
            "schema_version": 1,
            "story_id": self.story_id,
            "future_map_path": self.future_map_path,
            "nodes": [],
            "edges": [],
            "last_updated_episode": -1,
            "updated_at": "",
        }
        if not os.path.exists(self.future_driving_path):
            return empty
        try:
            with open(self.future_driving_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                return empty
            if loaded.get("story_id") not in (None, self.story_id):
                return empty
            loaded["nodes"] = _as_list(loaded.get("nodes"))
            loaded["edges"] = _as_list(loaded.get("edges"))
            loaded["schema_version"] = 1
            loaded["story_id"] = self.story_id
            loaded["future_map_path"] = self.future_map_path
            return loaded
        except (OSError, ValueError, json.JSONDecodeError):
            return empty

    def _load_episode_contributions(self) -> Dict[int, List[Dict[str, Any]]]:
        if not os.path.exists(self.episode_contributions_path):
            return {}
        try:
            with open(self.episode_contributions_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            episodes = (
                _as_list(loaded.get("episodes")) if isinstance(loaded, dict) else []
            )
            contributions_by_episode: Dict[int, List[Dict[str, Any]]] = {}
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                try:
                    zero_based_episode_id = int(episode.get("episode_id")) - 1
                except (TypeError, ValueError):
                    continue
                if zero_based_episode_id < 0:
                    continue
                cleaned_contributions = []
                seen_node_ids: Set[str] = set()
                for contribution in _as_list(episode.get("contributions")):
                    if not isinstance(contribution, dict):
                        continue
                    node_id = str(contribution.get("node_id", "")).strip().upper()
                    if not re.fullmatch("FM\\d+", node_id) or node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)
                    cleaned_contributions.append(
                        {
                            "node_id": node_id,
                            "contribution_type": str(
                                contribution.get("contribution_type", "")
                            ).strip(),
                            "importance": str(
                                contribution.get("importance", "")
                            ).strip(),
                            "contribution": _clip(
                                contribution.get("contribution"), 240
                            ),
                        }
                    )
                contributions_by_episode[zero_based_episode_id] = cleaned_contributions
            return contributions_by_episode
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _collect_future_map_subtree(
        root_id: str, children_by_parent: Dict[str, List[str]]
    ) -> Set[str]:
        subtree_ids: Set[str] = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in subtree_ids:
                continue
            subtree_ids.add(current)
            stack.extend(children_by_parent.get(current, []))
        return subtree_ids

    def _empty(self):
        return {
            "schema_version": 1,
            "story_id": self.story_id,
            "last_updated_episode": -1,
            "updated_at": "",
        }

    def _load(self):
        if not os.path.exists(self.state_path):
            return self._empty()
        with open(self.state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("story_id") != self.story_id:
            raise ValueError("Memory story_id does not match")
        return state

    @contextmanager
    def _update_lock(self):
        os.makedirs(self.story_dir, exist_ok=True)
        lock_path = os.path.join(self.story_dir, ".memory.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save(self):
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json_dump(self.state_path, self.data)

    def retrieve(
        self, episode_outline: str, episode_id: int
    ) -> Tuple[str, Dict[str, Any]]:
        subtree_limit = int(os.environ.get("DRAMA_SUBTREE_MEMORY_LIMIT", "10"))
        if subtree_limit < 0:
            raise ValueError(
                "DRAMA_SUBTREE_MEMORY_LIMIT must be nonnegative; 0 means unlimited"
            )
        _, _, future_map_snapshot = self._load_future_map()
        contributions_by_episode = self._load_episode_contributions()
        future_driving_graph = self._load_future_driving_graph()
        node_by_id: Dict[str, Dict[str, Any]] = {}
        children_by_parent: Dict[str, List[str]] = {}
        for raw_node in _as_list(future_map_snapshot.get("nodes")):
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("node_id", "")).strip().upper()
            if not re.fullmatch("FM\\d+", node_id):
                continue
            node_by_id[node_id] = raw_node
        for node_id, raw_node in node_by_id.items():
            parent_id = str(raw_node.get("parent_id") or "").strip().upper()
            if parent_id and parent_id in node_by_id:
                children_by_parent.setdefault(parent_id, []).append(node_id)
        direct_node_ids: List[str] = []
        for contribution in contributions_by_episode.get(episode_id, []):
            node_id = str(contribution.get("node_id", "")).strip().upper()
            if node_id in node_by_id and node_id not in direct_node_ids:
                direct_node_ids.append(node_id)
        direct_node_id_set = set(direct_node_ids)
        descendant_node_ids: Set[str] = set()
        for node_id in direct_node_ids:
            descendant_node_ids.update(
                self._collect_future_map_subtree(node_id, children_by_parent)
            )
        descendant_node_ids.difference_update(direct_node_id_set)
        direct_items: Dict[str, List[Dict[str, Any]]] = {
            "event": [],
            "detail": [],
            "clue": [],
            "foreshadowing": [],
        }
        subtree_items: List[Dict[str, Any]] = []
        seen_direct: Set[str] = set()
        seen_subtree: Set[str] = set()
        future_nodes = sorted(
            (
                node
                for node in _as_list(future_driving_graph.get("nodes"))
                if isinstance(node, dict)
            ),
            key=lambda node: (int(node.get("episode_id", -1)), str(node.get("id", ""))),
        )
        for future_node in future_nodes:
            node_episode_id = int(future_node.get("episode_id", -1))
            if node_episode_id < 0 or node_episode_id >= episode_id:
                continue
            kind = str(future_node.get("kind", "event")).strip().lower()
            if kind not in _ALLOWED_FUTURE_NODE_KINDS:
                kind = "event"
            linked_targets = {
                str(link.get("target", "")).strip().upper()
                for link in _as_list(future_node.get("fm_links"))
                if isinstance(link, dict)
            }
            item = {
                "id": str(future_node.get("id", "")),
                "episode_id": node_episode_id,
                "kind": kind,
                "summary": _clip(
                    future_node.get("summary") or future_node.get("title"), 220
                ),
                "long_term_importance": _number(
                    future_node.get("long_term_importance"), 0.0
                ),
            }
            if not item["summary"]:
                continue
            item_id = item["id"] or f"{node_episode_id}:{item['summary']}"
            if linked_targets & direct_node_id_set:
                if item_id not in seen_direct:
                    direct_items[kind].append(item)
                    seen_direct.add(item_id)
                continue
            if (
                kind != "foreshadowing"
                and linked_targets & descendant_node_ids
                and (item_id not in seen_subtree)
            ):
                subtree_items.append(item)
                seen_subtree.add(item_id)
        if subtree_limit and len(subtree_items) > subtree_limit:
            subtree_items = sorted(
                subtree_items,
                key=lambda item: _number(item.get("long_term_importance"), 0.0),
                reverse=True,
            )[:subtree_limit]
        payload = {
            "episode_id": episode_id,
            "query": episode_outline,
            "direct_node_ids": direct_node_ids,
            "descendant_node_ids": sorted(descendant_node_ids),
            "direct_items": direct_items,
            "subtree_items": subtree_items,
            "subtree_memory_limit": subtree_limit,
        }
        context = self._format_context(payload)
        payload["formatted_context"] = context
        return (context, payload)

    def _format_context(self, payload: Dict[str, Any]) -> str:
        direct_items = payload.get("direct_items", {})
        subtree_items = _as_list(payload.get("subtree_items"))
        sections = [
            (
                "Events most directly relevant to this episode",
                _as_list(direct_items.get("event")),
            ),
            ("detail", _as_list(direct_items.get("detail"))),
            ("clue", _as_list(direct_items.get("clue"))),
            (
                "Foreshadowing tied to similar goals (pay off when appropriate)",
                _as_list(direct_items.get("foreshadowing")),
            ),
            ("Other potentially relevant events", subtree_items),
        ]
        lines: List[str] = []
        for title, items in sections:
            lines.append(f"{title}:")
            if items:
                for item in items:
                    lines.append(
                        f"- Episode {int(item.get('episode_id', -1)) + 1}: {_clip(item.get('summary', ''), 220)}"
                    )
            else:
                lines.append("- None")
        return "\n".join(lines)

    def _build_update_service(self, prompt: str) -> LLM:
        memory_config = get_model_config("future_memory_model")
        return LLM(
            model=memory_config["model"],
            system_prompt=memory_config["system_prompt"],
            temperature=min(float(memory_config.get("temperature", 0.4)), 0.3),
            top_p=memory_config["top_p"],
            max_new_tokens=memory_config["max_new_tokens"],
            template=jinja2.Template(prompt),
        )

    async def update_from_episode(self, episode_id, episode_outline, final_script):
        future_map_text, valid_ids, snapshot = self._load_future_map()
        service = self._build_update_service(EXTRACT_MEMORY_PROMPT)
        try:
            extraction = await llm_inference_json(
                params={
                    "EpisodeNumber": str(episode_id + 1),
                    "EpisodeOutline": episode_outline,
                    "FutureMap": future_map_text,
                    "EpisodeScript": final_script,
                },
                llm_service=service,
            )
            if not isinstance(extraction, dict):
                extraction = {}
        except Exception:
            extraction = {}
        with self._update_lock():
            self.data = self._load()
            graph = self._load_future_driving_graph()
            if valid_ids:
                graph["nodes"] = [
                    node
                    for node in graph["nodes"]
                    if int(node.get("episode_id", -1)) < episode_id
                ]
                retained_ids = {node["id"] for node in graph["nodes"]}
                graph["edges"] = [
                    edge
                    for edge in graph["edges"]
                    if edge.get("source") in retained_ids
                ]
                applied = self._apply_future_driving_extraction(
                    episode_id, extraction, valid_ids, graph
                )
                graph["last_updated_episode"] = episode_id
                graph["updated_at"] = datetime.now().isoformat(timespec="seconds")
                graph["future_map_root_id"] = snapshot.get("root_id")
            else:
                applied = {"nodes": [], "edges": []}
            self.data["last_updated_episode"] = max(
                self.data["last_updated_episode"], episode_id
            )
            self._save()
            _atomic_json_dump(self.future_driving_path, graph)
        return {
            "future_driving_nodes": applied["nodes"],
            "future_map_edges": applied["edges"],
        }

    def _apply_future_driving_extraction(
        self,
        episode_id: int,
        extraction: Dict[str, Any],
        valid_future_map_ids: Set[str],
        future_driving_graph: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not valid_future_map_ids:
            return {"nodes": [], "edges": []}
        new_nodes: List[Dict[str, Any]] = []
        new_edges: List[Dict[str, Any]] = []
        for index, raw in enumerate(
            _as_list(extraction.get("future_driving_nodes")), start=1
        ):
            if not isinstance(raw, dict):
                continue
            summary = _clip(raw.get("summary"), 500)
            title = _clip(raw.get("title"), 40)
            if not summary or not title:
                continue
            kind = str(raw.get("kind", "event")).strip().lower()
            if kind not in _ALLOWED_FUTURE_NODE_KINDS:
                kind = "event"
            normalized_links: List[Dict[str, Any]] = []
            seen_targets: Set[str] = set()
            for raw_link in _as_list(raw.get("fm_links")):
                if not isinstance(raw_link, dict):
                    continue
                match = re.search(
                    "\\b(FM\\d+)\\b", str(raw_link.get("target", "")), re.IGNORECASE
                )
                target = match.group(1).upper() if match else ""
                contribution_type = (
                    str(raw_link.get("contribution_type", "")).strip().lower()
                )
                contribution = _clip(raw_link.get("contribution"), 500)
                if (
                    target not in valid_future_map_ids
                    or target in seen_targets
                    or contribution_type not in _ALLOWED_FUTURE_CONTRIBUTION_TYPES
                    or (not contribution)
                ):
                    continue
                seen_targets.add(target)
                normalized_links.append(
                    {
                        "target": target,
                        "contribution_type": contribution_type,
                        "contribution": contribution,
                        "confidence": _number(raw_link.get("confidence"), 0.6),
                    }
                )
            if not normalized_links:
                continue
            node_id = f"FD{episode_id + 1:03d}_{len(new_nodes) + 1:02d}"
            node = {
                "id": node_id,
                "episode_id": episode_id,
                "kind": kind,
                "title": title,
                "summary": summary,
                "participants": [
                    _clip(item, 40) for item in _as_list(raw.get("participants"))[:12]
                ],
                "entities": [
                    _clip(item, 60) for item in _as_list(raw.get("entities"))[:16]
                ],
                "evidence": _clip(raw.get("evidence"), 240),
                "importance": _number(raw.get("importance"), 0.7),
                "long_term_importance": _number(raw.get("long_term_importance"), 0.0),
                "fm_links": normalized_links,
            }
            new_nodes.append(node)
            for link in normalized_links:
                new_edges.append(
                    {
                        "source": node_id,
                        "target": link["target"],
                        "type": "serves",
                        "contribution_type": link["contribution_type"],
                        "contribution": link["contribution"],
                        "confidence": link["confidence"],
                        "created_episode": episode_id,
                    }
                )
        future_driving_graph["nodes"].extend(new_nodes)
        future_driving_graph["edges"].extend(new_edges)
        return {"nodes": new_nodes, "edges": new_edges}
