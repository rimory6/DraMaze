import os
import jinja2
import requests
from .api_client import LLM


def state_catalog(view):
    if not isinstance(view, list):
        raise ValueError("Scene review requires structured state records")
    field_map = {
        "characters": "character_state",
        "relationships": "relationship_state",
        "assets": "resources_and_evidence",
    }
    catalog = {}
    for index, record in enumerate(view):
        if not isinstance(record, dict) or record.get("category") not in field_map:
            raise ValueError(
                f"Scene review structured-state record {index} has invalid category"
            )
        entity, attribute, value = (
            record.get("entity"),
            record.get("attribute"),
            record.get("value"),
        )
        if any(
            (
                not isinstance(item, str) or not item.strip()
                for item in (entity, attribute, value)
            )
        ):
            raise ValueError(
                f"Scene review structured-state record {index} requires entity, attribute and value"
            )
        catalog[f"M{len(catalog) + 1:04d}"] = {
            "field": field_map[record["category"]],
            "text": f"{entity}｜{attribute}：{value}",
            "source": "structured_state_hybrid",
            "category": record["category"],
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "last_changed_episode": record.get("last_changed_episode", -1),
        }
    return catalog


def text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from text_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from text_values(item)


class SceneReview:
    def __init__(self, episode_number, params, view, base_url=None):
        self.episode = episode_number
        self.params = params
        self.catalog = state_catalog(view)
        self.base_url = base_url or os.environ.get("DRAMA_LLM_BASE_URL", "")

    @classmethod
    def from_memory(cls, memory, episode_index, params):
        if hasattr(memory, "data") and hasattr(memory, "story_dir"):
            record = memory.last_retrieval
            if record is None:
                raise ValueError("Retrieve state before scene review")
            from .state_memory import state_digest as structured_state_digest

            if (
                record.get("story_id") != memory.story_id
                or record.get("episode_id") != episode_index
                or record.get("last_updated_episode") != episode_index - 1
                or (
                    record.get("state_sha256")
                    != structured_state_digest(memory.data["state"])
                )
            ):
                raise ValueError(
                    "Scene review retrieval is not bound to current structured pre-episode state"
                )
            selected = record.get("selected")
            if not isinstance(selected, list):
                raise ValueError(
                    "Structured-state retrieval must contain the selected evidence records"
                )
            if not selected:
                return None
            return cls(
                episode_index + 1,
                params,
                selected,
            )
        raise TypeError("Scene review requires structured state memory")

    async def request(
        self, slot, system, user, validator, model=None, base_url=None, temperature=0
    ):
        model = model or os.environ.get("DRAMA_LLM_MODEL") or "your-model-name"
        url = base_url or self.base_url
        max_tokens = 8192 if temperature else 4096
        client = LLM(
            model=model,
            system_prompt=system,
            temperature=temperature,
            top_p=1,
            max_new_tokens=max_tokens,
            template=jinja2.Template("{{ Prompt }}"),
            base_url=url,
        )
        client.model = model
        client.max_tokens = max_tokens
        client.timeout = 600
        error_text = ""
        for attempt in range(1, 4):
            prompt = user + (
                f"\nFormat error: {error_text}. Keep the original task; correct the format and evidence references."
                if error_text
                else ""
            )
            raw = None
            client.last_output = None
            try:
                response = await client.request(
                    {"Prompt": prompt, "EpisodeNumber": str(self.episode)}
                )
                raw = response.response
                result = validator(raw)
            except (
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                requests.RequestException,
            ) as error:
                error_text = str(error)
            else:
                return result
        raise RuntimeError(
            f"Scene review {slot} schema/API retries exhausted: {error_text}"
        )
