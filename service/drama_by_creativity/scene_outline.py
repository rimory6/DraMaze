import json
import re
import jinja2
from .api_client import LLM
from .prompts.scene_repair import REPAIR_SCENE_OUTLINE_JSON_PROMPT
from .utils import get_model_config, post_process

TEXT_FIELDS = (
    "scene_id",
    "time_and_location",
    "core_function",
    "plot_summary",
    "ending_hook",
)
LIST_FIELDS = ("characters", "conflicts_and_tension", "key_actions")


def validate_scene_outline(raw, episode_number):
    value = json.loads(post_process(raw))
    if not isinstance(value, list) or not value:
        raise ValueError("Scene outline must be a nonempty JSON array of objects")
    seen = set()
    for scene in value:
        if not isinstance(scene, dict) or not scene:
            raise ValueError("Each scene must be a nonempty object")
        for key in TEXT_FIELDS:
            if not isinstance(scene.get(key), str) or not scene[key].strip():
                raise ValueError(f"Scene lacks a nonempty text field: {key}")
        for key in LIST_FIELDS:
            items = scene.get(key)
            if (
                not isinstance(items, list)
                or not items
                or any((not isinstance(x, str) or not x.strip() for x in items))
            ):
                raise ValueError(f"Scene field must be a nonempty string array: {key}")
        motives = scene.get("character_goals")
        if (
            not isinstance(motives, list)
            or not motives
            or any(
                (
                    not isinstance(m, dict)
                    or any(
                        (
                            not isinstance(m.get(k), str) or not m[k].strip()
                            for k in ("character", "motivation_and_goal")
                        )
                    )
                    for m in motives
                )
            )
        ):
            raise ValueError(
                "Character motivations must have nonempty character and goal fields"
            )
        match = re.fullmatch(
            "(\\d+)-(\\d+)(?:（[^）]*）|\\([^)]*\\))?", scene["scene_id"].strip()
        )
        if not match or int(match[1]) != int(episode_number) or int(match[2]) < 1:
            raise ValueError(
                f"Scene ID is invalid for this episode: {scene['scene_id']}"
            )
        scene["scene_id"] = scene["scene_id"].strip()
        scene_id = (int(match[1]), int(match[2]))
        if scene_id in seen:
            raise ValueError(f"Duplicate scene ID: {scene['scene_id']}")
        seen.add(scene_id)
    return value


def validate_repair_preserves_scenes(original, repaired):
    before = re.findall('"scene_id"\\s*:\\s*"\\s*(\\d+-\\d+)', original)
    after = [
        re.match("\\d+-\\d+", scene["scene_id"].strip()).group() for scene in repaired
    ]
    if before and before != after:
        raise ValueError(
            f"Repair changed scene count, IDs, or order: original {before}; repaired {after}"
        )


async def generate_scene_outline(
    params, llm_service, generation_attempts=3, repair_attempts=3
):
    if generation_attempts < 1 or repair_attempts < 1:
        raise ValueError("attempt counts must be positive")
    episode_number = params["EpisodeIdx"]
    last_error = None
    original = None

    async def attempt(service, request_params, preserve=None):
        nonlocal last_error
        raw = None
        try:
            raw = (await service.request(request_params)).response
            llm_service.last_output = raw
            result = validate_scene_outline(raw, episode_number)
            if preserve is not None:
                validate_repair_preserves_scenes(preserve, result)
            return (result, raw)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if raw is None:
                llm_service.last_output = getattr(service, "last_output", None)
            return (None, raw)

    for generation in range(1, generation_attempts + 1):
        result, raw = await attempt(llm_service, params)
        if result is not None:
            return result
        if raw is not None:
            original = raw
    if original is not None:
        config = get_model_config("json_repair_model")
        repair_service = LLM(
            **config, template=jinja2.Template(REPAIR_SCENE_OUTLINE_JSON_PROMPT)
        )
        for repair in range(1, repair_attempts + 1):
            result, _ = await attempt(
                repair_service,
                {
                    "EpisodeIdx": episode_number,
                    "Ret": original,
                    "ValidationError": last_error,
                },
                preserve=original,
            )
            if result is not None:
                return result
    return {}
