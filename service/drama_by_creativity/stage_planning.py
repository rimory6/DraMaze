from __future__ import annotations
import json
import os
import re
from pathlib import Path
import jinja2
from .api_client import LLM
from .llm_inference import llm_inference_json
from .realtime_output import save_realtime
from .utils import get_model_config
from .prompts.stage_episode_plan import GENERATE_STAGE_EPISODE_PLAN_PROMPT


def extract_stages(outline):
    if isinstance(outline, str):
        try:
            payload = json.loads(outline)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            outline = payload
        else:
            sections = re.findall(
                "^### ([^\\n]+)\\n(.*?)(?=^#{1,3} |\\Z)",
                outline,
                flags=re.MULTILINE | re.DOTALL,
            )
            outline = {
                "framework": [
                    {
                        "stage": name.strip(),
                        "event_list": [
                            {"event": event.strip()}
                            for event in re.findall(
                                "^- (.*?)(?=^- |\\Z)",
                                body,
                                flags=re.MULTILINE | re.DOTALL,
                            )
                        ],
                    }
                    for name, body in sections
                ]
            }
    if not isinstance(outline, dict) or not isinstance(outline.get("framework"), list):
        raise ValueError(
            "Story outline lacks a recognizable framework or phase event list"
        )
    stages = []
    for item in outline["framework"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("stage"), str)
            or (not item["stage"].strip())
        ):
            raise ValueError("Story outline contains an invalid phase name")
        events = item.get("event_list")
        if not isinstance(events, list) or not events:
            raise ValueError(
                f"Phase {item['stage']} has an empty or invalid event_list"
            )
        cleaned = []
        for event in events:
            text = event.get("event") if isinstance(event, dict) else event
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Phase {item['stage']} contains an empty event")
            cleaned.append({"event": text.strip()})
        stages.append({"stage": item["stage"].strip(), "event_list": cleaned})
    names = [s["stage"] for s in stages]
    if not names or len(names) != len(set(names)):
        raise ValueError("Story phases must be nonempty and unique")
    return stages


def validate_stage_plan(plan, stages, total):
    if not isinstance(plan, list) or len(plan) != len(stages):
        raise ValueError(f"Phase plan must be an array of {len(stages)} phases")
    next_id = 1
    for item, stage in zip(plan, stages):
        if not isinstance(item, dict) or item.get("stage") != stage["stage"]:
            raise ValueError(
                f"Phase names and order must match the outline; expected {stage['stage']}"
            )
        start, end = (item.get("start_episode_id"), item.get("end_episode_id"))
        if (
            type(start) is not int
            or type(end) is not int
            or start != next_id
            or (end < start)
            or (end > total)
        ):
            raise ValueError(
                f"Invalid range {start}–{end} for {stage['stage']}; start at {next_id} and end by {total}"
            )
        if (
            not isinstance(item.get("pacing_reason"), str)
            or not item["pacing_reason"].strip()
        ):
            raise ValueError(f"{stage['stage']} lacks a pacing rationale")
        next_id = end + 1
    if next_id != total + 1:
        raise ValueError(
            f"Phase ranges must end at Episode {total}, but end at {next_id - 1}"
        )


def validate_episode_batch(episodes, start, end):
    if not isinstance(episodes, list) or len(episodes) != end - start + 1:
        raise ValueError(
            f"Generate all {end - start + 1} outlines for Episodes {start}–{end}"
        )
    fields = (
        "title",
        "core_plot",
        "character_growth",
        "relationship_changes",
        "main_storyline_progression",
    )
    for expected, episode in zip(range(start, end + 1), episodes):
        if not isinstance(episode, dict) or set(episode) != {
            "episode_id",
            "title",
            "core_plot",
            "roles",
            "highlights",
            "character_growth",
            "relationship_changes",
            "main_storyline_progression",
        }:
            raise ValueError(
                "Each episode outline must contain exactly the eight required fields"
            )
        if (
            not isinstance(episode, dict)
            or type(episode.get("episode_id")) is not int
            or episode["episode_id"] != expected
        ):
            raise ValueError(
                f"Episode IDs must be sequential; expected Episode {expected} here"
            )
        for field in fields:
            if not isinstance(episode.get(field), str) or not episode[field].strip():
                raise ValueError(f"Episode {expected} {field} must be nonempty text")
        roles = episode.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or any((not isinstance(r, str) or not r.strip() for r in roles))
        ):
            raise ValueError(
                f"Episode {expected} roles must be a nonempty array of names"
            )
        highlights = episode.get("highlights")
        if not (
            isinstance(highlights, str)
            and highlights.strip()
            or (
                isinstance(highlights, list)
                and highlights
                and all((isinstance(h, str) and h.strip() for h in highlights))
            )
        ):
            raise ValueError(
                f"Episode {expected} highlights must be nonempty text or a text array"
            )


def make_outline_llm(prompt):
    config = get_model_config("episode_outline_model")
    return LLM(
        **{
            key: config[key]
            for key in (
                "model",
                "system_prompt",
                "temperature",
                "top_p",
                "max_new_tokens",
            )
        },
        template=jinja2.Template(prompt),
    )


async def _validated_call(llm, params, validate):
    feedback = ""
    for attempt in range(1, 4):
        actual = dict(params, RetryFeedback=feedback)
        result = await llm_inference_json(params=actual, llm_service=llm)
        try:
            validate(result)
        except ValueError as error:
            feedback = str(error)
        else:
            return result
    raise RuntimeError(
        f"Outline generation failed validation after 3 attempts: {feedback}"
    )


async def generate_stage_episode_outlines(
    story_outline,
    total,
    common_params,
    episode_prompt,
    *,
    batch_validator=None,
    review_callback,
):
    validate_batch = batch_validator or validate_episode_batch
    stages = extract_stages(story_outline)
    if type(total) is not int or total < len(stages):
        raise ValueError(
            f"Episode count must be an integer of at least {len(stages)}; got {total}"
        )
    output = os.environ.get("DRAMA_OUTPUT_DIR")
    if output and list((Path(output) / "episode_outline").glob("*.json")):
        raise FileExistsError(
            "Episode outlines already exist; use a new output directory"
        )
    plan_params = dict(
        common_params,
        FullOutline=story_outline,
        Stages=json.dumps(stages, ensure_ascii=False),
        TotalEpisodeNums=str(total),
    )
    plan = await _validated_call(
        make_outline_llm(GENERATE_STAGE_EPISODE_PLAN_PROMPT),
        plan_params,
        lambda value: validate_stage_plan(value, stages, total),
    )
    save_realtime(
        ["stage_plan", "stage_plan.json"],
        {"total_episode_nums": total, "stages": plan},
    )
    ranges = [
        {key: item[key] for key in ("stage", "start_episode_id", "end_episode_id")}
        for item in plan
    ]
    all_episodes = []
    llm = make_outline_llm(episode_prompt)
    for allocation, stage in zip(plan, stages):
        start, end = (allocation["start_episode_id"], allocation["end_episode_id"])
        params = dict(
            common_params,
            TotalEpisodeNums=str(total),
            StageName=stage["stage"],
            StageRanges=json.dumps(ranges, ensure_ascii=False),
            PacingReason=allocation["pacing_reason"],
            Outline=json.dumps(stage["event_list"], ensure_ascii=False),
            EpisodeOutline=json.dumps(all_episodes, ensure_ascii=False),
            StartEpisodeId=str(start),
            EndEpisodeId=str(end),
        )
        episodes = await _validated_call(
            llm,
            params,
            lambda value: validate_batch(value, start, end),
        )
        all_episodes.extend(episodes)
    validate_batch(all_episodes, 1, total)
    all_episodes = await review_callback(all_episodes, common_params)
    return all_episodes
