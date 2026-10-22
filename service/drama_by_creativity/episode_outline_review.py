import copy
import json
import math


def validate_review(value):
    if not isinstance(value, dict):
        raise ValueError("Result must be a JSON object containing total_score")
    score = value.get("total_score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or (not 0 <= score <= 100)
    ):
        raise ValueError("total_score must be a finite number between 0 and 100")


async def review_episode_outlines(
    episodes,
    common,
    *,
    batch_validator=None,
    review_prompt,
    rewrite_prompt,
):
    from .stage_planning import (
        _validated_call,
        make_outline_llm,
        save_realtime,
        validate_episode_batch,
    )

    validate_batch = batch_validator or validate_episode_batch
    total = len(episodes)
    validate_batch(episodes, 1, total)
    current = copy.deepcopy(episodes)
    background = {
        key: common[key]
        for key in ("Topic", "WorldView", "StoryOutline", "RoleInfo", "Reference")
        if common.get(key)
    }
    rounds = 2
    candidates = []
    for version in range(rounds + 1):
        params = {
            "Total": str(total),
            "Original": json.dumps(current, ensure_ascii=False),
            "Background": json.dumps(background, ensure_ascii=False),
        }
        review = await _validated_call(
            make_outline_llm(review_prompt),
            params,
            validate_review,
        )
        candidates.append(
            {
                "version": version,
                "score": review["total_score"],
                "episodes": copy.deepcopy(current),
                "review": review,
            }
        )
        if version < rounds:
            revised = await _validated_call(
                make_outline_llm(rewrite_prompt),
                dict(params, Review=json.dumps(review, ensure_ascii=False)),
                lambda value: validate_batch(value, 1, total),
            )
            current = copy.deepcopy(revised)
    best = max(candidates, key=lambda item: item["score"])
    save_realtime(
        ["episode_outline_review", "result.json"],
        {
            "selected_version": best["version"],
            "selected_score": best["score"],
            "version_scores": [
                {"version": item["version"], "score": item["score"]}
                for item in candidates
            ],
            "review": best["review"],
        },
    )
    return copy.deepcopy(best["episodes"])
