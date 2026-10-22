import os
from .utils import (
    require_text,
    calc_scene_nums_by_per_scene_word,
    calc_word_nums_by_word_count,
    get_model_config,
    read_yaml,
)
import asyncio
import json
from drama_local import models
from .scene_outline import generate_scene_outline
from .llm_inference import llm_inference_json
from .future_memory import FutureNarrativeMemory
from .state_memory import (
    STATE_UPDATE_PROMPT,
    StateMemory,
    validate_extraction as validate_state_extraction,
)
from .generation_assets import assets_from_request, preserve_generation_assets
from .scene_consistency_review import SceneConsistencyReview
from .prompts.screenplay import (
    GENERATE_SCENE_OUTLINE_PROMPT,
    GENERATE_WHOLE_EPISODE_PROMPT,
)
from .prompts.screenplay_format import SCRIPT_FORMAT_PROMPT
from .realtime_output import save_realtime
from pathlib import Path
import jinja2
from .api_client import LLM

DRAM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "drama_configs.yaml"
)
RETRY_TIMES = 3
RETRY_INTERVAL = 60


async def generate_episode_script(
    request: models.GenerateDramaByCreativityReq,
) -> models.GenerateDramaRsp:
    all_dram_configs = read_yaml(DRAM_CONFIG_PATH)
    drama_config = all_dram_configs["drama"]
    episode_outline_list = []
    if (
        len(request.generate_data.episode_outline.seasons) > 0
        and len(request.generate_data.episode_outline.seasons[0].episodes) > 0
    ):
        for episode_info in request.generate_data.episode_outline.seasons[0].episodes:
            episode_outline_list.append(episode_info.content)
    else:
        raise Exception("Episode outlines must not be empty")
    generated_assets = assets_from_request(request)
    preserve_generation_assets(generated_assets, os.environ["DRAMA_OUTPUT_DIR"])
    world_view = generated_assets["world_view"]
    role_info = generated_assets["role_info"]
    future_memory = FutureNarrativeMemory(request.story_info.story_id)
    state_memory = StateMemory(request.story_info.story_id)
    scene_outline_config = get_model_config("scene_outline_model")
    scene_outline_llm_service = LLM(
        model=scene_outline_config["model"],
        system_prompt=scene_outline_config["system_prompt"],
        temperature=scene_outline_config["temperature"],
        top_p=scene_outline_config["top_p"],
        max_new_tokens=scene_outline_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_SCENE_OUTLINE_PROMPT),
    )
    script_config = get_model_config("script_model")
    whole_episode_llm_service = LLM(
        model=script_config["model"],
        system_prompt=script_config["system_prompt"],
        temperature=script_config["temperature"],
        top_p=script_config["top_p"],
        max_new_tokens=script_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_WHOLE_EPISODE_PROMPT),
    )
    state_config = get_model_config("state_update_model")
    state_update_llm_service = LLM(
        model=state_config["model"],
        system_prompt=state_config["system_prompt"],
        temperature=min(float(state_config.get("temperature", 0.4)), 0.3),
        top_p=state_config["top_p"],
        max_new_tokens=state_config["max_new_tokens"],
        template=jinja2.Template(STATE_UPDATE_PROMPT),
    )
    final_rets = []
    for episode_id in range(len(episode_outline_list)):
        single_episode_outline = episode_outline_list[episode_id]
        prev_plot = ""
        prev_episode_full_script = ""
        if episode_id == 0:
            prev_plot = ""
            prev_episode_full_script = ""
        else:
            try:
                previous = (
                    Path(os.environ["DRAMA_OUTPUT_DIR"])
                    / "drama"
                    / f"episode_{episode_id:02d}.json"
                )
                previous_script = json.loads(previous.read_text(encoding="utf-8"))[
                    "content"
                ]
                prev_plot = previous_script.split("\n\n---\n\n")[-1]
                prev_episode_full_script = previous_script
            except Exception:
                prev_plot = ""
                prev_episode_full_script = ""
        plot_reference_for_prompt = request.generate_input.reference or ""
        memory_context, memory_retrieval = future_memory.retrieve(
            single_episode_outline, episode_id
        )
        state_context, state_retrieval = state_memory.retrieve(
            single_episode_outline, episode_id
        )
        recent_episode_summaries = state_memory.recent_summaries(episode_id)
        try:
            min_word_count = request.generate_input.common.min_word_count_per_episode
            max_word_count = request.generate_input.common.max_word_count_per_episode
            scene_nums_min, scene_nums_max = calc_scene_nums_by_per_scene_word(
                drama_config, min_word_count, max_word_count
            )
        except Exception:
            scene_nums_min = drama_config["scene_nums_min"]
            scene_nums_max = drama_config["scene_nums_max"]
        try:
            word_nums_for_outline = calc_word_nums_by_word_count(
                drama_config, min_word_count, max_word_count
            )
        except Exception:
            word_nums_for_outline = drama_config.get("word_nums", "500")
        nearby_start = max(0, episode_id - 2)
        nearby_end = min(len(episode_outline_list), episode_id + 3)
        nearby_episode_outline = ""
        for idx in range(nearby_start, nearby_end):
            nearby_episode_outline += (
                f"### Episode {idx + 1} outline\n{episode_outline_list[idx]}\n\n---\n\n"
            )
        nearby_episode_outline = (
            nearby_episode_outline.rstrip("\n-\n") if nearby_episode_outline else "None"
        )
        generate_scene_outline_params = {
            "WorldView": require_text(world_view, "world_view"),
            "RoleInfo": require_text(role_info, "role_info"),
            "EpisodeOutline": nearby_episode_outline or "None",
            "Outline": single_episode_outline or "None",
            "SceneNumsMin": str(scene_nums_min),
            "SceneNumsMax": str(scene_nums_max),
            "WordNums": word_nums_for_outline,
            "EpisodeIdx": str(episode_id + 1),
            "PlotReference": plot_reference_for_prompt,
            "PrevEpisodeScript": prev_episode_full_script,
            "NarrativeMemory": memory_context,
            "StateMemory": state_context,
            "RecentEpisodeSummaries": recent_episode_summaries,
        }
        scene_review = SceneConsistencyReview.factory(
            state_memory, episode_id, generate_scene_outline_params
        )
        scene_outline_ret_json = await generate_scene_outline(
            params=generate_scene_outline_params,
            llm_service=scene_outline_llm_service,
        )
        if (
            not isinstance(scene_outline_ret_json, list)
            or not scene_outline_ret_json
            or any(
                (
                    not isinstance(scene, dict) or not scene
                    for scene in scene_outline_ret_json
                )
            )
        ):
            continue
        if scene_review is not None:
            scene_outline_ret_json = await scene_review.apply(scene_outline_ret_json)
        save_realtime(
            ["scene_outlines", f"episode_{episode_id + 1:02d}.json"],
            scene_outline_ret_json,
        )
        episode_plot = ""
        prev_abs = memory_context
        try:
            min_word_count = request.generate_input.common.min_word_count_per_episode
            max_word_count = request.generate_input.common.max_word_count_per_episode
            word_nums = calc_word_nums_by_word_count(
                drama_config, min_word_count, max_word_count
            )
        except Exception:
            word_nums = drama_config.get("word_nums", "300")
        script_format = SCRIPT_FORMAT_PROMPT.replace(
            "{{EpisodeIndex}}", str(episode_id + 1)
        )
        generate_whole_episode_params = {
            "EpisodeIndex": str(episode_id + 1),
            "WorldView": require_text(world_view, "world_view"),
            "RoleInfo": require_text(role_info, "role_info"),
            "EpisodeOutline": single_episode_outline,
            "AllSceneOutlines": json.dumps(scene_outline_ret_json, ensure_ascii=False),
            "PrevPlot": prev_plot,
            "PrevAbs": prev_abs,
            "StateMemory": state_context,
            "ScriptFormat": script_format,
            "WordNums": word_nums,
            "PlotReference": plot_reference_for_prompt,
            "RecentEpisodeSummaries": recent_episode_summaries,
        }
        whole_plot_status = False
        for i in range(RETRY_TIMES):
            try:
                response = await whole_episode_llm_service.request(
                    generate_whole_episode_params
                )
                episode_plot = response.response
                if not isinstance(episode_plot, str) or not episode_plot.strip():
                    raise ValueError("Model API returned an empty screenplay")
                whole_plot_status = True
                break
            except Exception:
                await asyncio.sleep(RETRY_INTERVAL)
                continue
        if not whole_plot_status:
            continue
        save_realtime(
            ["drama", f"episode_{episode_id + 1:02d}.json"],
            {"episode_id": episode_id, "content": episode_plot},
        )

        async def update_future_memory():
            try:
                result = await future_memory.update_from_episode(
                    episode_id=episode_id,
                    episode_outline=single_episode_outline,
                    final_script=episode_plot,
                )
                return result
            except Exception:
                return {}

        async def update_current_state():
            validation_feedback = ""
            for attempt in range(1, 4):
                try:
                    result = await llm_inference_json(
                        params={
                            "EpisodeNumber": str(episode_id + 1),
                            "EpisodeOutline": single_episode_outline,
                            "PreviousState": state_memory.previous_state_json(),
                            "EpisodeScript": episode_plot,
                            "ValidationFeedback": validation_feedback,
                        },
                        llm_service=state_update_llm_service,
                    )
                    return state_memory.commit(
                        episode_id, validate_state_extraction(result)
                    )
                except Exception as error:
                    validation_feedback = str(error)
            raise RuntimeError(
                f"Episode {episode_id + 1} state update failed after 3 attempts: {validation_feedback}"
            )

        await asyncio.gather(update_future_memory(), update_current_state())
        episode_plot_for_return = episode_plot.replace("\n\n---\n\n", "\n\n")
        final_rets.append(
            models.Drama.Episode(
                season_id=0,
                episode_id=episode_id,
                script_content=episode_plot_for_return,
            )
        )
    return models.GenerateDramaRsp(
        result=models.Drama(
            seasons=[models.Drama.Season(episodes=final_rets, season_id=0)]
        )
    )
