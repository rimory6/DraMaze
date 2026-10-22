from .utils import require_text, get_model_config
from drama_local import models
from .llm_inference import llm_inference_json
from .prompts.story_planning import GENERATE_PLOT_POINT_PROMPT
from .realtime_output import save_realtime
import jinja2
from .api_client import LLM


def parse_plot_plan(plot_point_json):
    plot_planning = plot_point_json["plot_planning"]
    if not isinstance(plot_planning, list) or not plot_planning:
        raise ValueError("plot_planning must be a nonempty array")
    plot_plan = []
    for plot in plot_planning:
        chapters = []
        for chapter in plot.get("plot_desc", []):
            chapters.append(
                models.SingleScriptProposal.PlotPlanning.Chapter(
                    chapter_num=chapter.get("episode_id", 0),
                    content=require_text(chapter["plot_desc"], "plot_desc"),
                )
            )
        if plot.get("episode_range", [1, 1]) != []:
            plot_plan.append(
                models.SingleScriptProposal.PlotPlanning(
                    plot_id=plot.get("plot_id", -1),
                    chapter_from=plot.get("episode_range", [1, 1])[0],
                    chapter_to=plot.get("episode_range", [1, 1])[1],
                    title=plot.get("title", ""),
                    chapters=chapters,
                )
            )
    return plot_plan


async def generate_script_proposal(
    request: models.GenerateScriptProposalByCreativityReq,
) -> models.GenerateScriptProposalRsp:
    try:
        episode_nums = int(request.generate_input.common.episode_nums)
    except Exception:
        episode_nums = 60
    plot_point_config = get_model_config("plot_point_model")
    plot_point_llm_service = LLM(
        model=plot_point_config["model"],
        system_prompt=plot_point_config["system_prompt"],
        temperature=plot_point_config["temperature"],
        top_p=plot_point_config["top_p"],
        max_new_tokens=plot_point_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_PLOT_POINT_PROMPT),
    )
    final_world_view = (request.generate_input.world_view or "").strip()
    final_role = (request.generate_input.role_setting or "").strip()
    plot_point_params = {
        "Topic": require_text(request.generate_input.topic, "topic"),
        "WorldView": require_text(final_world_view, "world_view"),
        "StoryOutline": require_text(request.generate_input.core_story, "core_story"),
        "RoleInfo": require_text(final_role, "role_setting"),
        "Reference": request.generate_input.reference,
        "EpisodeNums": str(episode_nums),
    }
    plot_point_ret_json = await llm_inference_json(
        params=plot_point_params, llm_service=plot_point_llm_service
    )
    save_realtime(["four_stage_outline", "plot_plan.json"], plot_point_ret_json)
    plot_plan = parse_plot_plan(plot_point_ret_json)
    script_proposal = models.SingleScriptProposal(
        plot_planning=plot_plan,
        point_planning=models.SingleScriptProposal.PointPlanning(
            description=require_text(
                plot_point_ret_json["stuck_point_desc"], "stuck_point_desc"
            )
        ),
    )
    story_assets = models.StoryOutline(
        story_outline=[""], role_info=[final_role], world_building=[final_world_view]
    )
    return models.GenerateScriptProposalRsp(
        story_outline=story_assets,
        script_proposal=models.ScriptProposal(script_proposals=[script_proposal]),
    )
