from .utils import require_text, get_model_config
from drama_local import models
from .llm_inference import llm_inference_json
from .prompts.story_planning import GENERATE_OUTLINE_BY_SCRIPT_PROMPT
import jinja2
from .api_client import LLM
from .eventline_refinement import refine_eventlines
from .realtime_output import save_realtime


def transfer_outline_to_str(outline_json):
    outline_str = ""
    play_name = require_text(outline_json["title"], "title")
    core_story = require_text(outline_json["background"], "background")
    framework = outline_json["framework"]
    highlights = require_text(outline_json["highlights"], "highlights")
    summary = require_text(outline_json["summary"], "summary")
    outline_str = f"#### Series title: {play_name}\n\n"
    outline_str += f"#### Background\n{core_story}\n\n"
    outline_str += f"#### Distinctive highlights\n{highlights}\n\n"
    outline_str += "#### Story structure:\n"
    if not isinstance(framework, list) or not framework:
        raise ValueError("framework must be a nonempty array")
    for item in framework:
        stage = require_text(item["stage"], "stage")
        events = item["event_list"]
        if not isinstance(events, list) or not events:
            raise ValueError("event_list must be a nonempty array")
        outline_str += f"### {stage}\n"
        for event in events:
            event_content = require_text(event["event"], "event")
            outline_str += f"- {event_content}\n"
    outline_str += f"\n\n## Summary\n{summary}"
    return outline_str


def transfer_script_proposal_to_str(script_proposal):
    if not script_proposal or not script_proposal.script_proposals:
        raise ValueError("A nonempty preliminary plot plan is required")
    result_str = ""
    for proposal in script_proposal.script_proposals:
        result_str += "**Preliminary plot plan**\n\n"
        if proposal.plot_planning:
            result_str += "#### Plot segments\n"
            for plot in proposal.plot_planning:
                result_str += f"**{plot.plot_id}. {plot.title}** (Episodes {plot.chapter_from}–{plot.chapter_to})\n"
                result_str += f"{plot.description}\n"
                if plot.chapters:
                    for chapter in plot.chapters:
                        result_str += (
                            f"  - Episode {chapter.chapter_num}: {chapter.content}\n"
                        )
                result_str += "\n"
        if proposal.point_planning and proposal.point_planning.description:
            result_str += (
                f"Turning-point rationale: {proposal.point_planning.description}\n\n"
            )
        result_str += "---\n\n"
    return result_str.rstrip("---\n\n")


async def generate_story_outline(request: models.GenerateStoryOutlineByCreativityReq):
    script_proposal_str = transfer_script_proposal_to_str(
        request.generate_data.script_proposal
    )
    story_outline_config = get_model_config("story_outline_model")
    outline_llm_service = LLM(
        model=story_outline_config["model"],
        system_prompt=story_outline_config["system_prompt"],
        temperature=story_outline_config["temperature"],
        top_p=story_outline_config["top_p"],
        max_new_tokens=story_outline_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_OUTLINE_BY_SCRIPT_PROMPT),
    )
    outline_params = {
        "Topic": require_text(request.generate_input.topic, "topic"),
        "WorldView": require_text(request.generate_input.world_view, "world_view"),
        "StoryOutline": require_text(request.generate_input.core_story, "core_story"),
        "RoleInfo": require_text(request.generate_input.role_setting, "role_setting"),
        "Reference": request.generate_input.reference,
        "ScriptProposal": script_proposal_str,
    }
    outline_ret_json = await llm_inference_json(
        params=outline_params, llm_service=outline_llm_service
    )
    transfer_outline_to_str(outline_ret_json)
    save_realtime(["four_stage_outline", "outline.json"], outline_ret_json)
    outline_ret_json = await refine_eventlines(outline_ret_json)
    final_outline = transfer_outline_to_str(outline_ret_json)
    save_realtime(["refined_outline", "outline.json"], outline_ret_json)
    canonical_role = request.generate_input.role_setting or ""
    canonical_world = request.generate_input.world_view or ""
    story_assets = models.StoryOutline(
        story_outline=[final_outline],
        role_info=[canonical_role] if canonical_role else [],
        world_building=[canonical_world] if canonical_world else [],
    )
    return models.GenerateStoryOutlineRsp(story_outline=story_assets)
