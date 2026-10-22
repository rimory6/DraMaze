from .utils import require_text
from drama_local import models
from .realtime_output import save_realtime
from .future_map import generate_future_map_artifacts


def transfer_outline_to_str(outline_ret_json):
    all_outline_list = []
    all_outline_str = ""
    for episode in outline_ret_json:
        outline_str = ""
        episode_id = episode.get("episode_id", -1)
        episode_name = require_text(episode["title"], "title")
        episode_core_plot = require_text(episode["core_plot"], "core_plot")
        episode_roles = episode.get("roles", [])
        all_roles = ", ".join(episode_roles)
        episode_plot = require_text(episode["highlights"], "highlights")
        episode_role_growth = require_text(
            episode["character_growth"], "character_growth"
        )
        episode_relationship = require_text(
            episode["relationship_changes"], "relationship_changes"
        )
        episode_storyline = require_text(
            episode["main_storyline_progression"], "main_storyline_progression"
        )
        outline_str += f"### Episode {episode_id}: {episode_name}\n"
        outline_str += f"Characters: {all_roles}\n\n"
        outline_str += f"#### Core plot\n{episode_core_plot}\n\n"
        outline_str += f"#### Highlights\n{episode_plot}\n\n"
        outline_str += f"#### Character growth\n{episode_role_growth}\n\n"
        outline_str += f"#### Relationship changes\n{episode_relationship}\n\n"
        outline_str += f"#### Main-plot progress\n{episode_storyline}\n\n"
        all_outline_str += outline_str
        all_outline_str += "\n\n---\n\n"
        all_outline_list.append(outline_str)
    return (all_outline_str[:-7], all_outline_list)


async def generate_episode_outline(
    request: models.GenerateEpisodeOutlineByCreativityReq,
):
    from .episode_fields import generate_eight_fields

    data = request.generated_data.story_outline
    if not data.story_outline or not data.story_outline[0]:
        raise ValueError("A nonempty detailed story outline is required")
    story_outline = data.story_outline[0]
    episode_nums = request.generate_input.common.episode_nums
    common = {
        "Topic": require_text(request.generate_input.topic, "topic"),
        "WorldView": require_text(
            data.world_building[0]
            if data.world_building and data.world_building[0]
            else request.generate_input.world_view,
            "world_view",
        ),
        "StoryOutline": require_text(request.generate_input.core_story, "core_story"),
        "Reference": request.generate_input.reference or "",
        "RoleInfo": require_text(
            data.role_info[0]
            if data.role_info
            else request.generate_input.role_setting,
            "role_setting",
        ),
    }
    outlines = await generate_eight_fields(story_outline, episode_nums, common)
    save_realtime(["episode_outlines.json"], outlines)
    await generate_future_map_artifacts(outlines)
    final_outline_str, final_outline_list = transfer_outline_to_str(outlines)
    episodes = [
        models.EpisodeOutline.Episode(
            season_id=0, episode_id=item["episode_id"] - 1, content=content
        )
        for item, content in zip(outlines, final_outline_list)
    ]
    return models.GenerateEpisodeOutlineRsp(
        episode_outline=models.EpisodeOutline(
            seasons=[models.EpisodeOutline.Season(season_id=0, episodes=episodes)]
        )
    )
