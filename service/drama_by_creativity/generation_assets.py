import json
from pathlib import Path
from drama_local.runtime import atomic_json


def assets_from_request(request):
    story = request.generate_data.story_outline
    return {
        "world_view": "\n\n".join(story.world_building)
        or request.generate_input.world_view,
        "role_info": "\n\n".join(story.role_info)
        or request.generate_input.role_setting,
        "story_outline": "\n\n".join(story.story_outline),
        "episode_outlines": [
            episode.to_dict()
            for season in request.generate_data.episode_outline.seasons
            for episode in season.episodes
        ],
    }


def preserve_generation_assets(assets, output_root):
    path = Path(output_root) / "generation_assets.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != assets:
        raise ValueError("Generated story assets changed within this run")
    atomic_json(path, assets)
