from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an English short drama from an idea."
    )
    parser.add_argument("--records-file", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--index", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--min-words", type=int, default=700)
    parser.add_argument("--max-words", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from-outline", type=Path)
    parser.add_argument("--outline-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.outline_only and args.from_outline:
        parser.error("--outline-only cannot be combined with --from-outline")
    if (
        args.episodes < 4
        or args.index < 1
        or (not 0 < args.min_words <= args.max_words)
    ):
        parser.error(
            "Use at least four episodes (one per phase), a positive sample index, and 0 < min-words <= max-words"
        )
    return args


def select_input(args):
    source = (args.records_file or ROOT / "example/idea.jsonl").resolve()
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = records
    if args.sample_id:
        matches = [row for row in rows if row.get("sample_id") == args.sample_id]
        if len(matches) != 1:
            raise ValueError(
                f"sample_id={args.sample_id!r} matched {len(matches)} records"
            )
        row = matches[0]
    else:
        if args.index > len(rows):
            raise ValueError(
                f"Sample index {args.index} is out of range; record count: {len(rows)} records"
            )
        row = rows[args.index - 1]
    for field in ("sample_id", "core_plot", "topic", "role_setting", "world_view"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Sample requires a nonempty string field: {field}")
    if not isinstance(row.get("reference", ""), str):
        raise ValueError("reference must be a string")
    row = dict(row)
    row["core_story"] = row["core_plot"]
    return (source, row)


async def generate(args, row, story_id, output):
    from drama_local import models
    from drama_local.runtime import atomic_json
    from service.drama_by_creativity.generate_script_proposal import (
        generate_script_proposal,
    )
    from service.drama_by_creativity.generate_story_outline import (
        generate_story_outline,
    )
    from service.drama_by_creativity.generate_episode_outline import (
        generate_episode_outline,
    )
    from service.drama_by_creativity.generate_episode_script import (
        generate_episode_script,
    )

    story_info = models.StoryInfo(story_id=story_id)
    common = models.GenerateInputCommon(
        episode_nums=args.episodes,
        min_word_count_per_episode=args.min_words,
        max_word_count_per_episode=args.max_words,
    )

    def inputs(world, roles):
        return models.GenerateInputByCreativity(
            core_story=row["core_story"],
            topic=row["topic"],
            world_view=world,
            role_setting=roles,
            reference=row.get("reference", ""),
            common=common,
        )

    results = {}

    def save(name, result):
        results[name] = result.to_dict()
        atomic_json(output / "pipeline_result.json", results)

    if args.from_outline:
        source = args.from_outline.resolve()
        cached = json.loads((source / "pipeline_result.json").read_text())
        story = models.GenerateStoryOutlineRsp.from_dict(cached["story_outline"])
        outlines = models.GenerateEpisodeOutlineRsp.from_dict(cached["episode_outline"])
        for name in ("script_proposal", "story_outline", "episode_outline"):
            results[name] = cached[name]
        atomic_json(output / "pipeline_result.json", results)
        outline_files = ("episode_outlines.json",)
        outline_files += (
            "future_map/future_map.json",
            "future_map/episode_contributions.json",
        )
        for relative in outline_files:
            atomic_json(output / relative, json.loads((source / relative).read_text()))
        world = "\n\n".join(story.story_outline.world_building) or row["world_view"]
        roles = "\n\n".join(story.story_outline.role_info) or row["role_setting"]
    else:
        proposal = await generate_script_proposal(
            models.GenerateScriptProposalByCreativityReq(
                generate_input=inputs(row["world_view"], row["role_setting"]),
            )
        )
        save("script_proposal", proposal)
        world = "\n\n".join(proposal.story_outline.world_building) or row["world_view"]
        roles = "\n\n".join(proposal.story_outline.role_info) or row["role_setting"]
        story = await generate_story_outline(
            models.GenerateStoryOutlineByCreativityReq(
                generate_input=inputs(world, roles),
                generate_data=models.GenerateStoryOutline(
                    script_proposal=proposal.script_proposal
                ),
            )
        )
        save("story_outline", story)
        world = "\n\n".join(story.story_outline.world_building) or world
        roles = "\n\n".join(story.story_outline.role_info) or roles
        outlines = await generate_episode_outline(
            models.GenerateEpisodeOutlineByCreativityReq(
                generate_input=inputs(world, roles),
                generated_data=models.GeneratedData(story_outline=story.story_outline),
            )
        )
        save("episode_outline", outlines)
    if args.outline_only:
        episodes = [
            ep.episode_id
            for season in outlines.episode_outline.seasons
            for ep in season.episodes
        ]
        if episodes != list(range(args.episodes)):
            raise RuntimeError(
                f"Incomplete episode outlines: expected {args.episodes} episodes; received IDs: {episodes}"
            )
        for relative in (
            "episode_outlines.json",
            "future_map/future_map.json",
            "future_map/episode_contributions.json",
        ):
            json.loads((output / relative).read_text(encoding="utf-8"))
        print(f"Saved: {output}")
        return
    drama = await generate_episode_script(
        models.GenerateDramaByCreativityReq(
            story_info=story_info,
            generate_input=inputs(world, roles),
            generate_data=models.GenerateDrama(
                story_outline=story.story_outline,
                episode_outline=outlines.episode_outline,
            ),
        ),
    )
    drama.episode_outline = outlines.episode_outline
    actual = [
        episode.episode_id
        for season in drama.result.seasons
        for episode in season.episodes
    ]
    if sorted(actual) != list(range(args.episodes)):
        raise RuntimeError(
            f"Incomplete screenplay: expected {args.episodes} episodes; received IDs: {actual}"
        )
    from tools.validate_run import completed

    if completed(output, {"story_id": story_id}) != args.episodes:
        raise RuntimeError("Saved screenplay is incomplete")
    print(f"Saved: {output / 'drama'}")


def main():
    args = parse_args()
    if args.from_outline:
        saved = json.loads((args.from_outline / "input_manifest.json").read_text())
        source, row = (Path(saved["source"]), saved["sample"])
        args.episodes = saved["episodes"]
        args.min_words, args.max_words = saved["word_range"]
        cached = json.loads((args.from_outline / "pipeline_result.json").read_text())
        for name in ("script_proposal", "story_outline", "episode_outline"):
            if name not in cached:
                raise ValueError(f"Saved run is missing a required result: {name}")
        episodes = [
            ep
            for season in cached["episode_outline"]["episode_outline"]["seasons"]
            for ep in season["episodes"]
        ]
        if [ep["episode_id"] for ep in episodes] != list(range(args.episodes)):
            raise ValueError("Saved episode outlines are incomplete or out of order")
        outline_files = ("episode_outlines.json",)
        outline_files += (
            "future_map/future_map.json",
            "future_map/episode_contributions.json",
        )
        for relative in outline_files:
            json.loads((args.from_outline / relative).read_text())
    else:
        source, row = select_input(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    story_id = re.sub("[^A-Za-z0-9_-]", "_", f"dramaze_{row['sample_id']}_{stamp}")
    output = (args.output_dir or ROOT / "output" / story_id).resolve()
    settings = {
        "DRAMA_OUTPUT_DIR": str(output),
        "DRAMA_MEMORY_DIR": str(output / "future_narramem"),
        "DRAMA_STATE_MEMORY_DIR": str(output / "worldstate"),
        "DRAMA_STATE_MEMORY_LIMIT": "20",
        "DRAMA_LLM_MODEL": os.environ.get("DRAMA_LLM_MODEL") or "your-model-name",
        "DRAMA_LLM_BASE_URL": os.environ.get("DRAMA_LLM_BASE_URL") or "",
        "DRAMA_LLM_TIMEOUT": os.environ.get("DRAMA_LLM_TIMEOUT") or "1200",
        "DRAMA_LLM_MAX_TOKENS": os.environ.get("DRAMA_LLM_MAX_TOKENS") or "0",
        "DRAMA_EVENTLINE_STAGES": "Setup,Development,Climax,Resolution",
        "DRAMA_SUBTREE_MEMORY_LIMIT": "10",
    }
    os.environ.update(settings)
    manifest = {
        "state_memory_backend": "structured_state",
        "source": str(source),
        "sample": row,
        "sample_sha256": hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "story_id": story_id,
        "episodes": args.episodes,
        "from_outline": str(args.from_outline.resolve()) if args.from_outline else None,
        "word_range": [args.min_words, args.max_words],
        "algorithm": "dramaze",
        "outline_only": args.outline_only,
        "model": os.environ.get("DRAMA_LLM_MODEL", "your-model-name"),
        "base_url": os.environ.get("DRAMA_LLM_BASE_URL", ""),
        "settings": settings,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "episodes": args.episodes,
                    "model": settings["DRAMA_LLM_MODEL"],
                    "output_dir": str(output),
                },
                indent=2,
            )
        )
        return
    from tools.check_drama_dependencies import main as check_dependencies

    if check_dependencies():
        raise SystemExit(1)
    from drama_local.api_settings import validate_api_settings

    validate_api_settings(settings["DRAMA_LLM_BASE_URL"], settings["DRAMA_LLM_MODEL"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    asyncio.run(generate(args, row, story_id, output))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        message = str(error)
        if re.search("[㐀-鿿]", message):
            message = "Generation failed. Check the input and API settings."
        print(f"Error: {message[:240]}", file=sys.stderr)
        sys.exit(1)
