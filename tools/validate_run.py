import json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed(output, manifest):
    story = manifest["story_id"]
    files = sorted((output / "drama").glob("episode_*.json"))
    for i, path in enumerate(files):
        row = read(path)
        if (
            path.name != f"episode_{i + 1:02d}.json"
            or row.get("episode_id") != i
            or (not row.get("content", "").strip())
        ):
            raise ValueError(f"Screenplay files are incomplete or out of order: {path}")
    for relative in (
        f"future_narramem/{story}/memory_state.json",
        f"worldstate/{story}/state_memory.json",
    ):
        path = output / relative
        last = read(path).get("last_updated_episode", -1) if path.exists() else -1
        if last != len(files) - 1:
            raise ValueError(
                f"Screenplay and memory progress do not match: {path} last={last}, scripts={len(files)}; repair memory before continuing"
            )
    return len(files)
