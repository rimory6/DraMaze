from __future__ import annotations
import asyncio
import copy
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tools.outline_tree_search import search_stage_eventline


@dataclass(frozen=True)
class RefinementConfig:
    stages: tuple[str, ...] = ("Setup", "Development", "Climax", "Resolution")
    branch_factor: int = 4
    keep_top_k: int = 2
    parallel_workers: int = 32
    mutation_probability: float = 0.25
    model_name: str = "your-model-name"
    temperature: float = 0.9
    max_new_tokens: int = 32768


def parse_refinement_stages(value: str) -> tuple[str, ...]:
    stages = tuple((part for part in re.split("[,，\\s]+", value.strip()) if part))
    if not stages:
        raise ValueError("Search stages must not be empty")
    if len(stages) != len(set(stages)):
        raise ValueError(f"Search stages must be unique: {list(stages)}")
    return stages


def _load_config_from_env() -> RefinementConfig:
    stages = parse_refinement_stages(
        os.environ.get("DRAMA_EVENTLINE_STAGES", "Setup,Development,Climax,Resolution")
    )
    return RefinementConfig(
        stages=stages,
        branch_factor=int(os.environ.get("DRAMA_EVENTLINE_BRANCH_FACTOR", "4")),
        keep_top_k=int(os.environ.get("DRAMA_EVENTLINE_KEEP_TOP_K", "2")),
        parallel_workers=int(os.environ.get("DRAMA_EVENTLINE_PARALLEL_WORKERS", "32")),
        mutation_probability=float(
            os.environ.get("DRAMA_EVENTLINE_MUTATION_PROBABILITY", "0.25")
        ),
        model_name=os.environ.get("DRAMA_LLM_MODEL", "your-model-name"),
        temperature=float(os.environ.get("DRAMA_EVENTLINE_TEMPERATURE", "0.9")),
        max_new_tokens=int(os.environ.get("DRAMA_EVENTLINE_MAX_NEW_TOKENS", "32768")),
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def _replace_stage_events(
    outline: dict[str, Any], stage_name: str, events: list[str]
) -> None:
    for stage in outline.get("framework", []):
        if (
            isinstance(stage, dict)
            and str(stage.get("stage", "")).strip() == stage_name
        ):
            stage["event_list"] = [{"event": event} for event in events]
            return
    raise ValueError(f"Story stage not found: {stage_name}")


def _available_stage_names(outline: dict[str, Any]) -> list[str]:
    return [
        str(stage.get("stage", "")).strip()
        for stage in outline.get("framework", [])
        if isinstance(stage, dict) and str(stage.get("stage", "")).strip()
    ]


def _refine_story_outline_sync(
    outline: dict[str, Any], config: RefinementConfig, artifact_dir: Path
) -> dict[str, Any]:
    refined = copy.deepcopy(outline)
    available_stages = _available_stage_names(refined)
    stages = list(config.stages)
    if not stages or len(stages) != len(set(stages)):
        raise ValueError(f"Search stages must be nonempty and unique: {stages}")
    missing = [stage for stage in stages if stage not in available_stages]
    if missing:
        raise ValueError(
            f"Story outline is missing stages: {missing}; available stages: {available_stages}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trope_bank: list[dict[str, str]] | None = None
    for stage_name in stages:
        result = search_stage_eventline(
            outline=refined,
            stage_name=stage_name,
            branch_factor=config.branch_factor,
            keep_top_k=config.keep_top_k,
            local_max_depth=4,
            max_events_to_anchor=4,
            max_local_chains_per_anchor=4,
            parallel_workers=config.parallel_workers,
            mutation_probability=config.mutation_probability,
            model_name=config.model_name,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            initial_trope_bank=trope_bank,
        )
        final_event_line = result.get("final_event_line", [])
        if not isinstance(final_event_line, list) or not final_event_line:
            raise RuntimeError(
                f"Search for stage {stage_name} produced no valid final_event_line"
            )
        normalized_events = [
            str(event).strip() for event in final_event_line if str(event).strip()
        ]
        if not normalized_events:
            raise RuntimeError(
                f"Search for stage {stage_name} returned an empty event sequence"
            )
        _replace_stage_events(refined, stage_name, normalized_events)
        result_bank = result.get("trope_bank", [])
        if isinstance(result_bank, list) and result_bank:
            trope_bank = result_bank
    _save_json(artifact_dir / "trope_bank.json", trope_bank or [])
    return refined


async def _run_refinement_in_worker(
    outline: dict[str, Any], config: RefinementConfig, artifact_dir: Path
) -> dict[str, Any]:
    result_holder: list[dict[str, Any]] = []
    error_holder: list[BaseException] = []
    finished = threading.Event()

    def worker() -> None:
        try:
            result_holder.append(
                _refine_story_outline_sync(outline, config, artifact_dir)
            )
        except BaseException as error:
            error_holder.append(error)
        finally:
            finished.set()

    threading.Thread(
        target=worker, name="drama-outline-tree-search", daemon=True
    ).start()
    while not finished.is_set():
        await asyncio.sleep(0.1)
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise RuntimeError("Outline search worker returned no result")
    return result_holder[0]


async def refine_eventlines(outline: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(outline, dict):
        raise TypeError(
            f"Story outline must be a JSON object; got {type(outline).__name__}"
        )
    config = _load_config_from_env()
    output_root = os.environ.get("DRAMA_OUTPUT_DIR")
    if output_root:
        artifact_dir = (
            Path(output_root).expanduser().resolve()
            / "refined_outline"
            / "outline_tree_search"
        )
        return await _run_refinement_in_worker(outline, config, artifact_dir)
    with tempfile.TemporaryDirectory(prefix="drama_outline_tree_search_") as temp_dir:
        return await _run_refinement_in_worker(outline, config, Path(temp_dir))
