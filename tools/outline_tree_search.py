from __future__ import annotations
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable
from drama_local.eventline_llm import JSONResponseError, query_json
from service.drama_by_creativity.api_client import ModelAPIError
from requests.exceptions import RequestException
from tools.outline_tree_utils import (
    CANDIDATE_SCORE_DEFAULTS,
    NodeIdFactory,
    choose_top_candidates_for_expansion,
    collect_previous_stage_events,
    create_node_payload,
    extract_stage_payload,
    parse_candidate_step_scores,
    parse_candidates,
    sanitize_int,
)
from tools.outline_tree_prompts import (
    build_candidate_messages,
    build_candidate_step_score_messages,
    build_local_chain_score_messages,
    build_mutation_messages,
)


def retry_messages(
    messages: list[dict[str, str]],
    previous_response: str | None,
    previous_error: str | None,
) -> list[dict[str, str]]:
    if previous_error is None:
        return messages
    retry = list(messages)
    if previous_response:
        retry.append({"role": "assistant", "content": previous_response})
    retry.append(
        {
            "role": "user",
            "content": f"Your previous response failed validation. Validation error: {previous_error}\nKeep the original task and input unchanged. Correct the previous output and return a complete JSON response matching the requested format, without explanation.",
        }
    )
    return retry


def call_llm_with_schema_retry(
    *,
    runner: Callable[[str | None, str | None], tuple[Any, str]],
    parser: Callable[[Any, str], Any],
    max_attempts: int = 3,
) -> tuple[Any, str]:
    last_error: Exception | None = None
    previous_response: str | None = None
    previous_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        raw_response = None
        try:
            parsed, raw_response = runner(previous_response, previous_error)
        except JSONResponseError as error:
            last_error = error
            previous_response = error.raw_response
            previous_error = str(error)
        except ModelAPIError as error:
            if not error.retryable:
                raise
            last_error = error
        except RequestException as error:
            last_error = error
        except ValueError:
            raise
        except Exception as error:
            last_error = error
        else:
            try:
                result = parser(parsed, raw_response)
                return (result, raw_response)
            except Exception as error:
                last_error = error
                previous_response = raw_response
                previous_error = str(error)
        if attempt < max_attempts:
            time.sleep(float(attempt))
    assert last_error is not None
    raise RuntimeError(
        f"Outline search failed after {max_attempts} attempts"
    ) from last_error


def abstract_micro_motifs_once(
    *, events: list[str], source_label: str, model_name: str, max_new_tokens: int
) -> list[dict[str, str]]:
    if not events:
        return []
    schema = {
        "motifs": [
            {
                "cause": "pressure from above",
                "method": "public confrontation",
                "outcome": "pressure fails",
                "response": "change tactics",
            }
        ]
    }
    events_text = "\n".join((f"- {event}" for event in events))
    prompt = f"Analyze the narrative patterns in this already-written event chain of a short-form drama.\n\nRepresent each distinct pattern with four abstract links: cause -> method -> outcome -> response. Use a short English phrase for each link.\n\nDefinitions:\n- cause: The event or state that prompts the action.\n- method: The action taken by a character and the mechanism through which it works.\n- outcome: The change directly produced by that action.\n- response: A relevant character’s subsequent action or reaction to that outcome.\n\nRequirements:\n1. Do not include specific character names, places, or props.\n2. The phrases may be broad but must convey the mechanism of dramatic progress.\n3. List each distinct pattern in the chain.\n4. Output a JSON object only, without explanation.\n\nEvent chain ({source_label}):\n{events_text}\n\nSchema: {schema}\n"

    def run_query(
        previous_response: str | None, previous_error: str | None
    ) -> tuple[Any, str]:
        return query_json(
            retry_messages(
                [{"role": "user", "content": prompt}], previous_response, previous_error
            ),
            model_name=model_name,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
        )

    def parse_motifs(payload: Any, _raw_response: str) -> list[dict[str, str]]:
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        motifs: list[dict[str, str]] = []
        if isinstance(payload, dict):
            for item in payload.get("motifs", []):
                if not isinstance(item, dict):
                    continue
                motif = {
                    "cause": str(item.get("cause", "")).strip(),
                    "method": str(item.get("method", "")).strip(),
                    "outcome": str(item.get("outcome", "")).strip(),
                    "response": str(item.get("response", "")).strip(),
                }
                if all(motif.values()):
                    motifs.append(motif)
        return motifs

    motifs, _raw = call_llm_with_schema_retry(runner=run_query, parser=parse_motifs)
    return motifs


def compute_candidate_expansion_score(score_payload: dict[str, Any]) -> float:
    logic = float(score_payload["logic"])
    drive = float(score_payload["drive"])
    novelty = float(score_payload["novelty"])
    appeal = float(score_payload["appeal"])
    return round(0.3 * logic + 0.2 * drive + 0.3 * novelty + 0.2 * appeal, 4)


def score_candidates_for_expansion_once(
    *,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    current_segment: list[str],
    target_anchor: str,
    future_anchors: list[str],
    candidates: list[dict[str, Any]],
    trope_bank: list[dict[str, str]],
    model_name: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    def request_scores(batch):
        messages = build_candidate_step_score_messages(
            title=title,
            background=background,
            highlights=highlights,
            summary=summary,
            stage_name=stage_name,
            seed_events=seed_events,
            fixed_prefix=fixed_prefix,
            current_segment=current_segment,
            target_anchor=target_anchor,
            future_anchors=future_anchors,
            candidates=batch,
            trope_bank=trope_bank,
        )

        def run_query(previous_response, previous_error):
            return query_json(
                retry_messages(messages, previous_response, previous_error),
                model_name=model_name,
                temperature=0.2,
                max_new_tokens=max_new_tokens,
            )

        return call_llm_with_schema_retry(
            runner=run_query,
            parser=lambda payload, _raw_response: parse_candidate_step_scores(
                payload, len(batch)
            ),
        )

    scores, raw_response = request_scores(candidates)
    missing = [
        index
        for index, score in enumerate(scores)
        if score is None
        or any(score[field] is None for field in CANDIDATE_SCORE_DEFAULTS)
    ]
    if missing:
        supplemental_error = None
        try:
            recovered, supplemental_raw = request_scores(
                [candidates[index] for index in missing]
            )
        except ModelAPIError:
            raise
        except RuntimeError as error:
            recovered = [None] * len(missing)
            supplemental_raw = None
            supplemental_error = str(error)
        for index, score in zip(missing, recovered):
            scores[index] = {
                field: score[field]
                if score is not None and score[field] is not None
                else default
                for field, default in CANDIDATE_SCORE_DEFAULTS.items()
            }
            scores[index]["reason"] = score["reason"] if score is not None else ""
        trace = {
            "initial_response": raw_response,
            "supplemental_candidate_indices": [index + 1 for index in missing],
            "supplemental_response": supplemental_raw,
        }
        if supplemental_error is not None:
            trace["supplemental_error"] = supplemental_error
        raw_response = json.dumps(trace, ensure_ascii=False)
    return scores, raw_response


def parse_local_chain_score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Chain scores must be a JSON object")
    result = {}
    for key in ("coherence", "novelty", "budget_fit"):
        value = payload.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 100:
            raise ValueError(f"{key} must be a number between 0 and 100")
        result[key] = value
    for key in ("consistency_review", "novelty_review", "budget_review", "reason"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must include a nonempty explanation")
        result[key] = value.strip()
    return result


def compute_local_chain_score(score_payload: dict[str, Any]) -> float:
    return round(
        0.4 * score_payload["coherence"]
        + 0.3 * score_payload["novelty"]
        + 0.3 * score_payload["budget_fit"],
        4,
    )


def score_local_chain_once(
    *,
    title,
    background,
    highlights,
    summary,
    stage_name,
    seed_events,
    fixed_prefix,
    target_anchor,
    future_anchors,
    chain,
    trope_bank,
    anchor_remaining_budget,
    model_name,
    max_new_tokens,
):
    messages = build_local_chain_score_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        chain=chain,
        trope_bank=trope_bank,
        anchor_remaining_budget=anchor_remaining_budget,
    )

    def run_query(previous_response: str | None, previous_error: str | None):
        return query_json(
            retry_messages(messages, previous_response, previous_error),
            model_name=model_name,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
        )

    return call_llm_with_schema_retry(
        runner=run_query, parser=lambda payload, _raw: parse_local_chain_score(payload)
    )


def score_local_chains_in_parallel(
    *,
    title,
    background,
    highlights,
    summary,
    stage_name,
    seed_events,
    fixed_prefix,
    target_anchor,
    future_anchors,
    terminal_chains,
    trope_bank,
    anchor_remaining_budget,
    model_name,
    max_new_tokens,
    parallel_workers,
    anchor_index,
    total_anchors,
    round_index,
):
    if not terminal_chains:
        return ([], "", [])
    scores = [None] * len(terminal_chains)
    records = [None] * len(terminal_chains)
    with ThreadPoolExecutor(
        max_workers=max(1, min(parallel_workers, len(terminal_chains)))
    ) as executor:
        futures = {
            executor.submit(
                score_local_chain_once,
                title=title,
                background=background,
                highlights=highlights,
                summary=summary,
                stage_name=stage_name,
                seed_events=seed_events,
                fixed_prefix=fixed_prefix,
                target_anchor=target_anchor,
                future_anchors=future_anchors,
                chain=chain,
                trope_bank=trope_bank,
                anchor_remaining_budget=anchor_remaining_budget,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            ): index
            for index, chain in enumerate(terminal_chains)
        }
        for future in as_completed(futures):
            index = futures[future]
            score, raw = future.result()
            scores[index] = score
            records[index] = {
                "chain_index": index + 1,
                "candidate_count": 1,
                "anchor_remaining_budget_before_chain": anchor_remaining_budget,
                "candidate_event_count": len(terminal_chains[index]["events"]),
                "remaining_budget_after_chain": anchor_remaining_budget
                - len(terminal_chains[index]["events"]),
                "evaluation": score,
                "raw_response": raw,
            }
    raw_response = "\n\n".join(
        (
            f"=== chain {r['chain_index']}/{len(records)} ===\n{r['raw_response']}"
            for r in records
        )
    )
    return (scores, raw_response, records)


def choose_best_local_chain(
    scored_chains: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not scored_chains:
        return None
    return max(
        scored_chains,
        key=lambda item: (
            item["score"],
            item["coherence"],
            item["budget_fit"],
            item["novelty"],
        ),
    )


def generate_candidates_for_state_once(
    *,
    position: int,
    state: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    local_max_depth: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    anchor_remaining_budget: int,
) -> dict[str, Any]:
    path = state["path"]
    current_segment = [item["event"] for item in path]
    remaining_budget = local_max_depth - len(current_segment)
    prompt_budget = max(0, anchor_remaining_budget - len(current_segment))
    if remaining_budget <= 0:
        return {
            "position": position,
            "state": state,
            "current_segment": current_segment,
            "remaining_budget": remaining_budget,
            "candidates": [],
            "raw_generation_response": "",
            "skipped": True,
        }
    messages = build_candidate_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=current_segment,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        branch_factor=branch_factor,
        remaining_budget=prompt_budget,
    )

    def run_query(
        previous_response: str | None, previous_error: str | None
    ) -> tuple[Any, str]:
        return query_json(
            retry_messages(messages, previous_response, previous_error),
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    candidates, raw_generation_response = call_llm_with_schema_retry(
        runner=run_query,
        parser=lambda payload, _raw_response: parse_candidates(payload, branch_factor),
    )
    return {
        "position": position,
        "state": state,
        "current_segment": current_segment,
        "remaining_budget": remaining_budget,
        "candidates": candidates,
        "raw_generation_response": raw_generation_response,
        "skipped": False,
    }


def should_trigger_mutation(
    *,
    prepared_result: dict[str, Any],
    stage_name: str,
    anchor_index: int,
    round_index: int,
    depth: int,
    mutation_probability: float,
) -> tuple[bool, float]:
    if mutation_probability <= 0:
        return (False, 1.0)
    if mutation_probability >= 1:
        return (True, 0.0)
    current_segment = prepared_result.get("current_segment", [])
    fingerprint = "||".join(
        [
            stage_name,
            str(anchor_index),
            str(round_index),
            str(depth),
            str(prepared_result.get("position", 0)),
            *current_segment,
        ]
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    roll = int(digest[:12], 16) / float(16**12 - 1)
    return (roll < mutation_probability, roll)


def parse_mutation_candidate(payload: Any) -> list[dict[str, Any]]:
    candidates = parse_candidates(payload, 1)
    raw_payload = payload[0] if isinstance(payload, list) and payload else payload
    raw_candidates = (
        raw_payload.get("candidates", []) if isinstance(raw_payload, dict) else []
    )
    raw_candidate = next(
        (
            item
            for item in raw_candidates
            if isinstance(item, dict)
            and str(item.get("event", "")).strip() == candidates[0]["event"]
        ),
        None,
    )
    if raw_candidate is None:
        raise ValueError("Mutation candidate is missing original fields")
    for field in ("original_causal_implementation", "mutated_causal_implementation"):
        value = str(raw_candidate.get(field, "")).strip()
        if not value:
            raise ValueError(f"Mutation candidate is missing {field}")
        candidates[0][field] = value
    return candidates


def mutate_top_candidate_for_state_once(
    *,
    prepared_result: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    trope_bank: list[dict[str, str]],
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    anchor_index: int,
    total_anchors: int,
    depth: int,
) -> dict[str, Any]:
    scored_candidates = prepared_result.get("scored_candidates", [])
    if not scored_candidates:
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": "",
            "raw_mutation_score_response": "",
        }
    best_candidate = choose_top_candidates_for_expansion(scored_candidates, 1)[0]
    current_segment = prepared_result["current_segment"]
    messages = build_mutation_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=current_segment,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        best_candidate=best_candidate,
        scored_candidates=scored_candidates,
    )

    def run_query(
        previous_response: str | None, previous_error: str | None
    ) -> tuple[Any, str]:
        return query_json(
            retry_messages(messages, previous_response, previous_error),
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    mutation_candidates, raw_mutation_response = call_llm_with_schema_retry(
        runner=run_query,
        parser=lambda payload, _raw_response: parse_mutation_candidate(payload),
    )
    mutation_candidate = mutation_candidates[0] if mutation_candidates else None
    if not mutation_candidate:
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": raw_mutation_response,
            "raw_mutation_score_response": "",
        }
    existing_events = {candidate["event"] for candidate in scored_candidates}
    if mutation_candidate["event"] in existing_events:
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": raw_mutation_response,
            "raw_mutation_score_response": "",
        }
    mutation_scores, raw_mutation_score_response = score_candidates_for_expansion_once(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=current_segment,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        candidates=[mutation_candidate],
        trope_bank=trope_bank,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    score_payload = mutation_scores[0]
    return {
        "position": prepared_result["position"],
        "mutation_candidate": {
            **mutation_candidate,
            "logic": score_payload["logic"],
            "drive": score_payload["drive"],
            "novelty": score_payload["novelty"],
            "appeal": score_payload["appeal"],
            "evaluation_reason": score_payload["reason"],
            "expansion_score": compute_candidate_expansion_score(score_payload),
            "candidate_source": "mutation",
            "mutated_from_event": best_candidate["event"],
        },
        "raw_mutation_response": raw_mutation_response,
        "raw_mutation_score_response": raw_mutation_score_response,
    }


def score_candidates_for_state_once(
    *,
    generation_result: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    model_name: str,
    max_new_tokens: int,
    anchor_index: int,
    total_anchors: int,
    depth: int,
    trope_bank: list[dict[str, str]],
) -> dict[str, Any]:
    if generation_result["skipped"]:
        return {**generation_result, "step_scores": [], "raw_step_scoring_response": ""}
    step_scores, raw_step_scoring_response = score_candidates_for_expansion_once(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=generation_result["current_segment"],
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        candidates=generation_result["candidates"],
        trope_bank=trope_bank,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    return {
        **generation_result,
        "step_scores": step_scores,
        "raw_step_scoring_response": raw_step_scoring_response,
    }


def build_tree_with_parallel_generation(
    *,
    node_factory: NodeIdFactory,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    parallel_workers: int,
    mutation_probability: float,
    anchor_index: int,
    total_anchors: int,
    round_index: int,
    anchor_remaining_budget: int,
    trope_bank: list[dict[str, str]],
) -> dict[str, Any]:
    root = {
        "node_id": node_factory.new_id(),
        "event": "ROOT",
        "depth": 0,
        "progress_to_anchor": 0,
        "anchor_reached": False,
        "generation_reason": "root",
        "selected_for_expansion": True,
        "logic": None,
        "drive": None,
        "novelty": None,
        "appeal": None,
        "evaluation_reason": "",
        "expansion_score": None,
        "children": [],
    }
    active_states: list[dict[str, Any]] = [{"path": [], "tree_node": root}]
    completed_paths: list[dict[str, Any]] = []
    incomplete_paths: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    for depth in range(1, local_max_depth + 1):
        if not active_states:
            break
        layer_info: dict[str, Any] = {"depth": depth, "expanded_paths": []}
        generation_inputs: list[dict[str, Any]] = []
        for state in active_states:
            current_segment = [item["event"] for item in state["path"]]
            remaining_budget = local_max_depth - len(current_segment)
            if remaining_budget <= 0:
                incomplete_paths.append(
                    {
                        "events": current_segment,
                        "path": state["path"],
                        "terminal_reason": "budget_exhausted_before_generation",
                    }
                )
                continue
            generation_inputs.append(state)
        if not generation_inputs:
            break
        max_workers = max(1, min(parallel_workers, len(generation_inputs)))
        generation_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_position = {
                executor.submit(
                    generate_candidates_for_state_once,
                    position=position,
                    state=state,
                    title=title,
                    background=background,
                    highlights=highlights,
                    summary=summary,
                    stage_name=stage_name,
                    seed_events=seed_events,
                    fixed_prefix=fixed_prefix,
                    target_anchor=target_anchor,
                    future_anchors=future_anchors,
                    branch_factor=branch_factor,
                    local_max_depth=local_max_depth,
                    model_name=model_name,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    anchor_remaining_budget=anchor_remaining_budget,
                ): position
                for position, state in enumerate(generation_inputs)
            }
            for future in as_completed(future_to_position):
                generation_results.append(future.result())
        generation_results.sort(key=lambda item: item["position"])
        scoring_inputs = [
            result for result in generation_results if not result["skipped"]
        ]
        scoring_results_by_position: dict[int, dict[str, Any]] = {}
        if scoring_inputs:
            scoring_workers = max(1, min(parallel_workers, len(scoring_inputs)))
            with ThreadPoolExecutor(max_workers=scoring_workers) as executor:
                future_to_position = {
                    executor.submit(
                        score_candidates_for_state_once,
                        generation_result=result,
                        title=title,
                        background=background,
                        highlights=highlights,
                        summary=summary,
                        stage_name=stage_name,
                        seed_events=seed_events,
                        fixed_prefix=fixed_prefix,
                        target_anchor=target_anchor,
                        future_anchors=future_anchors,
                        model_name=model_name,
                        max_new_tokens=max_new_tokens,
                        anchor_index=anchor_index,
                        total_anchors=total_anchors,
                        depth=depth,
                        trope_bank=trope_bank,
                    ): result["position"]
                    for result in scoring_inputs
                }
                for future in as_completed(future_to_position):
                    scored_result = future.result()
                    scoring_results_by_position[scored_result["position"]] = (
                        scored_result
                    )
        prepared_results: list[dict[str, Any]] = []
        for result in generation_results:
            if result["skipped"]:
                prepared_results.append(result)
                continue
            scored_result = scoring_results_by_position[result["position"]]
            candidates = scored_result["candidates"]
            step_scores = scored_result["step_scores"]
            raw_step_scoring_response = scored_result["raw_step_scoring_response"]
            scored_candidates: list[dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                score_payload = step_scores[index]
                scored_candidates.append(
                    {
                        **candidate,
                        "logic": score_payload["logic"],
                        "drive": score_payload["drive"],
                        "novelty": score_payload["novelty"],
                        "appeal": score_payload["appeal"],
                        "evaluation_reason": score_payload["reason"],
                        "expansion_score": compute_candidate_expansion_score(
                            score_payload
                        ),
                        "candidate_source": "original",
                        "mutated_from_event": "",
                    }
                )
            prepared_results.append(
                {
                    **result,
                    "scored_candidates": scored_candidates,
                    "raw_step_scoring_response": raw_step_scoring_response,
                }
            )
        mutation_results_by_position: dict[int, dict[str, Any]] = {}
        mutation_inputs: list[dict[str, Any]] = []
        for result in prepared_results:
            if result.get("skipped") or not result.get("scored_candidates"):
                continue
            mutation_selected, mutation_roll = should_trigger_mutation(
                prepared_result=result,
                stage_name=stage_name,
                anchor_index=anchor_index,
                round_index=round_index,
                depth=depth,
                mutation_probability=mutation_probability,
            )
            result["mutation_roll"] = mutation_roll
            result["mutation_selected"] = mutation_selected
            if mutation_selected:
                mutation_inputs.append(result)
        if mutation_inputs:
            mutation_workers = max(1, min(parallel_workers, len(mutation_inputs)))
            with ThreadPoolExecutor(max_workers=mutation_workers) as executor:
                future_to_position = {
                    executor.submit(
                        mutate_top_candidate_for_state_once,
                        prepared_result=result,
                        title=title,
                        background=background,
                        highlights=highlights,
                        summary=summary,
                        stage_name=stage_name,
                        seed_events=seed_events,
                        fixed_prefix=fixed_prefix,
                        target_anchor=target_anchor,
                        future_anchors=future_anchors,
                        trope_bank=trope_bank,
                        model_name=model_name,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        anchor_index=anchor_index,
                        total_anchors=total_anchors,
                        depth=depth,
                    ): result["position"]
                    for result in mutation_inputs
                }
                for future in as_completed(future_to_position):
                    mutation_result = future.result()
                    mutation_results_by_position[mutation_result["position"]] = (
                        mutation_result
                    )
        next_active_states: list[dict[str, Any]] = []
        for result in prepared_results:
            state = result["state"]
            path = state["path"]
            parent_node = state["tree_node"]
            current_segment = result["current_segment"]
            if result["skipped"]:
                incomplete_paths.append(
                    {
                        "events": current_segment,
                        "path": path,
                        "terminal_reason": "budget_exhausted_before_generation",
                    }
                )
                continue
            scored_candidates = list(result["scored_candidates"])
            raw_step_scoring_response = result["raw_step_scoring_response"]
            mutation_result = mutation_results_by_position.get(result["position"], {})
            mutated_candidate = mutation_result.get("mutation_candidate")
            raw_mutation_response = mutation_result.get("raw_mutation_response", "")
            raw_mutation_score_response = mutation_result.get(
                "raw_mutation_score_response", ""
            )
            selected_original_candidates = choose_top_candidates_for_expansion(
                scored_candidates, keep_top_k
            )
            selected_candidates = list(selected_original_candidates)
            if mutated_candidate:
                selected_candidates.append(mutated_candidate)
            selected_node_keys = {
                (candidate["event"], candidate.get("candidate_source", "original"))
                for candidate in selected_candidates
            }
            expansion_record = {
                "parent_path": current_segment,
                "generation_raw_response": result["raw_generation_response"],
                "step_score_raw_response": raw_step_scoring_response,
                "mutation_raw_response": raw_mutation_response,
                "mutation_step_score_raw_response": raw_mutation_score_response,
                "generated_candidates": [],
            }
            all_candidates_for_tree = list(scored_candidates)
            if mutated_candidate:
                all_candidates_for_tree.append(mutated_candidate)
            for candidate in all_candidates_for_tree:
                node = create_node_payload(
                    node_id=node_factory.new_id(),
                    event=candidate["event"],
                    depth=depth,
                    progress_to_anchor=candidate["progress_to_anchor"],
                    anchor_reached=candidate["anchor_reached"],
                    reason=candidate.get("reason", ""),
                    selected_for_expansion=(
                        candidate["event"],
                        candidate.get("candidate_source", "original"),
                    )
                    in selected_node_keys,
                    logic=candidate["logic"],
                    drive=candidate["drive"],
                    novelty=candidate["novelty"],
                    appeal=candidate["appeal"],
                    evaluation_reason=candidate["evaluation_reason"],
                    expansion_score=candidate["expansion_score"],
                )
                node["candidate_source"] = candidate.get("candidate_source", "original")
                node["mutated_from_event"] = candidate.get("mutated_from_event", "")
                if node["candidate_source"] == "mutation":
                    node["original_causal_implementation"] = candidate[
                        "original_causal_implementation"
                    ]
                    node["mutated_causal_implementation"] = candidate[
                        "mutated_causal_implementation"
                    ]
                parent_node["children"].append(node)
                candidate_state = {
                    "node_id": node["node_id"],
                    "event": node["event"],
                    "depth": depth,
                    "progress_to_anchor": node["progress_to_anchor"],
                    "anchor_reached": node["anchor_reached"],
                    "generation_reason": node["generation_reason"],
                    "logic": node["logic"],
                    "drive": node["drive"],
                    "novelty": node["novelty"],
                    "appeal": node["appeal"],
                    "evaluation_reason": node["evaluation_reason"],
                    "expansion_score": node["expansion_score"],
                    "selected_for_expansion": node["selected_for_expansion"],
                    "candidate_source": node["candidate_source"],
                    "mutated_from_event": node["mutated_from_event"],
                }
                if node["candidate_source"] == "mutation":
                    candidate_state["original_causal_implementation"] = node[
                        "original_causal_implementation"
                    ]
                    candidate_state["mutated_causal_implementation"] = node[
                        "mutated_causal_implementation"
                    ]
                expansion_record["generated_candidates"].append(candidate_state)
                if not node["selected_for_expansion"]:
                    continue
                new_path = path + [candidate_state]
                if node["anchor_reached"]:
                    completed_paths.append(
                        {
                            "events": [item["event"] for item in new_path],
                            "path": new_path,
                            "terminal_reason": "anchor_reached",
                        }
                    )
                else:
                    next_active_states.append({"path": new_path, "tree_node": node})
            layer_info["expanded_paths"].append(expansion_record)
        layer_summaries.append(layer_info)
        active_states = next_active_states
    for state in active_states:
        incomplete_paths.append(
            {
                "events": [item["event"] for item in state["path"]],
                "path": state["path"],
                "terminal_reason": "local_max_depth_reached_without_anchor",
            }
        )
    return {
        "tree": root,
        "layer_summaries": layer_summaries,
        "completed_paths": completed_paths,
        "incomplete_paths": incomplete_paths,
    }


def collect_terminal_chains(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    terminal_chains: list[dict[str, Any]] = []
    for item in search_result.get("completed_paths", []):
        events = item.get("events", [])
        path = item.get("path", [])
        if not events or not path:
            continue
        terminal_chains.append(
            {
                "events": events,
                "path": path,
                "terminal_reason": str(item.get("terminal_reason", "anchor_reached")),
                "anchor_reached": True,
                "final_progress_to_anchor": sanitize_int(
                    path[-1].get("progress_to_anchor"),
                    default=100,
                    minimum=0,
                    maximum=100,
                ),
            }
        )
    for item in search_result.get("incomplete_paths", []):
        events = item.get("events", [])
        path = item.get("path", [])
        if not events or not path:
            continue
        terminal_chains.append(
            {
                "events": events,
                "path": path,
                "terminal_reason": str(item.get("terminal_reason", "incomplete")),
                "anchor_reached": False,
                "final_progress_to_anchor": sanitize_int(
                    path[-1].get("progress_to_anchor"),
                    default=0,
                    minimum=0,
                    maximum=100,
                ),
            }
        )
    return terminal_chains


def assemble_generation_result(
    *,
    stage_name: str,
    title: str,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    mutation_probability: float,
    model_name: str,
    temperature: float,
    generated_at: str,
    stage_payload: dict[str, Any],
    previous_stage_events: list[str],
    final_event_line: list[str],
    segment_results: list[dict[str, Any]],
    trope_bank: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "meta": {
            "mode": "outline_tree_search",
            "local_chain_judge": "individual_consistency_novelty_budget_100",
            "local_chain_score_weights": {
                "coherence": 0.4,
                "novelty": 0.3,
                "budget_fit": 0.3,
            },
            "stage": stage_name,
            "title": title,
            "branch_factor": branch_factor,
            "keep_top_k": keep_top_k,
            "local_max_depth": local_max_depth,
            "max_events_to_anchor": max_events_to_anchor,
            "max_local_chains_per_anchor": max_local_chains_per_anchor,
            "parallel_workers": parallel_workers,
            "mutation_probability": mutation_probability,
            "model_name": model_name,
            "temperature": temperature,
            "generated_at": generated_at,
            "completed_segments": len(segment_results),
            "total_segments": len(stage_payload.get("seed_events", [])),
            "is_finished": len(segment_results)
            == len(stage_payload.get("seed_events", [])),
        },
        "source_stage": stage_payload,
        "previous_stage_events": previous_stage_events,
        "final_event_line": final_event_line,
        "trope_bank": trope_bank,
        "segments": segment_results,
    }


def search_stage_eventline(
    *,
    outline: dict[str, Any],
    stage_name: str,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    mutation_probability: float = 0.25,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    initial_trope_bank: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    mutation_probability = max(0.0, min(1.0, mutation_probability))
    stage_payload = extract_stage_payload(outline, stage_name)
    seed_events = stage_payload["seed_events"]
    previous_stage_events = collect_previous_stage_events(outline, stage_name)
    title = str(outline.get("title", "")).strip()
    background = str(outline.get("background", "")).strip()
    highlights = str(outline.get("highlights", "")).strip()
    summary = str(outline.get("summary", "")).strip()
    generated_at = datetime.now().isoformat(timespec="seconds")
    mutation_probability = max(0.0, min(1.0, mutation_probability))
    node_factory = NodeIdFactory()
    final_event_line: list[str] = []
    segment_results: list[dict[str, Any]] = []
    trope_bank: list[dict[str, str]] = []
    if initial_trope_bank:
        trope_bank = list(initial_trope_bank)
    elif previous_stage_events:
        historical_motifs = abstract_micro_motifs_once(
            events=previous_stage_events,
            source_label="previous stages",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        trope_bank.extend(historical_motifs)
    for anchor_index, target_anchor in enumerate(seed_events):
        future_anchors = seed_events[anchor_index + 1 :]
        segment_prefix = previous_stage_events + final_event_line
        segment_committed: list[str] = []
        local_chain_rounds: list[dict[str, Any]] = []
        anchor_reached_by_search = False
        force_appended_anchor = False
        for round_index in range(1, max_local_chains_per_anchor + 1):
            if len(segment_committed) >= max_events_to_anchor:
                break
            remaining_total_budget = max_events_to_anchor - len(segment_committed)
            effective_local_depth = min(local_max_depth, remaining_total_budget)
            if effective_local_depth <= 0:
                break
            fixed_prefix_for_round = segment_prefix + segment_committed
            search_result = build_tree_with_parallel_generation(
                node_factory=node_factory,
                title=title,
                background=background,
                highlights=highlights,
                summary=summary,
                stage_name=stage_name,
                seed_events=seed_events,
                fixed_prefix=fixed_prefix_for_round,
                target_anchor=target_anchor,
                future_anchors=future_anchors,
                branch_factor=branch_factor,
                keep_top_k=keep_top_k,
                local_max_depth=effective_local_depth,
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                parallel_workers=parallel_workers,
                mutation_probability=mutation_probability,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
                round_index=round_index,
                anchor_remaining_budget=remaining_total_budget,
                trope_bank=trope_bank,
            )
            terminal_chains = collect_terminal_chains(search_result)
            local_chain_scores: list[dict[str, Any]] = []
            local_chain_score_raw_response = ""
            local_chain_score_records: list[dict[str, Any]] = []
            selected_local_chain: dict[str, Any] | None = None
            if terminal_chains:
                scores, local_chain_score_raw_response, local_chain_score_records = (
                    score_local_chains_in_parallel(
                        title=title,
                        background=background,
                        highlights=highlights,
                        summary=summary,
                        stage_name=stage_name,
                        seed_events=seed_events,
                        fixed_prefix=fixed_prefix_for_round,
                        target_anchor=target_anchor,
                        future_anchors=future_anchors,
                        terminal_chains=terminal_chains,
                        anchor_remaining_budget=remaining_total_budget,
                        model_name=model_name,
                        max_new_tokens=max_new_tokens,
                        parallel_workers=parallel_workers,
                        anchor_index=anchor_index,
                        total_anchors=len(seed_events),
                        round_index=round_index,
                        trope_bank=trope_bank,
                    )
                )
                for index, chain in enumerate(terminal_chains):
                    score_payload = scores[index]
                    local_chain_scores.append(
                        {
                            "chain_index": index + 1,
                            "events": chain["events"],
                            "path": chain["path"],
                            "terminal_reason": chain["terminal_reason"],
                            "anchor_reached": chain["anchor_reached"],
                            "final_progress_to_anchor": chain[
                                "final_progress_to_anchor"
                            ],
                            **score_payload,
                            "remaining_budget_after_chain": remaining_total_budget
                            - len(chain["events"]),
                            "score": compute_local_chain_score(score_payload),
                        }
                    )
                selected_local_chain = choose_best_local_chain(local_chain_scores)
            round_committed_events = (
                selected_local_chain["events"] if selected_local_chain else []
            )
            if round_committed_events:
                segment_committed.extend(round_committed_events)
            round_anchor_reached = bool(
                selected_local_chain and selected_local_chain["anchor_reached"]
            )
            local_chain_rounds.append(
                {
                    "round_index": round_index,
                    "anchor_remaining_budget_before_round": remaining_total_budget,
                    "fixed_prefix_used": fixed_prefix_for_round,
                    "tree": search_result["tree"],
                    "layer_summaries": search_result["layer_summaries"],
                    "completed_paths": search_result["completed_paths"],
                    "incomplete_paths": search_result["incomplete_paths"],
                    "terminal_chain_scores": local_chain_scores,
                    "terminal_chain_score_raw_response": local_chain_score_raw_response,
                    "terminal_chain_score_records": local_chain_score_records,
                    "selected_local_chain": selected_local_chain,
                    "round_committed": round_committed_events,
                    "anchor_reached_in_round": round_anchor_reached,
                }
            )
            if round_anchor_reached:
                anchor_reached_by_search = True
                break
            if not round_committed_events:
                break
        if not anchor_reached_by_search:
            if not segment_committed or segment_committed[-1] != target_anchor:
                segment_committed.append(target_anchor)
                force_appended_anchor = True
        final_event_line.extend(segment_committed)
        segment_motifs = abstract_micro_motifs_once(
            events=segment_committed,
            source_label=f"segment {anchor_index + 1}",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        trope_bank.extend(segment_motifs)
        segment_results.append(
            {
                "anchor_index": anchor_index,
                "target_anchor": target_anchor,
                "fixed_prefix_used": segment_prefix,
                "local_chain_rounds": local_chain_rounds,
                "segment_committed": segment_committed,
                "anchor_reached_by_search": anchor_reached_by_search,
                "force_appended_anchor": force_appended_anchor,
                "total_local_rounds": len(local_chain_rounds),
            }
        )
    return assemble_generation_result(
        stage_name=stage_name,
        title=title,
        branch_factor=branch_factor,
        keep_top_k=keep_top_k,
        local_max_depth=local_max_depth,
        max_events_to_anchor=max_events_to_anchor,
        max_local_chains_per_anchor=max_local_chains_per_anchor,
        parallel_workers=parallel_workers,
        mutation_probability=mutation_probability,
        model_name=model_name,
        temperature=temperature,
        generated_at=generated_at,
        stage_payload=stage_payload,
        previous_stage_events=previous_stage_events,
        final_event_line=final_event_line,
        segment_results=segment_results,
        trope_bank=trope_bank,
    )
