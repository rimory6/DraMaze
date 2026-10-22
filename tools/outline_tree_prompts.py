from __future__ import annotations
import json
from typing import Any

STEP_SCORING_FEWSHOT = "Example 1: In prior events, the heroine reclaimed control of the general's household by public theatrics and social pressure. The next anchor is the first debt collection. The branch now plans to pursue the Minister of Rites.\nCandidate 1: She again stages an angry public scene outside his mansion until he pays to end the embarrassment. Scores: logic 7, drive 7, novelty 3, appeal 5. It advances collection, but repeats the established mechanism of public pressure with a new target; novelty must be low.\nCandidate 2: The team learns he prizes paintings and fears a forgery scandal, then engineers a sham auction to expose him. Scores: logic 8, drive 8, novelty 8, appeal 8. It serves the same goal through a distinctly different scheme.\n\nExample 2: The leads have just begun to trust each other; the stepmother strikes back. The duke is a larger antagonist reserved for later.\nCandidate 1: The stepmother publicly reveals that the duke masterminded everything. Scores: logic 4, drive 5, novelty 2, appeal 4. This prematurely spends a later mystery rather than earning a turn.\nCandidate 2: She exploits a bookkeeping dispute to close the shop the heroine just recovered, forcing immediate self-defense. Scores: logic 8, drive 8, novelty 6, appeal 7. It fits the present situation while preserving room for the larger conspiracy."


def format_trope_bank(trope_bank: list[dict[str, str]]) -> str:
    if not trope_bank:
        return "None yet; this is the first committed segment."
    return "\n".join(
        (
            f"{i}. Cause: {m['cause']} -> Method: {m['method']} -> Outcome: {m['outcome']} -> Response: {m['response']}"
            for i, m in enumerate(trope_bank, 1)
        )
    )


def build_candidate_messages(
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
    branch_factor: int,
    remaining_budget: int,
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "event": f"One-sentence next event {index}",
                "progress_to_anchor": 72,
                "anchor_reached": False,
                "reason": "Why this candidate merits expansion",
            }
            for index in range(1, branch_factor + 1)
        ]
    }
    prompt = f"You are planning a serialized short-form drama through local tree search. Propose only the next possible events from this branch; do not write a whole phase or score candidates. Each event is one sentence. Make their mechanisms of progress meaningfully different while leading naturally toward the current anchor.\n\nTitle: {title}\nBackground: {background}\nDistinctive appeal: {highlights}\nStory summary: {summary}\nCurrent phase: {stage_name}\nThis phase's broad event anchors: {seed_events}\nFixed preceding events, including earlier phases and completed events here: {fixed_prefix}\nEvents on this branch so far: {current_segment}\nCurrent target anchor: {target_anchor}\nLater anchors, whose key mysteries must remain intact: {future_anchors}\n\nRequirements:\n1. Output exactly {branch_factor} candidates for the next event only; no dialogue, shots, or episode breakdowns.\n2. Generate a set of candidate events with substantive differences in how the plot unfolds or in the causal mechanisms involved, rather than differences in wording alone. Avoid repeating mechanisms in the fixed prefix.\n3. `progress_to_anchor` is 0–100, measuring proximity to the current anchor after that event. Set `anchor_reached` true only if the event actually reaches it.\n4. Ground candidate events in the preceding events that have already occurred. Do not skip necessary causal steps or prematurely resolve the later anchors.\n5. The branch has at most {remaining_budget} events left to reach the current anchor, though it may finish earlier.\n6. The top-level JSON value must be an object with a `candidates` key. Its value must be an array of exactly {branch_factor} candidate objects. Do not return the array by itself. Write all descriptive content in English. Output JSON only, without explanation.\n\nSchema: {json.dumps(schema)}"
    return [{"role": "user", "content": prompt}]


def build_candidate_step_score_messages(
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
) -> list[dict[str, str]]:
    schema = {
        "scores": [
            {
                "candidate_index": 1,
                "logic": 8,
                "drive": 8,
                "novelty": 7,
                "appeal": 7,
                "reason": "Two or three sentences explaining placement, progress, mechanism, and appeal.",
            }
        ]
    }
    prompt = f"You are an independent judge in a short-form drama event-tree search. Score every proposed next event at the same parent node. Judge its fit at this point, not an entire future chain.\n\nNovelty is a change in causal mechanism relative to the used trope bank, not mere oddity. Appeal means dramatic pull for an audience: conflict, a hook, contrast, suspense, or a fitting emotional payoff. An exciting-looking event that breaks motivation, skips causal steps, or spends a later anchor's mystery must score poorly on logic, novelty, and appeal.\n\n{STEP_SCORING_FEWSHOT}\n\nTitle: {title}\nBackground: {background}\nDistinctive appeal: {highlights}\nStory summary: {summary}\nCurrent phase: {stage_name}\nBroad anchors: {seed_events}\nFixed preceding events: {fixed_prefix}\nEvents on this branch: {current_segment}\nUsed causal-pattern bank (cause -> method -> outcome -> response):\n{format_trope_bank(trope_bank)}\nCurrent target anchor: {target_anchor}\nLater anchors: {future_anchors}\nCandidate next events: {json.dumps(candidates, ensure_ascii=False)}\n\nScore each candidate on a 1–10 scale:\n`logic`: fit with established motivation, causality, and event logic.\n`drive`: real progress toward the target rather than commotion in place.\n`novelty`: change in the cause–method–outcome–response mechanism relative to the bank and the fixed prefix. Give low scores for a re-skinned repeat; also consider repetition within this branch.\n`appeal`: strength of the next dramatic beat, including a hook, contrast, suspense, or a fitting payoff.\nIn `reason`, explain logic, progress, novelty, and appeal in two or three sentences.\nWrite all descriptive content in English. Output JSON only.\nSchema: {json.dumps(schema)}"
    return [{"role": "user", "content": prompt}]


def build_local_chain_score_messages(
    *,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    chain: dict[str, Any],
    trope_bank: list[dict[str, str]],
    anchor_remaining_budget: int,
) -> list[dict[str, str]]:
    remaining = anchor_remaining_budget - len(chain["events"])
    if remaining < 0:
        raise ValueError("The local chain exceeds the remaining event budget")
    schema = {
        "consistency_review": "Evidence-based check of states, causality, and transitions.",
        "novelty_review": "Compare mechanisms with relevant patterns in the bank.",
        "budget_review": "Whether the anchor is reached or can be reached within the budget.",
        "coherence": 80,
        "novelty": 70,
        "budget_fit": 80,
        "reason": "Brief summary consistent with the three scores.",
    }
    prompt = f"You are an independent judge of one local event chain in a short-form drama tree search. It follows the fixed events and aims to reach the next broad anchor. First give evidence-based reviews, then three scores of 0–100.\n\nTitle: {title}\nBackground: {background}\nDistinctive appeal: {highlights}\nSummary: {summary}\nCurrent phase: {stage_name}\nBroad anchors: {json.dumps(seed_events, ensure_ascii=False)}\nFixed preceding events (all earlier phases and committed events here):\n{json.dumps(fixed_prefix, ensure_ascii=False, indent=2)}\nCurrent local chain, and only this chain:\n{json.dumps(chain['events'], ensure_ascii=False, indent=2)}\nTarget anchor: {target_anchor}\nRemaining event budget after this chain: {remaining} events\nGenerator's report: reached={chain['anchor_reached']}, progress={chain['final_progress_to_anchor']}, stopping reason={chain['terminal_reason']}. Judge actual events, not the report alone.\nLater anchors: {json.dumps(future_anchors, ensure_ascii=False)}\nUsed causal-pattern bank:\n{format_trope_bank(trope_bank)}\n\nScoring:\n1. `coherence` (40%): Check continuity with the entire prefix and within the chain: identity, character and world states, relationships, motivation and knowledge, causes, actions and outcomes, travel and timing, and established results. Cite specific contradictions or gaps and their severity. 0–39: serious contradiction or causal break. 40–69: understandable plot but clear state, motivation, or transition gap. 70–89: coherent with minor room to improve. 90–100: highly coherent and well-motivated.\n2. `novelty` (30%): Compare cause -> method -> outcome -> response with the closest used patterns and identify meaningful differences or repetition, including repetition inside this chain. 0–39: mostly a change of characters, targets, or props. 40–69: some local variation but similar core mechanism. 70–89: clear mechanism change. 90–100: striking yet organic causal invention.\n3. `budget_fit` (30%): Judge how far the chain has actually progressed and whether remaining necessary steps fit naturally within {remaining} events. 0–39: far away, several key steps missing. 40–69: partial progress with a tight budget or major compression. 70–89: close with a plausible route. 90–100: naturally reached or ample room for a clear, complete route.\n\nThe program calculates 0.40*coherence + 0.30*novelty + 0.30*budget_fit. Give concrete evidence in each review. Write in English and output one JSON object only.\nSchema: {json.dumps(schema)}"
    return [{"role": "user", "content": prompt}]


def build_mutation_messages(
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
    best_candidate: dict[str, Any],
    scored_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "event": "One-sentence mutated next event",
                "progress_to_anchor": 72,
                "anchor_reached": False,
                "reason": "How its local causal mechanism differs",
                "original_causal_implementation": "The relevant links of the original cause -> method -> outcome -> response chain and how they work here",
                "mutated_causal_implementation": "The corresponding links and their different concrete implementation here",
            }
        ]
    }
    prompt = f"You are evolving one candidate in a short-form drama event-tree search. Starting from the highest-scoring next event, produce one mutation that changes the concrete implementation of relevant links in its local cause -> method -> outcome -> response chain. First identify which links the original candidate actually involves and how they work here; then design a different local implementation. You may change elements such as triggers, character strategies, modes of interaction, consequences of actions, or mechanisms of progress. Preserve motivation and causal logic, continue toward the target, and differ meaningfully from the original and its siblings.\n\nTitle: {title}\nBackground: {background}\nDistinctive appeal: {highlights}\nSummary: {summary}\nCurrent phase: {stage_name}\nBroad anchors: {seed_events}\nFixed preceding events: {fixed_prefix}\nEvents on this branch: {current_segment}\nCurrent target: {target_anchor}\nLater anchors, whose key mysteries must remain intact: {future_anchors}\nScored sibling candidates: {json.dumps(scored_candidates, ensure_ascii=False)}\nHighest-scoring candidate: {json.dumps(best_candidate, ensure_ascii=False)}\n\nRequirements:\n1. Output exactly one next-event candidate, not a multi-step sequence, scene, dialogue, or episode breakdown.\n2. Explicitly change the mechanism, not merely wording or surface details. For example, change direct confrontation to a trap or use of third-party leverage where the story supports it.\n3. A local causal trigger may change, but do not break motivation, skip necessary steps, or resolve later anchors early.\n4. `event` is one sentence. `progress_to_anchor` is 0–100; `anchor_reached` is true only if this event reaches the anchor.\n5. In `original_causal_implementation`, describe the concrete realization of only the relevant links in the original candidate's cause -> method -> outcome -> response chain at this node. In `mutated_causal_implementation`, describe the corresponding links after mutation. The two fields must directly reveal the causal change. Do not invent past events or present future multi-step events as accomplished facts.\n6. Write all descriptive content in English. Output JSON only.\nSchema: {json.dumps(schema)}"
    return [{"role": "user", "content": prompt}]
