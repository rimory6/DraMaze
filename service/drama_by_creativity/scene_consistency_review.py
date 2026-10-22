import json
from .scene_review import SceneReview
from .scene_outline import (
    validate_scene_outline,
    validate_repair_preserves_scenes,
)
from .scene_evidence import evidence_catalog
from .scene_review_rules import RULES, validate_conflicts

REPAIR_RULES = "Revise only confirmed conflicts marked classification=hard_conflict in the checks. Do not treat writing suggestions or uncertain items as grounds for change.\nOutput the complete scene JSON array only. Preserve scene count, IDs, order, fields, core tasks, character goals, conflicts, and length planning. Use exactly the same English schema keys as the original scenes: scene_id, time_and_location, characters, core_function, plot_summary, character_goals (with character and motivation_and_goal), conflicts_and_tension, key_actions, and ending_hook. Do not rename these keys.\nAn actually present character with the necessary ability and knowledge may take over an action. Explicit flashbacks, keepsakes, and valid transitions shown within a scene may remain.\nPlease note: Do not invent plot events in the current scene merely to explain away a contradiction. Do not treat future episode outlines as past events; do not empty scenes or rewrite worldbuilding, characters, or state memory.\n"


class SceneConsistencyReview(SceneReview):
    @classmethod
    def factory(cls, memory, episode_index, params):
        return cls.from_memory(memory, episode_index, params)

    async def check(self, scenes, slot="check_before"):
        packet = {
            "episode_number": self.episode,
            "state_catalog": self.catalog,
            "candidate_scenes": scenes,
            "scene_evidence_catalog": evidence_catalog(scenes),
        }
        try:
            return await self.request(
                slot,
                RULES,
                json.dumps(packet, ensure_ascii=False),
                lambda raw: validate_conflicts(raw, scenes, self.catalog),
            )
        except RuntimeError as error:
            return {
                "status": "uncertain",
                "action": "keep",
                "conflicts": [],
                "checks": [],
                "auto_repair_executed": False,
                "validation_fallback": {
                    "policy": "preserve_original_scene_outline",
                    "error": str(error),
                },
            }

    async def apply(self, scenes):
        original = validate_scene_outline(
            json.dumps(scenes, ensure_ascii=False), self.episode
        )
        before = await self.check(original)
        after, selected, candidate = (before, original, None)
        if before["status"] == "fail":
            hard_checks = before["conflicts"]
            if not hard_checks:
                raise ValueError("Repair requires a validated hard conflict")
            packet = {
                "original_generation_params": self.params,
                "state_catalog": self.catalog,
                "original_scenes": original,
                "hard_checks": hard_checks,
            }

            def validate_repair(raw):
                result = validate_scene_outline(raw, self.episode)
                validate_repair_preserves_scenes(
                    json.dumps(original, ensure_ascii=False), result
                )
                return result

            candidate = await self.request(
                "repair",
                REPAIR_RULES,
                json.dumps(packet, ensure_ascii=False),
                validate_repair,
                temperature=1,
            )
            after = await self.check(candidate, "check_after")
            if after["status"] == "pass":
                selected = candidate
        return selected
