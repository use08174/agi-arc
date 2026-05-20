from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, GameStatus, Transition
from arc_agi3.memory.action_effects import ActionEffectModel
from arc_agi3.memory.effect_uncertainty import ActionEffectEnsemble
from arc_agi3.memory.progress_model import ProgressModel
from arc_agi3.perception.map_parser import MapParser


class GeneralReasoningSignalsTest(unittest.TestCase):
    def test_progress_model_separates_hud_feedback_from_progress(self) -> None:
        model = ProgressModel()
        low = model.score_transition(
            transition=Transition(from_state="a", action=Action(name="ACTION1"), to_state="b", changed=True),
            after_notes={
                "changed_hud_cells": 3,
                "changed_playfield_cells": 0,
                "interaction_hint": "hud_or_counter_update",
                "likely_feedback_flash": True,
            },
            discovered_new_state=False,
            experiment_outcome=None,
        )
        high = model.score_transition(
            transition=Transition(from_state="a", action=Action(name="ACTION1"), to_state="c", changed=True, reward_delta=1.0),
            after_notes={
                "changed_playfield_cells": 4,
                "collectible_progress": True,
                "interaction_hint": "pickup_or_consume",
            },
            discovered_new_state=True,
            experiment_outcome=None,
        )

        self.assertFalse(low.is_progress)
        self.assertTrue(high.is_progress)
        self.assertGreater(high.score, low.score)

    def test_action_effect_model_captures_general_edit_signature(self) -> None:
        model = ActionEffectModel()
        transition = Transition(
            from_state="s0",
            action=Action(name="ACTION6", payload={"x": 3, "y": 3}),
            to_state="s1",
            changed=True,
            notes={},
        )

        signature = model.observe(
            action=transition.action,
            transition=transition,
            notes={
                "changed_cells": 6,
                "changed_playfield_cells": 6,
                "nonzero_delta": 0,
                "unique_color_delta": 1,
                "interaction_hint": "unknown",
                "region_bias": "playfield",
                "anchor_patch_changes": [],
                "collectible_progress": False,
                "semantic_player_moved": False,
                "semantic_region_roles": {"workspace_like": [{"anchor": "middle_center"}], "reference_like": [{"anchor": "top_left"}]},
            },
            progress_score=0.6,
            returned_previous=False,
            returned_initial=False,
            previous_action_key="ACTION5",
            latent_state={"workspace_signature": "5x5:3", "mode_state": "select_color"},
        )

        self.assertEqual(signature.transform_kind, "repaint_or_mode_change")
        self.assertEqual(signature.primary_region, "playfield")
        self.assertTrue(signature.progress_like)
        self.assertIn("workspace_like", signature.semantic_roles)
        self.assertIn("prev=ACTION5", signature.context_key)
        self.assertIn("mode=select_color", signature.context_key)
        contextual = model.summary_for_context(
            "ACTION6|x=3,y=3",
            previous_action_key="ACTION5",
            region_bias="playfield",
            latent_state={"workspace_signature": "5x5:3", "mode_state": "select_color"},
        )
        self.assertIsNotNone(contextual)
        self.assertGreater(contextual.progress_ratio, 0.0)

    def test_parser_infers_reference_and_workspace_roles(self) -> None:
        parser = MapParser()
        frame = Frame(
            grid=tuple(
                tuple(
                    2
                    if 1 <= x <= 2 and 1 <= y <= 2
                    else 3
                    if 5 <= x <= 9 and 4 <= y <= 8
                    else 4
                    if 1 <= x <= 2 and 13 <= y <= 14
                    else 0
                    for x in range(16)
                )
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )

        semantic = parser.parse(frame)
        roles = semantic.region_roles

        self.assertTrue(roles["reference_like"])
        self.assertTrue(roles["workspace_like"])
        self.assertTrue(roles["control_like"])
        self.assertIsNotNone(semantic.region_match)
        self.assertGreater(float((semantic.region_match or {}).get("alignment_score", 0.0) or 0.0), 0.0)

    def test_parser_extracts_object_relations(self) -> None:
        parser = MapParser()
        frame = Frame(
            grid=tuple(
                tuple(
                    2
                    if (2 <= x <= 3 and 2 <= y <= 3) or (8 <= x <= 9 and 2 <= y <= 3)
                    else 5
                    if 4 <= x <= 7 and 7 <= y <= 10
                    else 0
                    for x in range(16)
                )
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )

        semantic = parser.parse(frame)

        self.assertGreaterEqual(int(semantic.relation_summary.get("entity_count", 0) or 0), 2)
        self.assertGreaterEqual(float(semantic.relation_summary.get("nearest_same_color_distance", -1) or -1), 0.0)
        self.assertTrue(any("aligned" in line or "horizontal_offset" in line for line in semantic.relation_focus))

    def test_world_model_tracks_latent_workspace_and_reference_candidates(self) -> None:
        from arc_agi3.memory.world_model import WorldModel

        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_region_roles": {
                    "reference_like": [{"bbox": {"width": 2, "height": 2}, "color": 2, "anchor": "top_left"}],
                    "workspace_like": [{"bbox": {"width": 5, "height": 5}, "color": 3, "anchor": "middle_center"}],
                    "control_like": [{"bbox": {"width": 2, "height": 2}, "color": 4, "anchor": "bottom_left"}],
                },
                "interaction_hint": "spawn_or_unlock",
                "anchor_patch_changes": ["bottom_left"],
            }
        )

        self.assertIn("workspace_signature", world.latent_state_candidates)
        self.assertIn("reference_signature", world.latent_state_candidates)
        self.assertIn("mode_state", world.latent_state_candidates)
        self.assertIn("selected_color", world.latent_state_candidates)
        self.assertIn("selected_tool", world.latent_state_candidates)

    def test_progress_model_rewards_reference_workspace_alignment_improvement(self) -> None:
        model = ProgressModel()
        signal = model.score_transition(
            transition=Transition(from_state="a", action=Action(name="ACTION6", payload={"x": 2, "y": 2}), to_state="b", changed=True),
            after_notes={
                "changed_playfield_cells": 6,
                "reference_workspace_alignment_score": 0.62,
                "reference_workspace_alignment_delta": 0.18,
                "interaction_hint": "unknown",
            },
            discovered_new_state=False,
            experiment_outcome=None,
        )

        self.assertTrue(signal.score > 0.0)
        self.assertIn("reference_workspace_alignment_improved", signal.reasons)

    def test_progress_model_rewards_relation_improvement(self) -> None:
        model = ProgressModel()
        signal = model.score_transition(
            transition=Transition(from_state="a", action=Action(name="ACTION3"), to_state="b", changed=True),
            after_notes={
                "relation_nearest_marker_improvement": 3.0,
                "relation_best_overlap_delta": 0.2,
                "relation_best_alignment_delta": 0.15,
                "changed_playfield_cells": 4,
            },
            discovered_new_state=False,
            experiment_outcome=None,
        )

        self.assertGreater(signal.score, 0.0)
        self.assertIn("marker_alignment_improved", signal.reasons)
        self.assertIn("object_overlap_improved", signal.reasons)
        self.assertIn("object_alignment_improved", signal.reasons)

    def test_world_model_builds_explicit_reference_and_region_rules(self) -> None:
        from arc_agi3.memory.world_model import WorldModel

        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_region_roles": {
                    "reference_like": [{"bbox": {"width": 2, "height": 2}, "color": 2, "anchor": "top_left"}],
                    "workspace_like": [{"bbox": {"width": 5, "height": 5}, "color": 3, "anchor": "middle_center"}],
                },
                "reference_workspace_match": {
                    "reference_anchor": "top_left",
                    "workspace_anchor": "middle_center",
                    "alignment_score": 0.68,
                },
            }
        )
        world.learn_transition(
            Action(name="ACTION5"),
            before_notes={},
            after_notes={
                "interaction_hint": "board_or_room_transform",
                "anchor_patch_changes": ["bottom_left"],
                "changed_cells": 4,
                "changed_playfield_cells": 4,
                "region_bias": "playfield",
            },
            terminal_loss=False,
        )

        rule_lines = world.rule_library.lines(limit=8)
        self.assertTrue(any("reference_for" in line for line in rule_lines))
        self.assertTrue(any("mode_setting" in line for line in rule_lines))
        self.assertTrue(any("affects_region" in line for line in rule_lines))
        self.assertTrue(any("sets_mode" in line for line in rule_lines))
        self.assertTrue(any("changes_variable" in line for line in rule_lines))

    def test_world_model_supports_relation_hypotheses(self) -> None:
        from arc_agi3.memory.world_model import WorldModel

        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_relation_summary": {
                    "entity_count": 4,
                    "best_overlap_ratio": 0.24,
                    "best_alignment_score": 0.82,
                    "nearest_marker_distance": 3.0,
                },
                "semantic_relation_focus": ["item->goal kind=aligned dist=3 overlap=0.00 align=0.82"],
            }
        )

        hypothesis_lines = world.hypothesis_lines(limit=12)
        self.assertTrue(any("marker_alignment_objective_present" in line for line in hypothesis_lines))
        self.assertTrue(any("object_alignment_may_matter" in line for line in hypothesis_lines))

    def test_effect_uncertainty_drops_after_consistent_observations(self) -> None:
        ensemble = ActionEffectEnsemble()
        before = ensemble.uncertainty_score(
            action_key="ACTION3",
            family="simple:ACTION3",
            previous_action_key=None,
            region_bias="playfield",
            mode_state="none",
            workspace_signature="none",
        )
        for _ in range(4):
            ensemble.observe(
                action_key="ACTION3",
                family="simple:ACTION3",
                previous_action_key=None,
                region_bias="playfield",
                mode_state="none",
                workspace_signature="none",
                transform_kind="movement",
                interaction_hint="unknown",
                alignment_delta=0.0,
            )
        after = ensemble.uncertainty_score(
            action_key="ACTION3",
            family="simple:ACTION3",
            previous_action_key=None,
            region_bias="playfield",
            mode_state="none",
            workspace_signature="none",
        )

        self.assertGreater(before, after)


if __name__ == "__main__":
    unittest.main()
