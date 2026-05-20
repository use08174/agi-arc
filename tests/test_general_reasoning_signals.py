from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, GameStatus, Transition
from arc_agi3.memory.action_effects import ActionEffectModel
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
        )

        self.assertEqual(signature.transform_kind, "repaint_or_mode_change")
        self.assertEqual(signature.primary_region, "playfield")
        self.assertTrue(signature.progress_like)
        self.assertIn("workspace_like", signature.semantic_roles)

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


if __name__ == "__main__":
    unittest.main()
