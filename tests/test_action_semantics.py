from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, GameStatus, Transition
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.perception.hasher import StateHasher
from arc_agi3.perception.map_parser import MapParser


class ActionSemanticsTest(unittest.TestCase):
    def test_hasher_records_diff_notes(self) -> None:
        hasher = StateHasher()
        previous = Frame(grid=((0, 0), (1, 0)), status=GameStatus.IN_PROGRESS)
        current = Frame(grid=((0, 2), (1, 0)), status=GameStatus.IN_PROGRESS)

        observation = hasher.observe(current, previous=previous)

        self.assertEqual(observation.notes["changed_cells"], 1)
        self.assertEqual(observation.notes["nonzero_delta"], 1)
        self.assertEqual(observation.notes["motion_axis"], "area")

    def test_game_memory_builds_semantic_profile(self) -> None:
        memory = GameMemory()
        action = Action(name="ACTION2")

        memory.remember_effect(action.name, action.key, "changed_state")
        memory.remember_transition_signature(
            action.name,
            action.key,
            {
                "changed_cells": 2,
                "nonzero_delta": 1,
                "unique_color_delta": 0,
                "motion_axis": "vertical",
                "region_bias": "playfield",
                "interaction_hint": "spawn_or_unlock",
            },
        )
        profile = memory.semantic_profile(action.name, action.key)

        self.assertEqual(profile.uses, 1)
        self.assertEqual(profile.changed_uses, 1)
        self.assertAlmostEqual(profile.avg_changed_cells, 2.0)
        self.assertAlmostEqual(profile.avg_nonzero_delta, 1.0)
        self.assertIn("vertical", profile.top_motion_axes)
        self.assertTrue(profile.common_change_kinds)
        self.assertIn("playfield", profile.dominant_regions)
        self.assertIn("spawn_or_unlock", profile.interaction_hints)

    def test_hasher_keeps_bottom_band_in_state_on_small_board(self) -> None:
        hasher = StateHasher()
        previous = Frame(grid=((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), status=GameStatus.IN_PROGRESS)
        current = Frame(grid=((0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0)), status=GameStatus.IN_PROGRESS)

        previous_observation = hasher.observe(previous)
        current_observation = hasher.observe(current, previous=previous)

        self.assertEqual(current_observation.notes["semantic_hud_rows"], 0)
        self.assertNotEqual(previous_observation.state_key, current_observation.state_key)
        self.assertEqual(current_observation.notes["region_bias"], "playfield")

    def test_hasher_extracts_repeated_motifs_and_anchor_regions(self) -> None:
        hasher = StateHasher()
        frame = Frame(
            grid=(
                (1, 1, 0, 0, 0, 0, 2, 2),
                (1, 0, 0, 0, 0, 0, 2, 0),
                (0, 0, 0, 5, 5, 0, 0, 0),
                (0, 0, 0, 0, 5, 0, 0, 0),
                (3, 3, 0, 0, 0, 0, 4, 4),
                (3, 0, 0, 0, 0, 0, 4, 0),
            ),
            status=GameStatus.IN_PROGRESS,
        )

        observation = hasher.observe(frame)

        self.assertGreaterEqual(observation.notes["salient_region_count"], 4)
        self.assertTrue(observation.notes["repeated_motif_summary"])
        anchors = observation.notes["anchor_region_summary"]
        self.assertTrue(any(item["anchor"] == "top_left" for item in anchors))
        self.assertTrue(any(item["anchor"] == "top_right" for item in anchors))
        patches = observation.notes["anchor_patch_summary"]
        self.assertTrue(any(item["anchor"] == "bottom_left" for item in patches))
        self.assertTrue(any(item["anchor"] == "bottom_right" for item in patches))
        self.assertTrue(observation.notes["collectible_candidates"])

    def test_parser_detects_strong_bottom_panel_as_hud(self) -> None:
        parser = MapParser()
        frame = Frame(
            grid=tuple(
                tuple(8 if y >= 14 else 0 for _ in range(16))
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )

        semantic = parser.parse(frame)

        self.assertGreaterEqual(semantic.hud_rows, 2)
        self.assertGreaterEqual(semantic.hud_confidence, 0.65)

    def test_small_anchor_patch_flash_is_marked_as_feedback(self) -> None:
        hasher = StateHasher()
        previous = Frame(
            grid=tuple(
                tuple(
                    1 if (x, y) == (3, 3) else 2 if (x, y) == (0, 14) else 0
                    for x in range(16)
                )
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )
        current = Frame(
            grid=tuple(
                tuple(
                    1 if (x, y) == (4, 3) else 3 if (x, y) == (0, 14) else 0
                    for x in range(16)
                )
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )

        observation = hasher.observe(current, previous=previous)

        self.assertTrue(observation.notes["anchor_patch_changes"])
        self.assertTrue(observation.notes["likely_feedback_flash"])

    def test_bottom_small_object_is_display_candidate_not_item(self) -> None:
        parser = MapParser()
        frame = Frame(
            grid=tuple(
                tuple(
                    7
                    if 5 <= x <= 6 and 4 <= y <= 5
                    else 5
                    if 1 <= x <= 2 and 13 <= y <= 14
                    else 0
                    for x in range(16)
                )
                for y in range(16)
            ),
            status=GameStatus.IN_PROGRESS,
        )

        semantic = parser.parse(frame)

        bottom_item_centers = [
            ((item["bbox"]["min_x"] + item["bbox"]["max_x"]) // 2, (item["bbox"]["min_y"] + item["bbox"]["max_y"]) // 2)
            for item in semantic.items
        ]
        self.assertNotIn((1, 13), bottom_item_centers)

    def test_learned_action_semantics_infer_move_and_panel_change(self) -> None:
        memory = GameMemory()
        move = Transition(
            from_state="s0",
            action=Action(name="ACTION1"),
            to_state="s1",
            changed=True,
            notes={"semantic_player_moved": True},
        )
        panel = Transition(
            from_state="s1",
            action=Action(name="ACTION7"),
            to_state="s2",
            changed=True,
            notes={"anchor_patch_changes": ["bottom_left"]},
        )

        memory.learned_action_semantics.observe(move, move_vector=(0, -1), returned_previous=False, returned_initial=False)
        memory.learned_action_semantics.observe(panel, move_vector=None, returned_previous=False, returned_initial=False)

        self.assertEqual(memory.learned_action_semantics.meaning_for("ACTION1").best_label[0], "move")
        self.assertEqual(memory.learned_action_semantics.meaning_for("ACTION7").best_label[0], "panel_or_mode_change")


if __name__ == "__main__":
    unittest.main()
