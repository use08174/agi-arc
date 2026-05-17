from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, GameStatus
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.perception.hasher import StateHasher


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

    def test_hasher_marks_bottom_band_as_hud(self) -> None:
        hasher = StateHasher()
        previous = Frame(grid=((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), status=GameStatus.IN_PROGRESS)
        current = Frame(grid=((0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0)), status=GameStatus.IN_PROGRESS)

        observation = hasher.observe(current, previous=previous)

        self.assertEqual(observation.notes["region_bias"], "hud")
        self.assertEqual(observation.notes["interaction_hint"], "hud_or_counter_update")

    def test_hasher_extracts_repeated_motifs_and_anchor_regions(self) -> None:
        hasher = StateHasher()
        frame = Frame(
            grid=(
                (1, 1, 0, 0, 0, 0, 2, 2),
                (1, 0, 0, 0, 0, 0, 2, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
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


if __name__ == "__main__":
    unittest.main()
