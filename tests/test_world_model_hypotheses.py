from __future__ import annotations

import unittest

from arc_agi3.core.types import Action
from arc_agi3.memory.world_model import WorldModel
from arc_agi3.planning.path_planner import PathPlanner


class WorldModelHypothesesTest(unittest.TestCase):
    def test_collectible_plus_patch_change_supports_precondition_hypothesis(self) -> None:
        world = WorldModel()
        world.learn_transition(
            action=Action(name="ACTION1"),
            before_notes={},
            after_notes={
                "collectible_changes": {"removed": ["plus:10:10"], "added": []},
                "anchor_patch_changes": ["bottom_left"],
                "likely_feedback_flash": False,
            },
            terminal_loss=False,
        )

        self.assertTrue(world.has_precondition_evidence())
        self.assertIn("collection_changes_state", world.hypotheses)
        self.assertTrue(any("collectible_removed" in line for line in world.event_lines()))

    def test_same_color_regions_create_connection_hypothesis_candidate(self) -> None:
        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_objects": [
                    {"color": 2, "shape_signature": "a", "bbox": {"min_x": 1, "max_x": 2, "min_y": 1, "max_y": 2}, "area": 4},
                    {"color": 2, "shape_signature": "b", "bbox": {"min_x": 8, "max_x": 9, "min_y": 1, "max_y": 2}, "area": 4},
                ]
            }
        )

        self.assertTrue(world.relation_candidates)
        self.assertIn("same_color:2", world.relation_details)
        self.assertIn("connect_same_color_regions", world.hypotheses)
        self.assertGreater(world.hypothesis_library.confidence("connect_same_feature"), 0.10)

    def test_planner_prefers_visible_items_when_precondition_is_suspected(self) -> None:
        world = WorldModel(
            player_pos=(1, 0),
            known_traversable_cells={(0, 0), (1, 0), (2, 0)},
            visible_item_cells={(0, 0)},
            visible_goal_cells={(2, 0)},
            action_move_vectors={"ACTION1": (-1, 0), "ACTION2": (1, 0)},
        )
        world._support_hypothesis("collection_changes_state", 0.5, "test")

        path = PathPlanner().plan_to_nearest_item_or_goal(world, [Action(name="ACTION1"), Action(name="ACTION2")])

        self.assertEqual([action.name for action in path], ["ACTION1"])

    def test_generic_hypothesis_library_prefers_state_match_after_collection_and_feedback(self) -> None:
        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_items": [{"bbox": {"min_x": 3, "max_x": 3, "min_y": 3, "max_y": 3}}],
                "semantic_goals": [{"bbox": {"min_x": 8, "max_x": 8, "min_y": 8, "max_y": 8}}],
                "anchor_patch_summary": [{"anchor": "bottom_left", "signature": "a"}],
            }
        )
        world.learn_transition(
            action=Action(name="ACTION2"),
            before_notes={},
            after_notes={
                "collectible_changes": {"removed": ["plus:3:3"], "added": []},
                "anchor_patch_changes": ["bottom_left"],
                "likely_feedback_flash": True,
            },
            terminal_loss=False,
        )

        ranked = world.hypothesis_library.ranked()

        self.assertEqual(ranked[0].name, "state_match_before_goal")
        self.assertTrue(world.hypothesis_library.preferred_subgoals(world))
        self.assertTrue(world.hypothesis_library.proposed_tests(world))

    def test_bottom_display_becomes_mutable_panel_after_anchor_change(self) -> None:
        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_width": 16,
                "semantic_height": 16,
                "semantic_objects": [
                    {
                        "color": 5,
                        "shape_signature": "panel",
                        "bbox": {"min_x": 1, "max_x": 2, "min_y": 13, "max_y": 14},
                        "area": 4,
                        "anchor": "bottom_left",
                        "role": "display_candidate",
                        "confidence": 0.3,
                    }
                ],
            }
        )
        world.learn_transition(
            action=Action(name="ACTION2"),
            before_notes={},
            after_notes={"anchor_patch_changes": ["bottom_left"], "collectible_changes": {"removed": [], "added": []}},
            terminal_loss=False,
        )

        track = next(iter(world.object_tracks.values()))
        self.assertEqual(track.best_role[0], "mutable_panel")

    def test_safe_probe_prefers_move_toward_visible_goal(self) -> None:
        world = WorldModel(
            player_pos=(1, 1),
            visible_goal_cells={(1, 0)},
            action_move_vectors={"ACTION1": (-1, 0), "ACTION2": (0, -1)},
        )

        action = PathPlanner().safe_probe_action(world, [Action(name="ACTION1"), Action(name="ACTION2")])

        self.assertEqual(action.name, "ACTION2")

    def test_aligned_small_object_becomes_blocking_candidate(self) -> None:
        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_player_pos": (0, 1),
                "semantic_goals": [{"bbox": {"min_x": 4, "max_x": 4, "min_y": 1, "max_y": 1}}],
                "semantic_objects": [
                    {
                        "color": 6,
                        "shape_signature": "box",
                        "bbox": {"min_x": 2, "max_x": 2, "min_y": 1, "max_y": 1},
                        "area": 4,
                        "role": "wall",
                        "confidence": 0.4,
                    }
                ],
            }
        )

        track = next(iter(world.object_tracks.values()))
        self.assertEqual(track.best_affordance[0], "blocking_candidate")
        self.assertIn(track.track_id, world.likely_blocking_tracks)

    def test_disappearing_object_becomes_breakable_candidate(self) -> None:
        world = WorldModel()
        world.update_from_observation(
            {
                "semantic_objects": [
                    {
                        "color": 6,
                        "shape_signature": "box",
                        "bbox": {"min_x": 2, "max_x": 2, "min_y": 1, "max_y": 1},
                        "area": 4,
                    }
                ]
            }
        )
        track = next(iter(world.object_tracks.values()))
        track.disappeared_count = 1

        world.learn_transition(
            action=Action(name="ACTION1"),
            before_notes={},
            after_notes={"collectible_changes": {"removed": ["box"]}},
            terminal_loss=False,
        )

        self.assertEqual(track.best_affordance[0], "breakable_candidate")


if __name__ == "__main__":
    unittest.main()
