from __future__ import annotations

import unittest

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, ExperimentProposal, Transition
from arc_agi3.llm.transformers_local import TransformersLocalProvider
from arc_agi3.memory.experiments import ExperimentManager
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.world_model import WorldModel
from arc_agi3.planning.simple_planner import SimplePlanner


class ExperimentTest(unittest.TestCase):
    def test_llm_parser_accepts_exact_next_test_key(self) -> None:
        provider = TransformersLocalProvider(LLMConfig(enabled=True))
        experiments = [
            ExperimentProposal(
                key="collect_item:2,3",
                kind="collect_item",
                target=(2, 3),
                rationale="test collection",
            )
        ]
        bundle = provider._parse_response(
            '{"next_test":{"key":"collect_item:2,3","confidence":0.8,"reason":"most informative"},"ranked_actions":[]}',
            [Action(name="ACTION1")],
            experiments,
        )

        self.assertIsNotNone(bundle.next_test)
        self.assertEqual(bundle.next_test.key, "collect_item:2,3")
        self.assertEqual(bundle.next_test.source, "llm")

    def test_experiment_plan_routes_to_target(self) -> None:
        memory = GameMemory()
        memory.world_model.player_pos = (0, 0)
        memory.world_model.action_move_vectors["ACTION1"] = (1, 0)
        memory.world_model.known_traversable_cells.update({(0, 0), (1, 0), (2, 0)})
        proposal = ExperimentProposal(key="collect_item:2,0", kind="collect_item", target=(2, 0))

        plan = SimplePlanner().build_experiment_plan(
            proposal,
            [Action(name="ACTION1")],
            memory,
        )

        self.assertEqual(plan[0].action.name, "ACTION1")

    def test_collect_experiment_prefers_direct_click_when_available(self) -> None:
        memory = GameMemory()
        proposal = ExperimentProposal(key="collect_item:2,0", kind="collect_item", target=(2, 0))
        click = Action(name="ACTION6", payload={"x": 2, "y": 0})

        plan = SimplePlanner().build_experiment_plan(
            proposal,
            [Action(name="ACTION1"), click],
            memory,
        )

        self.assertEqual(plan[0].action.key, "ACTION6|x=2,y=0")

    def test_confirmed_collection_experiment_updates_hypothesis_family(self) -> None:
        world = WorldModel()
        manager = ExperimentManager(
            active=ExperimentProposal(key="collect_item:2,0", kind="collect_item", target=(2, 0))
        )
        before = world.hypothesis_library.confidence("collect_before_goal")
        outcome = manager.observe_transition(
            transition=Transition(
                from_state="s0",
                action=Action(name="ACTION1"),
                to_state="s1",
                changed=True,
            ),
            after_notes={"collectible_changes": {"removed": [(2, 0)]}},
            world=world,
        )

        self.assertIsNotNone(outcome)
        world.hypothesis_library.apply_experiment_result(
            kind=outcome.proposal.kind,
            status=outcome.status,
            evidence=outcome.evidence,
        )
        self.assertGreater(world.hypothesis_library.confidence("collect_before_goal"), before)

    def test_relation_experiment_is_available_and_plannable(self) -> None:
        memory = GameMemory()
        world = memory.world_model
        world.player_pos = (0, 0)
        world.action_move_vectors["ACTION1"] = (1, 0)
        world.known_traversable_cells.update({(0, 0), (1, 0), (2, 0)})
        world.update_from_observation(
            {
                "semantic_objects": [
                    {"color": 2, "shape_signature": "a", "bbox": {"min_x": 1, "max_x": 1, "min_y": 0, "max_y": 0}, "area": 1},
                    {"color": 2, "shape_signature": "b", "bbox": {"min_x": 2, "max_x": 2, "min_y": 0, "max_y": 0}, "area": 1},
                ]
            }
        )
        # Force a disconnected relation after the parser update.
        world.update_from_observation(
            {
                "semantic_objects": [
                    {"color": 2, "shape_signature": "a", "bbox": {"min_x": 1, "max_x": 1, "min_y": 0, "max_y": 0}, "area": 4},
                    {"color": 2, "shape_signature": "b", "bbox": {"min_x": 3, "max_x": 3, "min_y": 0, "max_y": 0}, "area": 4},
                ]
            }
        )
        proposals = memory.experiments.available(world, [Action(name="ACTION1")], set())
        relation = next(proposal for proposal in proposals if proposal.kind == "inspect_relation")

        plan = SimplePlanner().build_experiment_plan(relation, [Action(name="ACTION1")], memory)

        self.assertEqual(relation.key, "inspect_relation:same_color:2")
        self.assertEqual(plan[0].action.name, "ACTION1")

    def test_relation_distance_drop_confirms_experiment(self) -> None:
        world = WorldModel()
        world.relation_details = {
            "same_color:2": type(
                "Relation",
                (),
                {"min_distance": 2},
            )()
        }
        manager = ExperimentManager(
            active=ExperimentProposal(
                key="inspect_relation:same_color:2",
                kind="inspect_relation",
                target={"relation_key": "same_color:2", "baseline_distance": 4, "nearest_pair": ((1, 1), (5, 1))},
            )
        )

        outcome = manager.observe_transition(
            transition=Transition(
                from_state="s0",
                action=Action(name="ACTION1"),
                to_state="s1",
                changed=True,
            ),
            after_notes={},
            world=world,
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.status, "confirmed")

    def test_missing_vertical_axis_creates_discovery_experiment(self) -> None:
        memory = GameMemory()
        world = memory.world_model
        world.player_pos = (1, 1)
        world.visible_goal_cells = {(1, 4)}
        world.action_move_vectors["ACTION1"] = (1, 0)

        proposals = memory.experiments.available(
            world,
            [Action(name="ACTION1"), Action(name="ACTION2")],
            {"ACTION1"},
        )

        self.assertTrue(any(proposal.key == "discover_axis:vertical:ACTION2" for proposal in proposals))

    def test_axis_discovery_experiment_can_be_planned_directly(self) -> None:
        memory = GameMemory()
        proposal = ExperimentProposal(
            key="discover_axis:vertical:ACTION2",
            kind="discover_axis",
            target={"axis": "vertical", "action_key": "ACTION2"},
        )

        plan = SimplePlanner().build_experiment_plan(
            proposal,
            [Action(name="ACTION1"), Action(name="ACTION2")],
            memory,
        )

        self.assertEqual(plan[0].action.name, "ACTION2")

    def test_interactable_affordance_creates_experiment(self) -> None:
        memory = GameMemory()
        world = memory.world_model
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
        track.bump_affordance("breakable_candidate", 0.4)

        proposals = memory.experiments.available(world, [Action(name="ACTION1")], set())

        self.assertTrue(any(proposal.kind == "inspect_affordance" for proposal in proposals))

    def test_probe_action_is_available_again_in_new_state_context(self) -> None:
        memory = GameMemory()
        proposals = memory.experiments.available(
            memory.world_model,
            [Action(name="ACTION1"), Action(name="ACTION2")],
            {"ACTION1"},
        )

        self.assertTrue(any(proposal.key == "probe_action:ACTION2" for proposal in proposals))

    def test_active_experiment_is_not_replaced_while_in_progress(self) -> None:
        manager = ExperimentManager(
            active=ExperimentProposal(key="collect_item:1,1", kind="collect_item", target=(1, 1))
        )

        changed = manager.activate_if_idle(
            ExperimentProposal(key="collect_item:2,2", kind="collect_item", target=(2, 2))
        )

        self.assertFalse(changed)
        self.assertEqual(manager.active.key, "collect_item:1,1")


if __name__ == "__main__":
    unittest.main()
