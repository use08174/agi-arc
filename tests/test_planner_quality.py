from __future__ import annotations

import unittest

from arc_agi3.core.types import Action, Frame, Observation, Transition
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.planning.simple_planner import SimplePlanner


class PlannerQualityTest(unittest.TestCase):
    def test_generic_changed_move_is_not_treated_as_semantic_progress(self) -> None:
        memory = GameMemory()
        graph = StateGraph()
        action = Action(name="ACTION1")
        memory.promising_action_keys.add(action.key)
        memory.changed_action_keys.add(action.key)
        graph.record(
            Transition(
                from_state="s0",
                action=action,
                to_state="s1",
                changed=True,
            )
        )
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),)),
            changed=True,
        )

        plan = SimplePlanner().build_plan(
            observation=observation,
            actions=[action],
            graph=graph,
            game_memory=memory,
            recent_states={"s1"},
        )

        self.assertEqual(plan, [])

    def test_collectible_progress_remains_plannable(self) -> None:
        memory = GameMemory()
        graph = StateGraph()
        action = Action(name="ACTION1")
        memory.remember_effect(action.name, action.key, "changed")
        memory.remember_collectible_progress(action.key)
        graph.record(
            Transition(
                from_state="s0",
                action=action,
                to_state="s1",
                changed=True,
            )
        )
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),)),
            changed=True,
        )

        plan = SimplePlanner().build_plan(
            observation=observation,
            actions=[action],
            graph=graph,
            game_memory=memory,
            recent_states=set(),
        )

        self.assertEqual(plan[0].action.key, action.key)

    def test_alignment_improving_action_is_preferred(self) -> None:
        memory = GameMemory()
        graph = StateGraph()
        action_a = Action(name="ACTION3")
        action_b = Action(name="ACTION4")
        for action in (action_a, action_b):
            memory.remember_effect(action.name, action.key, "changed_state")
            graph.record(
                Transition(
                    from_state="s0",
                    action=action,
                    to_state=f"{action.key}_s1",
                    changed=True,
                )
            )
        memory.remember_alignment_signal(
            action_b.name,
            action_b.key,
            alignment_delta=0.18,
            family=memory.action_family(action_b.name, action_b.key),
        )
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),)),
            changed=True,
        )

        plan = SimplePlanner().build_plan(
            observation=observation,
            actions=[action_a, action_b],
            graph=graph,
            game_memory=memory,
            recent_states=set(),
        )

        self.assertEqual(plan[0].action.key, action_b.key)

    def test_planner_penalizes_recent_dominant_action_without_alignment_signal(self) -> None:
        memory = GameMemory()
        graph = StateGraph()
        action_a = Action(name="ACTION1")
        action_b = Action(name="ACTION2")
        for action in (action_a, action_b):
            memory.remember_effect(action.name, action.key, "changed_state")
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),)),
            changed=True,
        )

        plan = SimplePlanner().build_plan(
            observation=observation,
            actions=[action_a, action_b],
            graph=graph,
            game_memory=memory,
            recent_states=set(),
            recent_action_keys=["ACTION1"] * 6,
            recent_action_families=["movement"] * 6,
        )

        self.assertEqual(plan[0].action.key, action_b.key)


if __name__ == "__main__":
    unittest.main()
