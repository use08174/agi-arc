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


if __name__ == "__main__":
    unittest.main()
