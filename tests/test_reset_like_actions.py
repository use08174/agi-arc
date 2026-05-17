from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.types import Action, Frame, GameStatus, Transition
from arc_agi3.envs.base import StepResult


class ResetLikeEnvironment:
    """Tiny environment where ACTION7 jumps back to the initial state."""

    def __init__(self) -> None:
        self._x = 0

    def reset(self) -> Frame:
        self._x = 0
        return self._frame()

    def valid_actions(self) -> list[Action]:
        return [Action(name="ACTION3"), Action(name="ACTION7")]

    def step(self, action: Action) -> StepResult:
        if action.name == "ACTION3":
            self._x = 1
        elif action.name == "ACTION7":
            self._x = 0
        return StepResult(frame=self._frame(), reward_delta=0.0, done=False, won=False)

    def close(self) -> None:
        return None

    def _frame(self) -> Frame:
        return Frame(grid=((1 if self._x == 0 else 0, 1 if self._x == 1 else 0),), status=GameStatus.IN_PROGRESS)


class ResetLikeActionTest(unittest.TestCase):
    def test_agent_learns_and_filters_reset_like_action(self) -> None:
        agent = GraphSearchAgent()
        env = ResetLikeEnvironment()
        first = agent.hasher.observe(env.reset())
        agent.initial_state_key = first.state_key
        moved = agent.hasher.observe(env.step(Action(name="ACTION3")).frame, previous=first.frame)
        reset = agent.hasher.observe(env.step(Action(name="ACTION7")).frame, previous=moved.frame)
        agent._learn_meta_action(
            Transition(
                from_state=moved.state_key,
                action=Action(name="ACTION7"),
                to_state=reset.state_key,
                changed=reset.changed,
                reward_delta=0.0,
                terminal=False,
                won=False,
                notes=reset.notes,
            )
        )

        self.assertIn("ACTION7", agent.game_memory.reset_like_action_names)
        filtered = agent._filter_useless_actions(moved, [Action(name="ACTION3"), Action(name="ACTION7")])
        self.assertEqual([action.name for action in filtered], ["ACTION3"])


if __name__ == "__main__":
    unittest.main()
