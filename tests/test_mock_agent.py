from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.types import Action, Frame, GameStatus, Observation
from arc_agi3.envs.mock_env import MockArcEnvironment


class MockAgentTest(unittest.TestCase):
    def test_mock_agent_wins(self) -> None:
        env = MockArcEnvironment()
        agent = GraphSearchAgent()
        won, steps = agent.run_episode(env)
        self.assertTrue(won)
        self.assertLessEqual(steps, agent.config.budget.max_steps_per_level)

    def test_agent_filters_known_noop_actions(self) -> None:
        agent = GraphSearchAgent()
        observation = Observation(
            state_key="s0",
            frame=Frame(grid=((0,),), status=GameStatus.IN_PROGRESS),
            changed=False,
        )
        noop_action = Action(name="ACTION1")
        useful_action = Action(name="ACTION2")
        agent.graph.noop_counts[(observation.state_key, noop_action.key)] = 1

        filtered = agent._filter_useless_actions(observation, [noop_action, useful_action])

        self.assertEqual([action.key for action in filtered], [useful_action.key])

    def test_external_stepwise_mode_learns_from_next_frame(self) -> None:
        agent = GraphSearchAgent()
        start = Frame(grid=((1, 0),), status=GameStatus.IN_PROGRESS, info={"levels_completed": 0})
        next_frame = Frame(grid=((0, 1),), status=GameStatus.IN_PROGRESS, info={"levels_completed": 0})

        first_action = agent.choose_external_action(start, [Action(name="ACTION1")])
        agent.observe_external_frame(next_frame)

        self.assertEqual(first_action.name, "ACTION1")
        self.assertGreaterEqual(len(agent.graph.transitions), 1)


if __name__ == "__main__":
    unittest.main()
