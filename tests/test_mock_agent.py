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


if __name__ == "__main__":
    unittest.main()
