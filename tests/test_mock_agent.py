from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.envs.mock_env import MockArcEnvironment


class MockAgentTest(unittest.TestCase):
    def test_mock_agent_wins(self) -> None:
        env = MockArcEnvironment()
        agent = GraphSearchAgent()
        won, steps = agent.run_episode(env)
        self.assertTrue(won)
        self.assertLessEqual(steps, agent.config.budget.max_steps_per_level)


if __name__ == "__main__":
    unittest.main()
