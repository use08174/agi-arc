from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Observation


class ExplorationPressureTest(unittest.TestCase):
    def test_agent_reenters_exploration_after_novelty_stalls(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        agent.graph.touch("s0")
        agent.steps_since_new_state = agent.config.budget.novelty_patience_steps

        self.assertTrue(agent._should_force_exploration(step_idx=99, observation=observation))

    def test_agent_reenters_exploration_after_revisiting_same_state(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        agent.recent_states.extend(["s0"] * agent.config.budget.revisit_limit)

        self.assertTrue(agent._should_force_exploration(step_idx=99, observation=observation))


if __name__ == "__main__":
    unittest.main()
