from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation


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

    def test_agent_enters_counterfactual_mode_after_repetitive_low_progress_actions(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        agent.steps_since_semantic_progress = agent.config.budget.semantic_patience_steps
        agent.recent_actions.extend(["ACTION1", "ACTION2", "ACTION1", "ACTION2"])

        self.assertTrue(agent._should_force_counterfactual_exploration())

    def test_fallback_prefers_less_recent_action_instead_of_first_action(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        agent.recent_actions.extend(["ACTION3", "ACTION3", "ACTION3"])

        action = agent._fallback_action(
            observation,
            [Action(name="ACTION3"), Action(name="ACTION4")],
        )

        self.assertEqual(action.name, "ACTION4")

    def test_revisited_state_prefers_unseen_actions_during_exploration(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        agent.graph.touch("s0")
        agent.graph.nodes["s0"].outgoing["ACTION1"] = "s1"
        agent.recent_states.extend(["s0", "s0"])

        filtered = agent._prefer_unseen_actions(
            observation,
            [Action(name="ACTION1"), Action(name="ACTION2"), Action(name="ACTION3")],
            force_exploration=True,
        )

        self.assertEqual([action.name for action in filtered], ["ACTION2", "ACTION3"])

    def test_idle_agent_activates_axis_discovery_experiment(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True, notes={})  # type: ignore[arg-type]
        agent.game_memory.world_model.player_pos = (1, 1)
        agent.game_memory.world_model.visible_goal_cells = {(1, 4)}

        agent._activate_experiment_if_idle(
            observation,
            [Action(name="ACTION1"), Action(name="ACTION2")],
            force_exploration=True,
        )

        self.assertIsNotNone(agent.game_memory.experiments.active)
        self.assertEqual(agent.game_memory.experiments.active.kind, "discover_axis")

    def test_clicks_with_repeated_no_progress_are_filtered(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        stale_click = Action(name="ACTION6", payload={"x": 10, "y": 10})
        fresh_click = Action(name="ACTION6", payload={"x": 20, "y": 20})
        agent.click_no_progress_counts[stale_click.key] = 2

        filtered = agent._filter_useless_actions(observation, [stale_click, fresh_click])

        self.assertEqual([action.key for action in filtered], [fresh_click.key])

    def test_click_reprioritization_pushes_stale_clicks_back(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig())
        stale_click = Action(name="ACTION6", payload={"x": 10, "y": 10})
        fresh_click = Action(name="ACTION6", payload={"x": 20, "y": 20})
        agent.click_no_progress_counts[stale_click.key] = 1
        agent.recent_action_keys.extend([stale_click.key, stale_click.key])

        ordered = agent._reprioritize_click_actions([stale_click, fresh_click])

        self.assertEqual([action.key for action in ordered], [fresh_click.key, stale_click.key])


if __name__ == "__main__":
    unittest.main()
