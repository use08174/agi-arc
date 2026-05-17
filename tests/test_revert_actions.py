from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.types import Action, Frame, GameStatus, Transition


def observation(agent: GraphSearchAgent, grid: tuple[tuple[int, ...], ...], previous: Frame | None = None):
    return agent.hasher.observe(Frame(grid=grid, status=GameStatus.IN_PROGRESS), previous=previous)


class RevertActionTest(unittest.TestCase):
    def test_agent_learns_and_filters_restart_like_action(self) -> None:
        agent = GraphSearchAgent()
        first = observation(agent, ((1, 0, 0),))
        second = observation(agent, ((0, 1, 0),), previous=first.frame)
        third = observation(agent, ((0, 0, 1),), previous=second.frame)
        restarted = observation(agent, ((1, 0, 0),), previous=third.frame)
        agent.initial_state_key = first.state_key
        agent.recent_states.extend([first.state_key, second.state_key, third.state_key])

        agent._learn_meta_action(
            Transition(
                from_state=third.state_key,
                action=Action(name="ACTION8"),
                to_state=restarted.state_key,
                changed=restarted.changed,
                reward_delta=0.0,
                terminal=False,
                won=False,
                notes=restarted.notes,
            )
        )

        self.assertIn("ACTION8", agent.game_memory.restart_like_action_names)
        filtered = agent._filter_useless_actions(third, [Action(name="ACTION3"), Action(name="ACTION8")])
        self.assertEqual([action.name for action in filtered], ["ACTION3"])

    def test_agent_marks_one_step_reversion_as_undo_like_without_filtering_it(self) -> None:
        agent = GraphSearchAgent()
        first = observation(agent, ((1, 0, 0),))
        second = observation(agent, ((0, 1, 0),), previous=first.frame)
        third = observation(agent, ((0, 0, 1),), previous=second.frame)
        undone = observation(agent, ((0, 1, 0),), previous=third.frame)
        agent.initial_state_key = first.state_key
        agent.recent_states.extend([first.state_key, second.state_key, third.state_key])

        agent._learn_meta_action(
            Transition(
                from_state=third.state_key,
                action=Action(name="ACTION7"),
                to_state=undone.state_key,
                changed=undone.changed,
                reward_delta=0.0,
                terminal=False,
                won=False,
                notes=undone.notes,
            )
        )

        self.assertIn("ACTION7", agent.game_memory.undo_like_action_names)
        filtered = agent._filter_useless_actions(third, [Action(name="ACTION3"), Action(name="ACTION7")])
        self.assertEqual([action.name for action in filtered], ["ACTION3", "ACTION7"])
        agent._learn_action_semantics(
            Transition(
                from_state=third.state_key,
                action=Action(name="ACTION7"),
                to_state=undone.state_key,
                changed=undone.changed,
                reward_delta=0.0,
                terminal=False,
                won=False,
                notes=undone.notes,
            )
        )
        self.assertEqual(agent.game_memory.learned_action_semantics.meaning_for("ACTION7").best_label[0], "undo_like")


if __name__ == "__main__":
    unittest.main()
