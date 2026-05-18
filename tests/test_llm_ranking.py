from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig, LLMConfig
from arc_agi3.core.types import Action, ExperimentProposal, Observation, RankedAction
from arc_agi3.llm.provider import LLMProvider
from arc_agi3.llm.transformers_local import TransformersLocalProvider
from arc_agi3.llm.types import LLMContext, LLMDecisionBundle
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class FakeProvider:
    def analyze(self, context: LLMContext) -> LLMDecisionBundle:
        preferred = next(
            action for action in context.candidate_actions if action.name == "ACTION4"
        )
        return LLMDecisionBundle(
            ranked_actions=[
                RankedAction(action=preferred, score=1.0, reason="prefer action4")
            ]
        )

    def close(self) -> None:
        return None


class LLMRankingTest(unittest.TestCase):
    def test_ranked_actions_are_reordered(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig(), llm_config=LLMConfig(enabled=True))
        agent.llm.provider = FakeProvider()
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        actions = [Action(name="ACTION1"), Action(name="ACTION4"), Action(name="ACTION2")]
        ranked = agent.llm.rank_actions(
            observation=observation,
            candidate_actions=actions,
            graph=StateGraph(),
            game_memory=GameMemory(),
            recent_states=[],
            step_idx=8,
        )
        reordered = agent.explorer.reorder_with_rankings(
            observation=observation,
            actions=actions,
            ranked_actions=ranked,
            graph=StateGraph(),
            game_memory=GameMemory(),
            force_exploration=False,
        )
        self.assertEqual(reordered[0].name, "ACTION4")

    def test_transformers_provider_parses_fenced_json(self) -> None:
        provider = TransformersLocalProvider(LLMConfig(enabled=True))
        actions = [Action(name="ACTION3"), Action(name="ACTION4")]
        response = """```json
{
  "ranked_actions": [
    {"action": "ACTION3", "score": 0.9, "reason": "test direct move"}
  ],
  "hypotheses": [
    {"summary": "test hypothesis", "confidence": 0.6, "evidence": ["fact"]}
  ]
}
```"""
        bundle = provider._parse_response(response, actions)
        self.assertEqual(len(bundle.ranked_actions), 1)
        self.assertEqual(bundle.ranked_actions[0].action.name, "ACTION3")
        self.assertEqual(bundle.ranked_actions[0].reason, "test direct move")

    def test_transformers_provider_salvages_next_test_from_truncated_json(self) -> None:
        provider = TransformersLocalProvider(LLMConfig(enabled=True))
        experiments = [
            ExperimentProposal(
                key="collect_item:1,4",
                kind="collect_item",
                target=(1, 4),
                rationale="collect item",
            )
        ]
        response = """```json
{
  "next_test": {
    "key": "collect_item:1,4",
    "confidence": 0.9
  },
  "ranked_actions": [
    {"action": "ACTION3", "score": 90}
  ],
  "hypotheses": [
"""
        bundle = provider._parse_response(response, [Action(name="ACTION3")], experiments)

        self.assertIsNotNone(bundle.next_test)
        self.assertEqual(bundle.next_test.key, "collect_item:1,4")
        self.assertEqual(bundle.next_test.source, "llm_salvaged")

    def test_transformers_provider_normalizes_loose_coordinate_action_keys(self) -> None:
        provider = TransformersLocalProvider(LLMConfig(enabled=True))
        action = Action(name="ACTION6", payload={"x": 1, "y": 4})
        response = '{"ranked_actions":[{"action":"ACTION6|x=1,4","score":70}]}'

        bundle = provider._parse_response(response, [action])

        self.assertEqual(bundle.ranked_actions[0].action.key, "ACTION6|x=1,y=4")

    def test_unseen_actions_stay_ahead_of_seen_ranked_actions_during_exploration(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig(), llm_config=LLMConfig(enabled=True))
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        actions = [Action(name="ACTION1"), Action(name="ACTION2"), Action(name="ACTION3")]
        graph = StateGraph()
        graph.touch("s0")
        graph.nodes["s0"].outgoing["ACTION1"] = "s1"
        graph.nodes["s0"].outgoing["ACTION2"] = "s2"
        ranked = [
            RankedAction(action=actions[1], score=1.0, reason="prefer seen action2"),
            RankedAction(action=actions[0], score=0.9, reason="prefer seen action1"),
        ]
        reordered = agent.explorer.reorder_with_rankings(
            observation=observation,
            actions=actions,
            ranked_actions=ranked,
            graph=graph,
            game_memory=GameMemory(),
            force_exploration=True,
        )
        self.assertEqual(reordered[0].name, "ACTION3")

    def test_unseen_actions_beat_promising_seen_actions_during_exploration(self) -> None:
        agent = GraphSearchAgent(config=AgentConfig(), llm_config=LLMConfig(enabled=True))
        observation = Observation(state_key="s0", frame=None, changed=True)  # type: ignore[arg-type]
        actions = [Action(name="ACTION1"), Action(name="ACTION2")]
        graph = StateGraph()
        graph.touch("s0")
        graph.nodes["s0"].outgoing["ACTION1"] = "s1"
        memory = GameMemory()
        memory.promising_action_keys.add("ACTION1")
        ranked = [RankedAction(action=actions[0], score=1.0, reason="prefer seen promising action")]

        reordered = agent.explorer.reorder_with_rankings(
            observation=observation,
            actions=actions,
            ranked_actions=ranked,
            graph=graph,
            game_memory=memory,
            force_exploration=True,
        )

        self.assertEqual(reordered[0].name, "ACTION2")


if __name__ == "__main__":
    unittest.main()
