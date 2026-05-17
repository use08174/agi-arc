from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig, LLMConfig
from arc_agi3.core.types import Action, Observation, RankedAction
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
        reordered = agent.explorer.reorder_with_rankings(actions, ranked)
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


if __name__ == "__main__":
    unittest.main()
