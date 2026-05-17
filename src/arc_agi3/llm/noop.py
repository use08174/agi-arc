from __future__ import annotations

from arc_agi3.llm.types import LLMContext, LLMDecisionBundle


class NoOpLLMProvider:
    """Default provider used when no external LLM is configured."""

    def analyze(self, context: LLMContext) -> LLMDecisionBundle:
        del context
        return LLMDecisionBundle()

    def close(self) -> None:
        return None
