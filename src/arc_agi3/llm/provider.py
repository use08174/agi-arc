from __future__ import annotations

from typing import Protocol

from arc_agi3.llm.types import LLMContext, LLMDecisionBundle


class LLMProvider(Protocol):
    def analyze(self, context: LLMContext) -> LLMDecisionBundle:
        ...

    def close(self) -> None:
        ...
