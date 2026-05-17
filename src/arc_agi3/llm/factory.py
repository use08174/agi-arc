from __future__ import annotations

from arc_agi3.core.config import LLMConfig
from arc_agi3.llm.noop import NoOpLLMProvider
from arc_agi3.llm.provider import LLMProvider
from arc_agi3.llm.transformers_local import TransformersLocalProvider


def build_llm_provider(config: LLMConfig) -> LLMProvider:
    if not config.enabled or config.provider == "noop":
        return NoOpLLMProvider()
    if config.provider == "transformers_local":
        return TransformersLocalProvider(config)
    raise ValueError(f"Unknown LLM provider: {config.provider}")
