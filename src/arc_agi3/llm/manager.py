from __future__ import annotations

import hashlib

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import (
    Action,
    LLMDecisionTrace,
    Observation,
    RankedAction,
    RuleHypothesis,
)
from arc_agi3.llm.noop import NoOpLLMProvider
from arc_agi3.llm.prompting import PromptBuilder
from arc_agi3.llm.provider import LLMProvider
from arc_agi3.llm.types import LLMContext
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class LLMHookManager:
    """Optional helper that injects LLM advice without owning the policy.

    The policy still decides. The LLM can:
    - rank a shortlist of candidate actions
    - propose compact rule hypotheses
    """

    def __init__(self, config: LLMConfig | None = None, provider: LLMProvider | None = None) -> None:
        self.config = config or LLMConfig()
        self.provider = provider or NoOpLLMProvider()
        self.prompt_builder = PromptBuilder()
        self._cache: dict[str, list[RankedAction]] = {}
        self._last_call_step = -10**9
        self._call_count = 0
        self._traces: list[LLMDecisionTrace] = []

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def rank_actions(
        self,
        observation: Observation,
        candidate_actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: list[str],
        step_idx: int,
    ) -> list[RankedAction]:
        if not self.enabled or not candidate_actions:
            return []
        if step_idx < self.config.start_step:
            return []
        if self._call_count >= self.config.max_calls_per_episode:
            return []

        context = self._build_context(
            observation=observation,
            candidate_actions=candidate_actions,
            graph=graph,
            game_memory=game_memory,
            recent_states=recent_states,
        )
        cache_key = self._cache_key(context)
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        if step_idx - self._last_call_step < self.config.step_interval:
            return []
        bundle = self.provider.analyze(context)
        if self.config.trace_enabled:
            prompt = self.prompt_builder.build(context)
            self._traces.append(
                LLMDecisionTrace(
                    step_idx=step_idx,
                    state_key=observation.state_key,
                    prompt=prompt,
                    raw_response=bundle.raw_response,
                    ranked_actions=bundle.ranked_actions,
                    hypotheses=bundle.hypotheses,
                )
            )
        self._last_call_step = step_idx
        self._call_count += 1
        if bundle.hypotheses:
            game_memory.hypotheses.extend(bundle.hypotheses)
            game_memory.dedupe_hypotheses()
        ranked = bundle.ranked_actions[: self.config.max_ranked_actions]
        if self.config.cache_enabled:
            self._cache[cache_key] = ranked
        return ranked

    def build_prompt(
        self,
        observation: Observation,
        candidate_actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: list[str],
    ) -> str:
        context = self._build_context(
            observation=observation,
            candidate_actions=candidate_actions,
            graph=graph,
            game_memory=game_memory,
            recent_states=recent_states,
        )
        return self.prompt_builder.build(context)

    def reset_episode(self) -> None:
        self._cache = {}
        self._last_call_step = -10**9
        self._call_count = 0
        self._traces = []

    def close(self) -> None:
        self.provider.close()

    def recent_traces(self) -> list[LLMDecisionTrace]:
        return list(self._traces)

    def _build_context(
        self,
        observation: Observation,
        candidate_actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: list[str],
    ) -> LLMContext:
        latest_transitions = [
            (
                f"{transition.from_state} --{transition.action.key}--> "
                f"{transition.to_state} changed={transition.changed} "
                f"reward={transition.reward_delta} "
                f"terminal={transition.terminal} won={transition.won}"
            )
            for transition in graph.transitions[-8:]
        ]
        candidate_action_evidence: list[str] = []
        for action in candidate_actions:
            successor = graph.seen_successor(observation.state_key, action)
            if successor is None:
                candidate_action_evidence.append(
                    f"{action.key}: unseen-from-current-state; useful for testing a new rule"
                )
                continue
            changed = action.key in game_memory.changed_action_keys
            total_uses = game_memory.action_use_counts.get(action.key, 0)
            changed_uses = game_memory.action_changed_counts.get(action.key, 0)
            total_reward = game_memory.action_reward_counts.get(action.key, 0.0)
            danger_count = game_memory.action_terminal_loss_counts.get(action.key, 0)
            candidate_action_evidence.append(
                f"{action.key}: seen-successor={successor}; "
                f"changed_count={changed_uses}/{total_uses}; "
                f"total_reward={total_reward:.1f}; "
                f"terminal_losses={danger_count}; "
                f"changed_before={changed}"
            )
        return LLMContext(
            observation=observation,
            candidate_actions=candidate_actions,
            recent_states=recent_states,
            known_promising_actions=sorted(game_memory.promising_actions),
            known_dangerous_actions=sorted(game_memory.dangerous_action_keys),
            candidate_action_evidence=candidate_action_evidence,
            latest_transitions=latest_transitions,
            prior_hypotheses=game_memory.hypotheses,
        )

    def _cache_key(self, context: LLMContext) -> str:
        raw = {
            "state": context.observation.state_key,
            "actions": [action.key for action in context.candidate_actions],
            "recent_states": context.recent_states[-6:],
            "promising": context.known_promising_actions,
            "transitions": context.latest_transitions[-4:],
        }
        return hashlib.sha1(repr(raw).encode("utf-8")).hexdigest()[:16]
