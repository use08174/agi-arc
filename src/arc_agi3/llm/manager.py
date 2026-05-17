from __future__ import annotations

import hashlib

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, LLMDecisionTrace, Observation, RankedAction, RuleHypothesis
from arc_agi3.llm.noop import NoOpLLMProvider
from arc_agi3.llm.prompting import PromptBuilder
from arc_agi3.llm.provider import LLMProvider
from arc_agi3.llm.types import LLMContext
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class LLMHookManager:
    """Optional helper that injects LLM advice without owning the policy."""

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

    def rank_actions(self, observation: Observation, candidate_actions: list[Action], graph: StateGraph, game_memory: GameMemory, recent_states: list[str], step_idx: int) -> list[RankedAction]:
        if not self.enabled or not candidate_actions:
            return []
        if step_idx < self.config.start_step or self._call_count >= self.config.max_calls_per_episode:
            return []
        context = self._build_context(observation, candidate_actions, graph, game_memory, recent_states)
        cache_key = self._cache_key(context)
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        if step_idx - self._last_call_step < self.config.step_interval:
            return []
        bundle = self.provider.analyze(context)
        if self.config.trace_enabled:
            prompt = self.prompt_builder.build(context)
            self._traces.append(LLMDecisionTrace(step_idx=step_idx, state_key=observation.state_key, prompt=prompt, raw_response=bundle.raw_response, ranked_actions=bundle.ranked_actions, hypotheses=bundle.hypotheses))
        self._last_call_step = step_idx
        self._call_count += 1
        if bundle.hypotheses:
            game_memory.hypotheses.extend(bundle.hypotheses)
            game_memory.dedupe_hypotheses()
        ranked = bundle.ranked_actions[: self.config.max_ranked_actions]
        if self.config.cache_enabled:
            self._cache[cache_key] = ranked
        return ranked

    def build_prompt(self, observation: Observation, candidate_actions: list[Action], graph: StateGraph, game_memory: GameMemory, recent_states: list[str]) -> str:
        return self.prompt_builder.build(self._build_context(observation, candidate_actions, graph, game_memory, recent_states))

    def reset_episode(self) -> None:
        self._cache = {}
        self._last_call_step = -10**9
        self._call_count = 0
        self._traces = []

    def close(self) -> None:
        self.provider.close()

    def recent_traces(self) -> list[LLMDecisionTrace]:
        return list(self._traces)

    def _build_context(self, observation: Observation, candidate_actions: list[Action], graph: StateGraph, game_memory: GameMemory, recent_states: list[str]) -> LLMContext:
        latest_transitions = [
            f"{transition.from_state} --{transition.action.key}--> {transition.to_state} changed={transition.changed} reward={transition.reward_delta} terminal={transition.terminal} won={transition.won}"
            for transition in graph.transitions[-8:]
        ]
        candidate_action_evidence: list[str] = []
        world = game_memory.world_model
        for action in candidate_actions:
            profile = game_memory.semantic_profile(action.name, action.key)
            successor = graph.seen_successor(observation.state_key, action)
            target = world.predicted_target(world.player_pos, action.name)
            flags = []
            if world.is_unsafe_action(action):
                flags.append("UNSAFE_BY_WORLD_MODEL")
            if target is not None:
                flags.append(f"predicted_target={target}")
            if action.name == "ACTION6" and action.payload:
                flags.append(f"click=({action.payload.get('x')},{action.payload.get('y')})")
            seen = "unseen-from-current-state" if successor is None else f"seen-successor={successor}"
            candidate_action_evidence.append(f"{action.key}: {seen}; {'; '.join(flags)}; global_profile={self._format_profile(profile)}")
        subgoals = world.hypothesis_library.preferred_subgoals(world)
        if world.has_precondition_evidence() and (world.visible_item_cells or world.known_item_cells):
            precondition = {"type": "test_precondition_before_goal", "reason": "state-changing collectible or goal gate suspected"}
            if precondition not in subgoals:
                subgoals.insert(0, precondition)
        if world.relation_candidates:
            relation = {"type": "inspect_relation", "target": world.relation_candidates[0]}
            if relation not in subgoals:
                subgoals.append(relation)
        return LLMContext(
            observation=observation,
            candidate_actions=candidate_actions,
            recent_states=recent_states,
            known_promising_actions=sorted(game_memory.promising_action_keys or game_memory.promising_actions),
            known_dangerous_actions=sorted(game_memory.dangerous_action_keys),
            candidate_action_evidence=candidate_action_evidence,
            latest_transitions=latest_transitions,
            prior_hypotheses=game_memory.hypotheses,
            semantic_ascii_map=world.semantic_ascii_map or str(observation.notes.get("semantic_ascii_map", "")),
            world_model_summary=world.summary_lines(),
            recent_scene_events=world.event_lines(),
            goal_hypotheses=world.hypothesis_lines(),
            relation_candidates=world.relation_candidates,
            proposed_tests=world.hypothesis_library.proposed_tests(world),
            candidate_subgoals=subgoals,
        )

    def _cache_key(self, context: LLMContext) -> str:
        raw = {"state": context.observation.state_key, "actions": [action.key for action in context.candidate_actions], "recent_states": context.recent_states[-6:], "promising": context.known_promising_actions, "world": context.world_model_summary[:6], "transitions": context.latest_transitions[-4:]}
        return hashlib.sha1(repr(raw).encode("utf-8")).hexdigest()[:16]

    def _format_profile(self, profile) -> str:
        top_axes = ",".join(profile.top_motion_axes) or "none"
        change_kinds = ",".join(profile.common_change_kinds) or "unknown"
        dominant_regions = ",".join(profile.dominant_regions) or "unknown"
        interaction_hints = ",".join(profile.interaction_hints) or "unknown"
        return (
            f"uses={profile.uses}; changed={profile.changed_uses}; noop={profile.noop_uses}; "
            f"avg_changed_cells={profile.avg_changed_cells:.1f}; avg_nonzero_delta={profile.avg_nonzero_delta:.1f}; "
            f"avg_unique_color_delta={profile.avg_unique_color_delta:.1f}; axes={top_axes}; kinds={change_kinds}; "
            f"regions={dominant_regions}; hints={interaction_hints}; reward_total={profile.reward_total:.1f}; "
            f"feedback_flashes={profile.feedback_flashes}; terminal_losses={profile.terminal_losses}; collectible={profile.collectible_progress}"
        )
