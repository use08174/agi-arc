from __future__ import annotations

import hashlib

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, ExperimentProposal, LLMDecisionTrace, Observation, RankedAction, RuleHypothesis
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
        self._last_event_signature: tuple[str, ...] = ()

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
        trigger_reason: str = "",
    ) -> list[RankedAction]:
        if not self.enabled or not candidate_actions:
            return []
        if step_idx < self.config.start_step or self._call_count >= self.config.max_calls_per_episode:
            return []
        context = self._build_context(observation, candidate_actions, graph, game_memory, recent_states)
        cache_key = self._cache_key(context)
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        event_signature = tuple(context.recent_scene_events[-2:])
        event_triggered = bool(event_signature) and event_signature != self._last_event_signature
        if not event_triggered and not trigger_reason and step_idx - self._last_call_step < self.config.step_interval:
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
                    next_test=bundle.next_test,
                )
            )
        self._last_call_step = step_idx
        self._last_event_signature = event_signature
        self._call_count += 1
        if bundle.hypotheses:
            game_memory.hypotheses.extend(bundle.hypotheses)
            game_memory.dedupe_hypotheses()
        if bundle.next_test is not None:
            available = {proposal.key: proposal for proposal in context.available_experiments}
            selected = available.get(bundle.next_test.key)
            if selected is not None:
                selected.source = "llm"
                selected.confidence = bundle.next_test.confidence
                game_memory.experiments.activate_if_idle(selected)
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
        self._last_event_signature = ()

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
            learned_meaning = game_memory.learned_action_semantics.meaning_for(action.name)
            learned_label, learned_confidence = learned_meaning.best_label
            successor = graph.seen_successor(observation.state_key, action)
            target = world.predicted_target(world.player_pos, action.name)
            flags = []
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                flags.append("RESTART_LIKE")
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                flags.append("UNDO_LIKE")
            if action.name in game_memory.failure_revert_action_names or action.key in game_memory.failure_revert_action_keys:
                flags.append("FAILURE_REVERT_LIKE")
            if world.is_unsafe_action(action):
                flags.append("UNSAFE_BY_WORLD_MODEL")
            if target is not None:
                flags.append(f"predicted_target={target}")
            if action.name == "ACTION6" and action.payload:
                point = (int(action.payload.get("x", -999)), int(action.payload.get("y", -999)))
                flags.append(f"click={point}")
                if self._click_is_display_like(point, world):
                    flags.append("DISPLAY_LIKE_CLICK")
            seen = "unseen-from-current-state" if successor is None else f"seen-successor={successor}"
            candidate_action_evidence.append(
                f"{action.key}: {seen}; {'; '.join(flags)}; learned_label={learned_label}:{learned_confidence:.2f}; "
                f"global_profile={self._format_profile(profile)}"
            )
        subgoals = world.hypothesis_library.preferred_subgoals(world)
        if world.has_precondition_evidence() and (world.visible_item_cells or world.known_item_cells):
            precondition = {"type": "test_precondition_before_goal", "reason": "state-changing collectible or goal gate suspected"}
            if precondition not in subgoals:
                subgoals.insert(0, precondition)
        if world.relation_candidates:
            relation = {"type": "inspect_relation", "target": world.relation_candidates[0]}
            if relation not in subgoals:
                subgoals.append(relation)
        current_node = graph.nodes.get(observation.state_key)
        seen_action_keys = set(current_node.outgoing) if current_node is not None else set()
        available_experiments = game_memory.experiments.available(world, candidate_actions, seen_action_keys)
        return LLMContext(
            observation=observation,
            candidate_actions=candidate_actions,
            recent_states=recent_states,
            known_promising_actions=sorted(game_memory.promising_action_keys or game_memory.promising_actions),
            known_dangerous_actions=sorted(game_memory.dangerous_action_keys),
            known_restart_like_actions=sorted(game_memory.restart_like_action_keys or game_memory.restart_like_action_names),
            known_undo_like_actions=sorted(game_memory.undo_like_action_keys or game_memory.undo_like_action_names),
            known_failure_revert_actions=sorted(game_memory.failure_revert_action_keys or game_memory.failure_revert_action_names),
            candidate_action_evidence=candidate_action_evidence,
            learned_action_semantics=game_memory.learned_action_semantics.summaries(),
            latest_transitions=latest_transitions,
            prior_hypotheses=game_memory.hypotheses,
            semantic_ascii_map=world.semantic_ascii_map or str(observation.notes.get("semantic_ascii_map", "")),
            world_model_summary=world.summary_lines(),
            recent_scene_events=world.event_lines(),
            goal_hypotheses=world.hypothesis_lines(),
            relation_candidates=world.relation_candidates,
            proposed_tests=world.hypothesis_library.proposed_tests(world),
            candidate_subgoals=subgoals,
            available_experiments=available_experiments,
            experiment_history=game_memory.experiments.summary_lines(),
        )

    def _click_is_display_like(self, point: tuple[int, int], world) -> bool:
        x, y = point
        for obj in world.last_objects:
            bbox = obj.get("bbox")
            if not isinstance(bbox, dict):
                continue
            if not (int(bbox["min_x"]) <= x <= int(bbox["max_x"]) and int(bbox["min_y"]) <= y <= int(bbox["max_y"])):
                continue
            role = str(obj.get("role", "unknown"))
            anchor = str(obj.get("anchor", ""))
            return role in {"display_candidate", "static_display"} or anchor.startswith("bottom_")
        return False

    def _cache_key(self, context: LLMContext) -> str:
        raw = {
            "state": context.observation.state_key,
            "actions": [action.key for action in context.candidate_actions],
            "recent_states": context.recent_states[-6:],
            "promising": context.known_promising_actions,
            "world": context.world_model_summary[:6],
            "events": context.recent_scene_events[-4:],
            "experiments": [proposal.key for proposal in context.available_experiments],
            "transitions": context.latest_transitions[-4:],
        }
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
