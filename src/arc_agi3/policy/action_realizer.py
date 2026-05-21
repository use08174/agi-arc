from __future__ import annotations

from collections import Counter

from arc_agi3.abstraction.object_ops import AbstractIntent, ObjectOp
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class ActionRealizer:
    """Lower object-level intents into concrete ARC-AGI-3 actions."""

    def realize(
        self,
        intents: list[AbstractIntent],
        actions: list[Action],
        observation: Observation,
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> tuple[Action, str] | None:
        if not actions:
            return None
        for intent in intents:
            candidates = self._candidates_for_intent(intent, actions, game_memory, recent_action_keys)
            candidates = self._rank_candidates(candidates, observation, graph, game_memory, recent_action_keys)
            if candidates:
                return candidates[0], intent.rationale or intent.op.value
        return None

    def _candidates_for_intent(
        self,
        intent: AbstractIntent,
        actions: list[Action],
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> list[Action]:
        if intent.op == ObjectOp.PROBE_MOVE:
            return [action for action in actions if not action.payload and action.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}]
        if intent.op == ObjectOp.PROBE_MODE:
            return [action for action in actions if not action.payload and action.name in {"ACTION5", "ACTION7"}] or [action for action in actions if not action.payload]
        if intent.op == ObjectOp.PROBE_CLICK:
            click_actions = [action for action in actions if action.payload]
            targets = set(intent.target or [])
            if targets:
                preferred = [
                    action
                    for action in click_actions
                    if (int(action.payload.get("x", -1)), int(action.payload.get("y", -1))) in targets
                ]
                if preferred:
                    return preferred
            return click_actions
        if intent.op in {ObjectOp.PROBE_TRANSFORM, ObjectOp.PROBE_ALIGNMENT}:
            return [action for action in actions if not action.payload] + [action for action in actions if action.payload]
        if intent.op == ObjectOp.EXPLOIT_KNOWN_EFFECT:
            return sorted(
                actions,
                key=lambda action: (
                    -game_memory.semantic_profile(action.name, action.key).changed_uses,
                    -game_memory.action_alignment_score(action.key),
                ),
            )
        return actions

    def _rank_candidates(
        self,
        candidates: list[Action],
        observation: Observation,
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> list[Action]:
        key_counts = Counter(recent_action_keys[-8:])
        ranked = []
        previous_action_key = recent_action_keys[-1] if recent_action_keys else None
        for index, action in enumerate(candidates):
            semantic = game_memory.semantic_profile(action.name, action.key)
            successor = graph.seen_successor(observation.state_key, action)
            uncertainty = game_memory.uncertainty_score(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            immediate_repeat = int(previous_action_key == action.key)
            ranked.append(
                (
                    game_memory.is_dangerous(observation.state_key, action.key),
                    immediate_repeat,
                    graph.action_was_tried(observation.state_key, action),
                    key_counts.get(action.key, 0),
                    successor is not None,
                    -uncertainty,
                    -semantic.reward_total,
                    -semantic.changed_uses,
                    index,
                    action,
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked]
