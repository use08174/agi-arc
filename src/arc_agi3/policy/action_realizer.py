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
            candidates = self._rank_candidates(candidates, intent, observation, graph, game_memory, recent_action_keys)
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
                return sorted(click_actions, key=lambda action: self._target_distance(action, targets))
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
        intent: AbstractIntent,
        observation: Observation,
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> list[Action]:
        key_counts = Counter(recent_action_keys[-8:])
        recent_tail = recent_action_keys[-6:]
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

            # 추가: 최근 6스텝 안에 같은 action이 있었는지
            recently_used = int(action.key in recent_tail)

            # 추가: 최근 6스텝에서 2번 이상 쓴 action은 강하게 penalty
            repeated_in_window = int(key_counts.get(action.key, 0) >= 2)

            # 추가: 성과 없는 반복 action이면 더 강한 penalty
            no_progress_repeat = int(
                key_counts.get(action.key, 0) >= 2
                and semantic.reward_total <= 0
                and semantic.changed_uses == 0
            )

            target_distance = self._target_distance(action, set(intent.target or ())) if intent.op == ObjectOp.PROBE_CLICK else 0
            movement_alignment = self._movement_alignment_penalty(action, intent, observation, game_memory)

            ranked.append(
                (
                    game_memory.is_dangerous(observation.state_key, action.key),
                    no_progress_repeat,
                    repeated_in_window,
                    recently_used,
                    immediate_repeat,
                    graph.action_was_tried(observation.state_key, action),
                    key_counts.get(action.key, 0),
                    successor is not None,
                    movement_alignment,
                    target_distance,
                    -uncertainty,
                    -semantic.reward_total,
                    -semantic.changed_uses,
                    index,
                    action,
                )
            )

        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked]

    def _movement_alignment_penalty(
        self,
        action: Action,
        intent: AbstractIntent,
        observation: Observation,
        game_memory: GameMemory,
    ) -> int:
        if intent.op != ObjectOp.PROBE_MOVE or action.payload:
            return 0
        desired = intent.target or observation.notes.get("nav_next_step_delta")
        if not (isinstance(desired, tuple) and len(desired) == 2):
            return 0
        meaning = game_memory.learned_action_semantics.meaning_for(action.name)
        if not meaning.move_vectors:
            return 1
        best_vector = meaning.move_vectors.most_common(1)[0][0]
        return abs(int(best_vector[0]) - int(desired[0])) + abs(int(best_vector[1]) - int(desired[1]))

    def _target_distance(self, action: Action, targets: set[tuple[int, int]]) -> int:
        if not targets or not action.payload:
            return 0
        x = int(action.payload.get("x", -1))
        y = int(action.payload.get("y", -1))
        return min(abs(x - tx) + abs(y - ty) for tx, ty in targets)
