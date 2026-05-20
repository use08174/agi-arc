from __future__ import annotations

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class FrontierExplorer:
    """Prioritize informative, safe frontier actions."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def choose_action(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: set[str],
        force_exploration: bool = False,
    ) -> Action | None:
        world = game_memory.world_model
        known_axes = {
            axis
            for dx, dy in world.action_move_vectors.values()
            for axis in (
                "horizontal" if dx != 0 else "",
                "vertical" if dy != 0 else "",
            )
            if axis
        }
        needs_axis_discovery = len(known_axes) < 2
        ranked = []
        previous_action_key = None
        for index, action in enumerate(actions):
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            if world.is_unsafe_action(action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            profile = game_memory.semantic_profile(action.name, action.key)
            contextual = game_memory.contextual_effect_profile(
                action.key,
                previous_action_key=previous_action_key,
            )
            learned_label, learned_confidence = game_memory.learned_action_semantics.meaning_for(action.name).best_label
            unseen_from_state = successor is None
            known_changed = action.key in game_memory.changed_action_keys or action.key in game_memory.promising_action_keys
            known_reward = profile.reward_total > 0
            known_collectible = profile.collectible_progress > 0
            context_progress = contextual.progress_ratio if contextual is not None else 0.0
            context_transform = contextual.dominant_transform if contextual is not None else "unknown"
            globally_noop = profile.uses >= 2 and profile.changed_uses == 0 and profile.reward_total <= 0
            terminal_loss = profile.terminal_losses > 0
            blocked = world.is_blocked_action(action)
            loops_recent = successor in recent_states if successor is not None else False
            back_edge = graph.is_back_edge(observation.state_key, successor) if successor is not None else False
            target = world.predicted_target(world.player_pos, action.name)
            moves_toward_item = False
            item_targets = world.visible_item_cells or world.known_item_cells
            if target is not None and item_targets and world.player_pos is not None:
                before = min(abs(world.player_pos[0] - ix) + abs(world.player_pos[1] - iy) for ix, iy in item_targets)
                after = min(abs(target[0] - ix) + abs(target[1] - iy) for ix, iy in item_targets)
                moves_toward_item = after < before
            click_target_preferred = False
            if action.name == "ACTION6" and action.payload:
                xy = (int(action.payload.get("x", -999)), int(action.payload.get("y", -999)))
                click_target_preferred = xy in world.preferred_click_targets(limit=12)
            movement_discovery_candidate = (
                needs_axis_discovery
                and not action.payload
                and meaning_uses_below_threshold(profile.uses)
                and action.name not in world.action_move_vectors
            )
            ranked.append(
                (
                    terminal_loss,
                    learned_label in {"restart_like", "undo_like", "hud_only"} and learned_confidence >= 0.6,
                    blocked,
                    not movement_discovery_candidate,
                    not unseen_from_state if force_exploration else False,
                    globally_noop and not unseen_from_state,
                    loops_recent,
                    back_edge,
                    not click_target_preferred,
                    not moves_toward_item,
                    not known_reward,
                    not known_collectible,
                    -context_progress,
                    context_transform == "noop",
                    not known_changed,
                    learned_label == "noop" and learned_confidence >= 0.6,
                    not unseen_from_state,
                    graph.traversals_for(observation.state_key, action, successor) if successor is not None else 0,
                    graph.visits_for(successor) if successor is not None else 0,
                    profile.noop_uses,
                    index,
                    action,
                )
            )
        if ranked:
            ranked.sort(key=lambda item: item[:-1])
            return ranked[0][-1]
        return None

    def choose_counterfactual_action(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_names: list[str],
        recent_action_keys: list[str],
    ) -> Action | None:
        world = game_memory.world_model
        ranked = []
        recent = set(recent_action_names)
        recent_keys = set(recent_action_keys)
        previous_action_key = recent_action_keys[-1] if recent_action_keys else None
        for index, action in enumerate(actions):
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            if world.is_unsafe_action(action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            if graph.action_is_probably_useless(observation.state_key, action):
                continue
            meaning = game_memory.learned_action_semantics.meaning_for(action.name)
            learned_label, learned_confidence = meaning.best_label
            contextual = game_memory.contextual_effect_profile(
                action.key,
                previous_action_key=previous_action_key,
            )
            context_progress = contextual.progress_ratio if contextual is not None else 0.0
            ranked.append(
                (
                    successor is not None,
                    action.key in recent_keys,
                    action.name in recent,
                    -context_progress,
                    learned_label not in {"unknown"} and learned_confidence >= 0.6,
                    meaning.uses,
                    index,
                    action,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[:-1])
        return ranked[0][-1]


def meaning_uses_below_threshold(uses: int, limit: int = 2) -> bool:
    return uses < limit
