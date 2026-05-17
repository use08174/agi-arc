from __future__ import annotations

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation, RankedAction
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
    ) -> Action | None:
        world = game_memory.world_model
        ranked = []
        for index, action in enumerate(actions):
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            if world.is_unsafe_action(action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            profile = game_memory.semantic_profile(action.name, action.key)
            learned_label, learned_confidence = game_memory.learned_action_semantics.meaning_for(action.name).best_label
            unseen_from_state = successor is None
            known_changed = action.key in game_memory.changed_action_keys or action.key in game_memory.promising_action_keys
            known_reward = profile.reward_total > 0
            known_collectible = profile.collectible_progress > 0
            globally_noop = profile.uses >= 2 and profile.changed_uses == 0 and profile.reward_total <= 0
            terminal_loss = profile.terminal_losses > 0
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
            ranked.append(
                (
                    terminal_loss,
                    learned_label in {"restart_like", "undo_like", "hud_only"} and learned_confidence >= 0.6,
                    globally_noop and not unseen_from_state,
                    loops_recent,
                    back_edge,
                    not click_target_preferred,
                    not moves_toward_item,
                    not known_reward,
                    not known_collectible,
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

    def reorder_with_rankings(
        self,
        observation: Observation,
        actions: list[Action],
        ranked_actions: list[RankedAction],
        graph: StateGraph,
        game_memory: GameMemory,
        force_exploration: bool = False,
    ) -> list[Action]:
        if not ranked_actions:
            return actions
        world = game_memory.world_model
        score_by_key = {item.action.key: item.score for item in ranked_actions}
        index_by_key = {item.action.key: idx for idx, item in enumerate(ranked_actions)}

        def bucket(action: Action) -> tuple:
            unsafe = world.is_unsafe_action(action)
            restart_like = action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys
            undo_like = action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys
            learned_label, learned_confidence = game_memory.learned_action_semantics.meaning_for(action.name).best_label
            successor = graph.seen_successor(observation.state_key, action)
            unseen = successor is None
            score = -score_by_key.get(action.key, 0.0)
            rank_index = index_by_key.get(action.key, 10**6)
            if restart_like:
                return (10, rank_index, action.key)
            if undo_like:
                return (9, rank_index, action.key)
            if learned_label in {"restart_like", "undo_like", "hud_only"} and learned_confidence >= 0.6:
                return (8, rank_index, action.key)
            if unsafe:
                return (9, rank_index, action.key)
            if action.key in game_memory.promising_action_keys:
                return (0, score, rank_index, action.key)
            if unseen and force_exploration:
                return (1, score, rank_index, action.key)
            if unseen:
                return (2, score, rank_index, action.key)
            return (3, score, rank_index, action.key)

        return sorted(actions, key=bucket)
