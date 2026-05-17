from __future__ import annotations

from arc_agi3.core.types import Action, Observation, PlanStep
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.planning.path_planner import PathPlanner


class SimplePlanner:
    """World-model aware planner.

    It first follows BFS paths to semantic targets, then falls back to promising
    graph transitions. LLM still ranks candidates before this planner sees them.
    """

    def __init__(self) -> None:
        self.path_planner = PathPlanner()

    def build_plan(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: set[str],
    ) -> list[PlanStep]:
        world = game_memory.world_model
        path = self.path_planner.plan_to_nearest_item_or_goal(world, actions)
        if path:
            return [PlanStep(action=path[0], reason="following BFS safe path to semantic target")]

        probe = self.path_planner.safe_probe_action(world, actions)
        if probe is not None:
            return [PlanStep(action=probe, reason="probing safe adjacent frontier")]

        terminal_candidates = []
        for action in actions:
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            if world.is_unsafe_action(action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            if successor is None:
                continue
            node = graph.nodes.get(successor)
            if node is not None and node.winning_terminal:
                terminal_candidates.append(action)
        if terminal_candidates:
            return [PlanStep(action=terminal_candidates[0], reason="following known terminal transition")]

        ranked = []
        for action in actions:
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            profile = game_memory.semantic_profile(action.name, action.key)
            learned_label, learned_confidence = game_memory.learned_action_semantics.meaning_for(action.name).best_label
            is_promising = (
                action.key in game_memory.promising_action_keys
                or action.key in game_memory.changed_action_keys
                or profile.reward_total > 0
                or profile.collectible_progress > 0
            )
            if not is_promising:
                continue
            if learned_label in {"restart_like", "undo_like", "hud_only"} and learned_confidence >= 0.6:
                continue
            if world.is_unsafe_action(action):
                continue
            if graph.action_is_probably_useless(observation.state_key, action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            node = graph.nodes.get(successor) if successor is not None else None
            hints = set(profile.interaction_hints)
            ranked.append(
                (
                    successor is not None and node is not None and node.terminal and not node.winning_terminal,
                    "hud_or_counter_update" in hints and profile.reward_total <= 0,
                    successor is not None and successor == observation.state_key,
                    successor in recent_states if successor is not None else False,
                    graph.is_back_edge(observation.state_key, successor) if successor is not None else False,
                    graph.traversals_for(observation.state_key, action, successor) if successor is not None else 0,
                    graph.visits_for(successor) if successor is not None else 0,
                    -profile.reward_total,
                    -profile.collectible_progress,
                    action.key,
                    action,
                )
            )
        if ranked:
            ranked.sort(key=lambda item: item[:-1])
            return [PlanStep(action=ranked[0][-1], reason="following least-looping promising safe transition")]
        return []
