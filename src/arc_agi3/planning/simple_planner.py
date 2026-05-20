from __future__ import annotations

from arc_agi3.core.types import Action, ExperimentProposal, Observation, PlanStep
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph
from arc_agi3.planning.experiment_runner import ExperimentRunner
from arc_agi3.planning.path_planner import PathPlanner


class SimplePlanner:
    """World-model aware planner.

    It first follows BFS paths to semantic targets, then falls back to promising
    graph transitions.
    """

    def __init__(self) -> None:
        self.path_planner = PathPlanner()
        self.experiment_runner = ExperimentRunner()

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
        previous_action_key = None
        if recent_states and hasattr(game_memory, "action_semantics"):
            previous_action_key = None
        for action in actions:
            if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
                continue
            if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
                continue
            profile = game_memory.semantic_profile(action.name, action.key)
            contextual = game_memory.contextual_effect_profile(
                action.key,
                previous_action_key=previous_action_key,
            )
            family = game_memory.action_family(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            learned_label, learned_confidence = game_memory.learned_action_semantics.meaning_for(action.name).best_label
            hints = set(profile.interaction_hints)
            context_progress = contextual.progress_ratio if contextual is not None else 0.0
            context_transform = contextual.dominant_transform if contextual is not None else "unknown"
            alignment_score = game_memory.action_alignment_score(action.key)
            alignment_success = game_memory.action_alignment_success_rate(action.key)
            family_alignment_score = game_memory.family_alignment_score(family)
            family_alignment_success = game_memory.family_alignment_success_rate(family)
            is_semantically_promising = (
                profile.reward_total > 0
                or profile.collectible_progress > 0
                or bool(hints.intersection({"pickup_or_consume", "spawn_or_unlock", "board_or_room_transform"}))
                or context_progress >= 0.34
                or alignment_score > 0.02
                or family_alignment_score > 0.015
            )
            is_new_frontier = graph.seen_successor(observation.state_key, action) is None
            is_promising = is_semantically_promising or (
                is_new_frontier and action.key in game_memory.promising_action_keys
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
            ranked.append(
                (
                    successor is not None and node is not None and node.terminal and not node.winning_terminal,
                    "hud_or_counter_update" in hints and profile.reward_total <= 0,
                    successor is not None and successor == observation.state_key,
                    successor in recent_states if successor is not None else False,
                    graph.is_back_edge(observation.state_key, successor) if successor is not None else False,
                    alignment_score <= 0.0 and family_alignment_score <= 0.0 and context_progress < 0.20,
                    family_alignment_success <= 0.0 and alignment_success <= 0.0 and family in {"edit_or_mode", "simple:ACTION3", "simple:ACTION4", "simple:ACTION5"},
                    graph.traversals_for(observation.state_key, action, successor) if successor is not None else 0,
                    graph.visits_for(successor) if successor is not None else 0,
                    -alignment_score,
                    -family_alignment_score,
                    -alignment_success,
                    -family_alignment_success,
                    -context_progress,
                    context_transform == "noop",
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

    def build_experiment_plan(
        self,
        experiment: ExperimentProposal | None,
        actions: list[Action],
        game_memory: GameMemory,
    ) -> list[PlanStep]:
        step = self.experiment_runner.build_step(experiment, actions, game_memory)
        return [step] if step is not None else []
