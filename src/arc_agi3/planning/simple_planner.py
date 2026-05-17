from __future__ import annotations

from arc_agi3.core.types import Action, Observation, PlanStep
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class SimplePlanner:
    """Very small placeholder planner.

    Current behavior:
    - if game memory has a known promising action, try it
    - otherwise return no plan and let the explorer continue
    """

    def build_plan(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_states: set[str],
    ) -> list[PlanStep]:
        terminal_candidates = []
        for action in actions:
            successor = graph.seen_successor(observation.state_key, action)
            if successor is None:
                continue
            node = graph.nodes.get(successor)
            if node is not None and node.terminal:
                terminal_candidates.append(action)
        if terminal_candidates:
            return [
                PlanStep(
                    action=terminal_candidates[0],
                    reason="following known terminal transition",
                )
            ]

        ranked = []
        for action in actions:
            if action.name not in game_memory.promising_actions:
                continue
            if graph.action_is_probably_useless(observation.state_key, action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            ranked.append(
                (
                    successor is not None and successor == observation.state_key,
                    successor in recent_states if successor is not None else False,
                    graph.is_back_edge(observation.state_key, successor)
                    if successor is not None
                    else False,
                    graph.traversals_for(observation.state_key, action, successor)
                    if successor is not None
                    else 0,
                    graph.visits_for(successor) if successor is not None else 0,
                    action.key not in game_memory.changed_action_keys,
                    action,
                )
            )
        if ranked:
            ranked.sort(
                key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6].key)
            )
            return [
                PlanStep(
                    action=ranked[0][6],
                    reason="following the least-looping promising transition",
                )
            ]
        return []
