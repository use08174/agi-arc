from __future__ import annotations

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation, RankedAction
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class FrontierExplorer:
    """Prioritize untried or informative actions before planning deeply."""

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
        for action in actions:
            if action.key in game_memory.dangerous_action_keys:
                continue
            if graph.seen_successor(observation.state_key, action) is None:
                return action

        ranked = []
        for action in actions:
            if action.key in game_memory.dangerous_action_keys:
                continue
            if graph.action_is_probably_useless(observation.state_key, action):
                continue
            successor = graph.seen_successor(observation.state_key, action)
            if successor is None:
                return action
            ranked.append(
                (
                    action.key in game_memory.dangerous_action_keys,
                    action.name not in game_memory.promising_actions,
                    successor in recent_states,
                    graph.is_back_edge(observation.state_key, successor),
                    graph.traversals_for(observation.state_key, action, successor),
                    graph.visits_for(successor),
                    action,
                )
            )
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6].key))
            return ranked[0][6]

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

        def bucket(action: Action) -> int:
            unseen = graph.seen_successor(observation.state_key, action) is None
            dangerous = action.key in game_memory.dangerous_action_keys
            if dangerous:
                return 3
            if unseen:
                return 0
            if force_exploration:
                return 2
            return 1

        ranked_keys = [item.action.key for item in ranked_actions]
        by_key = {action.key: action for action in actions}
        score_index = {key: idx for idx, key in enumerate(ranked_keys)}
        ordered = sorted(
            actions,
            key=lambda action: (
                bucket(action),
                score_index.get(action.key, 10**6),
                action.key,
            ),
        )
        return ordered
