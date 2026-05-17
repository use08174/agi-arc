from __future__ import annotations

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation, RankedAction
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


class FrontierExplorer:
    """Prioritize untried or informative actions before planning deeply."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    # src/arc_agi3/exploration/frontier.py

    def choose_action(
        self,
        observation,
        actions,
        graph,
        game_memory,
        recent_states,
    ):
        ranked = []

        for action in actions:
            if action.key in game_memory.dangerous_action_keys:
                continue

            successor = graph.seen_successor(observation.state_key, action)
            profile = game_memory.semantic_profile(action.name, action.key)

            unseen_from_state = successor is None
            known_changed = action.key in game_memory.changed_action_keys
            known_reward = profile.reward_total > 0
            known_collectible = profile.collectible_progress > 0
            globally_noop = profile.uses >= 2 and profile.changed_uses == 0
            terminal_loss = profile.terminal_losses > 0

            loops_recent = successor in recent_states if successor is not None else False
            back_edge = (
                graph.is_back_edge(observation.state_key, successor)
                if successor is not None
                else False
            )

            ranked.append(
                (
                    terminal_loss,
                    globally_noop and not unseen_from_state,
                    loops_recent,
                    back_edge,
                    not known_reward,
                    not known_collectible,
                    not known_changed,
                    not unseen_from_state,
                    graph.traversals_for(observation.state_key, action, successor)
                    if successor is not None
                    else 0,
                    graph.visits_for(successor) if successor is not None else 0,
                    profile.noop_uses,
                    action.key,
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
