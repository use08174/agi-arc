from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from arc_agi3.core.types import Action, Transition


@dataclass(slots=True)
class StateNode:
    state_key: str
    visits: int = 0
    terminal: bool = False
    outgoing: dict[str, str] = field(default_factory=dict)


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, StateNode] = {}
        self.transitions: list[Transition] = []
        self.noop_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.edge_traversals: dict[tuple[str, str, str], int] = defaultdict(int)
        self.reverse_edge_traversals: dict[tuple[str, str], int] = defaultdict(int)

    def touch(self, state_key: str, terminal: bool = False) -> None:
        node = self.nodes.setdefault(state_key, StateNode(state_key=state_key))
        node.visits += 1
        node.terminal = node.terminal or terminal

    def record(self, transition: Transition) -> None:
        self.touch(transition.from_state)
        self.touch(transition.to_state, terminal=transition.terminal)
        node = self.nodes[transition.from_state]
        node.outgoing[transition.action.key] = transition.to_state
        self.transitions.append(transition)
        self.edge_traversals[
            (transition.from_state, transition.action.key, transition.to_state)
        ] += 1
        if transition.from_state != transition.to_state:
            self.reverse_edge_traversals[(transition.to_state, transition.from_state)] += 1
        if not transition.changed:
            self.noop_counts[(transition.from_state, transition.action.key)] += 1

    def action_is_probably_useless(self, state_key: str, action: Action) -> bool:
        return self.noop_counts[(state_key, action.key)] > 0

    def seen_successor(self, state_key: str, action: Action) -> str | None:
        node = self.nodes.get(state_key)
        if node is None:
            return None
        return node.outgoing.get(action.key)

    def visits_for(self, state_key: str) -> int:
        node = self.nodes.get(state_key)
        if node is None:
            return 0
        return node.visits

    def traversals_for(self, from_state: str, action: Action, to_state: str) -> int:
        return self.edge_traversals[(from_state, action.key, to_state)]

    def is_back_edge(self, from_state: str, to_state: str) -> bool:
        return self.reverse_edge_traversals[(from_state, to_state)] > 0
