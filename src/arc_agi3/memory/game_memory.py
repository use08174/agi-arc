from __future__ import annotations

from dataclasses import dataclass, field

from arc_agi3.core.types import RuleHypothesis


@dataclass(slots=True)
class GameMemory:
    """Cross-level reusable knowledge for one game family."""

    action_semantics: dict[str, str] = field(default_factory=dict)
    promising_actions: set[str] = field(default_factory=set)
    solved_level_paths: list[list[str]] = field(default_factory=list)
    changed_action_keys: set[str] = field(default_factory=set)
    hypotheses: list[RuleHypothesis] = field(default_factory=list)

    def remember_effect(self, action_name: str, action_key: str, effect: str) -> None:
        self.action_semantics[action_name] = effect
        if effect != "noop":
            self.promising_actions.add(action_name)
            self.changed_action_keys.add(action_key)
