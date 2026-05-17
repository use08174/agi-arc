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
    dangerous_action_keys: set[str] = field(default_factory=set)
    hypotheses: list[RuleHypothesis] = field(default_factory=list)
    action_use_counts: dict[str, int] = field(default_factory=dict)
    action_changed_counts: dict[str, int] = field(default_factory=dict)
    action_reward_counts: dict[str, float] = field(default_factory=dict)
    action_terminal_loss_counts: dict[str, int] = field(default_factory=dict)

    def remember_effect(self, action_name: str, action_key: str, effect: str) -> None:
        self.action_semantics[action_name] = effect
        if effect != "noop":
            self.promising_actions.add(action_name)
            self.changed_action_keys.add(action_key)
            self.action_changed_counts[action_key] = (
                self.action_changed_counts.get(action_key, 0) + 1
            )
        self.action_use_counts[action_key] = self.action_use_counts.get(action_key, 0) + 1

    def remember_danger(self, action_key: str) -> None:
        self.dangerous_action_keys.add(action_key)
        self.action_terminal_loss_counts[action_key] = (
            self.action_terminal_loss_counts.get(action_key, 0) + 1
        )

    def remember_reward(self, action_key: str, reward_delta: float) -> None:
        self.action_reward_counts[action_key] = (
            self.action_reward_counts.get(action_key, 0.0) + reward_delta
        )

    def dedupe_hypotheses(self, keep_last: int = 8) -> None:
        seen: set[str] = set()
        deduped: list[RuleHypothesis] = []
        for hypothesis in reversed(self.hypotheses):
            key = hypothesis.summary.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(hypothesis)
            if len(deduped) >= keep_last:
                break
        self.hypotheses = list(reversed(deduped))
