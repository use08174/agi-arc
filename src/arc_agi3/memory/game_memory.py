from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from arc_agi3.core.types import ActionSemanticProfile, RuleHypothesis


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
    action_noop_counts: dict[str, int] = field(default_factory=dict)
    action_changed_cells_total: dict[str, int] = field(default_factory=dict)
    action_nonzero_delta_total: dict[str, int] = field(default_factory=dict)
    action_unique_color_delta_total: dict[str, int] = field(default_factory=dict)
    action_motion_axes: dict[str, Counter[str]] = field(default_factory=dict)
    action_change_kinds: dict[str, Counter[str]] = field(default_factory=dict)
    action_region_biases: dict[str, Counter[str]] = field(default_factory=dict)
    action_interaction_hints: dict[str, Counter[str]] = field(default_factory=dict)
    action_feedback_counts: dict[str, int] = field(default_factory=dict)
    action_collectible_progress_counts: dict[str, int] = field(default_factory=dict)

    def remember_effect(self, action_name: str, action_key: str, effect: str) -> None:
        self.action_semantics[action_name] = effect
        if effect not in {"noop", "feedback_only"}:
            self.promising_actions.add(action_name)
            self.changed_action_keys.add(action_key)
            self.action_changed_counts[action_key] = (
                self.action_changed_counts.get(action_key, 0) + 1
            )
        elif effect == "noop":
            self.action_noop_counts[action_key] = (
                self.action_noop_counts.get(action_key, 0) + 1
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

    def remember_feedback(self, action_key: str) -> None:
        self.action_feedback_counts[action_key] = (
            self.action_feedback_counts.get(action_key, 0) + 1
        )

    def remember_collectible_progress(self, action_key: str) -> None:
        self.action_collectible_progress_counts[action_key] = (
            self.action_collectible_progress_counts.get(action_key, 0) + 1
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

    def remember_transition_signature(
        self,
        action_name: str,
        action_key: str,
        notes: dict[str, object],
    ) -> None:
        changed_cells = int(notes.get("changed_cells", 0) or 0)
        nonzero_delta = int(notes.get("nonzero_delta", 0) or 0)
        unique_color_delta = int(notes.get("unique_color_delta", 0) or 0)
        motion_axis = str(notes.get("motion_axis", "none"))
        region_bias = str(notes.get("region_bias", "none"))
        interaction_hint = str(notes.get("interaction_hint", "unknown"))

        self.action_changed_cells_total[action_key] = (
            self.action_changed_cells_total.get(action_key, 0) + changed_cells
        )
        self.action_nonzero_delta_total[action_key] = (
            self.action_nonzero_delta_total.get(action_key, 0) + nonzero_delta
        )
        self.action_unique_color_delta_total[action_key] = (
            self.action_unique_color_delta_total.get(action_key, 0) + unique_color_delta
        )
        self.action_motion_axes.setdefault(action_key, Counter())[motion_axis] += 1
        self.action_region_biases.setdefault(action_key, Counter())[region_bias] += 1
        self.action_interaction_hints.setdefault(action_key, Counter())[interaction_hint] += 1

        change_kind = self._classify_change_kind(notes)
        self.action_change_kinds.setdefault(action_key, Counter())[change_kind] += 1
        self.action_semantics[action_name] = interaction_hint if interaction_hint != "unknown" else change_kind

    def semantic_profile(self, action_name: str, action_key: str) -> ActionSemanticProfile:
        uses = self.action_use_counts.get(action_key, 0)
        changed_uses = self.action_changed_counts.get(action_key, 0)
        noop_uses = self.action_noop_counts.get(action_key, 0)
        axes = self.action_motion_axes.get(action_key, Counter())
        kinds = self.action_change_kinds.get(action_key, Counter())
        regions = self.action_region_biases.get(action_key, Counter())
        hints = self.action_interaction_hints.get(action_key, Counter())
        divisor = max(1, uses)
        return ActionSemanticProfile(
            action_key=action_key,
            action_name=action_name,
            uses=uses,
            changed_uses=changed_uses,
            noop_uses=noop_uses,
            reward_total=self.action_reward_counts.get(action_key, 0.0),
            feedback_flashes=self.action_feedback_counts.get(action_key, 0),
            collectible_progress=self.action_collectible_progress_counts.get(action_key, 0),
            terminal_losses=self.action_terminal_loss_counts.get(action_key, 0),
            terminal_wins=0,
            avg_changed_cells=self.action_changed_cells_total.get(action_key, 0) / divisor,
            avg_nonzero_delta=self.action_nonzero_delta_total.get(action_key, 0) / divisor,
            avg_unique_color_delta=self.action_unique_color_delta_total.get(action_key, 0) / divisor,
            top_motion_axes=[label for label, _ in axes.most_common(2)],
            common_change_kinds=[label for label, _ in kinds.most_common(2)],
            dominant_regions=[label for label, _ in regions.most_common(2)],
            interaction_hints=[label for label, _ in hints.most_common(3)],
        )

    def _classify_change_kind(self, notes: dict[str, object]) -> str:
        changed_cells = int(notes.get("changed_cells", 0) or 0)
        nonzero_delta = int(notes.get("nonzero_delta", 0) or 0)
        unique_color_delta = int(notes.get("unique_color_delta", 0) or 0)
        motion_axis = str(notes.get("motion_axis", "none"))
        region_bias = str(notes.get("region_bias", "none"))
        interaction_hint = str(notes.get("interaction_hint", "unknown"))

        if changed_cells == 0:
            return "noop"
        if interaction_hint == "hud_or_counter_update":
            return "hud_update"
        if nonzero_delta > 0 and changed_cells <= 4:
            return "spawn_or_extend"
        if nonzero_delta < 0 and changed_cells <= 4:
            return "erase_or_collect"
        if interaction_hint == "entity_move_or_push":
            return "entity_move_or_push"
        if interaction_hint == "pickup_or_consume":
            return "erase_or_collect"
        if abs(nonzero_delta) <= 1 and motion_axis in {"horizontal", "vertical"}:
            return f"{motion_axis}_shift_or_move"
        if unique_color_delta != 0:
            return "palette_or_mode_change"
        if region_bias == "hud":
            return "hud_update"
        if changed_cells >= 16:
            return "large_area_transform"
        return "local_transform"
