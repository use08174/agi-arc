from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import ExperimentOutcome, Transition


@dataclass(slots=True)
class ProgressSignal:
    score: float
    semantic_change: float
    information_gain: float
    goal_progress: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_progress(self) -> bool:
        return self.score >= 0.45

    def to_notes(self) -> dict[str, object]:
        return {
            "progress_score": round(self.score, 4),
            "semantic_change_score": round(self.semantic_change, 4),
            "information_gain_score": round(self.information_gain, 4),
            "goal_progress_score": round(self.goal_progress, 4),
            "progress_reasons": list(self.reasons),
        }


class ProgressModel:
    """Separate visual change from meaningful forward progress."""

    def score_transition(
        self,
        transition: Transition,
        after_notes: dict[str, Any],
        discovered_new_state: bool,
        experiment_outcome: ExperimentOutcome | None,
    ) -> ProgressSignal:
        reasons: list[str] = []
        semantic_change = 0.0
        information_gain = 0.0
        goal_progress = 0.0

        if transition.reward_delta > 0 or transition.won:
            goal_progress += 1.0
            reasons.append("reward_or_win")
        if bool(after_notes.get("collectible_progress", False)):
            goal_progress += 0.75
            reasons.append("collectible_progress")
        anchor_changes = list(after_notes.get("anchor_patch_changes", []) or [])
        if anchor_changes:
            semantic_change += 0.40
            goal_progress += 0.20
            reasons.append("anchor_patch_change")
        interaction_hint = str(after_notes.get("interaction_hint", "unknown"))
        if interaction_hint in {"spawn_or_unlock", "board_or_room_transform"}:
            semantic_change += 0.45
            goal_progress += 0.20
            reasons.append(interaction_hint)
        if bool(after_notes.get("semantic_player_moved", False)):
            semantic_change += 0.20
            reasons.append("player_moved")
        changed_playfield_cells = int(after_notes.get("changed_playfield_cells", 0) or 0)
        if changed_playfield_cells > 0:
            semantic_change += min(0.35, changed_playfield_cells / 48.0)
            reasons.append("playfield_change")
        if discovered_new_state:
            information_gain += 0.35
            reasons.append("new_state")
        if experiment_outcome is not None:
            if experiment_outcome.status == "confirmed":
                information_gain += 0.45
                reasons.append("experiment_confirmed")
            elif experiment_outcome.status in {"contradicted", "abandoned"}:
                information_gain += 0.20
                reasons.append("experiment_pruned")
        if bool(after_notes.get("likely_feedback_flash", False)) and goal_progress <= 0:
            semantic_change = max(0.0, semantic_change - 0.15)
            reasons.append("feedback_only")
        if interaction_hint == "hud_or_counter_update" and goal_progress <= 0:
            semantic_change = max(0.0, semantic_change - 0.10)
            reasons.append("hud_only")

        score = min(1.5, semantic_change + information_gain + goal_progress)
        return ProgressSignal(
            score=score,
            semantic_change=semantic_change,
            information_gain=information_gain,
            goal_progress=goal_progress,
            reasons=reasons,
        )
