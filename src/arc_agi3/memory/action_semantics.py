from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from arc_agi3.core.types import Transition


@dataclass(slots=True)
class LearnedActionMeaning:
    action_name: str
    uses: int = 0
    changed_uses: int = 0
    noop_uses: int = 0
    player_move_uses: int = 0
    collectible_uses: int = 0
    panel_change_uses: int = 0
    hud_only_uses: int = 0
    feedback_uses: int = 0
    terminal_loss_uses: int = 0
    reward_total: float = 0.0
    returned_previous_uses: int = 0
    returned_initial_uses: int = 0
    move_vectors: Counter[tuple[int, int]] = field(default_factory=Counter)

    def observe(
        self,
        transition: Transition,
        *,
        move_vector: tuple[int, int] | None,
        returned_previous: bool,
        returned_initial: bool,
    ) -> None:
        notes = transition.notes
        self.uses += 1
        self.changed_uses += int(bool(transition.changed))
        self.noop_uses += int(not transition.changed)
        self.player_move_uses += int(bool(notes.get("semantic_player_moved", False)))
        self.collectible_uses += int(bool(notes.get("collectible_progress", False)))
        self.panel_change_uses += int(bool(notes.get("anchor_patch_changes", [])))
        hud_only = int(notes.get("changed_hud_cells", 0) or 0) > 0 and int(notes.get("changed_playfield_cells", 0) or 0) == 0
        self.hud_only_uses += int(hud_only)
        self.feedback_uses += int(bool(notes.get("likely_feedback_flash", False)))
        self.terminal_loss_uses += int(transition.terminal and not transition.won)
        self.reward_total += transition.reward_delta
        self.returned_previous_uses += int(returned_previous)
        self.returned_initial_uses += int(returned_initial)
        if move_vector is not None and move_vector != (0, 0):
            self.move_vectors[move_vector] += 1

    @property
    def best_label(self) -> tuple[str, float]:
        if self.uses == 0:
            return "unknown", 0.0
        scores = {
            "restart_like": self.returned_initial_uses / self.uses,
            "undo_like": self.returned_previous_uses / self.uses,
            "move": self.player_move_uses / self.uses,
            "collect_or_consume": self.collectible_uses / self.uses,
            "panel_or_mode_change": self.panel_change_uses / self.uses,
            "hud_only": self.hud_only_uses / self.uses,
            "feedback_only": self.feedback_uses / self.uses,
            "noop": self.noop_uses / self.uses,
        }
        if self.reward_total > 0:
            scores["progress_or_goal"] = min(1.0, 0.5 + self.reward_total / max(1.0, self.uses))
        label, score = max(scores.items(), key=lambda item: item[1])
        return label, score

    def summary(self) -> str:
        label, confidence = self.best_label
        vector = self.move_vectors.most_common(1)[0][0] if self.move_vectors else None
        return (
            f"{self.action_name}: label={label} conf={confidence:.2f}; uses={self.uses}; "
            f"move={self.player_move_uses}; vector={vector}; collectible={self.collectible_uses}; "
            f"panel_change={self.panel_change_uses}; hud_only={self.hud_only_uses}; "
            f"return_prev={self.returned_previous_uses}; return_init={self.returned_initial_uses}; "
            f"terminal_loss={self.terminal_loss_uses}; reward={self.reward_total:.1f}"
        )


@dataclass(slots=True)
class ActionSemanticsModel:
    meanings: dict[str, LearnedActionMeaning] = field(default_factory=dict)

    def observe(
        self,
        transition: Transition,
        *,
        move_vector: tuple[int, int] | None,
        returned_previous: bool,
        returned_initial: bool,
    ) -> None:
        meaning = self.meanings.setdefault(transition.action.name, LearnedActionMeaning(action_name=transition.action.name))
        meaning.observe(
            transition,
            move_vector=move_vector,
            returned_previous=returned_previous,
            returned_initial=returned_initial,
        )

    def meaning_for(self, action_name: str) -> LearnedActionMeaning:
        return self.meanings.setdefault(action_name, LearnedActionMeaning(action_name=action_name))

    def summaries(self, limit: int = 8) -> list[str]:
        return [meaning.summary() for meaning in sorted(self.meanings.values(), key=lambda item: item.action_name)[:limit]]
