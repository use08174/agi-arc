from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action, Transition


@dataclass(slots=True)
class ActionEffectSignature:
    transform_kind: str
    spatial_scope: str
    primary_region: str
    interaction_hint: str
    semantic_roles: tuple[str, ...]
    reversible: bool
    progress_like: bool
    reward_like: bool


@dataclass(slots=True)
class ActionEffectAggregate:
    uses: int = 0
    progress_uses: int = 0
    reversible_uses: int = 0
    reward_total: float = 0.0
    transform_kinds: Counter[str] = field(default_factory=Counter)
    spatial_scopes: Counter[str] = field(default_factory=Counter)
    primary_regions: Counter[str] = field(default_factory=Counter)
    interaction_hints: Counter[str] = field(default_factory=Counter)
    semantic_roles: Counter[str] = field(default_factory=Counter)

    @property
    def dominant_transform(self) -> str:
        return self.transform_kinds.most_common(1)[0][0] if self.transform_kinds else "unknown"

    @property
    def dominant_region(self) -> str:
        return self.primary_regions.most_common(1)[0][0] if self.primary_regions else "unknown"

    @property
    def progress_ratio(self) -> float:
        return self.progress_uses / max(1, self.uses)


class ActionEffectModel:
    """General action-effect abstraction shared across all games."""

    def __init__(self) -> None:
        self.signatures: dict[str, ActionEffectAggregate] = {}

    def observe(
        self,
        action: Action,
        transition: Transition,
        notes: dict[str, Any],
        progress_score: float,
        returned_previous: bool,
        returned_initial: bool,
    ) -> ActionEffectSignature:
        signature = self._infer_signature(
            transition=transition,
            notes=notes,
            progress_score=progress_score,
            returned_previous=returned_previous,
            returned_initial=returned_initial,
        )
        aggregate = self.signatures.setdefault(action.key, ActionEffectAggregate())
        aggregate.uses += 1
        aggregate.reward_total += float(transition.reward_delta)
        aggregate.transform_kinds[signature.transform_kind] += 1
        aggregate.spatial_scopes[signature.spatial_scope] += 1
        aggregate.primary_regions[signature.primary_region] += 1
        aggregate.interaction_hints[signature.interaction_hint] += 1
        if signature.reversible:
            aggregate.reversible_uses += 1
        if signature.progress_like:
            aggregate.progress_uses += 1
        for role in signature.semantic_roles:
            aggregate.semantic_roles[role] += 1
        return signature

    def summary_for(self, action_key: str) -> ActionEffectAggregate | None:
        return self.signatures.get(action_key)

    def _infer_signature(
        self,
        transition: Transition,
        notes: dict[str, Any],
        progress_score: float,
        returned_previous: bool,
        returned_initial: bool,
    ) -> ActionEffectSignature:
        changed_cells = int(notes.get("changed_cells", 0) or 0)
        changed_playfield_cells = int(notes.get("changed_playfield_cells", 0) or 0)
        nonzero_delta = int(notes.get("nonzero_delta", 0) or 0)
        unique_color_delta = int(notes.get("unique_color_delta", 0) or 0)
        interaction_hint = str(notes.get("interaction_hint", "unknown"))
        region_bias = str(notes.get("region_bias", "unknown"))
        anchor_changes = list(notes.get("anchor_patch_changes", []) or [])
        collectible_progress = bool(notes.get("collectible_progress", False))
        semantic_player_moved = bool(notes.get("semantic_player_moved", False))

        if transition.terminal and not transition.won:
            transform_kind = "terminal_loss"
        elif semantic_player_moved:
            transform_kind = "movement"
        elif collectible_progress or nonzero_delta < 0:
            transform_kind = "collect_or_erase"
        elif anchor_changes or unique_color_delta != 0:
            transform_kind = "repaint_or_mode_change"
        elif interaction_hint in {"spawn_or_unlock", "board_or_room_transform"}:
            transform_kind = "toggle_or_unlock"
        elif changed_cells == 0:
            transform_kind = "noop"
        elif changed_playfield_cells >= 16:
            transform_kind = "large_transform"
        else:
            transform_kind = "local_transform"

        if changed_cells == 0:
            spatial_scope = "none"
        elif changed_cells <= 4:
            spatial_scope = "point"
        elif changed_cells <= 16:
            spatial_scope = "local"
        elif changed_cells <= 48:
            spatial_scope = "regional"
        else:
            spatial_scope = "global"

        semantic_region_roles = notes.get("semantic_region_roles", {}) or {}
        roles = tuple(
            sorted(role for role, items in semantic_region_roles.items() if isinstance(items, list) and items)
        )
        reversible = returned_previous or returned_initial
        return ActionEffectSignature(
            transform_kind=transform_kind,
            spatial_scope=spatial_scope,
            primary_region=region_bias,
            interaction_hint=interaction_hint,
            semantic_roles=roles,
            reversible=reversible,
            progress_like=progress_score >= 0.45 or transition.reward_delta > 0 or transition.won,
            reward_like=transition.reward_delta > 0 or transition.won,
        )
