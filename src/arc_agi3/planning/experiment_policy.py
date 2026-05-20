from __future__ import annotations

from typing import Any

from arc_agi3.core.types import ExperimentProposal
from arc_agi3.memory.world_model import WorldModel


class ExperimentPolicy:
    """Rank short-horizon experiments before fallback planning."""

    def select(
        self,
        proposals: list[ExperimentProposal],
        world: WorldModel,
        force_exploration: bool,
        mode_action_keys: set[str] | None = None,
        rule_priority_for: Any | None = None,
    ) -> ExperimentProposal | None:
        if not proposals:
            return None
        mode_action_keys = mode_action_keys or set()
        known_axes = {
            axis
            for dx, dy in world.action_move_vectors.values()
            for axis in (
                "horizontal" if dx != 0 else "",
                "vertical" if dy != 0 else "",
            )
            if axis
        }

        def priority(proposal: ExperimentProposal) -> tuple:
            kind = proposal.kind
            rule_bonus = float(rule_priority_for(proposal) if rule_priority_for is not None else 0.0)
            if kind == "discover_axis":
                axis = str((proposal.target or {}).get("axis", ""))
                return (0, axis in known_axes, -rule_bonus, proposal.key)
            if kind == "probe_action":
                target_key = str(proposal.target)
                return (1 if target_key in mode_action_keys else 2 if force_exploration else 3, -rule_bonus, proposal.key)
            if kind == "probe_action_pair":
                first_key = str((proposal.target or {}).get("first_action_key", ""))
                return (2 if first_key in mode_action_keys else 3 if force_exploration else 4, -rule_bonus, proposal.key)
            if kind == "inspect_affordance":
                return (5, -rule_bonus, -float(proposal.confidence or 0.0), proposal.key)
            if kind in {"collect_item", "activate_button", "go_to_goal"}:
                return (6, -rule_bonus, proposal.key)
            if kind == "inspect_relation":
                return (7, -rule_bonus, proposal.key)
            return (8, -rule_bonus, proposal.key)

        return sorted(proposals, key=priority)[0]
