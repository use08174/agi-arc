from __future__ import annotations

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
            if kind == "discover_axis":
                axis = str((proposal.target or {}).get("axis", ""))
                return (0, axis in known_axes, proposal.key)
            if kind == "probe_action":
                target_key = str(proposal.target)
                return (1 if target_key in mode_action_keys else 2 if force_exploration else 3, proposal.key)
            if kind == "probe_action_pair":
                first_key = str((proposal.target or {}).get("first_action_key", ""))
                return (2 if first_key in mode_action_keys else 3 if force_exploration else 4, proposal.key)
            if kind == "inspect_affordance":
                return (5, -float(proposal.confidence or 0.0), proposal.key)
            if kind in {"collect_item", "activate_button", "go_to_goal"}:
                return (6, proposal.key)
            if kind == "inspect_relation":
                return (7, proposal.key)
            return (8, proposal.key)

        return sorted(proposals, key=priority)[0]
