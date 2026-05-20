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
    ) -> ExperimentProposal | None:
        if not proposals:
            return None
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
                return (1 if force_exploration else 2, proposal.key)
            if kind == "inspect_affordance":
                return (3, -float(proposal.confidence or 0.0), proposal.key)
            if kind in {"collect_item", "activate_button", "go_to_goal"}:
                return (4, proposal.key)
            if kind == "inspect_relation":
                return (5, proposal.key)
            return (6, proposal.key)

        return sorted(proposals, key=priority)[0]
