from __future__ import annotations

from arc_agi3.core.types import Action, ExperimentProposal, PlanStep
from arc_agi3.memory.experiments import ExperimentSession
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.planning.path_planner import PathPlanner


class ExperimentRunner:
    """Turn an active experiment into the next executable action."""

    def __init__(self) -> None:
        self.path_planner = PathPlanner()

    def build_step_from_session(
        self,
        session: ExperimentSession | None,
        actions: list[Action],
        game_memory: GameMemory,
    ) -> PlanStep | None:
        if session is None:
            return None
        experiment = session.proposal
        if experiment.kind == "probe_action_pair" and isinstance(experiment.target, dict):
            first_key = str(experiment.target.get("first_action_key", ""))
            second_key = str(experiment.target.get("second_action_key", ""))
            target_key = first_key if session.step_count % 2 == 0 else second_key
            for action in actions:
                if action.key == target_key:
                    return PlanStep(
                        action=action,
                        reason=(
                            f"executing experiment {experiment.key} "
                            f"step={'first' if target_key == first_key else 'second'}"
                        ),
                    )
            return None
        return self.build_step(experiment, actions, game_memory)

    def build_step(
        self,
        experiment: ExperimentProposal | None,
        actions: list[Action],
        game_memory: GameMemory,
    ) -> PlanStep | None:
        if experiment is None:
            return None
        world = game_memory.world_model
        if experiment.kind in {"collect_item", "activate_button", "go_to_goal"} and isinstance(experiment.target, tuple):
            direct = self._direct_action_for_target(experiment.target, actions)
            if direct is not None and experiment.kind in {"collect_item", "activate_button"}:
                return PlanStep(action=direct, reason=f"executing experiment {experiment.key} via direct interaction")
            path = self.path_planner.plan_to_targets(world, actions, {experiment.target})
            if path:
                return PlanStep(action=path[0], reason=f"executing experiment {experiment.key}")
            return None
        if experiment.kind == "inspect_relation" and isinstance(experiment.target, dict):
            pair = tuple(experiment.target.get("nearest_pair", ()))
            relation_targets = {cell for cell in pair if isinstance(cell, tuple) and len(cell) == 2}
            path = self.path_planner.plan_to_targets(world, actions, relation_targets)
            if path:
                return PlanStep(action=path[0], reason=f"executing experiment {experiment.key}")
            return None
        if experiment.kind == "discover_axis" and isinstance(experiment.target, dict):
            action_key = experiment.target.get("action_key")
            for action in actions:
                if action.key == action_key:
                    return PlanStep(action=action, reason=f"executing experiment {experiment.key}")
            return None
        if experiment.kind == "inspect_affordance" and isinstance(experiment.target, dict):
            center = experiment.target.get("center")
            if isinstance(center, tuple):
                direct = self._direct_action_for_target(center, actions)
                if direct is not None:
                    return PlanStep(action=direct, reason=f"executing experiment {experiment.key} via direct interaction")
                path = self.path_planner.plan_to_targets(world, actions, {center})
                if path:
                    return PlanStep(action=path[0], reason=f"executing experiment {experiment.key}")
            return None
        if experiment.kind == "probe_action" and isinstance(experiment.target, str):
            for action in actions:
                if action.key == experiment.target:
                    return PlanStep(action=action, reason=f"executing experiment {experiment.key}")
            return None
        return None

    def _direct_action_for_target(self, target: tuple[int, int], actions: list[Action]) -> Action | None:
        tx, ty = target
        for action in actions:
            if action.name != "ACTION6" or not action.payload:
                continue
            if int(action.payload.get("x", -999)) == tx and int(action.payload.get("y", -999)) == ty:
                return action
        return None
