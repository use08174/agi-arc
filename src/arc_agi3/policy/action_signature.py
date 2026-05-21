from __future__ import annotations

from dataclasses import dataclass

from arc_agi3.core.types import Action, Observation


@dataclass(frozen=True, slots=True)
class RuntimeActionSignature:
    control_family: str
    grid_bucket: str
    recommended_strategy: str
    action_names: tuple[str, ...]
    has_coordinate: bool
    has_mode: bool
    has_movement: bool

    def to_notes(self) -> dict[str, object]:
        return {
            "runtime_control_family": self.control_family,
            "runtime_grid_bucket": self.grid_bucket,
            "runtime_recommended_strategy": self.recommended_strategy,
            "runtime_action_names": list(self.action_names),
            "runtime_has_coordinate": self.has_coordinate,
            "runtime_has_mode": self.has_mode,
            "runtime_has_movement": self.has_movement,
        }


def classify_runtime_signature(actions: list[Action], observation: Observation) -> RuntimeActionSignature:
    action_names = tuple(sorted({action.name for action in actions}))
    names = set(action_names)
    has_coordinate = "ACTION6" in names or any(action.payload for action in actions)
    has_movement = bool({"ACTION1", "ACTION2", "ACTION3", "ACTION4"} & names)
    has_mode = bool({"ACTION5", "ACTION7"} & names)
    if has_coordinate and not has_movement and not has_mode:
        control_family = "coordinate_only"
    elif has_coordinate and has_movement:
        control_family = "mixed_move_coordinate"
    elif has_coordinate and has_mode:
        control_family = "coordinate_mode"
    elif has_movement and has_mode:
        control_family = "movement_mode"
    elif has_movement:
        control_family = "movement_only"
    elif has_mode:
        control_family = "mode_only"
    else:
        control_family = "unknown"
    grid_bucket = _grid_bucket(observation)
    return RuntimeActionSignature(
        control_family=control_family,
        grid_bucket=grid_bucket,
        recommended_strategy=_recommended_strategy(control_family, grid_bucket),
        action_names=action_names,
        has_coordinate=has_coordinate,
        has_mode=has_mode,
        has_movement=has_movement,
    )


def _grid_bucket(observation: Observation) -> str:
    grid = observation.frame.grid
    if not grid:
        return "unknown"
    height = len(grid)
    width = len(grid[0]) if height else 0
    max_dim = max(width, height)
    if max_dim <= 12:
        return "small"
    if max_dim <= 32:
        return "medium"
    return "large"


def _recommended_strategy(control_family: str, grid_bucket: str) -> str:
    if control_family == "coordinate_only":
        return "object_coordinate_probe" if grid_bucket == "large" else "dense_coordinate_probe"
    if control_family in {"mixed_move_coordinate", "coordinate_mode"}:
        return "learn_simple_actions_then_object_clicks"
    if control_family == "movement_only":
        return "state_graph_bfs" if grid_bucket == "small" else "movement_axis_model"
    if control_family == "movement_mode":
        return "learn_mode_then_movement_model"
    if control_family == "mode_only":
        return "mode_cycle_probe"
    return "generic_refinement_probe"
