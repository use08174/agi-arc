from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from arc_agi3.abstraction.scene_blueprint import SceneBlueprint


class GoalKind(str, Enum):
    MATCH_SHAPE = "match_shape"
    PAINT_REFERENCE = "paint_reference"
    MOVE_TO_MARKER = "move_to_marker"
    CLEAR_OBSTACLE = "clear_obstacle"
    ALIGN_MARKERS = "align_markers"
    TRANSFORM_OBJECT = "transform_object"
    EXPLORE_CAUSAL_EFFECTS = "explore_causal_effects"


@dataclass(slots=True)
class GoalHypothesis:
    kind: GoalKind
    summary: str
    confidence: float
    targets: tuple[tuple[int, int], ...] = ()
    evidence: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.confidence + min(0.2, 0.03 * len(self.evidence))


class GoalInferencer:
    """Infer human-like goal candidates from scene blueprint and action signature."""

    def infer(self, blueprint: SceneBlueprint) -> list[GoalHypothesis]:
        notes = blueprint.notes
        control = str(notes.get("runtime_control_family", "unknown"))
        goals: list[GoalHypothesis] = []
        relation_kinds = {relation.kind for relation in blueprint.relations}
        targets = blueprint.goal_targets[:16]

        if blueprint.repeated_shape_groups:
            goals.append(
                GoalHypothesis(
                    kind=GoalKind.MATCH_SHAPE,
                    summary="similar shapes may need to match position, color, or orientation",
                    confidence=0.55,
                    targets=targets,
                    evidence=["repeated object shapes in scene"],
                )
            )
            if control in {"coordinate_only", "mixed_move_coordinate", "coordinate_mode"}:
                goals.append(
                    GoalHypothesis(
                        kind=GoalKind.PAINT_REFERENCE,
                        summary="coordinate actions may copy a reference pattern into a workspace",
                        confidence=0.58,
                        targets=targets,
                        evidence=["repeated shapes plus coordinate control"],
                    )
                )

        if "same_color_group" in relation_kinds or "near_same_color" in relation_kinds:
            goals.append(
                GoalHypothesis(
                    kind=GoalKind.ALIGN_MARKERS,
                    summary="same-color objects may be markers that should overlap or align",
                    confidence=0.5,
                    targets=targets,
                    evidence=["same-color relation groups"],
                )
            )

        if control in {"movement_only", "movement_mode", "mixed_move_coordinate"}:
            goals.append(
                GoalHypothesis(
                    kind=GoalKind.MOVE_TO_MARKER,
                    summary="movement controls may need to move an actor/object toward salient markers",
                    confidence=0.48,
                    targets=targets,
                    evidence=["movement-like action signature"],
                )
            )
            if blueprint.large_objects and blueprint.compact_objects:
                goals.append(
                    GoalHypothesis(
                        kind=GoalKind.CLEAR_OBSTACLE,
                        summary="large regions and compact objects suggest obstacles or movable blockers",
                        confidence=0.42,
                        targets=targets,
                        evidence=["large regions coexist with compact objects"],
                    )
                )

        if control in {"movement_mode", "mixed_move_coordinate"} and blueprint.relations:
            goals.append(
                GoalHypothesis(
                    kind=GoalKind.TRANSFORM_OBJECT,
                    summary="simple or mode actions may rotate/transform active objects before placement",
                    confidence=0.44,
                    targets=targets,
                    evidence=["relations plus simple/mode controls"],
                )
            )

        if not goals:
            goals.append(
                GoalHypothesis(
                    kind=GoalKind.EXPLORE_CAUSAL_EFFECTS,
                    summary="unknown goal; run short causal probes and compare object deltas",
                    confidence=0.35,
                    targets=targets,
                    evidence=["insufficient scene relations"],
                )
            )

        goals.sort(key=lambda goal: goal.score, reverse=True)
        return goals[:6]


def goals_to_notes(goals: list[GoalHypothesis]) -> dict[str, object]:
    merged_targets: list[tuple[int, int]] = []
    for goal in goals:
        for target in goal.targets:
            if target not in merged_targets:
                merged_targets.append(target)
    return {
        "scene_goal_kinds": [goal.kind.value for goal in goals],
        "scene_goal_summaries": [goal.summary for goal in goals],
        "scene_goal_confidences": [round(goal.confidence, 3) for goal in goals],
        "scene_goal_targets": merged_targets[:20],
    }
