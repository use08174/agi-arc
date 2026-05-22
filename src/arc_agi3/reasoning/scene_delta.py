from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from arc_agi3.abstraction.scene_blueprint import SceneBlueprintBuilder
from arc_agi3.abstraction.state_lifter import LiftedObject, StateLifter
from arc_agi3.core.types import Action, Observation


@dataclass(slots=True)
class ObjectDelta:
    kind: str
    before_id: str | None = None
    after_id: str | None = None
    distance: int = 0
    evidence: str = ""


@dataclass(slots=True)
class SceneDelta:
    object_deltas: list[ObjectDelta] = field(default_factory=list)
    moved_count: int = 0
    appeared_count: int = 0
    disappeared_count: int = 0
    color_changed_count: int = 0
    transformed_count: int = 0
    clicked_near_object: bool = False

    @property
    def primary_kind(self) -> str:
        if self.transformed_count:
            return "object_transform"
        if self.color_changed_count:
            return "color_edit"
        if self.moved_count:
            return "object_motion"
        if self.disappeared_count:
            return "object_removed"
        if self.appeared_count:
            return "object_appeared"
        return "no_object_delta"

    def to_notes(self) -> dict[str, object]:
        return {
            "scene_delta_kind": self.primary_kind,
            "scene_delta_moved_count": self.moved_count,
            "scene_delta_appeared_count": self.appeared_count,
            "scene_delta_disappeared_count": self.disappeared_count,
            "scene_delta_color_changed_count": self.color_changed_count,
            "scene_delta_transformed_count": self.transformed_count,
            "scene_delta_clicked_near_object": self.clicked_near_object,
            "scene_delta_summary": [delta.evidence for delta in self.object_deltas[:8]],
        }


class SceneDeltaInterpreter:
    """Compare consecutive scene blueprints at object/relation granularity."""

    def __init__(self) -> None:
        self.lifter = StateLifter()
        self.blueprints = SceneBlueprintBuilder()

    def interpret(self, before: Observation, after: Observation, action: Action) -> SceneDelta:
        before_blueprint = self.blueprints.build(self.lifter.lift(before))
        after_blueprint = self.blueprints.build(self.lifter.lift(after))
        before_objects = list(before_blueprint.objects)
        after_objects = list(after_blueprint.objects)
        matched_after: set[str] = set()
        deltas: list[ObjectDelta] = []

        for before_obj in before_objects[:48]:
            match = self._best_match(before_obj, after_objects, matched_after)
            if match is None:
                deltas.append(
                    ObjectDelta(
                        kind="disappeared",
                        before_id=before_obj.object_id,
                        evidence=f"{before_obj.object_id} disappeared",
                    )
                )
                continue
            after_obj, distance = match
            matched_after.add(after_obj.object_id)
            if before_obj.center != after_obj.center:
                deltas.append(
                    ObjectDelta(
                        kind="moved",
                        before_id=before_obj.object_id,
                        after_id=after_obj.object_id,
                        distance=distance,
                        evidence=f"{before_obj.object_id} moved by {distance}",
                    )
                )
            if before_obj.color != after_obj.color:
                deltas.append(
                    ObjectDelta(
                        kind="color_changed",
                        before_id=before_obj.object_id,
                        after_id=after_obj.object_id,
                        evidence=f"{before_obj.object_id} color {before_obj.color}->{after_obj.color}",
                    )
                )
            if self._shape_signature(before_obj) != self._shape_signature(after_obj):
                deltas.append(
                    ObjectDelta(
                        kind="transformed",
                        before_id=before_obj.object_id,
                        after_id=after_obj.object_id,
                        evidence=f"{before_obj.object_id} changed shape",
                    )
                )

        for after_obj in after_objects[:48]:
            if after_obj.object_id in matched_after:
                continue
            if self._nearest_distance(after_obj, before_objects) > 1:
                deltas.append(
                    ObjectDelta(
                        kind="appeared",
                        after_id=after_obj.object_id,
                        evidence=f"{after_obj.object_id} appeared",
                    )
                )

        delta = SceneDelta(object_deltas=deltas)
        delta.moved_count = sum(1 for item in deltas if item.kind == "moved")
        delta.appeared_count = sum(1 for item in deltas if item.kind == "appeared")
        delta.disappeared_count = sum(1 for item in deltas if item.kind == "disappeared")
        delta.color_changed_count = sum(1 for item in deltas if item.kind == "color_changed")
        delta.transformed_count = sum(1 for item in deltas if item.kind == "transformed")
        delta.clicked_near_object = self._clicked_near_object(action, before_objects)
        return delta

    def _best_match(
        self,
        before_obj: LiftedObject,
        after_objects: list[LiftedObject],
        matched_after: set[str],
    ) -> tuple[LiftedObject, int] | None:
        before_shape = self._shape_signature(before_obj)
        candidates: list[tuple[int, int, int, LiftedObject]] = []
        for after_obj in after_objects:
            if after_obj.object_id in matched_after:
                continue
            distance = self._distance(before_obj, after_obj)
            same_shape = self._shape_signature(after_obj) == before_shape
            same_color = after_obj.color == before_obj.color
            if not same_shape and not same_color:
                continue
            area_delta = abs(after_obj.area - before_obj.area)
            candidates.append((0 if same_shape else 1, distance, area_delta, after_obj))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:3])
        if candidates[0][1] > max(8, int(before_obj.area ** 0.5) + 4):
            return None
        return candidates[0][3], candidates[0][1]

    def _clicked_near_object(self, action: Action, objects: list[LiftedObject]) -> bool:
        if not action.payload:
            return False
        x = int(action.payload.get("x", -999))
        y = int(action.payload.get("y", -999))
        for obj in objects[:32]:
            min_x, min_y, max_x, max_y = obj.bbox
            if min_x - 1 <= x <= max_x + 1 and min_y - 1 <= y <= max_y + 1:
                return True
        return False

    def _nearest_distance(self, obj: LiftedObject, others: list[LiftedObject]) -> int:
        if not others:
            return 999
        return min(self._distance(obj, other) for other in others)

    def _distance(self, left: LiftedObject, right: LiftedObject) -> int:
        return abs(left.center[0] - right.center[0]) + abs(left.center[1] - right.center[1])

    def _shape_signature(self, obj: LiftedObject) -> str:
        min_x, min_y, _, _ = obj.bbox
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in obj.cells))
        return hashlib.sha1(repr((obj.shape, normalized)).encode("utf-8")).hexdigest()[:12]


class GoalProgressScorer:
    """Translate object deltas into goal-conditioned progress notes."""

    def score(
        self,
        *,
        before_notes: dict[str, Any],
        after_notes: dict[str, Any],
        delta: SceneDelta,
    ) -> dict[str, object]:
        click_role = str(after_notes.get("coordinate_click_role", ""))
        if click_role in {"top_band", "tool_or_palette", "reference"}:
            return {
                "scene_goal_progress_score": 0.0,
                "scene_goal_progress_reasons": ["top_band_click_not_goal_progress"],
            }
        goals = set(str(goal) for goal in before_notes.get("scene_goal_kinds", []))
        reasons: list[str] = []
        score = 0.0
        if {"paint_reference"} & goals and (delta.color_changed_count or delta.appeared_count):
            score += 0.25
            reasons.append("paint_goal_color_or_spawn_delta")
        if {"move_to_marker", "clear_obstacle"} & goals and delta.moved_count:
            score += 0.25
            reasons.append("movement_goal_object_motion")
        if "clear_obstacle" in goals and delta.disappeared_count:
            score += 0.30
            reasons.append("clear_goal_object_removed")
        if {"match_shape", "align_markers"} & goals:
            relation_delta = float(after_notes.get("relation_best_alignment_delta", 0.0) or 0.0)
            overlap_delta = float(after_notes.get("relation_best_overlap_delta", 0.0) or 0.0)
            same_color_improvement = float(after_notes.get("relation_nearest_same_color_improvement", 0.0) or 0.0)
            if relation_delta > 0 or overlap_delta > 0 or same_color_improvement > 0:
                score += min(0.35, 0.12 + relation_delta + overlap_delta + same_color_improvement / 24.0)
                reasons.append("relational_goal_improved")
        if "transform_object" in goals and delta.transformed_count:
            score += 0.25
            reasons.append("transform_goal_shape_delta")
        if delta.clicked_near_object and delta.primary_kind != "no_object_delta":
            score += 0.08
            reasons.append("object_directed_probe_changed_scene")
        if not goals and delta.primary_kind != "no_object_delta":
            score += 0.12
            reasons.append("unknown_goal_object_delta")
        return {
            "scene_goal_progress_score": round(min(1.0, score), 4),
            "scene_goal_progress_reasons": reasons,
        }
