from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

from arc_agi3.abstraction.state_lifter import LiftedObject, LiftedState


@dataclass(frozen=True, slots=True)
class SceneRelation:
    kind: str
    object_ids: tuple[str, ...]
    score: float
    evidence: str


@dataclass(frozen=True, slots=True)
class SceneBlueprint:
    objects: tuple[LiftedObject, ...]
    compact_objects: tuple[LiftedObject, ...]
    large_objects: tuple[LiftedObject, ...]
    repeated_shape_groups: tuple[tuple[LiftedObject, ...], ...]
    same_color_groups: tuple[tuple[LiftedObject, ...], ...]
    relations: tuple[SceneRelation, ...]
    goal_targets: tuple[tuple[int, int], ...]
    summary: tuple[str, ...]
    notes: dict[str, object] = field(default_factory=dict)

    def to_notes(self) -> dict[str, object]:
        return {
            "scene_object_count": len(self.objects),
            "scene_compact_object_count": len(self.compact_objects),
            "scene_large_object_count": len(self.large_objects),
            "scene_repeated_shape_group_count": len(self.repeated_shape_groups),
            "scene_same_color_group_count": len(self.same_color_groups),
            "scene_relations": [relation.kind for relation in self.relations[:12]],
            "scene_relation_evidence": [relation.evidence for relation in self.relations[:6]],
            "scene_goal_targets": list(self.goal_targets[:16]),
            "scene_summary": list(self.summary),
        }


class SceneBlueprintBuilder:
    """Build a compact object/relation blueprint from a lifted grid scene."""

    def build(self, state: LiftedState) -> SceneBlueprint:
        objects = state.objects
        compact_objects = tuple(
            obj for obj in objects if obj.area <= 64 and not self._touches_edge(obj, state.width, state.height)
        )
        large_objects = tuple(obj for obj in objects if obj.area > 64)
        shape_groups = self._groups_by_shape(objects)
        color_groups = self._groups_by_color(objects)
        relations = self._relations(objects, shape_groups, color_groups)
        goal_targets = self._goal_targets(state, relations, compact_objects)
        summary = self._summary(state, compact_objects, large_objects, shape_groups, relations)
        return SceneBlueprint(
            objects=objects,
            compact_objects=compact_objects,
            large_objects=large_objects,
            repeated_shape_groups=shape_groups,
            same_color_groups=color_groups,
            relations=relations,
            goal_targets=goal_targets,
            summary=summary,
            notes=dict(state.notes),
        )

    def _groups_by_shape(self, objects: tuple[LiftedObject, ...]) -> tuple[tuple[LiftedObject, ...], ...]:
        groups: dict[str, list[LiftedObject]] = {}
        for obj in objects:
            groups.setdefault(self._shape_signature(obj), []).append(obj)
        repeated = [tuple(sorted(group, key=lambda item: (item.center, item.color))) for group in groups.values() if len(group) >= 2]
        repeated.sort(key=lambda group: (-len(group), -sum(obj.area for obj in group), group[0].object_id))
        return tuple(repeated[:8])

    def _groups_by_color(self, objects: tuple[LiftedObject, ...]) -> tuple[tuple[LiftedObject, ...], ...]:
        groups: dict[int, list[LiftedObject]] = {}
        for obj in objects:
            groups.setdefault(obj.color, []).append(obj)
        repeated = [tuple(sorted(group, key=lambda item: (item.center, item.area))) for group in groups.values() if len(group) >= 2]
        repeated.sort(key=lambda group: (-len(group), -sum(obj.area for obj in group), group[0].color))
        return tuple(repeated[:8])

    def _relations(
        self,
        objects: tuple[LiftedObject, ...],
        shape_groups: tuple[tuple[LiftedObject, ...], ...],
        color_groups: tuple[tuple[LiftedObject, ...], ...],
    ) -> tuple[SceneRelation, ...]:
        relations: list[SceneRelation] = []
        for group in shape_groups[:4]:
            ids = tuple(obj.object_id for obj in group[:4])
            colors = sorted({obj.color for obj in group})
            relations.append(
                SceneRelation(
                    kind="same_shape_group",
                    object_ids=ids,
                    score=0.7 + min(0.2, 0.04 * len(group)),
                    evidence=f"{len(group)} objects share a shape across colors={colors}",
                )
            )
        for group in color_groups[:4]:
            ids = tuple(obj.object_id for obj in group[:4])
            relations.append(
                SceneRelation(
                    kind="same_color_group",
                    object_ids=ids,
                    score=0.55 + min(0.2, 0.03 * len(group)),
                    evidence=f"{len(group)} objects share color={group[0].color}",
                )
            )
        for left, right in self._close_pairs(objects):
            if left.color == right.color:
                kind = "near_same_color"
                score = 0.55
            elif self._shape_signature(left) == self._shape_signature(right):
                kind = "near_same_shape"
                score = 0.65
            else:
                continue
            relations.append(
                SceneRelation(
                    kind=kind,
                    object_ids=(left.object_id, right.object_id),
                    score=score,
                    evidence=f"{left.object_id} and {right.object_id} are spatially close",
                )
            )
        relations.sort(key=lambda relation: relation.score, reverse=True)
        return tuple(relations[:16])

    def _goal_targets(
        self,
        state: LiftedState,
        relations: list[SceneRelation],
        compact_objects: tuple[LiftedObject, ...],
    ) -> tuple[tuple[int, int], ...]:
        targets: list[tuple[int, int]] = []
        relation_ids = {object_id for relation in relations[:8] for object_id in relation.object_ids}
        for obj in state.objects:
            if obj.object_id in relation_ids:
                targets.append(obj.center)
        for marker in state.markers[:8]:
            targets.append(marker.center)
        for obj in compact_objects[:8]:
            targets.append(obj.center)
        for xy in state.candidate_clicks[:12]:
            targets.append(xy)
        deduped: list[tuple[int, int]] = []
        for target in targets:
            if target not in deduped:
                deduped.append(target)
        return tuple(deduped[:24])

    def _summary(
        self,
        state: LiftedState,
        compact_objects: tuple[LiftedObject, ...],
        large_objects: tuple[LiftedObject, ...],
        shape_groups: tuple[tuple[LiftedObject, ...], ...],
        relations: tuple[SceneRelation, ...],
    ) -> tuple[str, ...]:
        colors = len([color for color in state.color_histogram if color != 0])
        summary = [
            f"{len(state.objects)} objects, {colors} non-background colors",
            f"{len(compact_objects)} compact objects, {len(large_objects)} large regions",
        ]
        if shape_groups:
            summary.append(f"{len(shape_groups)} repeated-shape groups")
        if relations:
            summary.append("relations=" + ",".join(relation.kind for relation in relations[:4]))
        return tuple(summary)

    def _close_pairs(self, objects: tuple[LiftedObject, ...]) -> list[tuple[LiftedObject, LiftedObject]]:
        pairs: list[tuple[int, LiftedObject, LiftedObject]] = []
        for idx, left in enumerate(objects[:24]):
            for right in objects[idx + 1 : 24]:
                distance = abs(left.center[0] - right.center[0]) + abs(left.center[1] - right.center[1])
                if distance <= max(3, int((left.area + right.area) ** 0.5) + 2):
                    pairs.append((distance, left, right))
        pairs.sort(key=lambda item: item[0])
        return [(left, right) for _, left, right in pairs[:12]]

    def _shape_signature(self, obj: LiftedObject) -> str:
        min_x, min_y, _, _ = obj.bbox
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in obj.cells))
        return hashlib.sha1(repr((obj.shape, normalized)).encode("utf-8")).hexdigest()[:12]

    def _touches_edge(self, obj: LiftedObject, width: int, height: int) -> bool:
        min_x, min_y, max_x, max_y = obj.bbox
        return min_x <= 0 or min_y <= 0 or max_x >= width - 1 or max_y >= height - 1
