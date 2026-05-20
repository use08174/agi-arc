from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RelationSnapshot:
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    focus_relations: list[str]
    summary: dict[str, float | int | str]


class ObjectRelationModel:
    """Build reusable object-to-object relation abstractions.

    The goal is not to solve a specific game, but to expose generic structure:
    overlap, alignment, containment, relative distance, and small-marker pairing.
    """

    def infer(
        self,
        *,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        items: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        buttons: list[dict[str, Any]],
        displays: list[dict[str, Any]],
    ) -> RelationSnapshot:
        entities = self._select_entities(
            objects=objects,
            player=player,
            items=items,
            goals=goals,
            buttons=buttons,
            displays=displays,
        )
        relations: list[dict[str, Any]] = []
        best_overlap_ratio = 0.0
        best_containment_score = 0.0
        best_alignment_score = 0.0
        nearest_same_color_distance: int | None = None
        nearest_same_shape_distance: int | None = None
        nearest_marker_distance: int | None = None
        focus_relations: list[str] = []

        for idx, left in enumerate(entities):
            for right in entities[idx + 1 :]:
                relation = self._pair_relation(left, right)
                relations.append(relation)
                overlap_ratio = float(relation["overlap_ratio"])
                containment_score = float(relation["containment_score"])
                alignment_score = float(relation["alignment_score"])
                distance = int(relation["distance"])
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                if containment_score > best_containment_score:
                    best_containment_score = containment_score
                if alignment_score > best_alignment_score:
                    best_alignment_score = alignment_score
                if bool(relation["same_color"]):
                    nearest_same_color_distance = self._min_or(distance, nearest_same_color_distance)
                if bool(relation["same_shape"]):
                    nearest_same_shape_distance = self._min_or(distance, nearest_same_shape_distance)
                if bool(relation["marker_pair"]):
                    nearest_marker_distance = self._min_or(distance, nearest_marker_distance)
                if (
                    overlap_ratio >= 0.15
                    or containment_score >= 0.5
                    or alignment_score >= 0.7
                    or bool(relation["marker_pair"])
                ):
                    focus_relations.append(self._render_focus_relation(relation))

        focus_relations = focus_relations[:10]
        summary: dict[str, float | int | str] = {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "best_overlap_ratio": round(best_overlap_ratio, 4),
            "best_containment_score": round(best_containment_score, 4),
            "best_alignment_score": round(best_alignment_score, 4),
            "nearest_same_color_distance": float(nearest_same_color_distance if nearest_same_color_distance is not None else -1),
            "nearest_same_shape_distance": float(nearest_same_shape_distance if nearest_same_shape_distance is not None else -1),
            "nearest_marker_distance": float(nearest_marker_distance if nearest_marker_distance is not None else -1),
            "marker_pair_count": sum(1 for relation in relations if bool(relation["marker_pair"])),
        }
        return RelationSnapshot(
            entities=entities,
            relations=relations[:24],
            focus_relations=focus_relations,
            summary=summary,
        )

    def _select_entities(
        self,
        *,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        items: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        buttons: list[dict[str, Any]],
        displays: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def add(entity: dict[str, Any], forced_role: str | None = None) -> None:
            entity_id = str(entity.get("id", ""))
            if not entity_id or entity_id in seen_ids:
                return
            bbox = entity.get("bbox")
            center = entity.get("center")
            if not isinstance(bbox, dict) or not isinstance(center, tuple):
                return
            role = forced_role or str(entity.get("role", "unknown"))
            area = int(entity.get("area", 0) or 0)
            marker_like = area <= 9
            chosen.append(
                {
                    "id": entity_id,
                    "role": role,
                    "color": int(entity.get("color", 0) or 0),
                    "shape_signature": str(entity.get("shape_signature", "")),
                    "bbox": bbox,
                    "center": center,
                    "area": area,
                    "marker_like": marker_like,
                    "anchor": str(entity.get("anchor", "")),
                }
            )
            seen_ids.add(entity_id)

        if player is not None:
            add(player, forced_role="player")
        for group in (goals, items, buttons, displays, objects):
            for entity in group:
                add(entity)
        chosen.sort(
            key=lambda entity: (
                0 if entity["role"] in {"player", "goal", "item", "button"} else 1,
                entity["area"],
                entity["bbox"]["min_y"],
                entity["bbox"]["min_x"],
            )
        )
        return chosen[:12]

    def _pair_relation(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        left_bbox = left["bbox"]
        right_bbox = right["bbox"]
        left_center = left["center"]
        right_center = right["center"]
        dx = int(right_center[0]) - int(left_center[0])
        dy = int(right_center[1]) - int(left_center[1])
        distance = abs(dx) + abs(dy)
        overlap_area = self._overlap_area(left_bbox, right_bbox)
        min_area = max(1, min(int(left["area"]), int(right["area"])))
        overlap_ratio = overlap_area / min_area
        containment_score = self._containment_score(left_bbox, right_bbox)
        alignment_score = self._alignment_score(left_bbox, right_bbox, left_center, right_center)
        relation_kind = self._relation_kind(overlap_ratio, containment_score, alignment_score, dx, dy)
        marker_pair = bool(left["marker_like"] and right["marker_like"] and left["color"] == right["color"])
        return {
            "pair": f"{left['id']}->{right['id']}",
            "left_id": left["id"],
            "right_id": right["id"],
            "left_role": left["role"],
            "right_role": right["role"],
            "left_color": left["color"],
            "right_color": right["color"],
            "same_color": left["color"] == right["color"],
            "same_shape": left["shape_signature"] == right["shape_signature"],
            "distance": distance,
            "dx": dx,
            "dy": dy,
            "direction": self._direction(dx, dy),
            "overlap_ratio": round(overlap_ratio, 4),
            "containment_score": round(containment_score, 4),
            "alignment_score": round(alignment_score, 4),
            "marker_pair": marker_pair,
            "relation_kind": relation_kind,
        }

    def _render_focus_relation(self, relation: dict[str, Any]) -> str:
        return (
            f"{relation['left_role']}->{relation['right_role']} "
            f"kind={relation['relation_kind']} dist={relation['distance']} "
            f"overlap={float(relation['overlap_ratio']):.2f} "
            f"align={float(relation['alignment_score']):.2f}"
        )

    def _relation_kind(
        self,
        overlap_ratio: float,
        containment_score: float,
        alignment_score: float,
        dx: int,
        dy: int,
    ) -> str:
        if containment_score >= 0.8:
            return "contains"
        if overlap_ratio >= 0.4:
            return "overlap"
        if alignment_score >= 0.8:
            return "aligned"
        if abs(dx) > abs(dy):
            return "horizontal_offset"
        if abs(dy) > abs(dx):
            return "vertical_offset"
        return "diagonal_offset"

    def _overlap_area(self, left_bbox: dict[str, Any], right_bbox: dict[str, Any]) -> int:
        min_x = max(int(left_bbox["min_x"]), int(right_bbox["min_x"]))
        min_y = max(int(left_bbox["min_y"]), int(right_bbox["min_y"]))
        max_x = min(int(left_bbox["max_x"]), int(right_bbox["max_x"]))
        max_y = min(int(left_bbox["max_y"]), int(right_bbox["max_y"]))
        if min_x > max_x or min_y > max_y:
            return 0
        return (max_x - min_x + 1) * (max_y - min_y + 1)

    def _containment_score(self, left_bbox: dict[str, Any], right_bbox: dict[str, Any]) -> float:
        def contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
            return (
                int(outer["min_x"]) <= int(inner["min_x"])
                and int(outer["min_y"]) <= int(inner["min_y"])
                and int(outer["max_x"]) >= int(inner["max_x"])
                and int(outer["max_y"]) >= int(inner["max_y"])
            )

        if contains(left_bbox, right_bbox) or contains(right_bbox, left_bbox):
            return 1.0
        outer_margin_x = min(
            abs(int(left_bbox["min_x"]) - int(right_bbox["min_x"])),
            abs(int(left_bbox["max_x"]) - int(right_bbox["max_x"])),
        )
        outer_margin_y = min(
            abs(int(left_bbox["min_y"]) - int(right_bbox["min_y"])),
            abs(int(left_bbox["max_y"]) - int(right_bbox["max_y"])),
        )
        return max(0.0, 1.0 - (outer_margin_x + outer_margin_y) / 12.0)

    def _alignment_score(
        self,
        left_bbox: dict[str, Any],
        right_bbox: dict[str, Any],
        left_center: tuple[int, int],
        right_center: tuple[int, int],
    ) -> float:
        same_x = abs(int(left_center[0]) - int(right_center[0]))
        same_y = abs(int(left_center[1]) - int(right_center[1]))
        width_gap = abs(int(left_bbox["width"]) - int(right_bbox["width"]))
        height_gap = abs(int(left_bbox["height"]) - int(right_bbox["height"]))
        center_term = max(0.0, 1.0 - min(same_x, same_y) / 12.0)
        shape_term = max(0.0, 1.0 - (width_gap + height_gap) / 10.0)
        return 0.65 * center_term + 0.35 * shape_term

    def _direction(self, dx: int, dy: int) -> str:
        if dx == 0 and dy == 0:
            return "same_center"
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        if abs(dy) > abs(dx):
            return "down" if dy > 0 else "up"
        if dx > 0 and dy > 0:
            return "down_right"
        if dx > 0 and dy < 0:
            return "up_right"
        if dx < 0 and dy > 0:
            return "down_left"
        return "up_left"

    def _min_or(self, value: int, current: int | None) -> int:
        if current is None:
            return value
        return min(value, current)
