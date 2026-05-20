from __future__ import annotations

from typing import Any


class RegionRoleInferer:
    """Infer coarse region roles that generalize across game types."""

    def infer(
        self,
        *,
        width: int,
        height: int,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        walls: list[dict[str, Any]],
        items: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        buttons: list[dict[str, Any]],
        displays: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        player_id = player.get("id") if player else None
        wall_ids = {obj.get("id") for obj in walls}
        display_ids = {obj.get("id") for obj in displays}
        button_ids = {obj.get("id") for obj in buttons}
        item_ids = {obj.get("id") for obj in items}
        goal_ids = {obj.get("id") for obj in goals}

        reference_like: list[dict[str, Any]] = []
        workspace_like: list[dict[str, Any]] = []
        control_like: list[dict[str, Any]] = []
        target_like: list[dict[str, Any]] = []

        candidate_workspace = []
        for obj in objects:
            if obj.get("id") == player_id:
                continue
            bbox = obj.get("bbox", {})
            area = int(obj.get("area", 0) or 0)
            anchor = str(obj.get("anchor", "middle_center"))
            role = str(obj.get("role", "unknown"))
            touches_edge = bool(obj.get("touches_edge", False))
            item = self._summarize(obj)
            if obj.get("id") in display_ids or obj.get("id") in button_ids or anchor.startswith("bottom_"):
                control_like.append(item)
            if obj.get("id") in item_ids or obj.get("id") in goal_ids:
                target_like.append(item)
            if (
                anchor in {"top_left", "top_right"}
                and area <= max(24, max(1, width * height) // 12)
                and role != "wall"
            ):
                reference_like.append(item)
            if (
                not touches_edge
                and area >= max(9, max(1, width * height) // 24)
                and anchor not in {"top_left", "top_right", "bottom_left", "bottom_right"}
                and (
                    obj.get("id") not in wall_ids
                    or (bbox.get("width", 0) <= max(1, width // 2) and bbox.get("height", 0) <= max(1, height // 2))
                )
            ):
                candidate_workspace.append((area, item))

        candidate_workspace.sort(key=lambda item: (-item[0], item[1]["anchor"], item[1]["bbox"]["min_y"]))
        workspace_like = [item for _, item in candidate_workspace[:3]]

        if not reference_like:
            # Fallback: any compact stable-looking top structure can serve as a reference candidate.
            top_objects = [
                self._summarize(obj)
                for obj in objects
                if str(obj.get("anchor", "")).startswith("top_") and int(obj.get("area", 0) or 0) <= 20
            ]
            reference_like = top_objects[:3]

        return {
            "reference_like": reference_like,
            "workspace_like": workspace_like,
            "control_like": control_like[:4],
            "target_like": target_like[:6],
        }

    def _summarize(self, obj: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": obj.get("id"),
            "bbox": obj.get("bbox"),
            "anchor": obj.get("anchor"),
            "color": int(obj.get("color", 0) or 0),
            "area": int(obj.get("area", 0) or 0),
            "role": str(obj.get("role", "unknown")),
        }
