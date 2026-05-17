from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from arc_agi3.core.types import Frame


@dataclass(slots=True)
class SemanticMap:
    width: int
    height: int
    hud_rows: int
    objects: list[dict[str, Any]]
    player: dict[str, Any] | None
    walls: list[dict[str, Any]]
    items: list[dict[str, Any]]
    goals: list[dict[str, Any]]
    buttons: list[dict[str, Any]]
    hazards: list[dict[str, Any]]
    ascii_map: str

    def to_notes(self) -> dict[str, Any]:
        return {
            "semantic_width": self.width,
            "semantic_height": self.height,
            "semantic_hud_rows": self.hud_rows,
            "semantic_objects": self.objects[:24],
            "semantic_player": self.player,
            "semantic_walls": self.walls[:16],
            "semantic_items": self.items[:16],
            "semantic_goals": self.goals[:8],
            "semantic_buttons": self.buttons[:8],
            "semantic_hazards": self.hazards[:8],
            "semantic_ascii_map": self.ascii_map,
        }


class MapParser:
    """A conservative object-centric parser for rendered ARC-AGI frames.

    It does not pretend to know the exact game. It creates useful hypotheses:
    large/border components are wall candidates, small isolated components are
    item/button candidates, and moving components between frames are player candidates.
    """

    def parse(self, frame: Frame, previous: Frame | None = None, hud_rows: int | None = None) -> SemanticMap:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        hud_rows = self._hud_rows(frame) if hud_rows is None else hud_rows
        hud_start = max(0, height - hud_rows)
        objects = self._extract_components(frame, hud_rows)
        player = self._infer_player(objects, frame, previous, hud_rows)
        walls, items, goals, buttons, hazards = self._classify(objects, player, width, hud_start)
        ascii_map = self._ascii(width, hud_start, objects, player, walls, items, goals, buttons, hazards)
        return SemanticMap(
            width=width,
            height=hud_start,
            hud_rows=hud_rows,
            objects=objects,
            player=player,
            walls=walls,
            items=items,
            goals=goals,
            buttons=buttons,
            hazards=hazards,
            ascii_map=ascii_map,
        )

    def _hud_rows(self, frame: Frame) -> int:
        height = len(frame.grid)
        if height < 4:
            return 0
        return max(2, min(8, height // 6))

    def _extract_components(self, frame: Frame, hud_rows: int) -> list[dict[str, Any]]:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        hud_start = max(0, height - hud_rows)
        seen: set[tuple[int, int]] = set()
        out: list[dict[str, Any]] = []
        for y in range(hud_start):
            for x in range(width):
                color = int(grid[y][x])
                if color == 0 or (x, y) in seen:
                    continue
                stack = [(x, y)]
                seen.add((x, y))
                cells: list[tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < width and 0 <= ny < hud_start and (nx, ny) not in seen and int(grid[ny][nx]) == color:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                min_x = min(x for x, _ in cells)
                max_x = max(x for x, _ in cells)
                min_y = min(y for _, y in cells)
                max_y = max(y for _, y in cells)
                shape = tuple(
                    tuple(1 if (xx, yy) in set(cells) else 0 for xx in range(min_x, max_x + 1))
                    for yy in range(min_y, max_y + 1)
                )
                bbox = {
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                }
                out.append(
                    {
                        "id": hashlib.sha1(repr((color, bbox, len(cells))).encode("utf-8")).hexdigest()[:10],
                        "color": color,
                        "cells": cells[:64],
                        "bbox": bbox,
                        "area": len(cells),
                        "shape_signature": hashlib.sha1(repr(shape).encode("utf-8")).hexdigest()[:10],
                        "anchor": _anchor_name(min_x, min_y, max_x, max_y, width, hud_start),
                        "role": "unknown",
                        "confidence": 0.0,
                    }
                )
        out.sort(key=lambda item: (-int(item["area"]), item["bbox"]["min_y"], item["bbox"]["min_x"]))
        return out

    def _infer_player(self, objects: list[dict[str, Any]], frame: Frame, previous: Frame | None, hud_rows: int) -> dict[str, Any] | None:
        if previous is not None:
            prev_objects = self._extract_components(previous, hud_rows)
            candidates: list[tuple[int, dict[str, Any]]] = []
            prev_by_color = {}
            for obj in prev_objects:
                prev_by_color.setdefault(int(obj["color"]), []).append(obj)
            for obj in objects:
                if int(obj["area"]) > 32:
                    continue
                for prev in prev_by_color.get(int(obj["color"]), []):
                    if prev.get("shape_signature") != obj.get("shape_signature"):
                        continue
                    pc = _center(prev["bbox"])
                    cc = _center(obj["bbox"])
                    dist = abs(pc[0] - cc[0]) + abs(pc[1] - cc[1])
                    if 1 <= dist <= 5:
                        candidates.append((dist, obj))
            if candidates:
                player = min(candidates, key=lambda item: item[0])[1].copy()
                player["role"] = "player"
                player["confidence"] = 0.75
                return player
        # Fallback: a small component away from edges often represents the avatar.
        compact = [obj for obj in objects if 1 <= int(obj["area"]) <= 16]
        if not compact:
            return None
        compact.sort(key=lambda obj: (obj["bbox"]["min_y"], obj["bbox"]["min_x"]))
        player = compact[0].copy()
        player["role"] = "player_candidate"
        player["confidence"] = 0.35
        return player

    def _classify(
        self,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        walls: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        goals: list[dict[str, Any]] = []
        buttons: list[dict[str, Any]] = []
        hazards: list[dict[str, Any]] = []
        shape_counts = Counter(str(obj.get("shape_signature")) for obj in objects)
        player_id = player.get("id") if player else None
        for raw in objects:
            obj = raw.copy()
            if obj.get("id") == player_id:
                continue
            bbox = obj["bbox"]
            area = int(obj["area"])
            touches_edge = bbox["min_x"] == 0 or bbox["min_y"] == 0 or bbox["max_x"] >= width - 1 or bbox["max_y"] >= height - 1
            long_bar = bbox["width"] >= max(6, width // 4) or bbox["height"] >= max(6, height // 4)
            repeated = shape_counts[str(obj.get("shape_signature"))] >= 2
            anchor = str(obj.get("anchor", "middle_center"))
            if touches_edge or long_bar or area >= max(20, width * height // 18):
                obj["role"] = "wall"
                obj["confidence"] = 0.55
                walls.append(obj)
            elif anchor.startswith("bottom_") and area <= 32:
                obj["role"] = "display_candidate"
                obj["confidence"] = 0.30
            elif area <= 12 and repeated:
                obj["role"] = "item"
                obj["confidence"] = 0.55
                items.append(obj)
            elif area <= 16:
                obj["role"] = "item"
                obj["confidence"] = 0.35
                items.append(obj)
            elif area <= 32:
                obj["role"] = "button"
                obj["confidence"] = 0.25
                buttons.append(obj)
            else:
                obj["role"] = "goal"
                obj["confidence"] = 0.20
                goals.append(obj)
        return walls, items, goals, buttons, hazards

    def _ascii(
        self,
        width: int,
        height: int,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        walls: list[dict[str, Any]],
        items: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        buttons: list[dict[str, Any]],
        hazards: list[dict[str, Any]],
    ) -> str:
        if width <= 0 or height <= 0:
            return ""
        # Downsample large maps for the prompt.
        max_w, max_h = 32, 20
        sx = max(1, (width + max_w - 1) // max_w)
        sy = max(1, (height + max_h - 1) // max_h)
        aw = (width + sx - 1) // sx
        ah = (height + sy - 1) // sy
        canvas = [["." for _ in range(aw)] for _ in range(ah)]

        def mark(obj: dict[str, Any], ch: str) -> None:
            bbox = obj["bbox"]
            x0 = int(bbox["min_x"]) // sx
            x1 = int(bbox["max_x"]) // sx
            y0 = int(bbox["min_y"]) // sy
            y1 = int(bbox["max_y"]) // sy
            for yy in range(max(0, y0), min(ah, y1 + 1)):
                for xx in range(max(0, x0), min(aw, x1 + 1)):
                    canvas[yy][xx] = ch

        for obj in walls:
            mark(obj, "#")
        for obj in goals:
            mark(obj, "G")
        for obj in buttons:
            mark(obj, "B")
        for obj in items:
            mark(obj, "i")
        for obj in hazards:
            mark(obj, "H")
        if player is not None:
            mark(player, "P")
        return "\n".join("".join(row) for row in canvas)


def _center(bbox: dict[str, Any]) -> tuple[int, int]:
    return ((int(bbox["min_x"]) + int(bbox["max_x"])) // 2, (int(bbox["min_y"]) + int(bbox["max_y"])) // 2)


def _anchor_name(min_x: int, min_y: int, max_x: int, max_y: int, width: int, height: int) -> str:
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
    vertical = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.66 else "middle"
    return f"{vertical}_{horizontal}"
