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
    hud_confidence: float
    objects: list[dict[str, Any]]
    player: dict[str, Any] | None
    walls: list[dict[str, Any]]
    items: list[dict[str, Any]]
    goals: list[dict[str, Any]]
    buttons: list[dict[str, Any]]
    displays: list[dict[str, Any]]
    hazards: list[dict[str, Any]]
    ascii_map: str

    def to_notes(self) -> dict[str, Any]:
        role_counts = Counter(str(obj.get("role", "unknown")) for obj in self.objects)
        return {
            "semantic_width": self.width,
            "semantic_height": self.height,
            "semantic_hud_rows": self.hud_rows,
            "semantic_hud_confidence": self.hud_confidence,
            "semantic_objects": self.objects[:24],
            "semantic_player": self.player,
            "semantic_walls": self.walls[:16],
            "semantic_items": self.items[:16],
            "semantic_goals": self.goals[:8],
            "semantic_buttons": self.buttons[:8],
            "semantic_displays": self.displays[:8],
            "semantic_hazards": self.hazards[:8],
            "semantic_role_counts": dict(role_counts),
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
        inferred_hud_rows, hud_confidence = self._infer_hud_rows(frame)
        if hud_rows is None:
            hud_rows = inferred_hud_rows
        else:
            hud_confidence = 1.0 if hud_rows > 0 else 0.0
        hud_start = max(0, height - hud_rows)
        objects = self._extract_components(frame, hud_rows)
        player = self._infer_player(objects, frame, previous, hud_rows)
        walls, items, goals, buttons, displays, hazards = self._classify(objects, player, width, height, hud_start)
        ascii_map = self._ascii(width, height, player, walls, items, goals, buttons, displays, hazards)
        return SemanticMap(
            width=width,
            height=height,
            hud_rows=hud_rows,
            hud_confidence=hud_confidence,
            objects=objects,
            player=player,
            walls=walls,
            items=items,
            goals=goals,
            buttons=buttons,
            displays=displays,
            hazards=hazards,
            ascii_map=ascii_map,
        )

    def _infer_hud_rows(self, frame: Frame) -> tuple[int, float]:
        height = len(frame.grid)
        width = len(frame.grid[0]) if height else 0
        if height < 8 or width < 8:
            return 0, 0.0

        max_rows = min(8, max(0, height // 4))
        if max_rows <= 0:
            return 0, 0.0

        streak = 0
        score = 0.0
        for offset in range(1, max_rows + 1):
            row = frame.grid[height - offset]
            nonzero = sum(1 for value in row if int(value) != 0)
            if nonzero == 0:
                break
            fill_ratio = nonzero / max(1, width)
            colors = {int(value) for value in row if int(value) != 0}
            edge_run = self._edge_run_length(row)
            row_score = 0.0
            if fill_ratio >= 0.5:
                row_score += 0.45
            if len(colors) <= 2:
                row_score += 0.20
            if edge_run >= max(4, width // 4):
                row_score += 0.20
            if offset < height and row == frame.grid[height - offset - 1]:
                row_score += 0.15
            if row_score < 0.55:
                break
            streak += 1
            score += row_score

        if streak < 2:
            return 0, 0.0
        confidence = min(0.95, score / streak)
        if confidence < 0.65:
            return 0, confidence
        return streak, confidence

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
                        "center": _center(bbox),
                        "shape_signature": hashlib.sha1(repr(shape).encode("utf-8")).hexdigest()[:10],
                        "anchor": _anchor_name(min_x, min_y, max_x, max_y, width, hud_start),
                        "region": "hud" if min_y >= hud_start else "playfield",
                        "touches_edge": min_x == 0 or min_y == 0 or max_x >= width - 1 or max_y >= max(0, hud_start - 1),
                        "elongated": (max_x - min_x + 1) >= max(6, width // 4) or (max_y - min_y + 1) >= max(6, max(1, hud_start) // 4),
                        "role": "unknown",
                        "confidence": 0.0,
                        "role_reason": "unclassified",
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
                player["role_reason"] = "matched moving component with stable color/shape"
                return player
        # Fallback: prefer small, non-edge, non-bottom compact components near the center.
        playfield_height = max(1, len(frame.grid) - hud_rows)
        center_x = (len(frame.grid[0]) - 1) / 2 if frame.grid else 0.0
        center_y = (playfield_height - 1) / 2
        compact = [obj for obj in objects if 1 <= int(obj["area"]) <= 16 and str(obj.get("region")) == "playfield"]
        if not compact:
            return None
        compact.sort(
            key=lambda obj: (
                bool(obj.get("touches_edge", False)),
                str(obj.get("anchor", "")).startswith("bottom_"),
                abs(float(obj["center"][0]) - center_x) + abs(float(obj["center"][1]) - center_y),
                obj["bbox"]["min_y"],
                obj["bbox"]["min_x"],
            )
        )
        player = compact[0].copy()
        player["role"] = "player_candidate"
        player["confidence"] = 0.45 if not player.get("touches_edge", False) else 0.30
        player["role_reason"] = "small interior component chosen as avatar candidate"
        return player

    def _classify(
        self,
        objects: list[dict[str, Any]],
        player: dict[str, Any] | None,
        width: int,
        height: int,
        hud_start: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        walls: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        goals: list[dict[str, Any]] = []
        buttons: list[dict[str, Any]] = []
        displays: list[dict[str, Any]] = []
        hazards: list[dict[str, Any]] = []
        shape_counts = Counter(str(obj.get("shape_signature")) for obj in objects)
        player_id = player.get("id") if player else None
        for raw in objects:
            obj = raw.copy()
            if obj.get("id") == player_id:
                continue
            bbox = obj["bbox"]
            area = int(obj["area"])
            touches_edge = bool(obj.get("touches_edge", False))
            long_bar = bool(obj.get("elongated", False))
            repeated = shape_counts[str(obj.get("shape_signature"))] >= 2
            anchor = str(obj.get("anchor", "middle_center"))
            region = str(obj.get("region", "playfield"))
            near_bottom = bbox["max_y"] >= max(0, height - max(3, height // 6))
            aspect = max(bbox["width"], bbox["height"]) / max(1, min(bbox["width"], bbox["height"]))
            if touches_edge or long_bar or area >= max(20, width * max(1, hud_start) // 18):
                obj["role"] = "wall"
                obj["confidence"] = 0.55 if region == "playfield" else 0.35
                obj["role_reason"] = "large or boundary-touching structure"
                walls.append(obj)
            elif region == "hud" or (anchor.startswith("bottom_") and near_bottom and area <= 32 and aspect >= 1.0):
                obj["role"] = "display_candidate"
                obj["confidence"] = 0.45 if region == "hud" else 0.32
                obj["role_reason"] = "bottom-anchored compact structure likely to be display/panel"
                displays.append(obj)
            elif area <= 12 and repeated:
                obj["role"] = "item"
                obj["confidence"] = 0.55
                obj["role_reason"] = "small repeated motif likely to be collectible or token"
                items.append(obj)
            elif area <= 10 and not touches_edge:
                obj["role"] = "item"
                obj["confidence"] = 0.38
                obj["role_reason"] = "small interior object"
                items.append(obj)
            elif area <= 24 and near_bottom:
                obj["role"] = "button"
                obj["confidence"] = 0.28
                obj["role_reason"] = "near-bottom interactable candidate"
                buttons.append(obj)
            elif area <= 14 and not repeated and not touches_edge:
                obj["role"] = "goal"
                obj["confidence"] = 0.24
                obj["role_reason"] = "small distinct interior target candidate"
                goals.append(obj)
            else:
                obj["role"] = "goal"
                obj["confidence"] = 0.20
                obj["role_reason"] = "remaining salient non-wall object treated as goal candidate"
                goals.append(obj)
        return walls, items, goals, buttons, displays, hazards

    def _ascii(
        self,
        width: int,
        height: int,
        player: dict[str, Any] | None,
        walls: list[dict[str, Any]],
        items: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        buttons: list[dict[str, Any]],
        displays: list[dict[str, Any]],
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
        for obj in displays:
            mark(obj, "D")
        for obj in items:
            mark(obj, "i")
        for obj in hazards:
            mark(obj, "H")
        if player is not None:
            mark(player, "P")
        return "\n".join("".join(row) for row in canvas)

    def _edge_run_length(self, row: tuple[int, ...]) -> int:
        best = 0
        current = 0
        for value in row:
            if int(value) != 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best


def _center(bbox: dict[str, Any]) -> tuple[int, int]:
    return ((int(bbox["min_x"]) + int(bbox["max_x"])) // 2, (int(bbox["min_y"]) + int(bbox["max_y"])) // 2)


def _anchor_name(min_x: int, min_y: int, max_x: int, max_y: int, width: int, height: int) -> str:
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
    vertical = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.66 else "middle"
    return f"{vertical}_{horizontal}"
