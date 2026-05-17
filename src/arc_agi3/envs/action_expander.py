from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from arc_agi3.core.config import ClickExpansionConfig
from arc_agi3.core.types import Action, Frame


class ActionExpander:
    """Expands complex ARC actions into a small object-centric candidate set."""

    def __init__(self, config: ClickExpansionConfig) -> None:
        self.config = config

    def expand(self, actions: list[str], frame: Frame | None) -> list[Action]:
        expanded: list[Action] = []
        for action_name in actions:
            if action_name == "ACTION6":
                expanded.extend(self._expand_clicks(frame))
            else:
                expanded.append(Action(name=action_name))
        return expanded

    def _expand_clicks(self, frame: Frame | None) -> list[Action]:
        if frame is None or not frame.grid:
            return [Action(name="ACTION6", payload={"x": 32, "y": 32})]
        height = len(frame.grid)
        width = len(frame.grid[0]) if height else 0
        if width == 0:
            return [Action(name="ACTION6", payload={"x": 32, "y": 32})]
        points: list[tuple[int, int]] = []
        hud_rows = self._hud_rows(height)
        hud_start = max(1, height - hud_rows)
        points.extend(self._object_centers(frame))
        center = (width // 2, max(0, hud_start // 2))
        points.append(center)
        steps = max(1, self.config.grid_points_per_axis)
        for y in self._linspace_indices(hud_start, steps):
            for x in self._linspace_indices(width, steps):
                points.append((x, y))
        out: list[Action] = []
        seen: set[tuple[int, int]] = set()
        for x, y in points:
            if (x, y) in seen:
                continue
            seen.add((x, y))
            out.append(Action(name="ACTION6", payload={"x": int(x), "y": int(y)}))
            if len(out) >= self.config.max_candidates:
                break
        return out

    def _object_centers(self, frame: Frame) -> list[tuple[int, int]]:
        regions = self._extract_regions(frame)
        # Prefer compact repeated/non-edge objects likely to be items/buttons.
        shape_counts = Counter(region["shape_signature"] for region in regions)
        ranked = []
        height = len(frame.grid)
        width = len(frame.grid[0]) if height else 0
        for region in regions:
            bbox = region["bbox"]
            area = int(region["area"])
            touches_edge = bbox["min_x"] == 0 or bbox["min_y"] == 0 or bbox["max_x"] >= width - 1 or bbox["max_y"] >= height - 1
            compact = area <= 16 and bbox["width"] <= 6 and bbox["height"] <= 6
            repeated = shape_counts[region["shape_signature"]] >= 2
            score = (not compact, touches_edge, not repeated, area, bbox["min_y"], bbox["min_x"])
            if compact and not touches_edge:
                ranked.append((score, ((bbox["min_x"] + bbox["max_x"]) // 2, (bbox["min_y"] + bbox["max_y"]) // 2)))
        ranked.sort(key=lambda item: item[0])
        return [point for _, point in ranked[: self.config.max_candidates]]

    def _extract_regions(self, frame: Frame) -> list[dict[str, Any]]:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        hud_rows = self._hud_rows(height)
        hud_start = max(0, height - hud_rows)
        seen: set[tuple[int, int]] = set()
        regions: list[dict[str, Any]] = []
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
                min_x = min(px for px, _ in cells)
                max_x = max(px for px, _ in cells)
                min_y = min(py for _, py in cells)
                max_y = max(py for _, py in cells)
                shape = tuple(
                    tuple(1 if (xx, yy) in set(cells) else 0 for xx in range(min_x, max_x + 1))
                    for yy in range(min_y, max_y + 1)
                )
                regions.append(
                    {
                        "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1},
                        "area": len(cells),
                        "color": color,
                        "shape_signature": hashlib.sha1(repr(shape).encode("utf-8")).hexdigest()[:10],
                    }
                )
        return regions

    def _hud_rows(self, height: int) -> int:
        if height < 4:
            return 0
        return max(2, min(8, height // 6))

    def _linspace_indices(self, size: int, steps: int) -> list[int]:
        if steps <= 1:
            return [size // 2]
        max_index = max(0, size - 1)
        return sorted({min(max_index, round(idx * max_index / (steps - 1))) for idx in range(steps)})
