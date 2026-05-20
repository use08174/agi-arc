from __future__ import annotations

from typing import Any


class RegionMatcher:
    """Find likely reference/workspace pairs and score their alignment."""

    def infer(
        self,
        *,
        grid: tuple[tuple[int, ...], ...],
        region_roles: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        references = [item for item in region_roles.get("reference_like", []) if isinstance(item, dict)]
        workspaces = [item for item in region_roles.get("workspace_like", []) if isinstance(item, dict)]
        if not references or not workspaces:
            return None

        best: dict[str, Any] | None = None
        best_score = -1.0
        for reference in references[:3]:
            for workspace in workspaces[:3]:
                ref_bbox = reference.get("bbox")
                work_bbox = workspace.get("bbox")
                if not isinstance(ref_bbox, dict) or not isinstance(work_bbox, dict):
                    continue
                ref_patch = self._extract_patch(grid, ref_bbox)
                work_patch = self._extract_patch(grid, work_bbox)
                if not ref_patch or not work_patch:
                    continue
                ref_norm = self._normalize_patch(ref_patch)
                work_norm = self._normalize_patch(work_patch)
                layout_similarity = self._layout_similarity(ref_norm, work_norm)
                color_similarity = self._color_similarity(ref_patch, work_patch)
                shape_similarity = self._shape_similarity(ref_bbox, work_bbox)
                score = 0.55 * layout_similarity + 0.25 * color_similarity + 0.20 * shape_similarity
                if score <= best_score:
                    continue
                best_score = score
                best = {
                    "reference_id": reference.get("id"),
                    "workspace_id": workspace.get("id"),
                    "reference_anchor": reference.get("anchor", ""),
                    "workspace_anchor": workspace.get("anchor", ""),
                    "reference_bbox": ref_bbox,
                    "workspace_bbox": work_bbox,
                    "alignment_score": round(score, 4),
                    "layout_similarity": round(layout_similarity, 4),
                    "color_similarity": round(color_similarity, 4),
                    "shape_similarity": round(shape_similarity, 4),
                }
        return best

    def _extract_patch(
        self,
        grid: tuple[tuple[int, ...], ...],
        bbox: dict[str, Any],
    ) -> list[list[int]]:
        min_x = int(bbox.get("min_x", 0) or 0)
        max_x = int(bbox.get("max_x", -1) or -1)
        min_y = int(bbox.get("min_y", 0) or 0)
        max_y = int(bbox.get("max_y", -1) or -1)
        if min_x > max_x or min_y > max_y:
            return []
        patch: list[list[int]] = []
        for y in range(min_y, max_y + 1):
            if y < 0 or y >= len(grid):
                continue
            row = []
            for x in range(min_x, max_x + 1):
                if x < 0 or x >= len(grid[y]):
                    continue
                row.append(int(grid[y][x]))
            if row:
                patch.append(row)
        return patch

    def _normalize_patch(self, patch: list[list[int]], size: int = 8) -> list[list[int]]:
        src_h = len(patch)
        src_w = len(patch[0]) if src_h else 0
        if src_h == 0 or src_w == 0:
            return [[0 for _ in range(size)] for _ in range(size)]
        normalized: list[list[int]] = []
        for ny in range(size):
            src_y = min(src_h - 1, int(ny * src_h / size))
            row: list[int] = []
            for nx in range(size):
                src_x = min(src_w - 1, int(nx * src_w / size))
                row.append(int(patch[src_y][src_x]))
            normalized.append(row)
        return normalized

    def _layout_similarity(self, left: list[list[int]], right: list[list[int]]) -> float:
        if not left or not right:
            return 0.0
        total = 0
        same = 0
        for left_row, right_row in zip(left, right):
            for left_cell, right_cell in zip(left_row, right_row):
                total += 1
                if left_cell == right_cell:
                    same += 1
        return same / max(1, total)

    def _color_similarity(self, left: list[list[int]], right: list[list[int]]) -> float:
        left_counts = self._color_counts(left)
        right_counts = self._color_counts(right)
        colors = set(left_counts) | set(right_counts)
        if not colors:
            return 0.0
        overlap = sum(min(left_counts.get(color, 0.0), right_counts.get(color, 0.0)) for color in colors)
        return overlap

    def _color_counts(self, patch: list[list[int]]) -> dict[int, float]:
        counts: dict[int, int] = {}
        total = 0
        for row in patch:
            for value in row:
                counts[int(value)] = counts.get(int(value), 0) + 1
                total += 1
        if total <= 0:
            return {}
        return {color: count / total for color, count in counts.items()}

    def _shape_similarity(self, left_bbox: dict[str, Any], right_bbox: dict[str, Any]) -> float:
        left_w = max(1, int(left_bbox.get("width", 1) or 1))
        left_h = max(1, int(left_bbox.get("height", 1) or 1))
        right_w = max(1, int(right_bbox.get("width", 1) or 1))
        right_h = max(1, int(right_bbox.get("height", 1) or 1))
        width_ratio = min(left_w, right_w) / max(left_w, right_w)
        height_ratio = min(left_h, right_h) / max(left_h, right_h)
        return (width_ratio + height_ratio) / 2.0
