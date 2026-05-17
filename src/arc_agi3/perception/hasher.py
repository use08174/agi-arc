from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from arc_agi3.core.types import Frame, Observation


class StateHasher:
    """Normalizes frames into stable state keys.

    Replace this with richer object-centric hashing once a real ARC adapter exists.
    """

    def observe(self, frame: Frame, previous: Frame | None = None) -> Observation:
        encoded = repr((frame.grid, frame.info.get("levels_completed", 0))).encode(
            "utf-8"
        )
        state_key = hashlib.sha1(encoded).hexdigest()[:12]
        changed = previous is None or previous.grid != frame.grid
        nonzero_count = sum(1 for row in frame.grid for value in row if value != 0)
        colors = sorted({int(value) for row in frame.grid for value in row if value != 0})
        hud_rows = self._hud_rows(frame)
        hud_nonzero_count = sum(
            1
            for row in frame.grid[-hud_rows:]
            for value in row
            if value != 0
        ) if hud_rows > 0 else 0
        notes = {
            "status": frame.status.value,
            "levels_completed": frame.info.get("levels_completed", 0),
            "available_actions": frame.info.get("available_actions", []),
            "nonzero_count": nonzero_count,
            "unique_colors": colors,
            "hud_rows_hint": hud_rows,
            "hud_nonzero_count": hud_nonzero_count,
            "playfield_nonzero_count": nonzero_count - hud_nonzero_count,
        }
        notes.update(self._region_notes(frame, hud_rows))
        if previous is not None:
            notes.update(self._diff_notes(previous, frame))
        return Observation(state_key=state_key, frame=frame, changed=changed, notes=notes)

    def _region_notes(self, frame: Frame, hud_rows: int) -> dict[str, object]:
        regions = self._extract_regions(frame, hud_rows)
        repeated = Counter(region["shape_signature"] for region in regions if int(region["area"]) >= 3)
        repeated_summaries = []
        for signature, count in repeated.most_common():
            if count < 2:
                continue
            members = [region for region in regions if region["shape_signature"] == signature][:3]
            repeated_summaries.append(
                {
                    "count": count,
                    "bbox": members[0]["bbox"],
                    "region": members[0]["region"],
                    "area": members[0]["area"],
                }
            )
            if len(repeated_summaries) >= 4:
                break

        anchor_regions = sorted(
            regions,
            key=lambda region: (region["anchor_rank"], -int(region["area"])),
        )[:6]
        anchor_summary = [
            {
                "anchor": region["anchor_name"],
                "bbox": region["bbox"],
                "area": region["area"],
                "region": region["region"],
                "color": region["color"],
            }
            for region in anchor_regions
            if region["anchor_name"] != "center"
        ]
        anchor_patch_summary = self._anchor_patch_summary(frame, hud_rows)
        return {
            "salient_region_count": len(regions),
            "salient_regions": regions[:12],
            "repeated_motif_summary": repeated_summaries,
            "anchor_region_summary": anchor_summary[:4],
            "anchor_patch_summary": anchor_patch_summary,
        }

    def _diff_notes(self, previous: Frame, current: Frame) -> dict[str, object]:
        changed_cells = 0
        nonzero_delta = 0
        appeared_colors: Counter[int] = Counter()
        disappeared_colors: Counter[int] = Counter()
        changed_hud_cells = 0
        changed_playfield_cells = 0
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        hud_rows = self._hud_rows(current)
        hud_start = max(0, len(current.grid) - hud_rows)

        for y, (prev_row, curr_row) in enumerate(zip(previous.grid, current.grid)):
            for x, (prev_value, curr_value) in enumerate(zip(prev_row, curr_row)):
                if prev_value == curr_value:
                    continue
                changed_cells += 1
                if y >= hud_start:
                    changed_hud_cells += 1
                else:
                    changed_playfield_cells += 1
                if prev_value == 0 and curr_value != 0:
                    nonzero_delta += 1
                elif prev_value != 0 and curr_value == 0:
                    nonzero_delta -= 1

                prev_int = int(prev_value)
                curr_int = int(curr_value)
                if prev_int != 0:
                    disappeared_colors[prev_int] += 1
                if curr_int != 0:
                    appeared_colors[curr_int] += 1

                min_x = x if min_x is None else min(min_x, x)
                min_y = y if min_y is None else min(min_y, y)
                max_x = x if max_x is None else max(max_x, x)
                max_y = y if max_y is None else max(max_y, y)

        bbox = None
        if changed_cells > 0 and None not in {min_x, min_y, max_x, max_y}:
            bbox = {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "width": (max_x - min_x + 1) if max_x is not None and min_x is not None else 0,
                "height": (max_y - min_y + 1) if max_y is not None and min_y is not None else 0,
            }

        motion_axis = "none"
        if bbox is not None:
            width = int(bbox["width"])
            height = int(bbox["height"])
            if width > height:
                motion_axis = "horizontal"
            elif height > width:
                motion_axis = "vertical"
            elif width > 0:
                motion_axis = "area"

        moved_color_candidates = sorted(
            color
            for color in set(appeared_colors) | set(disappeared_colors)
            if appeared_colors.get(color, 0) > 0 and disappeared_colors.get(color, 0) > 0
        )
        region_bias = self._region_bias(
            changed_cells=changed_cells,
            changed_hud_cells=changed_hud_cells,
            changed_playfield_cells=changed_playfield_cells,
        )
        interaction_hint = self._interaction_hint(
            changed_cells=changed_cells,
            nonzero_delta=nonzero_delta,
            unique_color_delta=len(appeared_colors) - len(disappeared_colors),
            moved_color_candidates=moved_color_candidates,
            region_bias=region_bias,
            motion_axis=motion_axis,
        )
        region_change_summary = self._region_change_summary(previous, current)
        anchor_patch_changes = self._anchor_patch_changes(previous, current)
        likely_feedback_flash = self._likely_feedback_flash(
            width=len(current.grid[0]) if current.grid else 0,
            height=len(current.grid),
            changed_playfield_cells=changed_playfield_cells,
            changed_hud_cells=changed_hud_cells,
            anchor_patch_changes=anchor_patch_changes,
        )

        return {
            "changed_cells": changed_cells,
            "nonzero_delta": nonzero_delta,
            "appeared_colors": dict(appeared_colors),
            "disappeared_colors": dict(disappeared_colors),
            "changed_bbox": bbox,
            "motion_axis": motion_axis,
            "unique_color_delta": len(appeared_colors) - len(disappeared_colors),
            "changed_hud_cells": changed_hud_cells,
            "changed_playfield_cells": changed_playfield_cells,
            "region_bias": region_bias,
            "interaction_hint": interaction_hint,
            "moved_color_candidates": moved_color_candidates,
            "region_change_summary": region_change_summary,
            "anchor_patch_changes": anchor_patch_changes,
            "likely_feedback_flash": likely_feedback_flash,
        }

    def _extract_regions(self, frame: Frame, hud_rows: int) -> list[dict[str, Any]]:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        seen: set[tuple[int, int]] = set()
        regions: list[dict[str, Any]] = []
        hud_start = max(0, height - hud_rows)
        for y in range(height):
            for x in range(width):
                color = int(grid[y][x])
                if (x, y) in seen or color == 0:
                    continue
                stack = [(x, y)]
                seen.add((x, y))
                cells: list[tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and (nx, ny) not in seen
                            and int(grid[ny][nx]) == color
                        ):
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                min_x = min(cell[0] for cell in cells)
                max_x = max(cell[0] for cell in cells)
                min_y = min(cell[1] for cell in cells)
                max_y = max(cell[1] for cell in cells)
                area = len(cells)
                bbox_grid = tuple(
                    tuple(int(grid[row][col]) for col in range(min_x, max_x + 1))
                    for row in range(min_y, max_y + 1)
                )
                shape_grid = tuple(
                    tuple(1 if int(grid[row][col]) == color else 0 for col in range(min_x, max_x + 1))
                    for row in range(min_y, max_y + 1)
                )
                region = "hud" if min_y >= hud_start else "playfield"
                anchor_name, anchor_rank = self._anchor_name(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    width=width,
                    height=height,
                )
                regions.append(
                    {
                        "bbox": {
                            "min_x": min_x,
                            "min_y": min_y,
                            "max_x": max_x,
                            "max_y": max_y,
                            "width": max_x - min_x + 1,
                            "height": max_y - min_y + 1,
                        },
                        "area": area,
                        "region": region,
                        "color": color,
                        "anchor_name": anchor_name,
                        "anchor_rank": anchor_rank,
                        "signature": hashlib.sha1(repr(bbox_grid).encode("utf-8")).hexdigest()[:10],
                        "shape_signature": hashlib.sha1(repr(shape_grid).encode("utf-8")).hexdigest()[:10],
                    }
                )
        regions.sort(key=lambda item: (-int(item["area"]), item["bbox"]["min_y"], item["bbox"]["min_x"]))
        return regions

    def _anchor_patch_summary(self, frame: Frame, hud_rows: int) -> list[dict[str, Any]]:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        if height == 0 or width == 0:
            return []
        patch_h = max(4, min(10, height // 6))
        patch_w = max(4, min(10, width // 6))
        anchors = [
            ("top_left", 0, 0),
            ("top_right", max(0, width - patch_w), 0),
            ("bottom_left", 0, max(0, height - patch_h)),
            ("bottom_right", max(0, width - patch_w), max(0, height - patch_h)),
        ]
        summaries: list[dict[str, Any]] = []
        for name, start_x, start_y in anchors:
            patch = tuple(
                tuple(int(grid[y][x]) for x in range(start_x, min(width, start_x + patch_w)))
                for y in range(start_y, min(height, start_y + patch_h))
            )
            nonzero = sum(1 for row in patch for value in row if value != 0)
            colors = sorted({int(value) for row in patch for value in row if value != 0})
            region = "hud" if start_y >= max(0, height - hud_rows) else "playfield"
            summaries.append(
                {
                    "anchor": name,
                    "region": region,
                    "nonzero": nonzero,
                    "colors": colors[:6],
                    "signature": hashlib.sha1(repr(patch).encode("utf-8")).hexdigest()[:10],
                }
            )
        return summaries

    def _anchor_name(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        width: int,
        height: int,
    ) -> tuple[str, int]:
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
        vertical = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.66 else "middle"
        name = f"{vertical}_{horizontal}"
        rank_map = {
            "top_left": 0,
            "top_right": 1,
            "bottom_left": 2,
            "bottom_right": 3,
            "middle_left": 4,
            "middle_right": 5,
        }
        return name, rank_map.get(name, 9)

    def _region_change_summary(self, previous: Frame, current: Frame) -> list[str]:
        previous_regions = self._extract_regions(previous, self._hud_rows(previous))
        current_regions = self._extract_regions(current, self._hud_rows(current))
        prev_signatures = Counter(region["signature"] for region in previous_regions)
        curr_signatures = Counter(region["signature"] for region in current_regions)
        summary: list[str] = []
        for region in current_regions[:8]:
            signature = region["signature"]
            delta = curr_signatures[signature] - prev_signatures.get(signature, 0)
            if delta <= 0:
                continue
            summary.append(
                f"new_or_more_region anchor={region['anchor_name']} area={region['area']} region={region['region']}"
            )
            if len(summary) >= 4:
                break
        for region in previous_regions[:8]:
            signature = region["signature"]
            delta = prev_signatures[signature] - curr_signatures.get(signature, 0)
            if delta <= 0:
                continue
            summary.append(
                f"removed_or_less_region anchor={region['anchor_name']} area={region['area']} region={region['region']}"
            )
            if len(summary) >= 6:
                break
        return summary

    def _anchor_patch_changes(self, previous: Frame, current: Frame) -> list[str]:
        prev_patches = {
            item["anchor"]: item["signature"]
            for item in self._anchor_patch_summary(previous, self._hud_rows(previous))
        }
        curr_patches = {
            item["anchor"]: item["signature"]
            for item in self._anchor_patch_summary(current, self._hud_rows(current))
        }
        changed: list[str] = []
        for anchor, signature in curr_patches.items():
            if prev_patches.get(anchor) != signature:
                changed.append(anchor)
        return changed

    def _likely_feedback_flash(
        self,
        width: int,
        height: int,
        changed_playfield_cells: int,
        changed_hud_cells: int,
        anchor_patch_changes: list[str],
    ) -> bool:
        if width < 12 or height < 12:
            return False
        if not anchor_patch_changes:
            return False
        lower_panel_change = any("bottom" in anchor for anchor in anchor_patch_changes)
        if not lower_panel_change and changed_hud_cells == 0:
            return False
        if changed_playfield_cells <= 8:
            return True
        if changed_hud_cells > 0 and changed_playfield_cells <= 16:
            return True
        return False

    def _hud_rows(self, frame: Frame) -> int:
        height = len(frame.grid)
        if height == 0:
            return 0
        return max(2, min(8, height // 6))

    def _region_bias(
        self,
        changed_cells: int,
        changed_hud_cells: int,
        changed_playfield_cells: int,
    ) -> str:
        if changed_cells == 0:
            return "none"
        hud_ratio = changed_hud_cells / changed_cells
        if hud_ratio >= 0.7:
            return "hud"
        if changed_playfield_cells > 0 and hud_ratio <= 0.3:
            return "playfield"
        return "mixed"

    def _interaction_hint(
        self,
        changed_cells: int,
        nonzero_delta: int,
        unique_color_delta: int,
        moved_color_candidates: list[int],
        region_bias: str,
        motion_axis: str,
    ) -> str:
        if changed_cells == 0:
            return "noop"
        if region_bias == "hud":
            return "hud_or_counter_update"
        if nonzero_delta < 0 and changed_cells <= 8:
            return "pickup_or_consume"
        if nonzero_delta > 0 and changed_cells <= 8:
            return "spawn_or_unlock"
        if moved_color_candidates and changed_cells <= 8:
            return "entity_move_or_push"
        if unique_color_delta != 0:
            return "mode_or_palette_change"
        if motion_axis in {"horizontal", "vertical"} and changed_cells <= 8:
            return f"{motion_axis}_navigation"
        if changed_cells >= 24:
            return "board_or_room_transform"
        return "local_playfield_transform"
