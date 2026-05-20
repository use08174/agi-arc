from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from arc_agi3.core.types import Frame, Observation
from arc_agi3.perception.map_parser import MapParser


class StateHasher:
    """Normalizes frames into stable state keys plus semantic notes."""

    def __init__(self) -> None:
        self.map_parser = MapParser()

    def observe(self, frame: Frame, previous: Frame | None = None) -> Observation:
        semantic = self.map_parser.parse(frame, previous=previous)
        hud_rows = semantic.hud_rows
        semantic_notes = semantic.to_notes()
        full_grid = self._grid_signature(frame)

        # Keep the full grid in the state key. HUD guesses are too uncertain to crop aggressively.
        semantic_signature = (
            tuple(sorted((obj.get("color"), obj.get("shape_signature"), _bbox_tuple(obj.get("bbox"))) for obj in semantic.items[:16])),
            _bbox_tuple(semantic.player.get("bbox")) if semantic.player else None,
            frame.info.get("levels_completed", 0),
            frame.status.value,
        )
        encoded = repr((full_grid, semantic_signature)).encode("utf-8")
        state_key = hashlib.sha1(encoded).hexdigest()[:12]

        if previous is None:
            changed = True
        else:
            changed = self._grid_signature(previous) != full_grid

        nonzero_count = sum(1 for row in frame.grid for value in row if value != 0)
        colors = sorted({int(value) for row in frame.grid for value in row if value != 0})
        hud_nonzero_count = (
            sum(1 for row in frame.grid[-hud_rows:] for value in row if value != 0)
            if hud_rows > 0
            else 0
        )

        notes: dict[str, Any] = {
            "status": frame.status.value,
            "levels_completed": frame.info.get("levels_completed", 0),
            "available_actions": frame.info.get("available_actions", []),
            "nonzero_count": nonzero_count,
            "unique_colors": colors,
            "hud_rows_hint": hud_rows,
            "hud_confidence_hint": semantic.hud_confidence,
            "hud_nonzero_count": hud_nonzero_count,
            "playfield_nonzero_count": nonzero_count - hud_nonzero_count,
        }
        notes.update(semantic_notes)
        notes.update(self._region_notes(frame, hud_rows))
        if previous is not None:
            notes.update(self._diff_notes(previous, frame))
            notes.update(self._semantic_diff_notes(previous, frame, hud_rows))
        return Observation(state_key=state_key, frame=frame, changed=changed, notes=notes)

    def _grid_signature(self, frame: Frame) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in frame.grid)

    def _hud_rows(self, frame: Frame) -> int:
        hud_rows, _ = self.map_parser._infer_hud_rows(frame)
        return hud_rows

    def _region_notes(self, frame: Frame, hud_rows: int) -> dict[str, object]:
        regions = self._extract_regions(frame, hud_rows)
        repeated = Counter(region["shape_signature"] for region in regions if int(region["area"]) >= 3)
        repeated_summaries = []
        for signature, count in repeated.most_common():
            if count < 2:
                continue
            members = [region for region in regions if region["shape_signature"] == signature][:3]
            repeated_summaries.append(
                {"count": count, "bbox": members[0]["bbox"], "region": members[0]["region"], "area": members[0]["area"]}
            )
            if len(repeated_summaries) >= 4:
                break
        anchor_regions = sorted(regions, key=lambda region: (region["anchor_rank"], -int(region["area"])))[:6]
        anchor_summary = [
            {
                "anchor": region["anchor_name"],
                "bbox": region["bbox"],
                "area": region["area"],
                "region": region["region"],
                "color": region["color"],
            }
            for region in anchor_regions
            if region["anchor_name"] != "middle_center"
        ]
        collectible_candidates = self._collectible_candidates(regions, frame)
        return {
            "salient_region_count": len(regions),
            "salient_regions": regions[:12],
            "repeated_motif_summary": repeated_summaries,
            "anchor_region_summary": anchor_summary[:4],
            "anchor_patch_summary": self._anchor_patch_summary(frame, hud_rows),
            "collectible_candidates": collectible_candidates[:8],
            "collectible_candidate_count": len(collectible_candidates),
        }

    def _extract_regions(self, frame: Frame, hud_rows: int) -> list[dict[str, Any]]:
        grid = frame.grid
        height = len(grid)
        width = len(grid[0]) if height else 0
        hud_start = max(0, height - hud_rows)
        seen: set[tuple[int, int]] = set()
        regions: list[dict[str, Any]] = []
        for y in range(height):
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
                        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen and int(grid[ny][nx]) == color:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                min_x = min(px for px, _ in cells)
                max_x = max(px for px, _ in cells)
                min_y = min(py for _, py in cells)
                max_y = max(py for _, py in cells)
                shape_grid = tuple(
                    tuple(1 if int(grid[row][col]) == color else 0 for col in range(min_x, max_x + 1))
                    for row in range(min_y, max_y + 1)
                )
                region = "hud" if min_y >= hud_start else "playfield"
                bbox = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1}
                regions.append(
                    {
                        "bbox": bbox,
                        "area": len(cells),
                        "region": region,
                        "color": color,
                        "anchor_name": self._anchor_name(min_x, min_y, max_x, max_y, width, height)[0],
                        "anchor_rank": self._anchor_name(min_x, min_y, max_x, max_y, width, height)[1],
                        "signature": hashlib.sha1(repr((color, bbox, len(cells))).encode("utf-8")).hexdigest()[:10],
                        "shape_signature": hashlib.sha1(repr(shape_grid).encode("utf-8")).hexdigest()[:10],
                    }
                )
        regions.sort(key=lambda item: (-int(item["area"]), item["bbox"]["min_y"], item["bbox"]["min_x"]))
        return regions

    def _collectible_candidates(self, regions: list[dict[str, Any]], frame: Frame) -> list[dict[str, Any]]:
        height = len(frame.grid)
        width = len(frame.grid[0]) if height else 0
        candidates: list[dict[str, Any]] = []
        for region in regions:
            if region["region"] != "playfield":
                continue
            area = int(region["area"])
            bbox = region["bbox"]
            if area < 3 or area > 16:
                continue
            if bbox["width"] > 6 or bbox["height"] > 6:
                continue
            if bbox["min_x"] <= 0 or bbox["min_y"] <= 0 or bbox["max_x"] >= width - 1 or bbox["max_y"] >= height - 1:
                continue
            if region["anchor_name"] in {"bottom_left", "bottom_right", "top_left", "top_right"}:
                continue
            candidates.append({"bbox": bbox, "area": area, "color": region["color"], "shape_signature": region["shape_signature"], "anchor": region["anchor_name"]})
        candidates.sort(key=lambda item: (item["bbox"]["min_y"], item["bbox"]["min_x"]))
        return candidates

    def _diff_notes(self, previous: Frame, current: Frame) -> dict[str, object]:
        changed_cells = 0
        nonzero_delta = 0
        appeared_colors: Counter[int] = Counter()
        disappeared_colors: Counter[int] = Counter()
        changed_hud_cells = 0
        changed_playfield_cells = 0
        min_x = min_y = max_x = max_y = None
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
                if int(prev_value) != 0:
                    disappeared_colors[int(prev_value)] += 1
                if int(curr_value) != 0:
                    appeared_colors[int(curr_value)] += 1
                min_x = x if min_x is None else min(min_x, x)
                min_y = y if min_y is None else min(min_y, y)
                max_x = x if max_x is None else max(max_x, x)
                max_y = y if max_y is None else max(max_y, y)
        bbox = None
        if changed_cells > 0 and None not in {min_x, min_y, max_x, max_y}:
            bbox = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1}
        moved_color_candidates = sorted(
            color for color in set(appeared_colors) | set(disappeared_colors) if appeared_colors.get(color, 0) > 0 and disappeared_colors.get(color, 0) > 0
        )
        region_bias = self._region_bias(changed_cells, changed_hud_cells, changed_playfield_cells)
        motion_axis = "none"
        if bbox is not None:
            if int(bbox["width"]) > int(bbox["height"]):
                motion_axis = "horizontal"
            elif int(bbox["height"]) > int(bbox["width"]):
                motion_axis = "vertical"
            elif int(bbox["width"]) > 0:
                motion_axis = "area"
        interaction_hint = self._interaction_hint(changed_cells, nonzero_delta, len(appeared_colors) - len(disappeared_colors), moved_color_candidates, region_bias, motion_axis)
        collectible_changes = self._collectible_changes(previous, current)
        anchor_patch_changes = self._anchor_patch_changes(previous, current)
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
            "region_change_summary": self._region_change_summary(previous, current),
            "anchor_patch_changes": anchor_patch_changes,
            "collectible_changes": collectible_changes,
            "collectible_progress": bool(collectible_changes["removed"]) and changed_playfield_cells <= 32,
            "likely_feedback_flash": self._likely_feedback_flash(
                width=len(current.grid[0]) if current.grid else 0,
                height=len(current.grid),
                changed_playfield_cells=changed_playfield_cells,
                changed_hud_cells=changed_hud_cells,
                anchor_patch_changes=anchor_patch_changes,
            ),
        }

    def _semantic_diff_notes(self, previous: Frame, current: Frame, hud_rows: int) -> dict[str, object]:
        prev = self.map_parser.parse(previous, previous=None, hud_rows=hud_rows)
        curr = self.map_parser.parse(current, previous=previous, hud_rows=hud_rows)
        prev_pos = _center(prev.player.get("bbox")) if prev.player else None
        curr_pos = _center(curr.player.get("bbox")) if curr.player else None
        moved = prev_pos is not None and curr_pos is not None and prev_pos != curr_pos
        prev_match_score = float((prev.region_match or {}).get("alignment_score", 0.0) or 0.0)
        curr_match_score = float((curr.region_match or {}).get("alignment_score", 0.0) or 0.0)
        return {
            "semantic_previous_player_pos": prev_pos,
            "semantic_player_pos": curr_pos,
            "semantic_player_moved": moved,
            "reference_workspace_previous_alignment_score": prev_match_score,
            "reference_workspace_alignment_score": curr_match_score,
            "reference_workspace_alignment_delta": round(curr_match_score - prev_match_score, 4),
            "reference_workspace_layout_delta": round(
                float((curr.region_match or {}).get("layout_similarity", 0.0) or 0.0)
                - float((prev.region_match or {}).get("layout_similarity", 0.0) or 0.0),
                4,
            ),
        }

    def _collectible_changes(self, previous: Frame, current: Frame) -> dict[str, list[str]]:
        prev = {
            item["shape_signature"] + f":{item['bbox']['min_x']}:{item['bbox']['min_y']}"
            for item in self._collectible_candidates(self._extract_regions(previous, self._hud_rows(previous)), previous)
        }
        curr = {
            item["shape_signature"] + f":{item['bbox']['min_x']}:{item['bbox']['min_y']}"
            for item in self._collectible_candidates(self._extract_regions(current, self._hud_rows(current)), current)
        }
        return {"removed": sorted(prev - curr), "added": sorted(curr - prev)}

    def _anchor_name(self, min_x: int, min_y: int, max_x: int, max_y: int, width: int, height: int) -> tuple[str, int]:
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
            summaries.append(
                {
                    "anchor": name,
                    "region": "hud" if start_y >= max(0, height - hud_rows) else "playfield",
                    "nonzero": sum(1 for row in patch for value in row if value != 0),
                    "colors": sorted({int(value) for row in patch for value in row if value != 0})[:6],
                    "signature": hashlib.sha1(repr(patch).encode("utf-8")).hexdigest()[:10],
                }
            )
        return summaries

    def _region_change_summary(self, previous: Frame, current: Frame) -> list[str]:
        previous_regions = self._extract_regions(previous, self._hud_rows(previous))
        current_regions = self._extract_regions(current, self._hud_rows(current))
        prev_signatures = Counter(region["signature"] for region in previous_regions)
        curr_signatures = Counter(region["signature"] for region in current_regions)
        summary: list[str] = []
        for label, regions, left, right in (
            ("new_or_more_region", current_regions, curr_signatures, prev_signatures),
            ("removed_or_less_region", previous_regions, prev_signatures, curr_signatures),
        ):
            for region in regions[:8]:
                signature = region["signature"]
                if left[signature] - right.get(signature, 0) <= 0:
                    continue
                summary.append(f"{label} anchor={region['anchor_name']} area={region['area']} region={region['region']}")
                if len(summary) >= 6:
                    return summary
        return summary

    def _anchor_patch_changes(self, previous: Frame, current: Frame) -> list[str]:
        prev = {item["anchor"]: item["signature"] for item in self._anchor_patch_summary(previous, self._hud_rows(previous))}
        curr = {item["anchor"]: item["signature"] for item in self._anchor_patch_summary(current, self._hud_rows(current))}
        return [anchor for anchor, signature in curr.items() if prev.get(anchor) != signature]

    def _likely_feedback_flash(
        self,
        width: int,
        height: int,
        changed_playfield_cells: int,
        changed_hud_cells: int,
        anchor_patch_changes: list[str],
    ) -> bool:
        if width < 12 or height < 12 or not anchor_patch_changes:
            return False
        lower_panel_change = any("bottom" in anchor for anchor in anchor_patch_changes)
        if not lower_panel_change and changed_hud_cells == 0:
            return False
        return changed_playfield_cells <= 8 or (changed_hud_cells > 0 and changed_playfield_cells <= 16)

    def _region_bias(self, changed_cells: int, changed_hud_cells: int, changed_playfield_cells: int) -> str:
        if changed_cells == 0:
            return "none"
        hud_ratio = changed_hud_cells / changed_cells
        if hud_ratio >= 0.7:
            return "hud"
        if changed_playfield_cells > 0 and hud_ratio <= 0.3:
            return "playfield"
        return "mixed"

    def _interaction_hint(self, changed_cells: int, nonzero_delta: int, unique_color_delta: int, moved_color_candidates: list[int], region_bias: str, motion_axis: str) -> str:
        if changed_cells == 0:
            return "noop"
        if region_bias == "hud":
            return "hud_or_counter_update"
        if nonzero_delta < 0 and changed_cells <= 12:
            return "pickup_or_consume"
        if nonzero_delta > 0 and changed_cells <= 12:
            return "spawn_or_unlock"
        if moved_color_candidates and changed_cells <= 16:
            return "entity_move_or_push"
        if unique_color_delta != 0:
            return "mode_or_palette_change"
        if motion_axis in {"horizontal", "vertical"} and changed_cells <= 16:
            return f"{motion_axis}_navigation"
        if changed_cells >= 24:
            return "board_or_room_transform"
        return "local_playfield_transform"


def _center(bbox: dict[str, Any]) -> tuple[int, int]:
    return ((int(bbox["min_x"]) + int(bbox["max_x"])) // 2, (int(bbox["min_y"]) + int(bbox["max_y"])) // 2)


def _bbox_tuple(bbox: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, dict):
        return None
    return (int(bbox["min_x"]), int(bbox["min_y"]), int(bbox["max_x"]), int(bbox["max_y"]))
