from __future__ import annotations

import hashlib
from collections import Counter

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
        if previous is not None:
            notes.update(self._diff_notes(previous, frame))
        return Observation(state_key=state_key, frame=frame, changed=changed, notes=notes)

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
        }

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
