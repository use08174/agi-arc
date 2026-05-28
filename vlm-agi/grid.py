from __future__ import annotations

import re
from collections import deque
from typing import Any

import numpy as np


ARC_PALETTE_RGB = {
    0: (0xFF, 0xFF, 0xFF),
    1: (0xCC, 0xCC, 0xCC),
    2: (0x99, 0x99, 0x99),
    3: (0x66, 0x66, 0x66),
    4: (0x33, 0x33, 0x33),
    5: (0x00, 0x00, 0x00),
    6: (0xE5, 0x3A, 0xA3),
    7: (0xFF, 0x7B, 0xCC),
    8: (0xF9, 0x3C, 0x31),
    9: (0x1E, 0x93, 0xFF),
    10: (0x88, 0xD8, 0xF1),
    11: (0xFF, 0xDC, 0x00),
    12: (0xFF, 0x85, 0x1B),
    13: (0x92, 0x12, 0x31),
    14: (0x4F, 0xCC, 0x30),
    15: (0xA3, 0x56, 0xD6),
}


def as_list_grid(grid: Any) -> list[list[int]]:
    if grid is None:
        return []
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [[int(cell) for cell in row] for row in grid]


def raw_frame_stack(raw_frame: Any) -> list[Any]:
    return list(getattr(raw_frame, "frame", []) or [])


def latest_grid_from_raw(raw_frame: Any) -> list[list[int]]:
    frames = raw_frame_stack(raw_frame)
    return as_list_grid(frames[-1]) if frames else []


def is_valid_grid(grid: Any) -> bool:
    arr = np.asarray(as_list_grid(grid), dtype=np.int64)
    return arr.ndim == 2 and arr.size > 0


def grid_to_rgb_array(
    grid: Any,
    *,
    scale: int = 8,
    draw_grid: bool = False,
) -> np.ndarray:
    arr = np.asarray(as_list_grid(grid), dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape={arr.shape}")

    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for color, value in ARC_PALETTE_RGB.items():
        rgb[arr == color] = value

    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    if draw_grid and scale >= 6:
        rgb[::scale, :, :] = 40
        rgb[:, ::scale, :] = 40

    return rgb.astype(np.uint8)


def summarize_grid(grid: Any) -> dict[str, Any]:
    grid = as_list_grid(grid)
    if not grid:
        return {"shape": [0, 0], "colors": [], "nonzero": 0}
    arr = np.asarray(grid, dtype=int)
    return {
        "shape": [int(x) for x in arr.shape],
        "colors": [int(x) for x in sorted(np.unique(arr).tolist())],
        "nonzero": int(np.count_nonzero(arr)),
    }


def summarize_objects(grid: Any, max_objects: int = 10) -> list[dict[str, Any]]:
    arr = np.asarray(as_list_grid(grid), dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        return []

    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, Any]] = []

    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue

            color = int(arr[y0, x0])
            seen[y0, x0] = True
            if color == 0:
                continue

            queue = deque([(x0, y0)])
            cells = []
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if not seen[ny, nx] and int(arr[ny, nx]) == color:
                            seen[ny, nx] = True
                            queue.append((nx, ny))

            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            objects.append(
                {
                    "color_id": color,
                    "cell_count": len(cells),
                    "bbox": {
                        "x_min": int(min(xs)),
                        "x_max": int(max(xs)),
                        "y_min": int(min(ys)),
                        "y_max": int(max(ys)),
                    },
                    "center": {
                        "x": round(float(sum(xs) / len(xs)), 2),
                        "y": round(float(sum(ys) / len(ys)), 2),
                    },
                    "size": [int(max(xs) - min(xs) + 1), int(max(ys) - min(ys) + 1)],
                }
            )

    objects.sort(key=lambda obj: obj["cell_count"], reverse=True)
    return objects[:max_objects]


def summarize_spatial_layout(grid: Any, max_objects: int = 12) -> dict[str, Any]:
    arr = np.asarray(as_list_grid(grid), dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "shape": [0, 0],
            "nonzero_bbox": None,
            "top_colors": [],
            "object_count": 0,
            "large_objects": [],
            "edge_contacts": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        }

    h, w = arr.shape
    nonzero = arr != 0
    bbox = None
    if nonzero.any():
        ys, xs = np.where(nonzero)
        bbox = {
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
        }

    colors, counts = np.unique(arr[nonzero], return_counts=True) if nonzero.any() else ([], [])
    top_colors = [
        {"color_id": int(color), "count": int(count)}
        for color, count in sorted(zip(colors, counts), key=lambda item: -item[1])[:6]
    ]

    objects = summarize_objects(grid, max_objects=max_objects)
    large_objects = []
    for obj in objects[:6]:
        bbox_obj = obj["bbox"]
        touches = []
        if bbox_obj["y_min"] == 0:
            touches.append("top")
        if bbox_obj["y_max"] == h - 1:
            touches.append("bottom")
        if bbox_obj["x_min"] == 0:
            touches.append("left")
        if bbox_obj["x_max"] == w - 1:
            touches.append("right")
        large_objects.append(
            {
                "color_id": obj["color_id"],
                "cell_count": obj["cell_count"],
                "bbox": bbox_obj,
                "center": obj["center"],
                "touches_edge": touches,
            }
        )

    edge_contacts = {
        "top": int(np.count_nonzero(arr[0, :] != 0)),
        "bottom": int(np.count_nonzero(arr[-1, :] != 0)),
        "left": int(np.count_nonzero(arr[:, 0] != 0)),
        "right": int(np.count_nonzero(arr[:, -1] != 0)),
    }
    return {
        "shape": [int(h), int(w)],
        "nonzero_bbox": bbox,
        "top_colors": top_colors,
        "object_count": len(objects),
        "large_objects": large_objects,
        "edge_contacts": edge_contacts,
    }


def action6_candidates(
    grid: Any,
    *,
    max_candidates: int = 12,
    grid_points_per_axis: int = 3,
) -> list[dict[str, Any]]:
    arr = np.asarray(as_list_grid(grid), dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        return []

    height, width = arr.shape
    objects = summarize_objects(grid, max_objects=max_candidates * 2)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add_point(x: int, y: int, source: str, detail: str) -> None:
        if not (0 <= x < width and 0 <= y < height):
            return
        point = (int(x), int(y))
        if point in seen:
            return
        seen.add(point)
        candidates.append(
            {
                "x": point[0],
                "y": point[1],
                "action_text": f"ACTION6|x={point[0]},y={point[1]}",
                "source": source,
                "detail": detail,
            }
        )

    for obj in objects:
        bbox = obj["bbox"]
        cx = int(round(obj["center"]["x"]))
        cy = int(round(obj["center"]["y"]))
        add_point(cx, cy, "object_center", f"color={obj['color_id']} cells={obj['cell_count']}")
        add_point(bbox["x_min"], bbox["y_min"], "object_corner", "top_left")
        add_point(bbox["x_max"], bbox["y_max"], "object_corner", "bottom_right")

    add_point(width // 2, height // 2, "grid_center", "screen_center")

    xs = _linspace_indices(width, grid_points_per_axis)
    ys = _linspace_indices(height, grid_points_per_axis)
    for y in ys:
        for x in xs:
            add_point(x, y, "grid_probe", "uniform_probe")

    return candidates[:max_candidates]


def _linspace_indices(size: int, steps: int) -> list[int]:
    if steps <= 1:
        return [max(0, size // 2)]
    max_index = max(0, size - 1)
    return sorted(
        {
            min(max_index, round(idx * max_index / (steps - 1)))
            for idx in range(steps)
        }
    )


def display_grid(grid: Any, *, scale: int = 6, title: str | None = None) -> None:
    rgb = grid_to_rgb_array(grid, scale=scale, draw_grid=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()
    except Exception as exc:
        print("display failed:", repr(exc))
        print("grid summary:", summarize_grid(grid))


def normalize_action_name(action_obj: Any) -> str:
    name = getattr(action_obj, "name", None)
    text = str(name) if name else str(action_obj)

    if text.startswith("ACTION"):
        return text

    if match := re.search(r"ACTION\s*([0-9]+)", text):
        return f"ACTION{match.group(1)}"

    stripped = text.strip()
    if stripped.isdigit():
        return f"ACTION{stripped}"

    if match := re.fullmatch(r"[A-Za-z]*\(?([0-9]+)\)?", stripped):
        return f"ACTION{match.group(1)}"

    return text


def available_action_names(raw_frame: Any) -> list[str]:
    actions = list(getattr(raw_frame, "available_actions", []) or [])
    names = [normalize_action_name(action) for action in actions]
    return [name for name in names if name]


def frame_metadata(raw_frame: Any, *, game_id: str | None = None) -> dict[str, Any]:
    grid = latest_grid_from_raw(raw_frame)
    state = getattr(raw_frame, "state", None)
    state_name = getattr(state, "name", str(state)) if state is not None else None
    return {
        "game_id": game_id,
        "state": state_name,
        "available_actions": available_action_names(raw_frame),
        "levels_completed": getattr(raw_frame, "levels_completed", None),
        "win_levels": getattr(raw_frame, "win_levels", None),
        "frame_count": getattr(raw_frame, "frame_count", None),
        "latest_grid": summarize_grid(grid),
    }
