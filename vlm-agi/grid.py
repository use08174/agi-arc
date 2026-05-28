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
