from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

from arc_agi3.core.types import Frame, Observation


@dataclass(frozen=True, slots=True)
class LiftedObject:
    object_id: str
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    area: int
    center: tuple[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        min_x, min_y, max_x, max_y = self.bbox
        return max_x - min_x + 1, max_y - min_y + 1


@dataclass(frozen=True, slots=True)
class LiftedState:
    objects: tuple[LiftedObject, ...]
    markers: tuple[LiftedObject, ...]
    candidate_clicks: tuple[tuple[int, int], ...]
    color_histogram: dict[int, int]
    width: int
    height: int
    notes: dict[str, object] = field(default_factory=dict)


class StateLifter:
    """Lift raw frames into lightweight object-centric state."""

    def lift(self, observation: Observation) -> LiftedState:
        frame = observation.frame
        objects = tuple(self._connected_components(frame))
        color_histogram = self._color_histogram(frame)
        markers = tuple(sorted((obj for obj in objects if obj.area <= 9), key=lambda obj: (obj.area, obj.object_id)))
        candidate_clicks = self._candidate_clicks(objects, frame)
        width = len(frame.grid[0]) if frame.grid else 0
        height = len(frame.grid)
        return LiftedState(
            objects=objects,
            markers=markers,
            candidate_clicks=candidate_clicks,
            color_histogram=color_histogram,
            width=width,
            height=height,
            notes=dict(observation.notes),
        )

    def _connected_components(self, frame: Frame) -> Iterable[LiftedObject]:
        grid = frame.grid
        if not grid:
            return []
        height = len(grid)
        width = len(grid[0])
        seen: set[tuple[int, int]] = set()
        objects: list[LiftedObject] = []
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                if color == 0 or (x, y) in seen:
                    continue
                cells: list[tuple[int, int]] = []
                q: deque[tuple[int, int]] = deque([(x, y)])
                seen.add((x, y))
                while q:
                    cx, cy = q.popleft()
                    cells.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if (nx, ny) in seen or grid[ny][nx] != color:
                            continue
                        seen.add((nx, ny))
                        q.append((nx, ny))
                min_x = min(px for px, _ in cells)
                max_x = max(px for px, _ in cells)
                min_y = min(py for _, py in cells)
                max_y = max(py for _, py in cells)
                center = (round(sum(px for px, _ in cells) / len(cells)), round(sum(py for _, py in cells) / len(cells)))
                objects.append(
                    LiftedObject(
                        object_id=f"obj:{color}:{min_x},{min_y}:{len(cells)}",
                        color=int(color),
                        cells=tuple(sorted(cells)),
                        bbox=(min_x, min_y, max_x, max_y),
                        area=len(cells),
                        center=center,
                    )
                )
        return sorted(objects, key=lambda obj: (-obj.area, obj.color, obj.bbox))

    def _color_histogram(self, frame: Frame) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for row in frame.grid:
            counts.update(int(cell) for cell in row)
        return dict(counts)

    def _candidate_clicks(self, objects: tuple[LiftedObject, ...], frame: Frame) -> tuple[tuple[int, int], ...]:
        width = len(frame.grid[0]) if frame.grid else 0
        height = len(frame.grid)
        candidates: list[tuple[int, int]] = []
        compact = [obj for obj in objects if obj.area <= 64 and not self._touches_edge(obj, width, height)]
        large = [obj for obj in objects if obj.area > 64 and not self._touches_edge(obj, width, height)]
        ordered = compact + large + [obj for obj in objects if self._touches_edge(obj, width, height)]
        for obj in ordered[:20]:
            candidates.extend(self._object_click_points(obj))
        if width and height:
            candidates.extend(
                [
                    (0, 0),
                    (width - 1, 0),
                    (0, height - 1),
                    (width - 1, height - 1),
                    (width // 2, height // 2),
                ]
            )
        deduped: list[tuple[int, int]] = []
        for xy in candidates:
            if width and height and not (0 <= xy[0] < width and 0 <= xy[1] < height):
                continue
            if xy not in deduped:
                deduped.append(xy)
        return tuple(deduped[:40])

    def _object_click_points(self, obj: LiftedObject) -> list[tuple[int, int]]:
        min_x, min_y, max_x, max_y = obj.bbox
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        return [
            obj.center,
            (min_x, min_y),
            (max_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (mid_x, min_y),
            (mid_x, max_y),
            (min_x, mid_y),
            (max_x, mid_y),
        ]

    def _touches_edge(self, obj: LiftedObject, width: int, height: int) -> bool:
        min_x, min_y, max_x, max_y = obj.bbox
        return min_x <= 0 or min_y <= 0 or max_x >= width - 1 or max_y >= height - 1
