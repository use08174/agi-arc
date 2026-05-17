from __future__ import annotations

from arc_agi3.core.config import ClickExpansionConfig
from arc_agi3.core.types import Action, Frame


class ActionExpander:
    """Expands complex ARC actions into a small candidate set.

    Today only ACTION6 needs coordinates in the official toolkit.
    """

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
        steps = max(1, self.config.grid_points_per_axis)
        xs = self._linspace_indices(width, steps)
        ys = self._linspace_indices(height, steps)
        for y in ys:
            for x in xs:
                points.append((x, y))

        center = (width // 2, height // 2)
        if center not in points:
            points.insert(0, center)

        out: list[Action] = []
        seen: set[tuple[int, int]] = set()
        for x, y in points:
            if (x, y) in seen:
                continue
            seen.add((x, y))
            out.append(Action(name="ACTION6", payload={"x": x, "y": y}))
            if len(out) >= self.config.max_candidates:
                break
        return out

    def _linspace_indices(self, size: int, steps: int) -> list[int]:
        if steps <= 1:
            return [size // 2]
        max_index = max(0, size - 1)
        return sorted(
            {
                min(max_index, round(idx * max_index / (steps - 1)))
                for idx in range(steps)
            }
        )
