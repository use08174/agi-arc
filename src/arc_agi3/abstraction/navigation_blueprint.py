from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from arc_agi3.abstraction.state_lifter import LiftedObject, LiftedState


Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class NavigationBlueprint:
    walkable_color: int
    walkable_count: int
    component_count: int
    largest_component_size: int
    player_candidate: Cell | None
    target_candidates: tuple[Cell, ...]
    nearest_target: Cell | None
    path: tuple[Cell, ...]
    blocked_target_count: int
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def has_path(self) -> bool:
        return bool(self.path)

    @property
    def next_step_delta(self) -> Cell | None:
        if len(self.path) < 2:
            return None
        x0, y0 = self.path[0]
        x1, y1 = self.path[1]
        return x1 - x0, y1 - y0

    def to_notes(self) -> dict[str, object]:
        return {
            "nav_walkable_color": self.walkable_color,
            "nav_walkable_count": self.walkable_count,
            "nav_component_count": self.component_count,
            "nav_largest_component_size": self.largest_component_size,
            "nav_player_candidate": self.player_candidate,
            "nav_target_candidates": list(self.target_candidates[:12]),
            "nav_nearest_target": self.nearest_target,
            "nav_path_length": max(0, len(self.path) - 1),
            "nav_next_step_delta": self.next_step_delta,
            "nav_has_path": self.has_path,
            "nav_blocked_target_count": self.blocked_target_count,
        }


class NavigationBlueprintBuilder:
    """Infer traversable topology from background space and compact markers."""

    def build(self, state: LiftedState) -> NavigationBlueprint:
        grid = state.notes.get("_raw_grid")
        if not isinstance(grid, tuple):
            return self._empty(state)
        walkable_color = self._walkable_color(state)
        walkable = {
            (x, y)
            for y, row in enumerate(grid)
            for x, value in enumerate(row)
            if int(value) == walkable_color
        }
        components = self._components(walkable)
        player = self._player_candidate(state, walkable)
        targets = self._target_candidates(state, walkable, player)
        path: list[Cell] = []
        nearest_target = None
        blocked_target_count = 0
        if player is not None and targets:
            path, nearest_target = self._path_to_nearest(player, targets, walkable)
            blocked_target_count = sum(1 for target in targets if target not in walkable)
        return NavigationBlueprint(
            walkable_color=walkable_color,
            walkable_count=len(walkable),
            component_count=len(components),
            largest_component_size=max((len(component) for component in components), default=0),
            player_candidate=player,
            target_candidates=tuple(targets),
            nearest_target=nearest_target,
            path=tuple(path),
            blocked_target_count=blocked_target_count,
            notes=dict(state.notes),
        )

    def _empty(self, state: LiftedState) -> NavigationBlueprint:
        return NavigationBlueprint(
            walkable_color=0,
            walkable_count=0,
            component_count=0,
            largest_component_size=0,
            player_candidate=None,
            target_candidates=(),
            nearest_target=None,
            path=(),
            blocked_target_count=0,
            notes=dict(state.notes),
        )

    def _walkable_color(self, state: LiftedState) -> int:
        if 0 in state.color_histogram:
            return 0
        return max(state.color_histogram.items(), key=lambda item: item[1])[0] if state.color_histogram else 0

    def _player_candidate(self, state: LiftedState, walkable: set[Cell]) -> Cell | None:
        semantic_pos = state.notes.get("semantic_player_pos")
        if isinstance(semantic_pos, tuple) and len(semantic_pos) == 2:
            return self._nearest_walkable((int(semantic_pos[0]), int(semantic_pos[1])), walkable)
        compact = [obj for obj in state.objects if obj.area <= 16 and not self._touches_edge(obj, state.width, state.height)]
        if not compact:
            compact = [obj for obj in state.objects if obj.area <= 32]
        if not compact:
            return None
        compact.sort(key=lambda obj: (obj.area, self._edge_distance(obj, state.width, state.height), obj.center))
        return self._nearest_walkable(compact[0].center, walkable)

    def _target_candidates(self, state: LiftedState, walkable: set[Cell], player: Cell | None) -> list[Cell]:
        targets: list[Cell] = []
        candidates = [obj for obj in state.markers if obj.area <= 16]
        candidates.extend(obj for obj in state.objects if 2 <= obj.area <= 24 and obj not in candidates)
        for obj in candidates[:16]:
            target = self._nearest_walkable(obj.center, walkable)
            if player is not None and target is not None and abs(target[0] - player[0]) + abs(target[1] - player[1]) <= 1:
                continue
            if target is not None and target not in targets:
                targets.append(target)
        return targets[:16]

    def _nearest_walkable(self, start: Cell, walkable: set[Cell]) -> Cell | None:
        if start in walkable:
            return start
        sx, sy = start
        best: tuple[int, Cell] | None = None
        for cell in walkable:
            distance = abs(cell[0] - sx) + abs(cell[1] - sy)
            if best is None or distance < best[0]:
                best = (distance, cell)
        if best is None or best[0] > 3:
            return None
        return best[1]

    def _path_to_nearest(self, start: Cell, targets: list[Cell], walkable: set[Cell]) -> tuple[list[Cell], Cell | None]:
        target_set = set(targets)
        queue: deque[tuple[Cell, list[Cell]]] = deque([(start, [start])])
        seen = {start}
        while queue:
            cell, path = queue.popleft()
            if cell in target_set and cell != start:
                return path, cell
            for nxt in self._neighbors(cell):
                if nxt in seen or nxt not in walkable:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return [], None

    def _components(self, walkable: set[Cell]) -> list[set[Cell]]:
        remaining = set(walkable)
        components: list[set[Cell]] = []
        while remaining:
            start = remaining.pop()
            component = {start}
            queue = deque([start])
            while queue:
                cell = queue.popleft()
                for nxt in self._neighbors(cell):
                    if nxt not in remaining:
                        continue
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
            components.append(component)
        return components

    def _neighbors(self, cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
        x, y = cell
        return (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)

    def _touches_edge(self, obj: LiftedObject, width: int, height: int) -> bool:
        min_x, min_y, max_x, max_y = obj.bbox
        return min_x <= 0 or min_y <= 0 or max_x >= width - 1 or max_y >= height - 1

    def _edge_distance(self, obj: LiftedObject, width: int, height: int) -> int:
        x, y = obj.center
        return min(x, y, max(0, width - 1 - x), max(0, height - 1 - y))
