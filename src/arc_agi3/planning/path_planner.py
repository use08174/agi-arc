from __future__ import annotations

from collections import deque

from arc_agi3.core.types import Action
from arc_agi3.memory.world_model import Cell, WorldModel


class PathPlanner:
    """BFS planner over learned movement actions and known safe cells."""

    def plan_to_targets(self, world: WorldModel, actions: list[Action], targets: set[Cell]) -> list[Action]:
        start = world.player_pos
        if start is None or not targets:
            return []
        move_actions = [action for action in actions if action.name in world.action_move_vectors]
        if not move_actions:
            return []
        blocked = set(world.known_blocked_cells) | set(world.known_hazard_cells)
        visited: set[Cell] = {start}
        queue: deque[tuple[Cell, list[Action]]] = deque([(start, [])])
        max_depth = 24
        while queue:
            pos, path = queue.popleft()
            if pos in targets and path:
                return path
            if len(path) >= max_depth:
                continue
            for action in move_actions:
                if (pos, action.name) in world.deadly_edges:
                    continue
                target = world.predicted_target(pos, action.name)
                if target is None:
                    continue
                if target in visited or target in blocked:
                    continue
                # Stay roughly inside explored/visible region. Unknown is allowed only near known traversable cells.
                if world.known_traversable_cells and target not in world.known_traversable_cells and not _adjacent_to_any(target, world.known_traversable_cells):
                    continue
                visited.add(target)
                queue.append((target, path + [action]))
        return []

    def plan_to_nearest_item_or_goal(self, world: WorldModel, actions: list[Action]) -> list[Action]:
        visible_items = set(world.visible_item_cells)
        visible_goals = set(world.visible_goal_cells)
        visible_buttons = set(world.visible_button_cells)
        visible_displays = set(world.visible_display_cells)
        state_targets = visible_items or visible_displays or visible_buttons
        if world.should_defer_goal() and state_targets:
            targets = state_targets
        elif world.has_precondition_evidence() and visible_items:
            targets = visible_items
        else:
            targets = state_targets or visible_goals
        return self.plan_to_targets(world, actions, targets)

    def safe_probe_action(self, world: WorldModel, actions: list[Action]) -> Action | None:
        pos = world.player_pos
        if pos is None:
            return None
        if world.should_defer_goal():
            targets = world.visible_item_cells or world.visible_display_cells or world.visible_button_cells or world.visible_goal_cells
        else:
            targets = world.visible_item_cells or world.visible_goal_cells or world.visible_button_cells or world.visible_display_cells
        ranked: list[tuple[int, int, Action]] = []
        for action in actions:
            if action.name in world.action_move_vectors:
                if world.is_unsafe_action(action, pos):
                    continue
                target = world.predicted_target(pos, action.name)
                if target is not None and target not in world.known_blocked_cells and target not in world.known_hazard_cells:
                    before = _nearest_distance(pos, targets)
                    after = _nearest_distance(target, targets)
                    ranked.append((after - before, len(ranked), action))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1]))
            return ranked[0][2]
        return None


def _adjacent_to_any(cell: Cell, cells: set[Cell]) -> bool:
    x, y = cell
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if (nx, ny) in cells:
            return True
    return False


def _nearest_distance(cell: Cell, targets: set[Cell]) -> int:
    if not targets:
        return 0
    x, y = cell
    return min(abs(x - tx) + abs(y - ty) for tx, ty in targets)
