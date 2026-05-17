from __future__ import annotations

from arc_agi3.core.types import Action, DEFAULT_ACTIONS, Frame, GameStatus
from arc_agi3.envs.base import StepResult


class MockArcEnvironment:
    """Tiny environment for smoke-testing the agent architecture.

    The agent wins by moving the marker to the bottom-right corner of a 3x3 grid.
    ACTION1: up, ACTION2: down, ACTION3: left, ACTION4: right
    Remaining actions are no-ops.
    """

    def __init__(self) -> None:
        self._x = 0
        self._y = 0

    def reset(self) -> Frame:
        self._x = 0
        self._y = 0
        return self._frame()

    def valid_actions(self) -> list[Action]:
        return list(DEFAULT_ACTIONS)

    def step(self, action: Action) -> StepResult:
        before = (self._x, self._y)
        if action.name == "ACTION1":
            self._y = max(0, self._y - 1)
        elif action.name == "ACTION2":
            self._y = min(2, self._y + 1)
        elif action.name == "ACTION3":
            self._x = max(0, self._x - 1)
        elif action.name == "ACTION4":
            self._x = min(2, self._x + 1)
        after = (self._x, self._y)
        changed = before != after
        done = after == (2, 2)
        frame = self._frame(status=GameStatus.WIN if done else GameStatus.IN_PROGRESS)
        return StepResult(frame=frame, reward_delta=1.0 if done else 0.0, done=done)

    def _frame(self, status: GameStatus = GameStatus.IN_PROGRESS) -> Frame:
        rows = []
        for y in range(3):
            row = []
            for x in range(3):
                row.append(1 if (x, y) == (self._x, self._y) else 0)
            rows.append(tuple(row))
        return Frame(grid=tuple(rows), status=status)

    def close(self) -> None:
        return None
