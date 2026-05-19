from __future__ import annotations

import unittest

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.types import Action, Frame, GameStatus
from arc_agi3.envs.base import StepResult


class LosingShortcutEnvironment:
    """A tiny environment where a tempting direct move loses immediately.

    ACTION4 goes straight to a losing terminal.
    ACTION2 then ACTION4 is the winning route.
    """

    def __init__(self) -> None:
        self._setup_done = False
        self._done = False

    def reset(self) -> Frame:
        self._setup_done = False
        self._done = False
        return self._frame(GameStatus.IN_PROGRESS)

    def valid_actions(self) -> list[Action]:
        return [Action(name="ACTION4"), Action(name="ACTION2")]

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(frame=self._frame(GameStatus.GAME_OVER), reward_delta=0.0, done=True, won=False)
        if action.name == "ACTION4":
            if self._setup_done:
                self._done = True
                return StepResult(frame=self._frame(GameStatus.WIN), reward_delta=1.0, done=True, won=True)
            self._done = True
            return StepResult(frame=self._frame(GameStatus.GAME_OVER), reward_delta=0.0, done=True, won=False)
        if action.name == "ACTION2":
            self._setup_done = True
            return StepResult(frame=self._frame(GameStatus.IN_PROGRESS), reward_delta=0.0, done=False, won=False)
        return StepResult(frame=self._frame(GameStatus.IN_PROGRESS), reward_delta=0.0, done=False, won=False)

    def close(self) -> None:
        return None

    def _frame(self, status: GameStatus) -> Frame:
        marker = 2 if self._setup_done else 1
        return Frame(grid=((marker, 0), (0, 0)), status=status)


class TerminalAvoidanceTest(unittest.TestCase):
    def test_agent_learns_to_avoid_losing_terminal(self) -> None:
        env = LosingShortcutEnvironment()
        agent = GraphSearchAgent()

        won1, _ = agent.run_episode(env)
        won2, _ = agent.run_episode(env)
        self.assertTrue(won1 or won2)


if __name__ == "__main__":
    unittest.main()
