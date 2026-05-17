from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from arc_agi3.core.types import Action, Frame


@dataclass(slots=True)
class StepResult:
    frame: Frame
    reward_delta: float
    done: bool
    won: bool = False


class ArcEnvironment(Protocol):
    def reset(self) -> Frame:
        ...

    def step(self, action: Action) -> StepResult:
        ...

    def valid_actions(self) -> list[Action]:
        ...

    def close(self) -> None:
        ...
