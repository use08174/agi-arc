from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GameStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        if not self.payload:
            return self.name
        items = ",".join(f"{key}={self.payload[key]}" for key in sorted(self.payload))
        return f"{self.name}|{items}"


@dataclass(slots=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    status: GameStatus = GameStatus.IN_PROGRESS
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation:
    state_key: str
    frame: Frame
    changed: bool
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Transition:
    from_state: str
    action: Action
    to_state: str
    changed: bool
    reward_delta: float = 0.0
    terminal: bool = False
    won: bool = False


@dataclass(slots=True)
class PlanStep:
    action: Action
    reason: str


@dataclass(slots=True)
class RankedAction:
    action: Action
    score: float
    reason: str


@dataclass(slots=True)
class RuleHypothesis:
    summary: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LLMDecisionTrace:
    step_idx: int
    state_key: str
    prompt: str
    raw_response: str
    ranked_actions: list[RankedAction] = field(default_factory=list)
    hypotheses: list[RuleHypothesis] = field(default_factory=list)


DEFAULT_ACTIONS: tuple[Action, ...] = tuple(
    Action(name=f"ACTION{i}") for i in range(1, 7)
)
