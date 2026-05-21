from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisKind(str, Enum):
    MOVEMENT_AXIS = "movement_axis"
    MODE_SWITCH = "mode_switch"
    EDITABLE_REGION = "editable_region"
    OBJECT_TRANSFORM = "object_transform"
    ALIGNMENT_GOAL = "alignment_goal"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Hypothesis:
    kind: HypothesisKind
    summary: str
    confidence: float = 0.0
    uncertainty: float = 1.0
    target: Any = None
    evidence: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.confidence + (0.5 * self.uncertainty)
