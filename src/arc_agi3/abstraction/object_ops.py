from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObjectOp(str, Enum):
    PROBE_MOVE = "probe_move"
    PROBE_MODE = "probe_mode"
    PROBE_CLICK = "probe_click"
    PROBE_TRANSFORM = "probe_transform"
    PROBE_ALIGNMENT = "probe_alignment"
    EXPLOIT_KNOWN_EFFECT = "exploit_known_effect"


@dataclass(frozen=True, slots=True)
class AbstractIntent:
    """Object-level action request before lowering to ACTION1..ACTION7."""

    op: ObjectOp
    target: Any = None
    priority: float = 0.0
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def simple_action_family(action_name: str, action_key: str) -> str:
    if "|x=" in action_key:
        return "click"
    if action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        return "simple_axis"
    if action_name in {"ACTION5", "ACTION7"}:
        return "mode_or_tool"
    if action_name == "ACTION6":
        return "complex"
    return "simple"
