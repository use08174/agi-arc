from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from arc_agi3.core.types import Action, Frame


def _grid_to_list(grid: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(row) for row in grid]


@dataclass(slots=True)
class EpisodeTaskExporter:
    """Export an interactive episode as a pseudo ARC task."""

    task_id: str
    max_train_pairs: int = 8
    _frames: list[Frame] = field(default_factory=list)
    _actions: list[Action] = field(default_factory=list)

    def reset(self, frame: Frame) -> None:
        self._frames = [frame]
        self._actions = []

    def observe(self, action: Action | None, frame: Frame) -> None:
        if action is not None:
            self._actions.append(action)
        self._frames.append(frame)

    def build_problem(self) -> dict:
        if not self._frames:
            return {"train": [], "test": []}

        train_examples: list[dict] = []
        start_idx = max(0, len(self._frames) - 1 - self.max_train_pairs)
        for idx in range(start_idx, len(self._frames) - 1):
            train_examples.append(
                {
                    "input": _grid_to_list(self._frames[idx].grid),
                    "output": _grid_to_list(self._frames[idx + 1].grid),
                }
            )
        if not train_examples:
            only = _grid_to_list(self._frames[-1].grid)
            train_examples.append({"input": only, "output": only})

        return {
            "train": train_examples,
            "test": [{"input": _grid_to_list(self._frames[-1].grid)}],
        }

    def write_problem_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.build_problem(), handle)
        return output_path

    def metadata(self) -> dict[str, int]:
        return {
            "observed_frames": len(self._frames),
            "observed_actions": len(self._actions),
            "train_pairs": max(1, len(self._frames) - 1),
        }
