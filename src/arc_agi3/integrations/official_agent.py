from __future__ import annotations

import os
from typing import Any

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Frame, GameStatus
from arc_agi3.envs.action_expander import ActionExpander

try:
    from arcengine import FrameData, GameAction, GameState
except Exception:  # pragma: no cover - optional dependency
    FrameData = Any
    GameAction = None
    GameState = None


class OfficialAgentPolicy:
    """Bridge the official frame-by-frame API to our graph agent runtime."""

    def __init__(self, agent_config: AgentConfig | None = None) -> None:
        self.runtime = GraphSearchAgent(config=agent_config)
        self.expander = ActionExpander(self.runtime.config.click_expansion)
        self.awaiting_fresh_episode = True
        self.debug_frames = os.getenv("ARC_AGI3_DEBUG_FRAMES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.debug_frame_limit = max(
            0,
            int(os.getenv("ARC_AGI3_DEBUG_FRAME_LIMIT", "40") or 0),
        )
        self._debug_frame_counter = 0

    def close(self) -> None:
        return

    def choose_action(self, latest_frame: FrameData) -> Any:
        if GameAction is None or GameState is None:
            raise RuntimeError("arcengine is required for the official agent adapter")

        frame = self._convert_frame(latest_frame)
        if latest_frame.state is GameState.NOT_PLAYED:
            self.awaiting_fresh_episode = True
            return GameAction.RESET

        if self.awaiting_fresh_episode:
            self.runtime.begin_external_episode(frame)
            self.awaiting_fresh_episode = False
        elif latest_frame.state in {GameState.GAME_OVER, GameState.WIN}:
            self.runtime.observe_external_frame(frame)
            if latest_frame.state is GameState.GAME_OVER:
                self.awaiting_fresh_episode = True
                return GameAction.RESET
            return GameAction.RESET

        internal_actions = self._available_actions(latest_frame, frame)
        internal_action = self.runtime.choose_external_action(frame, internal_actions)
        return self._to_game_action(internal_action)

    def _available_actions(self, latest_frame: FrameData, frame: Frame) -> list[Action]:
        names = [
            normalized
            for action in list(getattr(latest_frame, "available_actions", []) or [])
            if (normalized := normalize_action_name(action)) is not None
        ]
        if not names:
            names = [action.name for action in GameAction if action is not GameAction.RESET]
        return self.expander.expand(names, frame)

    def _convert_frame(self, latest_frame: FrameData) -> Frame:
        rendered_frames = getattr(latest_frame, "frame", []) or []
        self._maybe_print_frame_debug(latest_frame, rendered_frames)
        final = rendered_frames[-1] if rendered_frames else []
        if hasattr(final, "tolist"):
            final = final.tolist()
        grid = tuple(tuple(int(cell) for cell in row) for row in final)
        return Frame(
            grid=grid,
            status=self._status(latest_frame.state),
            info={
                "guid": getattr(latest_frame, "guid", None),
                "levels_completed": getattr(latest_frame, "levels_completed", 0),
                "win_levels": getattr(latest_frame, "win_levels", 0),
                "available_actions": list(getattr(latest_frame, "available_actions", []) or []),
                "frame_count": len(rendered_frames),
                "tool_state": getattr(latest_frame.state, "name", str(latest_frame.state)),
            },
        )

    def _maybe_print_frame_debug(self, latest_frame: FrameData, rendered_frames: list[Any]) -> None:
        if not self.debug_frames:
            return
        if self.debug_frame_limit and self._debug_frame_counter >= self.debug_frame_limit:
            return
        normalized = [self._normalize_grid(frame) for frame in rendered_frames]
        frame_count = len(normalized)
        shapes = [f"{len(grid)}x{len(grid[0])}" if grid and grid[0] else "0x0" for grid in normalized]
        deltas_between = [
            self._grid_delta_count(normalized[index - 1], normalized[index])
            for index in range(1, frame_count)
        ]
        first_last_delta = (
            self._grid_delta_count(normalized[0], normalized[-1]) if frame_count >= 2 else 0
        )
        grid = normalized[-1] if normalized else []
        preview = self._grid_preview(grid)
        available_actions = [normalize_action_name(action) for action in list(getattr(latest_frame, "available_actions", []) or [])]
        state_name = getattr(latest_frame.state, "name", str(latest_frame.state))
        guid = getattr(latest_frame, "guid", None)
        levels_completed = getattr(latest_frame, "levels_completed", 0)
        win_levels = getattr(latest_frame, "win_levels", 0)
        print(
            "input_frame "
            f"idx={self._debug_frame_counter} "
            f"guid={guid} "
            f"state={state_name} "
            f"levels_completed={levels_completed} "
            f"win_levels={win_levels} "
            f"frame_count={frame_count} "
            f"frame_shapes={shapes} "
            f"deltas_between={deltas_between} "
            f"first_last_delta={first_last_delta} "
            f"available_actions={available_actions} "
            f"final_grid_preview={preview}"
        )
        self._debug_frame_counter += 1

    def _normalize_grid(self, frame: Any) -> list[list[int]]:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        return [
            [int(cell) for cell in row]
            for row in (frame or [])
        ]

    def _grid_delta_count(self, before: list[list[int]], after: list[list[int]]) -> int:
        if not before and not after:
            return 0
        max_rows = max(len(before), len(after))
        max_cols = max(
            max((len(row) for row in before), default=0),
            max((len(row) for row in after), default=0),
        )
        changed = 0
        for y in range(max_rows):
            before_row = before[y] if y < len(before) else []
            after_row = after[y] if y < len(after) else []
            for x in range(max_cols):
                before_value = before_row[x] if x < len(before_row) else None
                after_value = after_row[x] if x < len(after_row) else None
                if before_value != after_value:
                    changed += 1
        return changed

    def _grid_preview(self, grid: list[list[int]], rows: int = 4, cols: int = 12) -> str:
        if not grid:
            return "[]"
        preview_rows = []
        for row in grid[:rows]:
            head = row[:cols]
            suffix = "..." if len(row) > cols else ""
            preview_rows.append("[" + ",".join(str(cell) for cell in head) + suffix + "]")
        if len(grid) > rows:
            preview_rows.append("...")
        return " ".join(preview_rows)

    def _status(self, state: Any) -> GameStatus:
        name = getattr(state, "name", str(state))
        if name == "WIN":
            return GameStatus.WIN
        if name == "GAME_OVER":
            return GameStatus.GAME_OVER
        if name == "NOT_PLAYED":
            return GameStatus.NOT_STARTED
        return GameStatus.IN_PROGRESS

    def _to_game_action(self, action: Action) -> Any:
        tool_action = GameAction.from_name(action.name)
        if action.payload:
            tool_action.set_data(dict(action.payload))
        return tool_action


def normalize_action_name(action: Any) -> str | None:
    """Normalize official action encodings into internal ACTION* names."""
    name = getattr(action, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(action, int):
        return "RESET" if action == 0 else f"ACTION{action}"
    text = str(action).strip()
    if not text:
        return None
    if text.isdigit():
        value = int(text)
        return "RESET" if value == 0 else f"ACTION{value}"
    upper = text.upper()
    if upper == "RESET" or upper.startswith("ACTION"):
        return upper
    return None
