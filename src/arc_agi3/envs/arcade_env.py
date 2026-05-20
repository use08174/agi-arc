from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from arc_agi3.core.config import ClickExpansionConfig
from arc_agi3.core.logging_utils import get_logger
from arc_agi3.core.types import Action, Frame, GameStatus
from arc_agi3.envs.action_expander import ActionExpander
from arc_agi3.envs.base import StepResult

try:
    from arc_agi import Arcade, OperationMode
    from arc_agi.scorecard import EnvironmentScorecard
    from arcengine import FrameDataRaw, GameAction, GameState
except Exception:  # pragma: no cover - optional dependency
    Arcade = None
    OperationMode = None
    EnvironmentScorecard = Any
    FrameDataRaw = Any
    GameAction = None
    GameState = None


@dataclass(slots=True)
class ArcadeEnvConfig:
    game_id: str
    mode: str = "offline"
    environments_dir: Path = Path("environment_files")
    recordings_dir: Path = Path("recordings")
    render_mode: str | None = None
    save_recording: bool = False
    include_frame_data: bool = True
    arc_api_key: str = ""
    arc_base_url: str = "https://three.arcprize.org"
    source_url: str | None = None
    tags: list[str] | None = None


class ArcadeEnvironment:
    def __init__(
        self,
        config: ArcadeEnvConfig,
        click_config: ClickExpansionConfig | None = None,
    ) -> None:
        if Arcade is None or OperationMode is None or GameAction is None:
            raise RuntimeError(
                "arc-agi is not installed. Install with `pip install -e \".[arc]\"`."
            )

        self.config = config
        self.logger = get_logger("arc_agi3.arcade")
        self.arcade = Arcade(
            arc_api_key=config.arc_api_key,
            arc_base_url=config.arc_base_url,
            operation_mode=self._operation_mode(config.mode),
            environments_dir=str(config.environments_dir),
            recordings_dir=str(config.recordings_dir),
        )
        self.scorecard_id = self.arcade.create_scorecard(
            source_url=config.source_url,
            tags=config.tags or ["arc-agi3-starter"],
        )
        self.wrapper = self.arcade.make(
            config.game_id,
            scorecard_id=self.scorecard_id,
            save_recording=config.save_recording,
            include_frame_data=config.include_frame_data,
            render_mode=config.render_mode,
        )
        if self.wrapper is None:
            raise RuntimeError(f"Failed to create ARC environment for {config.game_id}")
        self.expander = ActionExpander(click_config or ClickExpansionConfig())
        self._last_frame: Frame | None = None
        self.last_closed_scorecard: EnvironmentScorecard | None = None
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

    def reset(self) -> Frame:
        raw = self.wrapper.reset()
        if raw is None:
            raise RuntimeError("ARC environment reset returned no frame")
        self._last_frame = self._convert_frame(raw)
        return self._last_frame

    def step(self, action: Action) -> StepResult:
        tool_action = GameAction.from_name(action.name)
        raw = self.wrapper.step(tool_action, data=action.payload or None)
        if raw is None:
            raise RuntimeError(f"ARC environment step failed for {action.key}")
        next_frame = self._convert_frame(raw)
        reward_delta = next_frame.info.get("levels_completed", 0) - (
            self._last_frame.info.get("levels_completed", 0) if self._last_frame else 0
        )
        done = next_frame.status in {GameStatus.WIN, GameStatus.GAME_OVER}
        won = next_frame.status == GameStatus.WIN
        self._last_frame = next_frame
        return StepResult(
            frame=next_frame,
            reward_delta=float(reward_delta),
            done=done,
            won=won,
        )

    def valid_actions(self) -> list[Action]:
        names = [action.name for action in self.wrapper.action_space]
        return self.expander.expand(names, self._last_frame)

    def list_available_games(self) -> list[str]:
        return sorted(env.game_id for env in self.arcade.get_environments())

    def close(self) -> None:
        try:
            self.last_closed_scorecard = self.arcade.close_scorecard(self.scorecard_id)
        except Exception:
            self.logger.exception("Failed to close scorecard cleanly")

    def get_scorecard(self) -> EnvironmentScorecard | None:
        return self.arcade.get_scorecard(self.scorecard_id)

    def scorecard_url(self) -> str:
        return f"{self.config.arc_base_url.rstrip('/')}/scorecards/{self.scorecard_id}"

    def has_online_replays(self) -> bool:
        return self.config.mode in {"online", "competition"}

    def _operation_mode(self, mode: str) -> OperationMode:
        try:
            return OperationMode(mode.lower())
        except Exception as exc:
            raise ValueError(f"Unsupported ARC mode: {mode}") from exc

    def _convert_frame(self, raw: FrameDataRaw) -> Frame:
        self._maybe_print_frame_debug(raw)
        final_grid = self._extract_final_grid(raw)
        status = self._map_status(raw.state)
        info = {
            "guid": getattr(raw, "guid", None),
            "levels_completed": getattr(raw, "levels_completed", 0),
            "win_levels": getattr(raw, "win_levels", 0),
            "available_actions": getattr(raw, "available_actions", []),
            "frame_count": len(getattr(raw, "frame", []) or []),
            "tool_state": getattr(raw.state, "name", str(raw.state)),
        }
        return Frame(grid=final_grid, status=status, info=info)

    def _maybe_print_frame_debug(self, raw: FrameDataRaw) -> None:
        if not self.debug_frames:
            return
        if self.debug_frame_limit and self._debug_frame_counter >= self.debug_frame_limit:
            return
        rendered_frames = getattr(raw, "frame", []) or []
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
        final_grid = normalized[-1] if normalized else []
        preview = self._grid_preview(final_grid)
        available_actions = [getattr(action, "name", str(action)) for action in list(getattr(raw, "available_actions", []) or [])]
        state_name = getattr(raw.state, "name", str(raw.state))
        guid = getattr(raw, "guid", None)
        levels_completed = getattr(raw, "levels_completed", 0)
        win_levels = getattr(raw, "win_levels", 0)
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

    def _extract_final_grid(self, raw: FrameDataRaw) -> tuple[tuple[int, ...], ...]:
        frames = getattr(raw, "frame", [])
        if not frames:
            return tuple()
        final = frames[-1]
        if hasattr(final, "tolist"):
            data = final.tolist()
        else:
            data = final
        return tuple(tuple(int(cell) for cell in row) for row in data)

    def _normalize_grid(self, frame: Any) -> list[list[int]]:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        return [[int(cell) for cell in row] for row in (frame or [])]

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

    def _map_status(self, state: Any) -> GameStatus:
        name = getattr(state, "name", str(state))
        if name == "WIN":
            return GameStatus.WIN
        if name == "GAME_OVER":
            return GameStatus.GAME_OVER
        if name == "NOT_PLAYED":
            return GameStatus.NOT_STARTED
        return GameStatus.IN_PROGRESS
