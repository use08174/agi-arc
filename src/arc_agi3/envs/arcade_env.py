from __future__ import annotations

from dataclasses import dataclass
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
        self._last_frame = next_frame
        return StepResult(frame=next_frame, reward_delta=float(reward_delta), done=done)

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

    def _map_status(self, state: Any) -> GameStatus:
        name = getattr(state, "name", str(state))
        if name == "WIN":
            return GameStatus.WIN
        if name == "GAME_OVER":
            return GameStatus.GAME_OVER
        if name == "NOT_PLAYED":
            return GameStatus.NOT_STARTED
        return GameStatus.IN_PROGRESS
