from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_paths() -> None:
    repo_root = Path(os.getenv("AGI_ARC_REPO", "")).expanduser()
    candidates = []
    if repo_root:
        candidates.extend([repo_root / "vlm-agi", repo_root / "src"])
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "vlm-agi",
            Path(__file__).resolve().parents[1] / "src",
            Path.cwd() / "vlm-agi",
            Path.cwd() / "src",
            Path("/kaggle/working/agi-arc/vlm-agi"),
            Path("/kaggle/working/agi-arc/src"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)


_ensure_repo_paths()

try:
    from agents.agent import Agent
except Exception:  # pragma: no cover - only absent outside the official runner
    class Agent:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

try:
    from arcengine import FrameData, GameAction, GameState
except Exception:  # pragma: no cover - only absent outside ARC runtime
    FrameData = Any

    class _MissingGameAction:
        RESET = "RESET"

        @staticmethod
        def from_name(name: str) -> "_MissingGameAction":
            raise RuntimeError(f"arcengine is required to create GameAction: {name}")

    class _MissingGameState:
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"
        NOT_PLAYED = "NOT_PLAYED"

    GameAction = _MissingGameAction
    GameState = _MissingGameState

from config import AppConfig
from grid import latest_grid_from_raw
from model import VLMManager
from session import VLMArcRunner, build_default_session_state


class MyAgent(Agent):
    """Official ARC-AGI-3 wrapper for the VLM-based policy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = AppConfig.from_env().with_overrides(
            game_id=os.getenv("ARC_GAME_ID", "kaggle"),
            arc_mode="competition",
            print_vlm_output=os.getenv("ARC_VLM_PRINT_OUTPUT", "0") == "1",
        )
        self.runner = VLMArcRunner(config, VLMManager(config))
        self.awaiting_fresh_episode = True
        self.fail_fast = os.getenv("ARC_VLM_FAIL_FAST", "1") == "1"
        self.max_fallbacks = int(os.getenv("ARC_VLM_MAX_FALLBACKS", "0"))
        self.fallback_count = 0

    def _handle_vlm_failure(
        self,
        exc: Exception,
        latest_frame: FrameData,
    ) -> tuple[list[str], dict[str, Any]]:
        self.fallback_count += 1
        message = f"VLM planning failed (count={self.fallback_count}): {repr(exc)}"
        if self.fail_fast and self.fallback_count > self.max_fallbacks:
            raise RuntimeError(message) from exc
        fallback = self.runner.fallback_action(latest_frame)
        return [fallback], {
            "reasoning": f"fallback due to error: {repr(exc)}",
            "chosen_actions": [fallback],
        }

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _begin_episode_if_needed(self) -> None:
        if self.awaiting_fresh_episode:
            self.runner.session = build_default_session_state()
            self.awaiting_fresh_episode = False

    def _observe_latest_frame(self, latest_frame: FrameData) -> None:
        previous_raw = self.runner.session.get("raw")
        previous_grid = self.runner.session.get("prev_grid")
        if (
            previous_raw is None
            or previous_grid is None
            or self.runner.session.get("last_action") is None
        ):
            return

        current_grid = latest_grid_from_raw(latest_frame)
        transition = self.runner.summarize_transition(
            previous_grid,
            current_grid,
            after_raw=latest_frame,
        )
        self.runner.session["last_transition"] = transition
        clear_queue, _ = self.runner.should_clear_action_queue(
            previous_raw,
            latest_frame,
            transition,
        )
        if clear_queue:
            self.runner.session["action_queue"] = []

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        del frames

        if latest_frame.state is GameState.NOT_PLAYED:
            self.awaiting_fresh_episode = True
            self.runner.session = build_default_session_state()
            return GameAction.RESET

        self._begin_episode_if_needed()
        self._observe_latest_frame(latest_frame)

        if latest_frame.state in {GameState.GAME_OVER, GameState.WIN}:
            self.awaiting_fresh_episode = True
            return GameAction.RESET

        queue = self.runner.session.get("action_queue", [])
        if not queue:
            try:
                action_texts, plan, _ = self.runner.ask_vlm_for_action_sequence(latest_frame)
            except Exception as exc:
                action_texts, plan = self._handle_vlm_failure(exc, latest_frame)
            self.runner.session["action_queue"] = list(action_texts)
            self.runner.session["current_plan"] = plan

        action_text = self.runner.session["action_queue"].pop(0)
        action, payload = self.runner.parse_action(action_text)

        self.runner.session["prev_grid"] = latest_grid_from_raw(latest_frame)
        self.runner.session["last_action"] = action_text
        self.runner.session["raw"] = latest_frame

        if payload:
            action.set_data(payload)
        return action
