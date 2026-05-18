from __future__ import annotations

from typing import Any

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig, LLMConfig
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

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.runtime = GraphSearchAgent(config=agent_config, llm_config=llm_config)
        self.expander = ActionExpander(self.runtime.config.click_expansion)
        self.awaiting_fresh_episode = True

    def close(self) -> None:
        self.runtime.close()

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
            getattr(action, "name", str(action))
            for action in list(getattr(latest_frame, "available_actions", []) or [])
        ]
        if not names:
            names = [action.name for action in GameAction if action is not GameAction.RESET]
        return self.expander.expand(names, frame)

    def _convert_frame(self, latest_frame: FrameData) -> Frame:
        rendered_frames = getattr(latest_frame, "frame", []) or []
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
