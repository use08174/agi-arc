from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_src_on_path() -> None:
    candidates = []
    env_repo = os.getenv("AGI_ARC_REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "src")
    candidates.extend(
        [
            Path("/kaggle/working/agi-arc/src"),
            Path("/kaggle/input/agi-arc/src"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
            return


_ensure_repo_src_on_path()

from agents.agent import Agent
from arc_agi3.core.config import AgentConfig
from arc_agi3.integrations.official_agent import OfficialAgentPolicy
from arcengine import FrameData, GameAction, GameState


class MyAgent(Agent):
    """Official ARC-AGI-3 agent wrapper used by the offline evaluation notebook."""

    MAX_ACTIONS = int(os.getenv("ARC_AGI3_MAX_ACTIONS", "500"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        agent_config = AgentConfig()
        agent_config.budget.max_steps_per_level = int(os.getenv("ARC_AGI3_MAX_STEPS", "128"))
        agent_config.budget.explore_phase_steps = int(os.getenv("ARC_AGI3_EXPLORE_STEPS", "24"))
        agent_config.budget.novelty_patience_steps = int(os.getenv("ARC_AGI3_NOVELTY_PATIENCE_STEPS", "6"))
        agent_config.budget.revisit_limit = int(os.getenv("ARC_AGI3_REVISIT_LIMIT", "3"))
        self.policy = OfficialAgentPolicy(agent_config=agent_config)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return self.policy.choose_action(latest_frame)
