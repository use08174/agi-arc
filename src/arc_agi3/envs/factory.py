from __future__ import annotations

from arc_agi3.core.config import AgentConfig, RuntimeConfig
from arc_agi3.envs.arcade_env import ArcadeEnvConfig, ArcadeEnvironment
from arc_agi3.envs.mock_env import MockArcEnvironment


def build_environment(runtime: RuntimeConfig, agent_config: AgentConfig):
    if runtime.backend == "mock":
        return MockArcEnvironment()

    if runtime.backend == "arcade":
        return ArcadeEnvironment(
            ArcadeEnvConfig(
                game_id=runtime.game_id,
                mode=runtime.mode,
                environments_dir=runtime.environments_dir,
                recordings_dir=runtime.recordings_dir,
                render_mode=runtime.render_mode,
                save_recording=runtime.save_recording,
                include_frame_data=runtime.include_frame_data,
            ),
            click_config=agent_config.click_expansion,
        )

    raise ValueError(f"Unknown backend: {runtime.backend}")
