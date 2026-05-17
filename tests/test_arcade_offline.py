from __future__ import annotations

import unittest
from pathlib import Path

from arc_agi3.agents.graph_agent import GraphSearchAgent
from arc_agi3.core.config import AgentConfig
from arc_agi3.envs.arcade_env import ArcadeEnvConfig, ArcadeEnvironment

try:
    import arc_agi  # noqa: F401
except Exception:  # pragma: no cover
    arc_agi = None


@unittest.skipIf(arc_agi is None, "arc-agi optional dependency is not installed")
class ArcadeOfflineTest(unittest.TestCase):
    def test_offline_demo_game_runs(self) -> None:
        env = ArcadeEnvironment(
            ArcadeEnvConfig(
                game_id="ms00",
                mode="offline",
                environments_dir=Path("environment_files"),
                recordings_dir=Path("recordings"),
            )
        )
        try:
            agent = GraphSearchAgent(config=AgentConfig())
            won, steps = agent.run_episode(env)
            self.assertTrue(won)
            self.assertGreaterEqual(steps, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
