from __future__ import annotations

from arc_agi3.core.types import Observation
from arc_agi3.envs.base import ArcEnvironment
from arc_agi3.memory.game_memory import GameMemory


class EnvUnderstandingAgent:
    """Very lightweight first-pass world model initializer.

    This does not spend extra actions by default. It registers semantic map notes
    from the first frame. The controlled probing should happen through normal
    safe planner/explorer choices so action budget is not wasted blindly.
    """

    def inspect(self, env: ArcEnvironment, initial_observation: Observation, game_memory: GameMemory) -> None:
        game_memory.world_model.update_from_observation(initial_observation.notes)

