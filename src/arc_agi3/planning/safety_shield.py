from __future__ import annotations

from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory


class SafetyShield:
    """Final veto layer before env.step."""

    def validate_or_replace(
        self,
        action: Action,
        observation: Observation,
        actions: list[Action],
        game_memory: GameMemory,
    ) -> Action:
        world = game_memory.world_model
        if not self.is_safe(action, observation, game_memory):
            non_meta_candidates = [
                candidate
                for candidate in actions
                if candidate.name not in game_memory.restart_like_action_names
                and candidate.key not in game_memory.restart_like_action_keys
                and candidate.name not in game_memory.undo_like_action_names
                and candidate.key not in game_memory.undo_like_action_keys
            ]
            for candidate in non_meta_candidates:
                if self.is_safe(candidate, observation, game_memory):
                    return candidate
            if non_meta_candidates:
                return non_meta_candidates[0]
        return action

    def is_safe(self, action: Action, observation: Observation, game_memory: GameMemory) -> bool:
        world = game_memory.world_model
        if action.name in game_memory.restart_like_action_names or action.key in game_memory.restart_like_action_keys:
            return False
        if action.name in game_memory.undo_like_action_names or action.key in game_memory.undo_like_action_keys:
            return False
        if action.key in game_memory.dangerous_action_keys:
            return False
        if world.is_unsafe_action(action):
            return False
        profile = game_memory.semantic_profile(action.name, action.key)
        if action.payload and profile.terminal_losses > 0:
            return False
        if profile.uses >= 3 and profile.changed_uses == 0 and profile.reward_total <= 0:
            return False
        return True
