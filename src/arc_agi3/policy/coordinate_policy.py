from __future__ import annotations

from dataclasses import dataclass

from arc_agi3.abstraction.object_ops import AbstractIntent
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory


@dataclass(frozen=True, slots=True)
class CoordinateCandidate:
    action: Action
    role: str
    score_key: tuple[float, int, int, int, str]


class CoordinateInteractionPolicy:
    """Phase-aware coordinate action policy with role/cooldown memory."""

    def rank(
        self,
        *,
        actions: list[Action],
        intent: AbstractIntent,
        observation: Observation,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> list[Action]:
        click_actions = [action for action in actions if action.payload]
        if not click_actions:
            return []
        height = len(observation.frame.grid)
        width = len(observation.frame.grid[0]) if height else 0
        targets = set(intent.target or observation.notes.get("scene_goal_targets", []) or [])
        phase = self._phase(observation, game_memory, recent_action_keys)
        candidates: list[CoordinateCandidate] = []
        for action in click_actions:
            x = int(action.payload.get("x", -1))
            y = int(action.payload.get("y", -1))
            if x < 0 or y < 0 or (width and x >= width) or (height and y >= height):
                continue
            role = self._role(x, y, width, height, observation)
            score = self._score(
                action=action,
                role=role,
                x=x,
                y=y,
                phase=phase,
                targets=targets,
                observation=observation,
                game_memory=game_memory,
                recent_action_keys=recent_action_keys,
            )
            candidates.append(CoordinateCandidate(action=action, role=role, score_key=score))
        candidates.sort(key=lambda item: item.score_key)
        ranked = [candidate.action for candidate in candidates]
        return ranked or click_actions

    def classify_transition(self, action: Action, notes: dict[str, object]) -> tuple[str, bool, int] | None:
        if not action.payload:
            return None
        x = int(action.payload.get("x", -1))
        y = int(action.payload.get("y", -1))
        height = int(notes.get("grid_height", 0) or 0)
        width = int(notes.get("grid_width", 0) or 0)
        role = self._role_from_notes(x, y, width, height, notes)
        scene_progress = float(notes.get("scene_goal_progress_score", 0.0) or 0.0)
        scene_delta = str(notes.get("scene_delta_kind", "no_object_delta"))
        changed_playfield = int(notes.get("changed_playfield_cells", 0) or 0)
        changed_hud = int(notes.get("changed_hud_cells", 0) or 0)
        likely_feedback = bool(notes.get("likely_feedback_flash", False))
        hud_only = changed_hud > 0 and changed_playfield == 0
        meaningful_delta = scene_delta not in {"no_object_delta", "object_appeared"} or changed_playfield > 0
        success = scene_progress > 0 or (meaningful_delta and not likely_feedback and not hud_only)
        if role in {"top_band", "tool_or_palette", "reference"} and not bool(notes.get("won", False)) and float(notes.get("reward_delta", 0.0) or 0.0) <= 0:
            success = False
        if hud_only and scene_progress <= 0:
            role = "tool_or_palette"
            success = False
        return role, success, y

    def _phase(self, observation: Observation, game_memory: GameMemory, recent_action_keys: list[str]) -> str:
        control = str(observation.notes.get("runtime_control_family", "unknown"))
        coordinate_count = sum(1 for key in recent_action_keys if key.startswith("ACTION6|"))
        if control == "coordinate_only":
            if game_memory.coordinate_top_band_uses < 2 and coordinate_count < 4:
                return "tool_probe"
            return "workspace_probe"
        if control in {"mixed_move_coordinate", "coordinate_mode"}:
            if game_memory.coordinate_top_band_uses < 2 and coordinate_count < 3:
                return "tool_probe"
            return "workspace_probe"
        return "workspace_probe"

    def _score(
        self,
        *,
        action: Action,
        role: str,
        x: int,
        y: int,
        phase: str,
        targets: set[tuple[int, int]],
        observation: Observation,
        game_memory: GameMemory,
        recent_action_keys: list[str],
    ) -> tuple[float, int, int, int, str]:
        target_distance = self._target_distance(x, y, targets)
        role_penalty = self._role_penalty(role, phase, game_memory)
        cooldown = game_memory.coordinate_key_cooldowns.get(action.key, 0)
        row_cooldown = game_memory.coordinate_row_cooldowns.get(y, 0)
        recent_penalty = sum(1 for key in recent_action_keys[-12:] if key == action.key)
        danger = 100.0 if game_memory.is_dangerous(observation.state_key, action.key) else 0.0
        if role in {"top_band", "tool_or_palette", "reference"} and game_memory.coordinate_top_band_uses >= 3:
            role_penalty += 8.0
        if row_cooldown:
            role_penalty += 4.0
        return (
            danger + role_penalty - game_memory.coordinate_role_score(role),
            cooldown,
            recent_penalty,
            target_distance,
            action.key,
        )

    def _role_penalty(self, role: str, phase: str, game_memory: GameMemory) -> float:
        if phase == "tool_probe":
            preferred = {"tool_or_palette", "reference", "top_band"}
            return 0.0 if role in preferred else 2.0
        preferred = {"workspace_object", "workspace_empty", "goal_marker", "changed_region"}
        if role in preferred:
            return 0.0
        if role in {"tool_or_palette", "reference", "top_band"}:
            return 5.0 + game_memory.coordinate_top_band_uses
        return 2.0

    def _role(self, x: int, y: int, width: int, height: int, observation: Observation) -> str:
        top_limit = max(2, height // 6) if height else 0
        if y <= top_limit:
            return "top_band"
        changed_bbox = observation.notes.get("changed_bbox")
        if isinstance(changed_bbox, dict):
            if int(changed_bbox.get("min_x", -999)) - 1 <= x <= int(changed_bbox.get("max_x", -999)) + 1 and int(changed_bbox.get("min_y", -999)) - 1 <= y <= int(changed_bbox.get("max_y", -999)) + 1:
                return "changed_region"
        for target in observation.notes.get("scene_goal_targets", [])[:12]:
            if isinstance(target, tuple) and len(target) == 2 and abs(x - int(target[0])) + abs(y - int(target[1])) <= 1:
                return "goal_marker"
        grid = observation.frame.grid
        if grid and int(grid[y][x]) != 0:
            return "workspace_object"
        return "workspace_empty"

    def _role_from_notes(self, x: int, y: int, width: int, height: int, notes: dict[str, object]) -> str:
        top_limit = max(2, height // 6) if height else 0
        if y <= top_limit:
            return "top_band"
        changed_bbox = notes.get("changed_bbox")
        if isinstance(changed_bbox, dict):
            if int(changed_bbox.get("min_x", -999)) - 1 <= x <= int(changed_bbox.get("max_x", -999)) + 1 and int(changed_bbox.get("min_y", -999)) - 1 <= y <= int(changed_bbox.get("max_y", -999)) + 1:
                return "changed_region"
        return "workspace_object"

    def _target_distance(self, x: int, y: int, targets: set[tuple[int, int]]) -> int:
        if not targets:
            return 0
        return min(abs(x - int(tx)) + abs(y - int(ty)) for tx, ty in targets)
