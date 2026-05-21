from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from arc_agi3.core.config import AgentConfig
from arc_agi3.core.types import Action, Observation
from arc_agi3.memory.game_memory import GameMemory
from arc_agi3.memory.state_graph import StateGraph


@dataclass(slots=True)
class BackendDecision:
    action: Action
    source: str
    reason: str


@dataclass(slots=True)
class BackendTaskProfile:
    family: str
    probe_bias: str
    observed_frames: int
    same_size: bool
    color_count: int


class BackendActionController:
    """Thin controller that treats vendored external reasoners as the main brains.

    The controller itself stays intentionally small. It only:
    - reads backend hints from observation notes
    - chooses whether we are still probing or already exploiting
    - translates those hints into a concrete action choice
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def choose_action(
        self,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
        recent_action_families: list[str],
        recent_states: list[str],
    ) -> BackendDecision | None:
        if not actions:
            return None
        profile = self._profile_from_notes(observation.notes)
        phase = self._phase(profile, game_memory)
        if phase == "probe":
            ranked = self._rank_probe_actions(
                observation=observation,
                actions=actions,
                graph=graph,
                game_memory=game_memory,
                recent_action_keys=recent_action_keys,
                recent_action_families=recent_action_families,
                profile=profile,
            )
            if ranked:
                action = ranked[0]
                return BackendDecision(
                    action=action,
                    source="backend_probe",
                    reason=f"probing {profile.family} structure via {profile.probe_bias}",
                )
        ranked = self._rank_exploit_actions(
            observation=observation,
            actions=actions,
            graph=graph,
            game_memory=game_memory,
            recent_action_keys=recent_action_keys,
            recent_action_families=recent_action_families,
            recent_states=recent_states,
            profile=profile,
        )
        if not ranked:
            return None
        action = ranked[0]
        return BackendDecision(
            action=action,
            source="backend_exploit",
            reason=f"following backend-guided {profile.family} hypothesis",
        )

    def _profile_from_notes(self, notes: dict[str, object]) -> BackendTaskProfile:
        metadata = notes.get("external_reasoner_metadata") or {}
        observed_frames = int(getattr(metadata, "get", lambda *_: 0)("observed_frames", 0) or 0)
        same_size = bool(notes.get("compressarc_in_out_same_size")) and bool(notes.get("compressarc_all_in_same_size"))
        color_count = int(notes.get("compressarc_n_colors", 0) or 0)
        family = str(notes.get("external_task_family", "unknown"))
        probe_bias = str(notes.get("external_probe_bias", "simple_actions_first"))
        return BackendTaskProfile(
            family=family,
            probe_bias=probe_bias,
            observed_frames=observed_frames,
            same_size=same_size,
            color_count=color_count,
        )

    def _phase(self, profile: BackendTaskProfile, game_memory: GameMemory) -> str:
        useful_actions = sum(1 for key in game_memory.action_use_counts if game_memory.semantic_profile(key.split("|", 1)[0], key).changed_uses > 0)
        if profile.observed_frames <= max(6, self.config.budget.explore_phase_steps // 2):
            return "probe"
        if useful_actions < 2:
            return "probe"
        return "exploit"

    def _rank_probe_actions(
        self,
        *,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
        recent_action_families: list[str],
        profile: BackendTaskProfile,
    ) -> list[Action]:
        previous_action_key = recent_action_keys[-1] if recent_action_keys else None
        key_counts = Counter(recent_action_keys[-8:])
        family_counts = Counter(recent_action_families[-8:])
        ranked: list[tuple[object, ...]] = []
        for index, action in enumerate(actions):
            family = game_memory.action_family(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            semantic = game_memory.semantic_profile(action.name, action.key)
            successor = graph.seen_successor(observation.state_key, action)
            tried_here = graph.action_was_tried(observation.state_key, action)
            is_click = bool(action.payload)
            uncertainty = game_memory.uncertainty_score(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            ranked.append(
                (
                    self._probe_priority(profile, family=family, is_click=is_click),
                    tried_here,
                    semantic.uses,
                    key_counts.get(action.key, 0),
                    family_counts.get(family, 0),
                    successor is not None,
                    -uncertainty,
                    index,
                    action,
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked]

    def _rank_exploit_actions(
        self,
        *,
        observation: Observation,
        actions: list[Action],
        graph: StateGraph,
        game_memory: GameMemory,
        recent_action_keys: list[str],
        recent_action_families: list[str],
        recent_states: list[str],
        profile: BackendTaskProfile,
    ) -> list[Action]:
        previous_action_key = recent_action_keys[-1] if recent_action_keys else None
        key_counts = Counter(recent_action_keys[-8:])
        family_counts = Counter(recent_action_families[-8:])
        ranked: list[tuple[object, ...]] = []
        for index, action in enumerate(actions):
            family = game_memory.action_family(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            semantic = game_memory.semantic_profile(action.name, action.key)
            contextual = game_memory.contextual_effect_profile(
                action.key,
                previous_action_key=previous_action_key,
            )
            context_progress = contextual.progress_ratio if contextual is not None else 0.0
            alignment_score = game_memory.action_alignment_score(action.key)
            family_alignment_score = game_memory.family_alignment_score(family)
            uncertainty = game_memory.uncertainty_score(
                action.name,
                action.key,
                previous_action_key=previous_action_key,
            )
            successor = graph.seen_successor(observation.state_key, action)
            loop_penalty = successor in recent_states if successor is not None else False
            state_traversals = graph.traversals_for(observation.state_key, action, successor) if successor is not None else 0
            utility = (
                (4.0 * semantic.reward_total)
                + (2.0 * semantic.collectible_progress)
                + (1.5 * (semantic.changed_uses / max(1, semantic.uses)))
                + (4.0 * alignment_score)
                + (2.0 * family_alignment_score)
                + (1.25 * context_progress)
                + (0.75 * uncertainty)
            )
            utility += self._family_bias(profile, family=family, action=action)
            ranked.append(
                (
                    -utility,
                    loop_penalty,
                    key_counts.get(action.key, 0),
                    family_counts.get(family, 0),
                    state_traversals,
                    index,
                    action,
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked]

    def _probe_priority(self, profile: BackendTaskProfile, *, family: str, is_click: bool) -> int:
        if profile.probe_bias == "movement_first":
            if family == "movement":
                return 0
            if not is_click:
                return 1
            return 3
        if profile.probe_bias == "simple_then_click":
            if not is_click:
                return 0
            if family in {"direct_click", "edit_or_mode"}:
                return 1
            return 2
        if profile.probe_bias == "click_enabled":
            if family in {"edit_or_mode", "direct_click"}:
                return 0
            if not is_click:
                return 1
            return 2
        return 0 if not is_click else 1

    def _family_bias(self, profile: BackendTaskProfile, *, family: str, action: Action) -> float:
        if profile.family == "navigation":
            if family == "movement":
                return 1.0
            if action.payload:
                return -1.0
            return 0.0
        if profile.family == "editing":
            if family in {"edit_or_mode", "direct_click"}:
                return 1.0
            if family == "movement":
                return -0.5
            return 0.0
        if profile.family == "transform":
            if family in {"edit_or_mode", "movement"}:
                return 0.5
            return 0.0
        return 0.0
