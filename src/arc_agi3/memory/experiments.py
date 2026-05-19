from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action, ExperimentOutcome, ExperimentProposal, Transition


@dataclass(slots=True)
class ExperimentManager:
    """Tracks testable subgoals and scores their observed outcomes."""

    active: ExperimentProposal | None = None
    history: list[ExperimentOutcome] = field(default_factory=list)
    completed_keys: set[str] = field(default_factory=set)
    active_unplannable_steps: int = 0

    def available(self, world: Any, actions: list[Action], seen_action_keys: set[str]) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        for target in sorted(world.visible_item_cells)[:6]:
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"collect_item:{target[0]},{target[1]}",
                    kind="collect_item",
                    target=target,
                    rationale="visible item-like object can test whether collection changes later state",
                    expected_if_true="object disappears or inventory-like/panel state changes",
                    failure_signal="target reached but no collectible or panel change occurs",
                ),
            )
        if self._button_tests_are_supported(world):
            for target in sorted(world.visible_button_cells)[:4]:
                self._append_if_new(
                    proposals,
                    ExperimentProposal(
                        key=f"activate_button:{target[0]},{target[1]}",
                        kind="activate_button",
                        target=target,
                        rationale="button-like object can test whether a passage or board state changes",
                        expected_if_true="blocked region, room, or board layout changes",
                        failure_signal="interaction produces only HUD/noop feedback",
                    ),
                )
        if not world.has_precondition_evidence():
            for target in sorted(world.visible_goal_cells)[:4]:
                self._append_if_new(
                    proposals,
                    ExperimentProposal(
                    key=f"go_to_goal:{target[0]},{target[1]}",
                    kind="go_to_goal",
                    target=target,
                    rationale="visible goal candidate can test direct navigation only when gating evidence is weak",
                    expected_if_true="reward or win occurs",
                    failure_signal="feedback flash or no progress at goal",
                    ),
                )
        for relation in world.relation_details.values():
            if relation.min_distance <= 1:
                continue
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"inspect_relation:{relation.key}",
                    kind="inspect_relation",
                    target={
                        "relation_key": relation.key,
                        "baseline_distance": relation.min_distance,
                        "nearest_pair": relation.nearest_pair,
                    },
                    rationale="disconnected same-feature regions can test whether the task requires connecting them",
                    expected_if_true="actions reduce the distance or make the regions connected",
                    failure_signal="after inspecting endpoints, relation distance never decreases",
                ),
            )
        missing_axis = self._missing_axis_for_visible_targets(world)
        if missing_axis is not None:
            for action in actions:
                if action.key in seen_action_keys:
                    continue
                self._append_if_new(
                    proposals,
                    ExperimentProposal(
                        key=f"discover_axis:{missing_axis}:{action.key}",
                        kind="discover_axis",
                        target={"axis": missing_axis, "action_key": action.key},
                        rationale=f"visible target requires {missing_axis} movement, but that axis is not learned yet",
                        expected_if_true=f"action reveals {missing_axis} player movement",
                        failure_signal="action does not move the player on the missing axis",
                    ),
                )
        for track in world.likely_interactable_tracks():
            affordance, confidence = track.best_affordance
            if confidence < 0.18:
                continue
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"inspect_affordance:{track.track_id}",
                    kind="inspect_affordance",
                    target={
                        "track_id": track.track_id,
                        "center": track.center,
                        "affordance": affordance,
                        "baseline_blocking": track.best_affordance[0] == "blocking_candidate",
                    },
                    rationale=f"object {track.track_id} may be {affordance} and affect the route",
                    expected_if_true="object moves, disappears, or opens nearby cells after interaction",
                    failure_signal="reaching or acting near the object causes no object-level change",
                    confidence=confidence,
                ),
            )
        for action in actions:
            if action.key in seen_action_keys:
                continue
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"probe_action:{action.key}",
                    kind="probe_action",
                    target=action.key,
                    rationale="unseen action can reveal a new rule or control dimension",
                    expected_if_true="state or player relation changes in a new way",
                    failure_signal="noop, undo, restart, or HUD-only effect",
                ),
            )
        return proposals

    def activate(self, proposal: ExperimentProposal) -> None:
        self.active = proposal
        self.active_unplannable_steps = 0

    def activate_if_idle(self, proposal: ExperimentProposal) -> bool:
        if self.active is not None:
            return False
        self.active = proposal
        self.active_unplannable_steps = 0
        return True

    def note_plan_result(self, plannable: bool, limit: int = 3) -> ExperimentOutcome | None:
        proposal = self.active
        if proposal is None:
            return None
        if plannable:
            self.active_unplannable_steps = 0
            return None
        self.active_unplannable_steps += 1
        if self.active_unplannable_steps < limit:
            return None
        outcome = ExperimentOutcome(
            proposal=proposal,
            status="inconclusive",
            evidence="no_executable_plan_from_current_world_model",
        )
        self.history.append(outcome)
        self.completed_keys.add(proposal.key)
        self.active = None
        self.active_unplannable_steps = 0
        del self.history[:-24]
        return outcome

    def observe_transition(
        self,
        transition: Transition,
        after_notes: dict[str, Any],
        world: Any,
    ) -> ExperimentOutcome | None:
        proposal = self.active
        if proposal is None:
            return None
        outcome = self._score(proposal, transition, after_notes, world)
        if outcome is None:
            return None
        self.history.append(outcome)
        self.completed_keys.add(proposal.key)
        self.active = None
        self.active_unplannable_steps = 0
        del self.history[:-24]
        return outcome

    def summary_lines(self, limit: int = 6) -> list[str]:
        lines: list[str] = []
        if self.active is not None:
            lines.append(f"active={self.active.key} expected={self.active.expected_if_true}")
        for outcome in self.history[-limit:]:
            lines.append(f"{outcome.proposal.key} -> {outcome.status}: {outcome.evidence}")
        return lines

    def _score(
        self,
        proposal: ExperimentProposal,
        transition: Transition,
        after_notes: dict[str, Any],
        world: Any,
    ) -> ExperimentOutcome | None:
        removed = list((after_notes.get("collectible_changes", {}) or {}).get("removed", []) or [])
        anchor_changes = list(after_notes.get("anchor_patch_changes", []) or [])
        feedback = bool(after_notes.get("likely_feedback_flash", False))
        interaction_hint = str(after_notes.get("interaction_hint", "unknown"))
        player_pos = world.player_pos

        if proposal.kind == "collect_item":
            if removed or bool(after_notes.get("collectible_progress", False)):
                evidence = f"collectible_removed={removed[:2]} anchor_changes={anchor_changes or 'none'}"
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence=evidence)
            if player_pos == proposal.target and transition.changed:
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="target reached without collectible evidence")
            return None

        if proposal.kind == "activate_button":
            if interaction_hint in {"spawn_or_unlock", "board_or_room_transform"}:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence=f"interaction_hint={interaction_hint}")
            if player_pos == proposal.target and (interaction_hint in {"hud_or_counter_update", "unknown"} or not transition.changed):
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence=f"interaction_hint={interaction_hint}")
            return None

        if proposal.kind == "go_to_goal":
            if transition.won or transition.reward_delta > 0:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="reward_or_win_observed")
            if feedback:
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="feedback_flash_without_success")
            if player_pos == proposal.target and transition.changed:
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="goal_reached_without_reward")
            return None

        if proposal.kind == "probe_action":
            if transition.action.key != proposal.target:
                return None
            if transition.changed and not feedback:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="new action produced non-feedback state change")
            return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="action produced noop_or_feedback_only")

        if proposal.kind == "discover_axis" and isinstance(proposal.target, dict):
            if transition.action.key != proposal.target.get("action_key"):
                return None
            axis = str(proposal.target.get("axis", ""))
            before = after_notes.get("semantic_previous_player_pos")
            after = after_notes.get("semantic_player_pos")
            if isinstance(before, tuple) and isinstance(after, tuple):
                dx = int(after[0]) - int(before[0])
                dy = int(after[1]) - int(before[1])
                moved_on_axis = (axis == "horizontal" and dx != 0) or (axis == "vertical" and dy != 0)
                if moved_on_axis:
                    return ExperimentOutcome(proposal=proposal, status="confirmed", evidence=f"revealed_{axis}_movement")
            return ExperimentOutcome(proposal=proposal, status="contradicted", evidence=f"no_{axis}_movement")

        if proposal.kind == "inspect_affordance" and isinstance(proposal.target, dict):
            track_id = str(proposal.target.get("track_id", ""))
            center = proposal.target.get("center")
            track = world.object_tracks.get(track_id)
            if track is None:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="object_disappeared")
            affordance, _ = track.best_affordance
            if affordance in {"breakable_candidate", "pushable_candidate", "door_candidate"}:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence=f"affordance={affordance}")
            if player_pos == center and transition.changed:
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="reached_object_without_affordance_change")
            return None

        if proposal.kind == "inspect_relation" and isinstance(proposal.target, dict):
            key = str(proposal.target.get("relation_key", ""))
            baseline = int(proposal.target.get("baseline_distance", 0) or 0)
            relation = world.relation_for(key)
            if relation is None:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="relation disappeared_or_merged")
            if relation.min_distance < baseline:
                return ExperimentOutcome(
                    proposal=proposal,
                    status="confirmed",
                    evidence=f"relation_distance {baseline}->{relation.min_distance}",
                )
            nearest_pair = tuple(proposal.target.get("nearest_pair", ()))
            if player_pos in nearest_pair and transition.changed:
                return ExperimentOutcome(
                    proposal=proposal,
                    status="contradicted",
                    evidence=f"inspected endpoint without relation progress distance={relation.min_distance}",
                )
            return None

        return None

    def _append_if_new(self, proposals: list[ExperimentProposal], proposal: ExperimentProposal) -> None:
        if proposal.key in self.completed_keys:
            return
        proposals.append(proposal)

    def _button_tests_are_supported(self, world: Any) -> bool:
        if not world.visible_button_cells:
            return False
        switch_family = world.hypothesis_library.families.get("switch_opens_path")
        direct_switch_evidence = False
        if switch_family is not None:
            direct_switch_evidence = any(
                "caused" in evidence or "interaction_hint=" in evidence
                for evidence in switch_family.evidence
            )
        if direct_switch_evidence and world.hypothesis_library.confidence("switch_opens_path") >= 0.28:
            return True
        for track in world.likely_interactable_tracks():
            affordance, confidence = track.best_affordance
            if affordance == "door_candidate" and confidence >= 0.24:
                return True
        return False

    def _missing_axis_for_visible_targets(self, world: Any) -> str | None:
        if world.player_pos is None:
            return None
        targets = world.visible_item_cells or world.visible_goal_cells or world.visible_button_cells
        if not targets:
            return None
        px, py = world.player_pos
        needs_horizontal = any(tx != px for tx, _ in targets)
        needs_vertical = any(ty != py for _, ty in targets)
        learned_vectors = list(world.action_move_vectors.values())
        has_horizontal = any(dx != 0 for dx, _ in learned_vectors)
        has_vertical = any(dy != 0 for _, dy in learned_vectors)
        if needs_vertical and not has_vertical:
            return "vertical"
        if needs_horizontal and not has_horizontal:
            return "horizontal"
        return None
