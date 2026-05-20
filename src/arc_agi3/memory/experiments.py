from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action, ExperimentOutcome, ExperimentProposal, Transition


@dataclass(slots=True)
class ExperimentSession:
    proposal: ExperimentProposal
    target_ref: str
    max_trials: int
    max_steps: int
    trial_count: int = 0
    step_count: int = 0
    repeated_action_count: int = 0
    last_action_key: str = ""


class ExperimentManager:
    """Tracks testable subgoals and runs them as short bounded sessions."""

    def __init__(
        self,
        active: ExperimentProposal | None = None,
        history: list[ExperimentOutcome] | None = None,
        completed_keys: set[str] | None = None,
        active_unplannable_steps: int = 0,
    ) -> None:
        self.history = list(history or [])
        self.completed_keys = set(completed_keys or set())
        self.active_unplannable_steps = active_unplannable_steps
        self.active_session: ExperimentSession | None = None
        self.mode_action_keys: set[str] = set()
        self.macro_pair_counts: Counter[tuple[str, str]] = Counter()
        self.failed_macro_pairs: set[tuple[str, str]] = set()
        if active is not None:
            self.activate(active)

    @property
    def active(self) -> ExperimentProposal | None:
        return self.active_session.proposal if self.active_session is not None else None

    def available(
        self,
        world: Any,
        actions: list[Action],
        seen_action_keys: set[str],
        family_for: Any | None = None,
        rule_focus_action_keys: set[str] | None = None,
    ) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        rule_focus_action_keys = rule_focus_action_keys or set()
        for target in sorted(world.visible_item_cells)[:6]:
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"collect_item:{target[0]},{target[1]}",
                    kind="collect_item",
                    target=target,
                    rationale="visible item-like object can test whether collection changes later state",
                    expected_if_true="object disappears or inventory-like/panel state changes",
                    failure_signal="target reached or clicked repeatedly without collectible evidence",
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
                        failure_signal="repeated interaction produces only HUD/noop feedback",
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
                        failure_signal="feedback flash or repeated no-progress attempts at goal",
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
            if action.key in seen_action_keys and action.key not in rule_focus_action_keys:
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
        for first_key, second_key in self._macro_pair_candidates(actions, family_for=family_for):
            self._append_if_new(
                proposals,
                ExperimentProposal(
                    key=f"probe_action_pair:{first_key}->{second_key}",
                    kind="probe_action_pair",
                    target={"first_action_key": first_key, "second_action_key": second_key},
                    rationale="some actions only become meaningful after a preceding setup or mode-setting action",
                    expected_if_true="the paired action sequence produces a new nontrivial effect",
                    failure_signal="the paired sequence still produces noop or feedback-only changes",
                ),
            )
        return proposals

    def activate(self, proposal: ExperimentProposal) -> None:
        self.active_session = ExperimentSession(
            proposal=proposal,
            target_ref=self._target_ref(proposal),
            max_trials=self._max_trials(proposal),
            max_steps=self._max_steps(proposal),
        )
        self.active_unplannable_steps = 0

    def activate_if_idle(self, proposal: ExperimentProposal) -> bool:
        if self.active_session is not None:
            return False
        self.activate(proposal)
        return True

    def note_execution(self, action: Action) -> None:
        session = self.active_session
        if session is None:
            return
        session.step_count += 1
        if session.last_action_key == action.key:
            session.repeated_action_count += 1
        else:
            session.repeated_action_count = 1
            session.last_action_key = action.key
        session.trial_count += 1

    def remember_macro_effect(
        self,
        first_action_key: str,
        second_action_key: str,
        *,
        mode_setting: bool,
        productive: bool,
    ) -> None:
        if mode_setting:
            self.mode_action_keys.add(first_action_key)
        if productive:
            self.macro_pair_counts[(first_action_key, second_action_key)] += 1

    def note_plan_result(self, plannable: bool, limit: int = 2) -> ExperimentOutcome | None:
        session = self.active_session
        if session is None:
            return None
        if plannable:
            self.active_unplannable_steps = 0
            return None
        self.active_unplannable_steps += 1
        if self.active_unplannable_steps < limit:
            return None
        return self._finish(
            ExperimentOutcome(
                proposal=session.proposal,
                status="abandoned",
                evidence="no_executable_plan_from_current_world_model",
            )
        )

    def observe_transition(
        self,
        transition: Transition,
        after_notes: dict[str, Any],
        world: Any,
    ) -> ExperimentOutcome | None:
        session = self.active_session
        if session is None:
            return None
        outcome = self._score(session, transition, after_notes, world)
        if outcome is not None:
            return self._finish(outcome)
        if self._should_abandon(session, transition, after_notes, world):
            return self._finish(
                ExperimentOutcome(
                    proposal=session.proposal,
                    status="contradicted",
                    evidence=self._abandon_evidence(session, transition, after_notes),
                )
            )
        return None

    def summary_lines(self, limit: int = 6) -> list[str]:
        lines: list[str] = []
        if self.active_session is not None:
            session = self.active_session
            lines.append(
                f"active={session.proposal.key} expected={session.proposal.expected_if_true} "
                f"trials={session.trial_count}/{session.max_trials} steps={session.step_count}/{session.max_steps}"
            )
        for outcome in self.history[-limit:]:
            lines.append(f"{outcome.proposal.key} -> {outcome.status}: {outcome.evidence}")
        return lines

    def _score(
        self,
        session: ExperimentSession,
        transition: Transition,
        after_notes: dict[str, Any],
        world: Any,
    ) -> ExperimentOutcome | None:
        proposal = session.proposal
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
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="target_reached_without_collectible_evidence")
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
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="new_action_produced_non_feedback_state_change")
            return ExperimentOutcome(proposal=proposal, status="contradicted", evidence="action_produced_noop_or_feedback_only")

        if proposal.kind == "probe_action_pair" and isinstance(proposal.target, dict):
            first_key = str(proposal.target.get("first_action_key", ""))
            second_key = str(proposal.target.get("second_action_key", ""))
            if transition.action.key not in {first_key, second_key}:
                return None
            if session.step_count < 2:
                return None
            if transition.changed and not feedback and (
                bool(after_notes.get("collectible_progress", False))
                or bool(after_notes.get("anchor_patch_changes", []))
                or interaction_hint not in {"unknown", "hud_or_counter_update"}
            ):
                return ExperimentOutcome(
                    proposal=proposal,
                    status="confirmed",
                    evidence=f"paired_actions {first_key}->{second_key} produced {interaction_hint}",
                )
            if not transition.changed and interaction_hint in {"unknown", "hud_or_counter_update"}:
                return ExperimentOutcome(
                    proposal=proposal,
                    status="contradicted",
                    evidence=f"paired_actions {first_key}->{second_key} produced no useful effect",
                )
            return None

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
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence="relation_disappeared_or_merged")
            if relation.min_distance < baseline:
                return ExperimentOutcome(proposal=proposal, status="confirmed", evidence=f"relation_distance {baseline}->{relation.min_distance}")
            nearest_pair = tuple(proposal.target.get("nearest_pair", ()))
            if player_pos in nearest_pair and transition.changed:
                return ExperimentOutcome(proposal=proposal, status="contradicted", evidence=f"inspected_endpoint_without_relation_progress distance={relation.min_distance}")
            return None

        return None

    def _should_abandon(
        self,
        session: ExperimentSession,
        transition: Transition,
        after_notes: dict[str, Any],
        world: Any,
    ) -> bool:
        if session.step_count >= session.max_steps or session.trial_count >= session.max_trials:
            return True
        proposal = session.proposal
        if proposal.kind in {"collect_item", "activate_button", "inspect_affordance"} and transition.action.payload:
            if session.repeated_action_count >= 2 and not self._goal_relevant_change(after_notes):
                return True
        if proposal.kind == "go_to_goal" and world.should_defer_goal():
            return True
        return False

    def _goal_relevant_change(self, after_notes: dict[str, Any]) -> bool:
        return bool(
            after_notes.get("collectible_progress", False)
            or after_notes.get("anchor_patch_changes", [])
            or str(after_notes.get("interaction_hint", "unknown")) in {"spawn_or_unlock", "board_or_room_transform"}
            or (after_notes.get("collectible_changes", {}) or {}).get("removed", [])
        )

    def _abandon_evidence(self, session: ExperimentSession, transition: Transition, after_notes: dict[str, Any]) -> str:
        proposal = session.proposal
        if proposal.kind in {"collect_item", "activate_button", "inspect_affordance"} and transition.action.payload:
            return f"repeated_direct_interaction_without_goal_relevant_change trials={session.trial_count}"
        if proposal.kind == "go_to_goal":
            return "goal_deferred_or_no_reward_after_attempts"
        if proposal.kind == "probe_action_pair" and isinstance(proposal.target, dict):
            return (
                "macro_pair_limit_reached "
                f"{proposal.target.get('first_action_key')}->{proposal.target.get('second_action_key')}"
            )
        return f"trial_limit_reached trials={session.trial_count} steps={session.step_count}"

    def _finish(self, outcome: ExperimentOutcome) -> ExperimentOutcome:
        if outcome.proposal.kind == "probe_action_pair" and isinstance(outcome.proposal.target, dict):
            first_key = str(outcome.proposal.target.get("first_action_key", ""))
            second_key = str(outcome.proposal.target.get("second_action_key", ""))
            if (
                first_key
                and second_key
                and outcome.status in {"contradicted", "abandoned"}
            ):
                self.failed_macro_pairs.add((first_key, second_key))
        self.history.append(outcome)
        self.completed_keys.add(outcome.proposal.key)
        self.active_session = None
        self.active_unplannable_steps = 0
        del self.history[:-24]
        return outcome

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
            direct_switch_evidence = any("caused" in evidence or "interaction_hint=" in evidence for evidence in switch_family.evidence)
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
        targets = world.visible_item_cells or world.visible_goal_cells or world.visible_button_cells or world.visible_display_cells
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

    def _target_ref(self, proposal: ExperimentProposal) -> str:
        if isinstance(proposal.target, tuple):
            return f"cell:{proposal.target[0]},{proposal.target[1]}"
        if isinstance(proposal.target, dict):
            if "track_id" in proposal.target:
                return f"track:{proposal.target['track_id']}"
            if "relation_key" in proposal.target:
                return f"relation:{proposal.target['relation_key']}"
            if "action_key" in proposal.target:
                return f"action:{proposal.target['action_key']}"
        if isinstance(proposal.target, str):
            return f"action:{proposal.target}"
        return proposal.key

    def _max_trials(self, proposal: ExperimentProposal) -> int:
        if proposal.kind in {"probe_action", "discover_axis"}:
            return 1
        if proposal.kind == "probe_action_pair":
            return 2
        if proposal.kind in {"collect_item", "activate_button", "go_to_goal", "inspect_affordance"}:
            return 2
        return 3

    def _max_steps(self, proposal: ExperimentProposal) -> int:
        if proposal.kind in {"probe_action", "discover_axis"}:
            return 1
        if proposal.kind == "probe_action_pair":
            return 2
        if proposal.kind in {"collect_item", "activate_button", "go_to_goal", "inspect_affordance"}:
            return 3
        return 4

    def _macro_pair_candidates(
        self,
        actions: list[Action],
        family_for: Any | None = None,
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        actions_by_key = {action.key: action for action in actions}
        action_keys = list(actions_by_key)

        def pair_is_valid(first_key: str, second_key: str) -> bool:
            if not first_key or not second_key or first_key == second_key:
                return False
            if (first_key, second_key) in self.failed_macro_pairs:
                return False
            first_action = actions_by_key.get(first_key)
            second_action = actions_by_key.get(second_key)
            if first_action is None or second_action is None:
                return False
            if family_for is not None:
                first_family = str(family_for(first_action))
                second_family = str(family_for(second_action))
                if first_family and second_family and first_family == second_family:
                    return False
            return True

        for (first_key, second_key), _ in sorted(
            self.macro_pair_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        ):
            if (
                first_key in action_keys
                and second_key in action_keys
                and (first_key, second_key) not in seen
                and pair_is_valid(first_key, second_key)
            ):
                seen.add((first_key, second_key))
                candidates.append((first_key, second_key))
                if len(candidates) >= 3:
                    return candidates
        for first_key in sorted(self.mode_action_keys):
            if first_key not in action_keys:
                continue
            for second_key in action_keys:
                pair = (first_key, second_key)
                if pair in seen or not pair_is_valid(first_key, second_key):
                    continue
                seen.add(pair)
                candidates.append(pair)
                if len(candidates) >= 3:
                    return candidates
        return candidates
