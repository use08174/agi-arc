from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HypothesisFamily:
    """Reusable task-family hypothesis updated from interaction evidence."""

    name: str
    prior: float
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = self.prior

    def support(self, amount: float, evidence: str) -> None:
        self.confidence = min(1.0, self.confidence + amount)
        if evidence not in self.evidence:
            self.evidence.append(evidence)

    def contradict(self, amount: float, evidence: str) -> None:
        self.confidence = max(0.0, self.confidence - amount)
        if evidence not in self.contradictions:
            self.contradictions.append(evidence)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        return []

    def proposed_tests(self, world: Any) -> list[str]:
        return []


class PlainNavigationHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="plain_navigation", prior=0.22)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        return [{"type": "go_to_goal", "target": cell} for cell in sorted(world.visible_goal_cells)[:4]]

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Try a safe path toward a visible goal if no gating evidence exists."]


class CollectBeforeGoalHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="collect_before_goal", prior=0.14)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        return [{"type": "collect_item", "target": cell} for cell in sorted(world.visible_item_cells)[:8]]

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Visit a small disappearing object and check whether another state changes."]


class StateMatchBeforeGoalHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="state_match_before_goal", prior=0.12)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        goals = [{"type": "collect_item", "target": cell} for cell in sorted(world.visible_item_cells)[:8]]
        if world.anchor_patch_states:
            goals.append({"type": "compare_anchor_patches", "target": sorted(world.anchor_patch_states)})
        return goals

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Change a mutable panel candidate, then compare it with stable anchor patches before entering a goal."]


class ConnectionHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="connect_same_feature", prior=0.10)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        return [{"type": "inspect_relation", "target": item} for item in world.relation_candidates[:4]]

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Test whether actions reduce distance or increase connectivity between repeated same-feature regions."]


class SwitchPathHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="switch_opens_path", prior=0.10)

    def preferred_subgoals(self, world: Any) -> list[dict[str, Any]]:
        return [{"type": "activate_button", "target": cell} for cell in sorted(world.visible_button_cells)[:6]]

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Interact with a button candidate and inspect whether blocked regions or passages change."]


class SequenceHypothesis(HypothesisFamily):
    def __init__(self) -> None:
        super().__init__(name="sequence_required", prior=0.07)

    def proposed_tests(self, world: Any) -> list[str]:
        return ["Track whether multiple interactions change later outcomes depending on order."]


class HypothesisLibrary:
    def __init__(self) -> None:
        self.families: dict[str, HypothesisFamily] = {
            family.name: family
            for family in (
                PlainNavigationHypothesis(),
                CollectBeforeGoalHypothesis(),
                StateMatchBeforeGoalHypothesis(),
                ConnectionHypothesis(),
                SwitchPathHypothesis(),
                SequenceHypothesis(),
            )
        }

    def observe_scene(self, world: Any) -> None:
        if world.visible_goal_cells:
            self._support("plain_navigation", 0.02, "visible goal candidate exists")
        if world.visible_item_cells:
            self._support("collect_before_goal", 0.02, "visible item candidate exists")
        if world.visible_button_cells:
            self._support("switch_opens_path", 0.03, "visible button candidate exists")
        if world.relation_candidates:
            self._support("connect_same_feature", 0.05, world.relation_candidates[0])
        if any(track.best_role[0] == "mutable_panel" for track in world.object_tracks.values()):
            self._support("state_match_before_goal", 0.08, "mutable panel candidate observed")
        if any(track.best_role[0] == "static_display" for track in world.object_tracks.values()):
            self._contradict("collect_before_goal", 0.01, "static display candidates should not be treated as items")

    def observe_transition(self, world: Any, after_notes: dict[str, Any]) -> None:
        removed = list((after_notes.get("collectible_changes", {}) or {}).get("removed", []) or [])
        anchor_changes = list(after_notes.get("anchor_patch_changes", []) or [])
        feedback = bool(after_notes.get("likely_feedback_flash", False))
        interaction_hint = str(after_notes.get("interaction_hint", "unknown"))
        if removed:
            self._support("collect_before_goal", 0.22, f"collectible removed {removed[:2]}")
            self._support("sequence_required", 0.04, "at least one interaction changed inventory-like state")
            self._contradict("plain_navigation", 0.04, "collectible interaction mattered")
        if removed and anchor_changes:
            self._support("state_match_before_goal", 0.34, f"collectible removal changed anchor patches {anchor_changes}")
            self._contradict("plain_navigation", 0.10, "panel-like state changed after collection")
        if feedback:
            self._support("state_match_before_goal", 0.24, "feedback flash without success")
            self._contradict("plain_navigation", 0.12, "direct progress produced failure feedback")
        if interaction_hint in {"spawn_or_unlock", "board_or_room_transform"} and world.visible_button_cells:
            self._support("switch_opens_path", 0.18, f"button-like interaction caused {interaction_hint}")
        if interaction_hint == "hud_or_counter_update":
            self._contradict("plain_navigation", 0.01, "HUD-only update is not task progress")
        if len([event for event in world.recent_events if event.kind == "collectible_removed"]) >= 2:
            self._support("sequence_required", 0.14, "multiple collectible interactions observed")

    def ranked(self) -> list[HypothesisFamily]:
        return sorted(self.families.values(), key=lambda family: (-family.confidence, family.name))

    def lines(self, limit: int = 6) -> list[str]:
        lines: list[str] = []
        for family in self.ranked()[:limit]:
            evidence = "; ".join(family.evidence[-3:]) or "prior only"
            lines.append(f"{family.name} confidence={family.confidence:.2f} evidence={evidence}")
        return lines

    def proposed_tests(self, world: Any, limit: int = 6) -> list[str]:
        tests: list[str] = []
        for family in self.ranked():
            for test in family.proposed_tests(world):
                if test not in tests:
                    tests.append(test)
                if len(tests) >= limit:
                    return tests
        return tests

    def preferred_subgoals(self, world: Any, limit: int = 8) -> list[dict[str, Any]]:
        subgoals: list[dict[str, Any]] = []
        for family in self.ranked():
            for subgoal in family.preferred_subgoals(world):
                if subgoal not in subgoals:
                    subgoals.append(subgoal)
                if len(subgoals) >= limit:
                    return subgoals
        return subgoals

    def confidence(self, name: str) -> float:
        family = self.families.get(name)
        return family.confidence if family is not None else 0.0

    def apply_experiment_result(self, kind: str, status: str, evidence: str) -> None:
        if kind == "collect_item":
            if status == "confirmed":
                self._support("collect_before_goal", 0.18, evidence)
                self._support("sequence_required", 0.04, "collection experiment confirmed")
            elif status == "contradicted":
                self._contradict("collect_before_goal", 0.08, evidence)
        elif kind == "activate_button":
            if status == "confirmed":
                self._support("switch_opens_path", 0.18, evidence)
            elif status == "contradicted":
                self._contradict("switch_opens_path", 0.08, evidence)
        elif kind == "go_to_goal":
            if status == "confirmed":
                self._support("plain_navigation", 0.24, evidence)
            elif status == "contradicted":
                self._contradict("plain_navigation", 0.16, evidence)
                self._support("state_match_before_goal", 0.12, evidence)
        elif kind == "inspect_relation":
            if status == "confirmed":
                self._support("connect_same_feature", 0.22, evidence)
            elif status == "contradicted":
                self._contradict("connect_same_feature", 0.08, evidence)

    def _support(self, name: str, amount: float, evidence: str) -> None:
        self.families[name].support(amount, evidence)

    def _contradict(self, name: str, amount: float, evidence: str) -> None:
        self.families[name].contradict(amount, evidence)
