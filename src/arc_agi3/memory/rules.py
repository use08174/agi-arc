from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action


@dataclass(slots=True)
class RuleCandidate:
    category: str
    subject: str
    target: str
    support_count: int = 0
    contradiction_count: int = 0
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.category}:{self.subject}->{self.target}"

    @property
    def total_observations(self) -> int:
        return self.support_count + self.contradiction_count

    @property
    def confidence(self) -> float:
        total = self.total_observations
        if total <= 0:
            return 0.5
        return self.support_count / total

    @property
    def uncertainty(self) -> float:
        total_term = 1.0 / (1.0 + self.total_observations)
        balance_term = 1.0 - abs(self.confidence - 0.5) * 2.0
        return max(0.0, min(1.0, 0.55 * total_term + 0.45 * balance_term))

    def support(self, evidence: str) -> None:
        self.support_count += 1
        if evidence not in self.evidence:
            self.evidence.append(evidence)

    def contradict(self, evidence: str) -> None:
        self.contradiction_count += 1
        if evidence not in self.contradictions:
            self.contradictions.append(evidence)


class RuleLibrary:
    """Explicit rule candidates used to drive focused experiments."""

    def __init__(self) -> None:
        self.rules: dict[str, RuleCandidate] = {}

    def observe_scene(self, world: Any) -> None:
        match = getattr(world, "reference_workspace_match", None)
        if not isinstance(match, dict):
            return
        reference_anchor = str(match.get("reference_anchor", ""))
        workspace_anchor = str(match.get("workspace_anchor", ""))
        alignment_score = float(match.get("alignment_score", 0.0) or 0.0)
        if not reference_anchor or not workspace_anchor or alignment_score <= 0.0:
            return
        rule = self._get_or_create("reference_for", reference_anchor, workspace_anchor)
        if alignment_score >= 0.45:
            rule.support(f"alignment_score={alignment_score:.2f}")
        elif alignment_score < 0.20:
            rule.contradict(f"weak_alignment={alignment_score:.2f}")

    def observe_transition(
        self,
        world: Any,
        action: Action,
        before_notes: dict[str, Any],
        after_notes: dict[str, Any],
        before_latent: dict[str, str] | None = None,
        after_latent: dict[str, str] | None = None,
    ) -> None:
        before_latent = before_latent or {}
        after_latent = after_latent or {}
        interaction_hint = str(after_notes.get("interaction_hint", "unknown"))
        anchor_changes = list(after_notes.get("anchor_patch_changes", []) or [])
        semantic_player_moved = bool(after_notes.get("semantic_player_moved", False))
        changed_cells = int(after_notes.get("changed_cells", 0) or 0)
        changed_playfield_cells = int(after_notes.get("changed_playfield_cells", 0) or 0)
        region_bias = str(after_notes.get("region_bias", "unknown"))
        alignment_delta = float(after_notes.get("reference_workspace_alignment_delta", 0.0) or 0.0)
        latent_changes = {
            key: (before_latent.get(key), after_latent.get(key))
            for key in set(before_latent) | set(after_latent)
            if before_latent.get(key) != after_latent.get(key) and after_latent.get(key) is not None
        }

        mode_rule = self._get_or_create("mode_setting", action.key, "latent_mode")
        if (
            not semantic_player_moved
            and (
                anchor_changes
                or interaction_hint in {"spawn_or_unlock", "board_or_room_transform"}
                or alignment_delta > 0.08
            )
        ):
            mode_rule.support(
                f"hint={interaction_hint} anchors={len(anchor_changes)} alignment_delta={alignment_delta:.2f}"
            )
        elif changed_cells == 0 and interaction_hint == "unknown":
            mode_rule.contradict("noop_without_mode_evidence")

        for variable_name, (_, after_value) in latent_changes.items():
            if after_value is None:
                continue
            variable_rule = self._get_or_create("changes_variable", action.key, variable_name)
            variable_rule.support(f"{variable_name}={after_value}")
            if variable_name in {"mode_state", "selected_tool", "selected_color", "control_mode"}:
                set_rule = self._get_or_create("sets_mode", action.key, str(after_value))
                set_rule.support(f"{variable_name}={after_value}")

        if not latent_changes and changed_cells == 0 and interaction_hint == "unknown":
            for variable_name in ("mode_state", "selected_tool", "selected_color", "control_mode"):
                self._get_or_create("changes_variable", action.key, variable_name).contradict("no_latent_delta")
            mode_rule.contradict("no_latent_delta")

        affect_target = "workspace" if alignment_delta != 0 else region_bias
        affect_rule = self._get_or_create("affects_region", action.key, affect_target)
        if changed_playfield_cells > 0 or alignment_delta != 0:
            affect_rule.support(
                f"playfield_change={changed_playfield_cells} alignment_delta={alignment_delta:.2f}"
            )
        elif changed_cells == 0:
            affect_rule.contradict("no_region_change")

    def focus_action_keys(self, actions: list[Action], limit: int = 4) -> set[str]:
        available = {action.key for action in actions}
        ranked = [
            rule
            for rule in self.rules.values()
            if rule.category in {"mode_setting", "affects_region", "sets_mode", "changes_variable"} and rule.subject in available
        ]
        ranked.sort(
            key=lambda rule: (
                -(rule.uncertainty + (0.20 if rule.total_observations <= 2 else 0.0)),
                rule.category,
                rule.subject,
            )
        )
        out: list[str] = []
        for rule in ranked:
            if rule.subject in out:
                continue
            out.append(rule.subject)
            if len(out) >= limit:
                break
        return set(out)

    def priority_for_proposal(self, proposal: Any) -> float:
        kind = getattr(proposal, "kind", "")
        target = getattr(proposal, "target", None)
        if kind == "probe_action" and isinstance(target, str):
            return self._action_rule_priority(target)
        if kind == "probe_action_pair" and isinstance(target, dict):
            first_key = str(target.get("first_action_key", ""))
            second_key = str(target.get("second_action_key", ""))
            return self._action_rule_priority(first_key) + self._action_rule_priority(second_key) * 0.5
        if kind in {"collect_item", "activate_button", "go_to_goal"}:
            reference_rules = [rule for rule in self.rules.values() if rule.category == "reference_for"]
            if any(rule.confidence >= 0.6 and rule.uncertainty >= 0.2 for rule in reference_rules):
                return 0.15
        return 0.0

    def lines(self, limit: int = 6) -> list[str]:
        lines: list[str] = []
        for rule in self.ranked()[:limit]:
            lines.append(
                f"{rule.category} {rule.subject}->{rule.target} "
                f"conf={rule.confidence:.2f} unc={rule.uncertainty:.2f}"
            )
        return lines

    def ranked(self) -> list[RuleCandidate]:
        return sorted(
            self.rules.values(),
            key=lambda rule: (-rule.uncertainty, -rule.confidence, rule.category, rule.subject, rule.target),
        )

    def _action_rule_priority(self, action_key: str) -> float:
        priorities = [
            rule.uncertainty
            for rule in self.rules.values()
            if rule.subject == action_key and rule.category in {"mode_setting", "affects_region", "sets_mode", "changes_variable"}
        ]
        return max(priorities, default=0.0)

    def _get_or_create(self, category: str, subject: str, target: str) -> RuleCandidate:
        key = f"{category}:{subject}->{target}"
        rule = self.rules.get(key)
        if rule is None:
            rule = RuleCandidate(category=category, subject=subject, target=target)
            self.rules[key] = rule
        return rule
