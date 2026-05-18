from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from arc_agi3.core.types import Action
from arc_agi3.memory.hypotheses import HypothesisLibrary

Cell = tuple[int, int]
Edge = tuple[Cell, str]
SafeEdge = tuple[Cell, str, Cell]


@dataclass(slots=True)
class ObjectRule:
    """Generalized object interaction rule.

    Example:
      color=3, shape_signature=abcd, role=item,
      interaction=click_removes, action_name=ACTION6
    means: objects of this type should be clicked.
    """

    color: int
    shape_signature: str | None
    role: str
    interaction: str
    action_name: str
    support: int = 0
    failures: int = 0

    @property
    def confidence(self) -> float:
        total = self.support + self.failures
        if total <= 0:
            return 0.0
        return self.support / total

    @property
    def key(self) -> str:
        return object_rule_key(
            color=self.color,
            shape_signature=self.shape_signature,
            action_name=self.action_name,
            interaction=self.interaction,
        )


@dataclass(slots=True)
class ObjectTrack:
    track_id: str
    color: int
    shape_signature: str | None
    center: Cell
    area: int
    seen_count: int = 1
    moved_count: int = 0
    disappeared_count: int = 0
    changed_count: int = 0
    role_scores: dict[str, float] = field(default_factory=dict)

    def bump_role(self, role: str, amount: float) -> None:
        self.role_scores[role] = min(1.0, self.role_scores.get(role, 0.0) + amount)

    @property
    def best_role(self) -> tuple[str, float]:
        if not self.role_scores:
            return "unknown", 0.0
        return max(self.role_scores.items(), key=lambda item: item[1])


@dataclass(slots=True)
class SceneEvent:
    kind: str
    detail: str
    support: float = 1.0


@dataclass(slots=True)
class RelationCandidate:
    key: str
    color: int
    centers: list[Cell]
    nearest_pair: tuple[Cell, Cell]
    min_distance: int

    def summary(self) -> str:
        return (
            f"same_color_regions color={self.color} count={len(self.centers)} "
            f"min_distance={self.min_distance} nearest_pair={self.nearest_pair}"
        )


@dataclass(slots=True)
class GoalHypothesis:
    name: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def support(self, amount: float, evidence: str) -> None:
        self.confidence = min(1.0, self.confidence + amount)
        if evidence not in self.evidence:
            self.evidence.append(evidence)


def object_rule_key(
    color: int,
    shape_signature: str | None,
    action_name: str,
    interaction: str,
) -> str:
    return f"color={color}|shape={shape_signature or '*'}|action={action_name}|interaction={interaction}"


@dataclass(slots=True)
class WorldModel:
    """Compact world model updated online from semantic observations.

    It is deliberately conservative: unknown cells are not treated as hazards,
    but known hazards/deadly edges are never selected by the safety shield.
    """

    player_pos: Cell | None = None
    previous_player_pos: Cell | None = None
    known_traversable_cells: set[Cell] = field(default_factory=set)
    known_blocked_cells: set[Cell] = field(default_factory=set)
    known_hazard_cells: set[Cell] = field(default_factory=set)
    known_item_cells: set[Cell] = field(default_factory=set)
    known_goal_cells: set[Cell] = field(default_factory=set)
    known_button_cells: set[Cell] = field(default_factory=set)
    visible_item_cells: set[Cell] = field(default_factory=set)
    visible_goal_cells: set[Cell] = field(default_factory=set)
    visible_button_cells: set[Cell] = field(default_factory=set)
    safe_edges: set[SafeEdge] = field(default_factory=set)
    deadly_edges: set[Edge] = field(default_factory=set)
    action_move_vectors: dict[str, Cell] = field(default_factory=dict)
    action_move_votes: dict[str, Counter[Cell]] = field(default_factory=dict)
    object_rules: dict[str, ObjectRule] = field(default_factory=dict)
    object_tracks: dict[str, ObjectTrack] = field(default_factory=dict)
    recent_events: list[SceneEvent] = field(default_factory=list)
    hypotheses: dict[str, GoalHypothesis] = field(default_factory=dict)
    hypothesis_library: HypothesisLibrary = field(default_factory=HypothesisLibrary)
    relation_candidates: list[str] = field(default_factory=list)
    relation_details: dict[str, RelationCandidate] = field(default_factory=dict)
    anchor_patch_states: dict[str, str] = field(default_factory=dict)
    anchor_patch_change_counts: Counter[str] = field(default_factory=Counter)
    semantic_ascii_map: str = ""
    last_objects: list[dict[str, Any]] = field(default_factory=list)
    last_notes: dict[str, Any] = field(default_factory=dict)

    def update_from_observation(self, notes: dict[str, Any]) -> None:
        self.last_notes = dict(notes)
        self.semantic_ascii_map = str(notes.get("semantic_ascii_map", ""))
        self.anchor_patch_states = {
            str(item.get("anchor")): str(item.get("signature"))
            for item in notes.get("anchor_patch_summary", []) or []
            if isinstance(item, dict) and item.get("anchor") and item.get("signature")
        }
        player = notes.get("semantic_player")
        if isinstance(player, dict):
            pos = _center_from_bbox(player.get("bbox"))
            if pos is not None:
                self.previous_player_pos = self.player_pos
                self.player_pos = pos
                self.known_traversable_cells.add(pos)
        elif isinstance(notes.get("semantic_player_pos"), tuple):
            pos = notes["semantic_player_pos"]
            self.previous_player_pos = self.player_pos
            self.player_pos = pos
            self.known_traversable_cells.add(pos)

        self.visible_item_cells = set()
        self.visible_goal_cells = set()
        self.visible_button_cells = set()
        for item in notes.get("semantic_items", []) or []:
            center = _center_from_bbox(item.get("bbox"))
            if center is not None:
                self.known_item_cells.add(center)
                self.visible_item_cells.add(center)
        for item in notes.get("semantic_goals", []) or []:
            center = _center_from_bbox(item.get("bbox"))
            if center is not None:
                self.known_goal_cells.add(center)
                self.visible_goal_cells.add(center)
        for item in notes.get("semantic_buttons", []) or []:
            center = _center_from_bbox(item.get("bbox"))
            if center is not None:
                self.known_button_cells.add(center)
                self.visible_button_cells.add(center)
        for item in notes.get("semantic_walls", []) or []:
            bbox = item.get("bbox")
            if isinstance(bbox, dict):
                for y in range(int(bbox["min_y"]), int(bbox["max_y"]) + 1):
                    for x in range(int(bbox["min_x"]), int(bbox["max_x"]) + 1):
                        self.known_blocked_cells.add((x, y))

        objects = notes.get("semantic_objects", [])
        if isinstance(objects, list):
            self.last_objects = [obj for obj in objects if isinstance(obj, dict)]
            self._update_tracks(self.last_objects)
            self._update_relation_candidates(self.last_objects)
        self.hypothesis_library.observe_scene(self)

    def learn_transition(self, action: Action, before_notes: dict[str, Any], after_notes: dict[str, Any], terminal_loss: bool) -> None:
        before_player = _player_pos(before_notes)
        after_player = _player_pos(after_notes)
        if before_player is not None:
            self.known_traversable_cells.add(before_player)
        if after_player is not None:
            self.known_traversable_cells.add(after_player)
            self.previous_player_pos = before_player
            self.player_pos = after_player

        # Learn movement vector per action name, not per coordinate key.
        if before_player is not None and after_player is not None:
            dx = after_player[0] - before_player[0]
            dy = after_player[1] - before_player[1]
            if (dx, dy) != (0, 0):
                self.action_move_votes.setdefault(action.name, Counter())[(dx, dy)] += 1
                self.action_move_vectors[action.name] = self.action_move_votes[action.name].most_common(1)[0][0]
                self.safe_edges.add((before_player, action.name, after_player))
            else:
                target = self.predicted_target(before_player, action.name)
                if target is not None:
                    self.known_blocked_cells.add(target)

        if terminal_loss and before_player is not None:
            target = self.predicted_target(before_player, action.name)
            if target is not None:
                self.deadly_edges.add((before_player, action.name))
                self.known_hazard_cells.add(target)

        self._learn_object_rules(action, before_notes, after_notes)
        self._learn_events_and_hypotheses(before_notes, after_notes)
        self.hypothesis_library.observe_transition(self, after_notes)

    def predicted_target(self, pos: Cell | None, action_name: str) -> Cell | None:
        if pos is None:
            return None
        vec = self.action_move_vectors.get(action_name)
        if vec is None:
            return None
        return (pos[0] + vec[0], pos[1] + vec[1])

    def is_unsafe_action(self, action: Action, pos: Cell | None = None) -> bool:
        pos = pos if pos is not None else self.player_pos
        if pos is None:
            return False
        if (pos, action.name) in self.deadly_edges:
            return True
        target = self.predicted_target(pos, action.name)
        if target is None:
            return False
        return target in self.known_hazard_cells or target in self.known_blocked_cells

    def preferred_click_targets(self, limit: int = 8) -> list[Cell]:
        targets: list[Cell] = []
        for obj in self.last_objects:
            role = str(obj.get("role", "unknown"))
            color = int(obj.get("color", 0) or 0)
            shape = str(obj.get("shape_signature", "")) or None
            center = _center_from_bbox(obj.get("bbox"))
            if center is None:
                continue
            anchor = str(obj.get("anchor", ""))
            if role in {"display_candidate", "static_display"} or anchor.startswith("bottom_"):
                continue
            if role in {"item", "button", "goal"}:
                targets.append(center)
                continue
            # Apply learned generalized rule such as green-box click_removes.
            for rule in self.object_rules.values():
                if rule.action_name != "ACTION6":
                    continue
                if rule.confidence < 0.5 or rule.support <= 0:
                    continue
                if rule.color == color and (rule.shape_signature in {None, shape}):
                    targets.append(center)
                    break
        # Deduplicate while preserving order.
        out: list[Cell] = []
        seen: set[Cell] = set()
        for cell in targets:
            if cell in seen:
                continue
            seen.add(cell)
            out.append(cell)
            if len(out) >= limit:
                break
        return out

    def summary_lines(self) -> list[str]:
        lines = []
        lines.append(f"player={self.player_pos}")
        if self.action_move_vectors:
            moves = ", ".join(f"{name}->{vec}" for name, vec in sorted(self.action_move_vectors.items()))
            lines.append(f"learned_moves={moves}")
        if self.known_item_cells:
            lines.append(f"visible_items={sorted(self.visible_item_cells)[:8]} known_items={sorted(self.known_item_cells)[:8]}")
        if self.known_goal_cells:
            lines.append(f"known_goals={sorted(self.known_goal_cells)[:8]}")
        if self.known_blocked_cells:
            lines.append(f"blocked_count={len(self.known_blocked_cells)} sample={sorted(self.known_blocked_cells)[:8]}")
        if self.known_hazard_cells:
            lines.append(f"hazards={sorted(self.known_hazard_cells)[:8]}")
        if self.deadly_edges:
            lines.append(f"deadly_edges={sorted(self.deadly_edges)[:8]}")
        if self.object_rules:
            rendered = []
            for rule in sorted(self.object_rules.values(), key=lambda r: (-r.support, r.key))[:8]:
                rendered.append(
                    f"color={rule.color} shape={rule.shape_signature} role={rule.role} "
                    f"interaction={rule.interaction} action={rule.action_name} "
                    f"support={rule.support} failures={rule.failures} conf={rule.confidence:.2f}"
                )
            lines.append("object_rules=" + " | ".join(rendered))
        if self.object_tracks:
            rendered_tracks = []
            for track in sorted(self.object_tracks.values(), key=lambda item: (-item.seen_count, item.track_id))[:8]:
                role, confidence = track.best_role
                rendered_tracks.append(
                    f"{track.track_id} color={track.color} center={track.center} shape={track.shape_signature} "
                    f"seen={track.seen_count} moved={track.moved_count} disappeared={track.disappeared_count} "
                    f"role={role}:{confidence:.2f}"
                )
            lines.append("object_tracks=" + " | ".join(rendered_tracks))
        if self.relation_candidates:
            lines.append("relation_candidates=" + " | ".join(self.relation_candidates[:6]))
        if self.hypotheses:
            rendered_hypotheses = []
            for hypothesis in sorted(self.hypotheses.values(), key=lambda item: (-item.confidence, item.name))[:6]:
                rendered_hypotheses.append(f"{hypothesis.name}:{hypothesis.confidence:.2f}")
            lines.append("goal_hypotheses=" + " | ".join(rendered_hypotheses))
        lines.append("hypothesis_families=" + " | ".join(f"{item.name}:{item.confidence:.2f}" for item in self.hypothesis_library.ranked()[:6]))
        return lines

    def event_lines(self, limit: int = 10) -> list[str]:
        return [f"{event.kind}: {event.detail}" for event in self.recent_events[-limit:]]

    def hypothesis_lines(self, limit: int = 6) -> list[str]:
        lines = self.hypothesis_library.lines(limit=limit)
        for hypothesis in sorted(self.hypotheses.values(), key=lambda item: (-item.confidence, item.name))[:limit]:
            evidence = "; ".join(hypothesis.evidence[-3:])
            line = f"{hypothesis.name} confidence={hypothesis.confidence:.2f} evidence={evidence}"
            if line not in lines and len(lines) < limit:
                lines.append(line)
        return lines

    def has_precondition_evidence(self) -> bool:
        legacy = any(
            self.hypotheses.get(name, GoalHypothesis(name)).confidence >= 0.30
            for name in ("collection_changes_state", "goal_requires_state_match")
        )
        return legacy or self.hypothesis_library.confidence("state_match_before_goal") >= 0.30

    def relation_for(self, key: str) -> RelationCandidate | None:
        return self.relation_details.get(key)

    def _learn_object_rules(self, action: Action, before_notes: dict[str, Any], after_notes: dict[str, Any]) -> None:
        if action.name != "ACTION6":
            return
        clicked = None
        if action.payload:
            x = int(action.payload.get("x", -1))
            y = int(action.payload.get("y", -1))
            clicked = _object_at(before_notes, (x, y))
        if clicked is None:
            clicked = _nearest_small_object(before_notes)
        if clicked is None:
            return

        removed = (after_notes.get("collectible_changes", {}) or {}).get("removed", []) or []
        progress = bool(after_notes.get("collectible_progress", False)) or bool(removed)
        color = int(clicked.get("color", 0) or 0)
        shape = str(clicked.get("shape_signature", "")) or None
        role = str(clicked.get("role", "item" if progress else "unknown"))
        interaction = "click_removes" if progress else "click_no_effect"
        key = object_rule_key(color, shape, action.name, interaction)
        rule = self.object_rules.get(key)
        if rule is None:
            rule = ObjectRule(color=color, shape_signature=shape, role=role, interaction=interaction, action_name=action.name)
            self.object_rules[key] = rule
        if progress:
            rule.support += 1
        else:
            rule.failures += 1

    def _update_tracks(self, objects: list[dict[str, Any]]) -> None:
        unmatched = set(self.object_tracks)
        for obj in objects:
            center = _center_from_bbox(obj.get("bbox"))
            if center is None:
                continue
            color = int(obj.get("color", 0) or 0)
            shape = str(obj.get("shape_signature", "")) or None
            match = self._best_track_match(color, shape, center, unmatched)
            if match is None:
                track_id = f"obj_{len(self.object_tracks) + 1}"
                track = ObjectTrack(track_id=track_id, color=color, shape_signature=shape, center=center, area=int(obj.get("area", 0) or 0))
                self.object_tracks[track_id] = track
            else:
                track = self.object_tracks[match]
                if track.center != center:
                    track.moved_count += 1
                track.center = center
                track.area = int(obj.get("area", track.area) or track.area)
                track.seen_count += 1
                unmatched.discard(match)
            role = str(obj.get("role", "unknown"))
            if role != "unknown":
                track.bump_role(role, float(obj.get("confidence", 0.0) or 0.0) * 0.25)
            anchor = str(obj.get("anchor", ""))
            if anchor.startswith("bottom_"):
                track.bump_role("display_candidate", 0.06)
        for track_id in unmatched:
            self.object_tracks[track_id].disappeared_count += 1

    def _best_track_match(self, color: int, shape: str | None, center: Cell, candidates: set[str]) -> str | None:
        ranked: list[tuple[int, str]] = []
        for track_id in candidates:
            track = self.object_tracks[track_id]
            if track.color != color or track.shape_signature != shape:
                continue
            dist = abs(track.center[0] - center[0]) + abs(track.center[1] - center[1])
            if dist <= 6:
                ranked.append((dist, track_id))
        return min(ranked)[1] if ranked else None

    def _update_relation_candidates(self, objects: list[dict[str, Any]]) -> None:
        by_color: dict[int, list[dict[str, Any]]] = {}
        for obj in objects:
            area = int(obj.get("area", 0) or 0)
            if 2 <= area <= 64:
                by_color.setdefault(int(obj.get("color", 0) or 0), []).append(obj)
        candidates: list[str] = []
        details: dict[str, RelationCandidate] = {}
        for color, members in by_color.items():
            if len(members) < 2:
                continue
            centers = [_center_from_bbox(item.get("bbox")) for item in members]
            valid = [center for center in centers if center is not None]
            if len(valid) < 2:
                continue
            best_distance, best_pair = min(
                (
                    abs(a[0] - b[0]) + abs(a[1] - b[1]),
                    (a, b),
                )
                for idx, a in enumerate(valid)
                for b in valid[idx + 1 :]
            )
            key = f"same_color:{color}"
            relation = RelationCandidate(
                key=key,
                color=color,
                centers=valid,
                nearest_pair=best_pair,
                min_distance=best_distance,
            )
            details[key] = relation
            candidates.append(relation.summary())
            if best_distance > 1:
                self._support_hypothesis(
                    "connect_same_color_regions",
                    0.08,
                    f"same-color regions color={color} remain disconnected distance={best_distance}",
                )
        self.relation_candidates = candidates
        self.relation_details = details

    def _learn_events_and_hypotheses(self, before_notes: dict[str, Any], after_notes: dict[str, Any]) -> None:
        removed = list((after_notes.get("collectible_changes", {}) or {}).get("removed", []) or [])
        anchor_changes = list(after_notes.get("anchor_patch_changes", []) or [])
        feedback = bool(after_notes.get("likely_feedback_flash", False))
        if removed:
            detail = f"collectible_removed={removed[:3]}"
            self._remember_event("collectible_removed", detail)
        if anchor_changes:
            detail = f"anchor_patches_changed={anchor_changes}"
            self._remember_event("anchor_patch_changed", detail)
            for anchor in anchor_changes:
                self.anchor_patch_change_counts[anchor] += 1
        if removed and anchor_changes:
            self._support_hypothesis("collection_changes_state", 0.35, f"{removed[:2]} followed by patch changes {anchor_changes}")
            self._mark_recent_disappeared_as("collectible", 0.45)
        if feedback:
            self._remember_event("feedback_flash", f"anchor_changes={anchor_changes or 'none'}")
            self._support_hypothesis("goal_requires_state_match", 0.25, "feedback flash observed without reward")
            if anchor_changes:
                self._support_hypothesis("goal_requires_state_match", 0.15, f"feedback coincided with patch changes {anchor_changes}")
        self._update_display_roles(anchor_changes)

    def _remember_event(self, kind: str, detail: str) -> None:
        event = SceneEvent(kind=kind, detail=detail)
        if self.recent_events and self.recent_events[-1].kind == kind and self.recent_events[-1].detail == detail:
            return
        self.recent_events.append(event)
        del self.recent_events[:-24]

    def _support_hypothesis(self, name: str, amount: float, evidence: str) -> None:
        self.hypotheses.setdefault(name, GoalHypothesis(name=name)).support(amount, evidence)

    def _mark_recent_disappeared_as(self, role: str, amount: float) -> None:
        for track in self.object_tracks.values():
            if track.disappeared_count > 0:
                track.bump_role(role, amount)

    def _update_display_roles(self, anchor_changes: list[str]) -> None:
        for track in self.object_tracks.values():
            role, _ = track.best_role
            if role == "display_candidate" and track.moved_count == 0 and track.disappeared_count == 0:
                track.bump_role("static_display", 0.04)
        for anchor in anchor_changes:
            for track in self.object_tracks.values():
                anchor_guess = self._track_anchor(track)
                if anchor_guess == anchor:
                    track.changed_count += 1
                    track.bump_role("mutable_panel", 0.22)

    def _track_anchor(self, track: ObjectTrack) -> str:
        width = int(self.last_notes.get("semantic_width", 0) or 0)
        height = int(self.last_notes.get("semantic_height", 0) or 0)
        if width <= 0 or height <= 0:
            return ""
        x, y = track.center
        horizontal = "left" if x < width * 0.33 else "right" if x > width * 0.66 else "center"
        vertical = "top" if y < height * 0.33 else "bottom" if y > height * 0.66 else "middle"
        return f"{vertical}_{horizontal}"


def _center_from_bbox(bbox: Any) -> Cell | None:
    if not isinstance(bbox, dict):
        return None
    try:
        return ((int(bbox["min_x"]) + int(bbox["max_x"])) // 2, (int(bbox["min_y"]) + int(bbox["max_y"])) // 2)
    except Exception:
        return None


def _player_pos(notes: dict[str, Any]) -> Cell | None:
    player = notes.get("semantic_player")
    if isinstance(player, dict):
        return _center_from_bbox(player.get("bbox"))
    pos = notes.get("semantic_player_pos")
    if isinstance(pos, tuple) and len(pos) == 2:
        return int(pos[0]), int(pos[1])
    return None


def _object_at(notes: dict[str, Any], point: Cell) -> dict[str, Any] | None:
    x, y = point
    for obj in notes.get("semantic_objects", []) or []:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, dict):
            continue
        if int(bbox["min_x"]) <= x <= int(bbox["max_x"]) and int(bbox["min_y"]) <= y <= int(bbox["max_y"]):
            return obj
    return None


def _nearest_small_object(notes: dict[str, Any]) -> dict[str, Any] | None:
    objects = [obj for obj in notes.get("semantic_objects", []) or [] if isinstance(obj, dict)]
    objects = [obj for obj in objects if int(obj.get("area", 9999) or 9999) <= 16]
    return objects[0] if objects else None
