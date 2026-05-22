from __future__ import annotations

from arc_agi3.abstraction.state_lifter import LiftedState
from arc_agi3.reasoning.hypotheses import Hypothesis, HypothesisKind


class HypothesisRefiner:
    """Generate compact structure hypotheses from a lifted state."""

    def generate(self, state: LiftedState) -> list[Hypothesis]:
        notes = state.notes
        family = str(notes.get("external_task_family", "unknown"))
        control_family = str(notes.get("runtime_control_family", "unknown"))
        strategy = str(notes.get("runtime_recommended_strategy", "generic_refinement_probe"))
        movement_probe_done = bool(notes.get("runtime_movement_probe_done", False))
        mode_probe_done = bool(notes.get("runtime_mode_probe_done", True))
        coordinate_probe_count = int(notes.get("runtime_coordinate_probe_count", 0) or 0)
        scene_goals = [str(goal) for goal in notes.get("scene_goal_kinds", [])]
        scene_targets = tuple(notes.get("scene_goal_targets", []) or state.candidate_clicks[:12])
        hypotheses: list[Hypothesis] = []

        hypotheses.extend(self._goal_conditioned_hypotheses(scene_goals, scene_targets))

        if control_family == "coordinate_only":
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.EDITABLE_REGION,
                    summary=f"{strategy}: probe object-derived coordinate candidates",
                    target=scene_targets[:12],
                    confidence=0.8,
                    uncertainty=0.25,
                    evidence=["ACTION6 is the only effective control family"],
                )
            )
            if len(state.markers) >= 2:
                hypotheses.append(
                    Hypothesis(
                        kind=HypothesisKind.ALIGNMENT_GOAL,
                        summary="coordinate-only task may be solved by selecting matching markers",
                        target=scene_targets[:12] or tuple(marker.center for marker in state.markers[:6]),
                        confidence=0.45,
                        uncertainty=0.35,
                        evidence=["compact marker candidates"],
                    )
                )
            return hypotheses

        if control_family == "movement_only":
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.MOVEMENT_AXIS,
                    summary=f"{strategy}: learn ACTION1-4 movement axes",
                    confidence=0.75,
                    uncertainty=0.25,
                    evidence=["only simple movement-like actions are available"],
                )
            )
            return hypotheses

        if control_family == "movement_mode":
            if not mode_probe_done:
                hypotheses.append(
                    Hypothesis(
                        kind=HypothesisKind.MODE_SWITCH,
                        summary="probe ACTION5/ACTION7 before committing to movement model",
                        confidence=0.8,
                        uncertainty=0.3,
                        evidence=["movement plus mode/tool action signature"],
                    )
                )
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.MOVEMENT_AXIS,
                    summary=f"{strategy}: compare movement effects before and after mode actions",
                    confidence=0.65,
                    uncertainty=0.35,
                    evidence=["movement plus mode/tool action signature"],
                )
            )
            return hypotheses

        if control_family in {"mixed_move_coordinate", "coordinate_mode"}:
            if not mode_probe_done:
                hypotheses.append(
                    Hypothesis(
                        kind=HypothesisKind.MODE_SWITCH,
                        summary="probe tool/mode action before coordinate interactions",
                        confidence=0.75,
                        uncertainty=0.3,
                        evidence=["coordinate task exposes ACTION5/ACTION7"],
                    )
                )
            if not movement_probe_done and control_family == "mixed_move_coordinate":
                hypotheses.append(
                    Hypothesis(
                        kind=HypothesisKind.MOVEMENT_AXIS,
                        summary="briefly learn simple action effects, then switch to object clicks",
                        confidence=0.62,
                        uncertainty=0.35,
                        evidence=["mixed movement and coordinate action signature"],
                    )
                )
            if control_family == "mixed_move_coordinate" and not movement_probe_done:
                click_confidence = 0.45
            elif coordinate_probe_count == 0:
                click_confidence = 0.78
            else:
                click_confidence = 0.68
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.EDITABLE_REGION,
                    summary=f"{strategy}: use object-aware ACTION6 candidates",
                    target=scene_targets[:12],
                    confidence=click_confidence,
                    uncertainty=0.25,
                    evidence=["public source priors favor mixed move-coordinate controls"],
                )
            )
            if len(state.markers) >= 2:
                hypotheses.append(
                    Hypothesis(
                        kind=HypothesisKind.ALIGNMENT_GOAL,
                        summary="small marker/object relations may define coordinate targets",
                        target=scene_targets[:12] or tuple(marker.center for marker in state.markers[:6]),
                        confidence=0.5,
                        uncertainty=0.35,
                        evidence=["multiple compact marker candidates"],
                    )
                )
            return hypotheses

        if family in {"navigation", "unknown"}:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.MOVEMENT_AXIS,
                    summary="simple actions may control movement axes",
                    evidence=["compressarc same-size/simple-color prior"],
                )
            )
        if family in {"editing", "unknown"} or len(state.color_histogram) >= 5:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.EDITABLE_REGION,
                    summary="actions may edit a workspace or select a tool",
                    target=scene_targets[:8],
                    evidence=["high color/layout complexity"],
                )
            )
        if family in {"transform", "unknown"}:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.OBJECT_TRANSFORM,
                    summary="simple actions may transform active objects",
                    evidence=["same-size multi-color transition prior"],
                )
            )
        if len(state.markers) >= 2:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.ALIGNMENT_GOAL,
                    summary="small marker/object relations may define progress",
                    target=tuple(marker.center for marker in state.markers[:4]),
                    evidence=["multiple compact marker candidates"],
                )
            )
        if not hypotheses:
            hypotheses.append(Hypothesis(kind=HypothesisKind.UNKNOWN, summary="unknown structure; probe cheapest actions"))
        return hypotheses

    def _goal_conditioned_hypotheses(
        self,
        scene_goals: list[str],
        scene_targets: tuple[object, ...],
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        if "paint_reference" in scene_goals:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.EDITABLE_REGION,
                    summary="scene goal suggests copying/painting a reference pattern",
                    target=scene_targets[:14],
                    confidence=0.7,
                    uncertainty=0.25,
                    evidence=["scene blueprint inferred paint_reference"],
                )
            )
        if "match_shape" in scene_goals or "align_markers" in scene_goals:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.ALIGNMENT_GOAL,
                    summary="scene goal suggests matching shapes or aligning markers",
                    target=scene_targets[:14],
                    confidence=0.68,
                    uncertainty=0.3,
                    evidence=["scene blueprint inferred relational goal"],
                )
            )
        if "move_to_marker" in scene_goals or "clear_obstacle" in scene_goals:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.MOVEMENT_AXIS,
                    summary="scene goal suggests moving toward markers or clearing blockers",
                    target=scene_targets[:10],
                    confidence=0.58,
                    uncertainty=0.35,
                    evidence=["scene blueprint inferred movement/blocker goal"],
                )
            )
        if "transform_object" in scene_goals:
            hypotheses.append(
                Hypothesis(
                    kind=HypothesisKind.OBJECT_TRANSFORM,
                    summary="scene goal suggests rotating or transforming an active object",
                    target=scene_targets[:10],
                    confidence=0.55,
                    uncertainty=0.4,
                    evidence=["scene blueprint inferred transform goal"],
                )
            )
        return hypotheses

    def refine(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        seen: set[tuple[HypothesisKind, str]] = set()
        refined: list[Hypothesis] = []
        for hypothesis in hypotheses:
            key = (hypothesis.kind, hypothesis.summary)
            if key in seen:
                continue
            seen.add(key)
            refined.append(hypothesis)
        return refined
