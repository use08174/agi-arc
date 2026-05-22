from __future__ import annotations

from arc_agi3.abstraction.state_lifter import LiftedState
from arc_agi3.reasoning.hypotheses import Hypothesis, HypothesisKind


class HypothesisCritic:
    """Score hypotheses from lifted state and CompressARC-style priors."""

    def critique(self, hypotheses: list[Hypothesis], state: LiftedState) -> list[Hypothesis]:
        colors = len([color for color in state.color_histogram if color != 0])
        same_size = bool(state.notes.get("compressarc_in_out_same_size")) and bool(state.notes.get("compressarc_all_in_same_size"))
        scene_goals = set(str(goal) for goal in state.notes.get("scene_goal_kinds", []))
        for hypothesis in hypotheses:
            if hypothesis.kind == HypothesisKind.MOVEMENT_AXIS:
                hypothesis.confidence += 0.25 if same_size and colors <= 3 else 0.05
                if {"move_to_marker", "clear_obstacle"} & scene_goals:
                    hypothesis.confidence += 0.12
                hypothesis.uncertainty = max(0.1, hypothesis.uncertainty - 0.15)
            elif hypothesis.kind == HypothesisKind.EDITABLE_REGION:
                hypothesis.confidence += 0.25 if colors >= 4 else 0.05
                if "paint_reference" in scene_goals:
                    hypothesis.confidence += 0.15
            elif hypothesis.kind == HypothesisKind.OBJECT_TRANSFORM:
                hypothesis.confidence += 0.2 if same_size and colors >= 4 else 0.08
                if "transform_object" in scene_goals:
                    hypothesis.confidence += 0.12
            elif hypothesis.kind == HypothesisKind.ALIGNMENT_GOAL:
                hypothesis.confidence += 0.15 if len(state.markers) >= 2 else 0.02
                if {"match_shape", "align_markers"} & scene_goals:
                    hypothesis.confidence += 0.15
        return sorted(hypotheses, key=lambda item: item.score, reverse=True)
