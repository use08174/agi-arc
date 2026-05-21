from __future__ import annotations

from arc_agi3.abstraction.state_lifter import LiftedState
from arc_agi3.reasoning.hypotheses import Hypothesis, HypothesisKind


class HypothesisRefiner:
    """Generate compact structure hypotheses from a lifted state."""

    def generate(self, state: LiftedState) -> list[Hypothesis]:
        notes = state.notes
        family = str(notes.get("external_task_family", "unknown"))
        hypotheses: list[Hypothesis] = []
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
                    target=state.candidate_clicks[:6],
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
