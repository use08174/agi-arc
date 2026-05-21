from __future__ import annotations

from arc_agi3.abstraction.object_ops import AbstractIntent, ObjectOp
from arc_agi3.abstraction.state_lifter import LiftedState
from arc_agi3.reasoning.hypotheses import Hypothesis, HypothesisKind


class HypothesisVerifier:
    """Turn current best hypotheses into abstract probes."""

    def intents_for(self, hypotheses: list[Hypothesis], state: LiftedState) -> list[AbstractIntent]:
        intents: list[AbstractIntent] = []
        for hypothesis in hypotheses[:4]:
            if hypothesis.kind == HypothesisKind.MOVEMENT_AXIS:
                intents.append(AbstractIntent(ObjectOp.PROBE_MOVE, priority=hypothesis.score, rationale=hypothesis.summary))
            elif hypothesis.kind == HypothesisKind.MODE_SWITCH:
                intents.append(AbstractIntent(ObjectOp.PROBE_MODE, priority=hypothesis.score, rationale=hypothesis.summary))
            elif hypothesis.kind == HypothesisKind.EDITABLE_REGION:
                intents.append(
                    AbstractIntent(
                        ObjectOp.PROBE_CLICK,
                        target=tuple(state.candidate_clicks[:8]),
                        priority=hypothesis.score,
                        rationale=hypothesis.summary,
                    )
                )
            elif hypothesis.kind == HypothesisKind.OBJECT_TRANSFORM:
                intents.append(AbstractIntent(ObjectOp.PROBE_TRANSFORM, priority=hypothesis.score, rationale=hypothesis.summary))
            elif hypothesis.kind == HypothesisKind.ALIGNMENT_GOAL:
                intents.append(AbstractIntent(ObjectOp.PROBE_ALIGNMENT, target=hypothesis.target, priority=hypothesis.score, rationale=hypothesis.summary))
        intents.append(AbstractIntent(ObjectOp.EXPLOIT_KNOWN_EFFECT, priority=0.1, rationale="reuse observed useful effects"))
        return sorted(intents, key=lambda item: item.priority, reverse=True)
