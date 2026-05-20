from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True)
class EnsembleBucket:
    labels: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.labels.values())

    @property
    def dominant_label(self) -> str:
        return self.labels.most_common(1)[0][0] if self.labels else "unknown"


class ActionEffectEnsemble:
    """Lightweight disagreement model inspired by Plan2Explore.

    Instead of a learned neural ensemble, we keep several predictors at different
    abstraction levels and estimate uncertainty from their disagreement and support.
    """

    def __init__(self) -> None:
        self.buckets: dict[str, EnsembleBucket] = {}

    def observe(
        self,
        *,
        action_key: str,
        family: str,
        previous_action_key: str | None,
        region_bias: str,
        mode_state: str,
        workspace_signature: str,
        transform_kind: str,
        interaction_hint: str,
        alignment_delta: float,
    ) -> None:
        label = self._label(transform_kind, interaction_hint, alignment_delta)
        for key in self._keys(
            action_key=action_key,
            family=family,
            previous_action_key=previous_action_key,
            region_bias=region_bias,
            mode_state=mode_state,
            workspace_signature=workspace_signature,
        ):
            self.buckets.setdefault(key, EnsembleBucket()).labels[label] += 1

    def uncertainty_score(
        self,
        *,
        action_key: str,
        family: str,
        previous_action_key: str | None,
        region_bias: str,
        mode_state: str,
        workspace_signature: str,
    ) -> float:
        keys = self._keys(
            action_key=action_key,
            family=family,
            previous_action_key=previous_action_key,
            region_bias=region_bias,
            mode_state=mode_state,
            workspace_signature=workspace_signature,
        )
        predictions: list[str] = []
        supports: list[int] = []
        for key in keys:
            bucket = self.buckets.get(key)
            if bucket is None or bucket.total <= 0:
                continue
            predictions.append(bucket.dominant_label)
            supports.append(bucket.total)
        if not predictions:
            return 1.0
        distinct_ratio = len(set(predictions)) / max(1, len(predictions))
        support = sum(supports) / max(1, len(supports))
        low_support_bonus = 1.0 / (1.0 + support)
        return max(0.0, min(1.0, 0.60 * distinct_ratio + 0.40 * low_support_bonus))

    def _keys(
        self,
        *,
        action_key: str,
        family: str,
        previous_action_key: str | None,
        region_bias: str,
        mode_state: str,
        workspace_signature: str,
    ) -> list[str]:
        prev = previous_action_key or "none"
        return [
            f"exact:{action_key}|prev={prev}|region={region_bias}|mode={mode_state}|ws={workspace_signature}",
            f"coarse:{family}|region={region_bias}|mode={mode_state}",
            f"family:{family}",
        ]

    def _label(self, transform_kind: str, interaction_hint: str, alignment_delta: float) -> str:
        if alignment_delta > 0.05:
            alignment_bucket = "align_up"
        elif alignment_delta < -0.05:
            alignment_bucket = "align_down"
        else:
            alignment_bucket = "align_flat"
        hint = interaction_hint if interaction_hint != "unknown" else "generic"
        return f"{transform_kind}|{hint}|{alignment_bucket}"
